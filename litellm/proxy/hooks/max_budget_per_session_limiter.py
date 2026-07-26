"""
Per-Session Budget Limiter for LiteLLM Proxy.

Enforces a dollar-amount cap per session (identified by `session_id` /
`x-litellm-trace-id`). Admission atomically reserves the request's estimated
worst-case cost against the session counter, so concurrent requests for the
same session cannot all read the same below-budget value and pass. The
reservation is reconciled to the actual response cost once the call finishes.
When the spend accumulated before a request (committed spend plus in-flight
reservations) reaches `max_budget_per_session` (configured in agent
litellm_params), that request receives a 429.

Note: trace-id enforcement (require_trace_id_on_calls_by_agent) is handled
separately in auth_checks.py at the agent level, not in this hook.

Works across multiple proxy instances via DualCache (in-memory + Redis).
Follows the same pattern as max_iterations_limiter.py.
"""

import os
from typing import TYPE_CHECKING, Any, Optional, Union

from litellm import DualCache
from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.exceptions import RateLimitType
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.proxy_rate_limit_error import ProxyRateLimitError
from litellm.proxy.hooks.rate_limiter_utils import resolve_llm_provider_for_rate_limit

if TYPE_CHECKING:
    from litellm.proxy.utils import InternalUsageCache as _InternalUsageCache

    InternalUsageCache = _InternalUsageCache
else:
    InternalUsageCache = Any


# Redis Lua script for atomic float increment with TTL.
# INCRBYFLOAT returns the new value as a string.
# Only sets EXPIRE on first call (when prior value was nil).
MAX_BUDGET_SESSION_INCREMENT_SCRIPT = """
local key = KEYS[1]
local amount = ARGV[1]
local ttl = tonumber(ARGV[2])

local existed = redis.call('EXISTS', key)
local new_val = redis.call('INCRBYFLOAT', key, amount)
if existed == 0 then
    redis.call('EXPIRE', key, ttl)
end

return new_val
"""

# Default TTL for session budget counters (1 hour)
DEFAULT_MAX_BUDGET_PER_SESSION_TTL = 3600

RESERVED_COST_KEY = "litellm_session_budget_reserved_cost"
RESERVED_SESSION_ID_KEY = "litellm_session_budget_reserved_session_id"
RESERVATION_RELEASED_KEY = "litellm_session_budget_reservation_released"

_METADATA_CHANNELS = ("metadata", "litellm_metadata")


def _to_float(value: Any) -> float:
    return float(value) if isinstance(value, (int, float)) else 0.0


class _PROXY_MaxBudgetPerSessionHandler(CustomLogger):
    """
    Pre-call hook that enforces max_budget_per_session.

    Configuration (set in agent litellm_params):
        - max_budget_per_session: dollar cap per session_id

    Cache key pattern:
        {session_budget:<session_id>}:spend
    """

    def __init__(self, internal_usage_cache: InternalUsageCache):
        self.internal_usage_cache = internal_usage_cache
        self.ttl = int(
            os.getenv(
                "LITELLM_MAX_BUDGET_PER_SESSION_TTL",
                DEFAULT_MAX_BUDGET_PER_SESSION_TTL,
            )
        )

        if self.internal_usage_cache.dual_cache.redis_cache is not None:
            self.increment_script = self.internal_usage_cache.dual_cache.redis_cache.async_register_script(
                MAX_BUDGET_SESSION_INCREMENT_SCRIPT
            )
        else:
            self.increment_script = None

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: str,
    ) -> Optional[Union[Exception, str, dict]]:
        """
        Before each LLM call, check if max_budget_per_session is set and
        whether accumulated spend exceeds the budget (429 if so).
        """
        max_budget = self._get_max_budget_per_session(user_api_key_dict)

        session_id = self._get_session_id(data)

        if max_budget is None or session_id is None:
            return None

        max_budget = float(max_budget)
        cache_key = self._make_cache_key(session_id)
        reserved_cost = self._estimate_request_cost(data=data, user_api_key_dict=user_api_key_dict)

        if reserved_cost is None:
            spend_before = await self._get_current_spend(cache_key)
        else:
            spend_before = await self._increment_spend(cache_key, reserved_cost) - reserved_cost

        verbose_proxy_logger.debug(
            "MaxBudgetPerSessionHandler: session_id=%s, spend=%.4f, reserved=%s, max=%.2f",
            session_id,
            spend_before,
            reserved_cost,
            max_budget,
        )

        if spend_before >= max_budget:
            if reserved_cost is not None:
                await self._increment_spend(cache_key, -reserved_cost)
            resolved_model, llm_provider = resolve_llm_provider_for_rate_limit(data.get("model") if data else None)
            raise ProxyRateLimitError(
                detail=(
                    f"Session budget exceeded for session {session_id}. "
                    f"Current spend: ${spend_before:.4f}, "
                    f"max_budget_per_session: ${max_budget:.2f}."
                ),
                rate_limit_type=RateLimitType.BUDGET,
                model=resolved_model,
                llm_provider=llm_provider,
            )

        if reserved_cost is not None:
            self._stash_reservation(data=data, session_id=session_id, reserved_cost=reserved_cost)

        return None

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        """
        After a successful LLM call, settle the session spend against the response cost.
        """
        try:
            response_cost = _to_float(kwargs.get("response_cost"))
            if await self._reconcile_reservation(data=kwargs, actual_cost=response_cost):
                return

            litellm_params = kwargs.get("litellm_params") or {}
            metadata = litellm_params.get("metadata") or {}
            session_id = metadata.get("session_id")
            if session_id is None:
                return

            agent_id = metadata.get("agent_id")
            if agent_id is None:
                return

            from litellm.proxy.agent_endpoints.agent_registry import (
                global_agent_registry,
            )

            agent = global_agent_registry.get_agent_by_id(agent_id=str(agent_id))
            if agent is None:
                return

            agent_litellm_params = agent.litellm_params or {}
            max_budget = agent_litellm_params.get("max_budget_per_session")
            if max_budget is None:
                return

            if response_cost <= 0:
                return

            cache_key = self._make_cache_key(str(session_id))
            await self._increment_spend(cache_key, response_cost)

            verbose_proxy_logger.debug(
                "MaxBudgetPerSessionHandler: incremented session %s spend by %.6f",
                session_id,
                response_cost,
            )
        except Exception as e:
            verbose_proxy_logger.warning(
                "MaxBudgetPerSessionHandler: error in async_log_success_event: %s",
                str(e),
            )

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        """
        Release the admission reservation when the LLM call fails.
        """
        await self._reconcile_reservation(data=kwargs, actual_cost=_to_float(kwargs.get("response_cost")))

    async def async_post_call_failure_hook(
        self,
        request_data: dict,
        original_exception: Exception,
        user_api_key_dict: UserAPIKeyAuth,
        traceback_str: str | None = None,
    ) -> None:
        """
        Release the admission reservation when the request fails before or
        during the LLM call (proxy-level rejection, provider error, timeout).
        """
        await self._reconcile_reservation(data=request_data, actual_cost=0.0)

    def _estimate_request_cost(self, data: dict, user_api_key_dict: UserAPIKeyAuth) -> float | None:
        """
        Worst-case cost of this request, used as the admission reservation.

        Returns None when the model has no token pricing to estimate from; the
        caller then falls back to read-only enforcement of committed spend.
        """
        from litellm.proxy.proxy_server import llm_router
        from litellm.proxy.spend_tracking.budget_reservation import (
            estimate_request_max_cost,
        )

        estimated_cost = estimate_request_max_cost(
            request_body=data,
            route=user_api_key_dict.request_route or "",
            llm_router=llm_router,
        )
        if estimated_cost is None or estimated_cost <= 0:
            return None
        return float(estimated_cost)

    @staticmethod
    def _stash_reservation(data: dict, session_id: str, reserved_cost: float) -> None:
        """
        Record the reservation on every metadata channel a later callback can
        read it from. ``data["metadata"]`` and ``kwargs["litellm_params"]["metadata"]``
        are the same dict by the time logging callbacks fire, so writes here reach
        both the success and the failure path.
        """
        channel_dicts = tuple(
            channel_dict for channel in _METADATA_CHANNELS if isinstance((channel_dict := data.get(channel)), dict)
        )
        if not channel_dicts:
            data["metadata"] = {}
            channel_dicts = (data["metadata"],)
        for channel_dict in channel_dicts:
            channel_dict[RESERVED_COST_KEY] = reserved_cost
            channel_dict[RESERVED_SESSION_ID_KEY] = session_id

    @staticmethod
    def _metadata_dicts(data: Any) -> tuple[dict[str, Any], ...]:
        if not isinstance(data, dict):
            return ()
        litellm_params = data.get("litellm_params")
        standard_logging_object = data.get("standard_logging_object")
        candidates = (
            *(data.get(channel) for channel in _METADATA_CHANNELS),
            *(
                (litellm_params.get(channel) for channel in _METADATA_CHANNELS)
                if isinstance(litellm_params, dict)
                else ()
            ),
            standard_logging_object.get("metadata") if isinstance(standard_logging_object, dict) else None,
        )
        return tuple(candidate for candidate in candidates if isinstance(candidate, dict))

    @classmethod
    def _get_reservation(cls, data: Any) -> tuple[str, float] | None:
        """``(session_id, reserved_cost)`` for an unreleased reservation, else None."""
        metadata_dicts = cls._metadata_dicts(data)
        if any(metadata.get(RESERVATION_RELEASED_KEY) for metadata in metadata_dicts):
            return None
        for metadata in metadata_dicts:
            session_id = metadata.get(RESERVED_SESSION_ID_KEY)
            reserved_cost = metadata.get(RESERVED_COST_KEY)
            if isinstance(session_id, str) and isinstance(reserved_cost, (int, float)):
                return session_id, float(reserved_cost)
        return None

    @classmethod
    def _mark_reservation_released(cls, data: Any) -> None:
        for metadata in cls._metadata_dicts(data):
            metadata[RESERVATION_RELEASED_KEY] = True

    async def _reconcile_reservation(self, data: Any, actual_cost: float) -> bool:
        """
        Settle this request's admission reservation to its actual cost.

        Returns False when the request never reserved (no estimate was
        available), so the caller can fall back to plain post-hoc accounting.
        Releasing is idempotent: the first caller marks the reservation
        released, so sibling callbacks for the same request no-op.
        """
        reservation = self._get_reservation(data)
        if reservation is None:
            return False

        session_id, reserved_cost = reservation
        self._mark_reservation_released(data)

        delta = actual_cost - reserved_cost
        if delta != 0:
            await self._increment_spend(self._make_cache_key(session_id), delta)

        verbose_proxy_logger.debug(
            "MaxBudgetPerSessionHandler: reconciled session %s reservation %.6f to actual %.6f",
            session_id,
            reserved_cost,
            actual_cost,
        )
        return True

    def _get_session_id(self, data: dict) -> Optional[str]:
        """Extract session_id from request metadata."""
        metadata = data.get("metadata") or {}
        session_id = metadata.get("session_id")
        if session_id is not None:
            return str(session_id)

        litellm_metadata = data.get("litellm_metadata") or {}
        session_id = litellm_metadata.get("session_id")
        if session_id is not None:
            return str(session_id)

        return None

    def _get_max_budget_per_session(self, user_api_key_dict: UserAPIKeyAuth) -> Optional[float]:
        """Extract max_budget_per_session from agent litellm_params."""
        agent_id = user_api_key_dict.agent_id
        if agent_id is None:
            return None

        from litellm.proxy.agent_endpoints.agent_registry import global_agent_registry

        agent = global_agent_registry.get_agent_by_id(agent_id=agent_id)
        if agent is None:
            return None

        litellm_params = agent.litellm_params or {}
        max_budget = litellm_params.get("max_budget_per_session")
        if max_budget is not None:
            return float(max_budget)
        return None

    def _make_cache_key(self, session_id: str) -> str:
        return f"{{session_budget:{session_id}}}:spend"

    async def _get_current_spend(self, cache_key: str) -> float:
        """Read current accumulated spend for a session."""
        if self.internal_usage_cache.dual_cache.redis_cache is not None:
            try:
                result = await self.internal_usage_cache.dual_cache.redis_cache.async_get_cache(key=cache_key)
                if result is not None:
                    return float(result)
                return 0.0
            except Exception as e:
                verbose_proxy_logger.warning(
                    "MaxBudgetPerSessionHandler: Redis GET failed, falling back to in-memory: %s",
                    str(e),
                )

        result = await self.internal_usage_cache.async_get_cache(
            key=cache_key,
            litellm_parent_otel_span=None,
            local_only=True,
        )
        if result is not None:
            return float(result)
        return 0.0

    async def _increment_spend(self, cache_key: str, amount: float) -> float:
        """Atomically increment the session spend and return the new value."""
        if self.increment_script is not None:
            try:
                result = await self.increment_script(
                    keys=[cache_key],
                    args=[str(amount), self.ttl],
                )
                return float(result)
            except Exception as e:
                verbose_proxy_logger.warning(
                    "MaxBudgetPerSessionHandler: Redis INCRBYFLOAT failed, falling back to in-memory: %s",
                    str(e),
                )

        return await self._in_memory_increment_spend(cache_key, amount)

    async def _in_memory_increment_spend(self, cache_key: str, amount: float) -> float:
        current = await self.internal_usage_cache.async_get_cache(
            key=cache_key,
            litellm_parent_otel_span=None,
            local_only=True,
        )
        new_value = (float(current) if current is not None else 0.0) + amount
        await self.internal_usage_cache.async_set_cache(
            key=cache_key,
            value=new_value,
            ttl=self.ttl,
            litellm_parent_otel_span=None,
            local_only=True,
        )
        return new_value
