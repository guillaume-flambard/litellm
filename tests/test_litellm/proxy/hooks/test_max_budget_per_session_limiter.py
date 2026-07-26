"""
Unit Tests for the per-session budget limiter for the proxy.

Tests that session-scoped budget tracking works correctly:
- Enforces max_budget_per_session per session_id (read from agent litellm_params)
- Different sessions have independent budgets
- Requests under budget pass through
- Requests without agent_id pass through
- Concurrent requests for one session cannot all be admitted against the same
  below-budget read (the admission reservation)
"""

import asyncio
from unittest.mock import patch

import pytest
from fastapi import HTTPException

from litellm.caching.caching import DualCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.hooks.max_budget_per_session_limiter import (
    _PROXY_MaxBudgetPerSessionHandler,
)
from litellm.proxy.utils import InternalUsageCache
from litellm.types.agents import AgentResponse

# gpt-4o-mini is priced in the bundled cost map, so the limiter can estimate a
# worst-case cost for this body and reserve it at admission.
PRICED_REQUEST_BODY = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]}


def _make_mock_agent(max_budget_per_session: float) -> AgentResponse:
    return AgentResponse(
        agent_id="agent-budget-123",
        agent_name="budget-agent",
        litellm_params={"max_budget_per_session": max_budget_per_session},
        agent_card_params={"name": "budget-agent", "version": "1.0.0"},
    )


def _make_handler() -> _PROXY_MaxBudgetPerSessionHandler:
    return _PROXY_MaxBudgetPerSessionHandler(internal_usage_cache=InternalUsageCache(DualCache()))


def _make_request(session_id: str) -> dict:
    return {**PRICED_REQUEST_BODY, "metadata": {"session_id": session_id}}


@pytest.mark.asyncio
async def test_budget_per_session_under_budget_passes():
    """
    Requests under budget should pass through without error.
    """
    local_cache = DualCache()
    handler = _PROXY_MaxBudgetPerSessionHandler(
        internal_usage_cache=InternalUsageCache(local_cache),
    )
    user_api_key_dict = UserAPIKeyAuth(
        api_key="sk-test-key-budget",
        agent_id="agent-budget-123",
    )

    mock_agent = _make_mock_agent(max_budget_per_session=5.0)

    with patch(
        "litellm.proxy.agent_endpoints.agent_registry.global_agent_registry"
    ) as mock_registry:
        mock_registry.get_agent_by_id.return_value = mock_agent

        result = await handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=local_cache,
            data={"metadata": {"session_id": "session-budget-1"}},
            call_type="",
        )
        assert result is None


@pytest.mark.asyncio
async def test_budget_per_session_exceeds_budget():
    """
    After accumulating spend beyond max_budget_per_session, the next
    pre-call check should raise 429.
    """
    local_cache = DualCache()
    handler = _PROXY_MaxBudgetPerSessionHandler(
        internal_usage_cache=InternalUsageCache(local_cache),
    )
    user_api_key_dict = UserAPIKeyAuth(
        api_key="sk-test-key-budget",
        agent_id="agent-budget-123",
    )

    session_id = "session-over-budget"
    cache_key = handler._make_cache_key(session_id)
    await handler._increment_spend(cache_key, 1.50)

    mock_agent = _make_mock_agent(max_budget_per_session=1.0)

    with patch(
        "litellm.proxy.agent_endpoints.agent_registry.global_agent_registry"
    ) as mock_registry:
        mock_registry.get_agent_by_id.return_value = mock_agent

        with pytest.raises(HTTPException) as exc_info:
            await handler.async_pre_call_hook(
                user_api_key_dict=user_api_key_dict,
                cache=local_cache,
                data={"metadata": {"session_id": session_id}},
                call_type="",
            )
        assert exc_info.value.status_code == 429
        assert "budget exceeded" in str(exc_info.value.detail).lower()


@pytest.mark.asyncio
async def test_budget_per_session_independent_sessions():
    """
    Different session_ids have independent budget counters.
    Exhausting session A does not affect session B.
    """
    local_cache = DualCache()
    handler = _PROXY_MaxBudgetPerSessionHandler(
        internal_usage_cache=InternalUsageCache(local_cache),
    )
    user_api_key_dict = UserAPIKeyAuth(
        api_key="sk-test-key-budget",
        agent_id="agent-budget-123",
    )

    cache_key_a = handler._make_cache_key("session-A")
    await handler._increment_spend(cache_key_a, 3.0)

    mock_agent = _make_mock_agent(max_budget_per_session=2.0)

    with patch(
        "litellm.proxy.agent_endpoints.agent_registry.global_agent_registry"
    ) as mock_registry:
        mock_registry.get_agent_by_id.return_value = mock_agent

        # Session A should be blocked
        with pytest.raises(HTTPException) as exc_info:
            await handler.async_pre_call_hook(
                user_api_key_dict=user_api_key_dict,
                cache=local_cache,
                data={"metadata": {"session_id": "session-A"}},
                call_type="",
            )
        assert exc_info.value.status_code == 429

        # Session B should still pass
        result = await handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=local_cache,
            data={"metadata": {"session_id": "session-B"}},
            call_type="",
        )
        assert result is None


@pytest.mark.asyncio
async def test_no_agent_id_passes():
    """
    When no agent_id is set on the key, all requests pass through.
    """
    local_cache = DualCache()
    handler = _PROXY_MaxBudgetPerSessionHandler(
        internal_usage_cache=InternalUsageCache(local_cache),
    )
    user_api_key_dict = UserAPIKeyAuth(
        api_key="sk-test-key-no-agent",
    )

    result = await handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=local_cache,
        data={"metadata": {"session_id": "any-session"}},
        call_type="",
    )
    assert result is None


@pytest.mark.asyncio
async def test_concurrent_requests_cannot_all_be_admitted_against_same_spend_read():
    """
    Regression test for the check-then-increment race: two requests that reach
    the pre-call hook before either has recorded spend must not both be
    admitted when their combined cost would blow the budget.

    The estimated worst-case cost of the request body below (~$0.0098) already
    exceeds the $0.005 budget, so exactly one of the two concurrent requests
    may be admitted.
    """
    handler = _make_handler()
    user_api_key_dict = UserAPIKeyAuth(api_key="sk-test-key-budget", agent_id="agent-budget-123")
    session_id = "session-concurrent"

    with patch("litellm.proxy.agent_endpoints.agent_registry.global_agent_registry") as mock_registry:
        mock_registry.get_agent_by_id.return_value = _make_mock_agent(max_budget_per_session=0.005)

        results = await asyncio.gather(
            *(
                handler.async_pre_call_hook(
                    user_api_key_dict=user_api_key_dict,
                    cache=handler.internal_usage_cache.dual_cache,
                    data=_make_request(session_id),
                    call_type="completion",
                )
                for _ in range(2)
            ),
            return_exceptions=True,
        )

    admitted = [result for result in results if not isinstance(result, Exception)]
    rejected = [result for result in results if isinstance(result, HTTPException)]
    assert len(admitted) == 1
    assert len(rejected) == 1
    assert rejected[0].status_code == 429


@pytest.mark.asyncio
async def test_reservation_is_reconciled_to_actual_cost_on_success():
    """
    The admission reservation is provisional: once the call finishes, the
    session counter must hold the actual response cost, not the estimate.
    """
    handler = _make_handler()
    user_api_key_dict = UserAPIKeyAuth(api_key="sk-test-key-budget", agent_id="agent-budget-123")
    session_id = "session-reconcile"
    data = _make_request(session_id)

    with patch("litellm.proxy.agent_endpoints.agent_registry.global_agent_registry") as mock_registry:
        mock_registry.get_agent_by_id.return_value = _make_mock_agent(max_budget_per_session=0.005)

        await handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=handler.internal_usage_cache.dual_cache,
            data=data,
            call_type="completion",
        )
        reserved_spend = await handler._get_current_spend(handler._make_cache_key(session_id))
        assert reserved_spend > 0.005

        await handler.async_log_success_event(
            kwargs={"litellm_params": {"metadata": data["metadata"]}, "response_cost": 0.001},
            response_obj=None,
            start_time=None,
            end_time=None,
        )

    assert await handler._get_current_spend(handler._make_cache_key(session_id)) == pytest.approx(0.001)


@pytest.mark.asyncio
async def test_reservation_is_released_when_request_fails():
    """A failed request must not leave its reservation pinned on the counter."""
    handler = _make_handler()
    user_api_key_dict = UserAPIKeyAuth(api_key="sk-test-key-budget", agent_id="agent-budget-123")
    session_id = "session-failure"
    data = _make_request(session_id)

    with patch("litellm.proxy.agent_endpoints.agent_registry.global_agent_registry") as mock_registry:
        mock_registry.get_agent_by_id.return_value = _make_mock_agent(max_budget_per_session=0.005)

        await handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=handler.internal_usage_cache.dual_cache,
            data=data,
            call_type="completion",
        )

    await handler.async_post_call_failure_hook(
        request_data=data,
        original_exception=Exception("boom"),
        user_api_key_dict=user_api_key_dict,
    )
    # Both failure paths fire for the same request; releasing must be idempotent.
    await handler.async_log_failure_event(
        kwargs={"litellm_params": {"metadata": data["metadata"]}},
        response_obj=None,
        start_time=None,
        end_time=None,
    )

    assert await handler._get_current_spend(handler._make_cache_key(session_id)) == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_unpriced_model_falls_back_to_post_hoc_accounting():
    """
    Without a cost estimate there is nothing to reserve, so the limiter keeps
    enforcing against committed spend and keeps accumulating it after the call.
    """
    handler = _make_handler()
    user_api_key_dict = UserAPIKeyAuth(api_key="sk-test-key-budget", agent_id="agent-budget-123")
    session_id = "session-unpriced"
    data = {
        "model": "model-with-no-pricing-in-cost-map",
        "messages": [{"role": "user", "content": "hi"}],
        "metadata": {"session_id": session_id, "agent_id": "agent-budget-123"},
    }

    with patch("litellm.proxy.agent_endpoints.agent_registry.global_agent_registry") as mock_registry:
        mock_registry.get_agent_by_id.return_value = _make_mock_agent(max_budget_per_session=0.005)

        await handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=handler.internal_usage_cache.dual_cache,
            data=data,
            call_type="completion",
        )
        assert await handler._get_current_spend(handler._make_cache_key(session_id)) == pytest.approx(0.0)

        await handler.async_log_success_event(
            kwargs={"litellm_params": {"metadata": data["metadata"]}, "response_cost": 0.006},
            response_obj=None,
            start_time=None,
            end_time=None,
        )
        assert await handler._get_current_spend(handler._make_cache_key(session_id)) == pytest.approx(0.006)

        with pytest.raises(HTTPException) as exc_info:
            await handler.async_pre_call_hook(
                user_api_key_dict=user_api_key_dict,
                cache=handler.internal_usage_cache.dual_cache,
                data=data,
                call_type="completion",
            )
        assert exc_info.value.status_code == 429
