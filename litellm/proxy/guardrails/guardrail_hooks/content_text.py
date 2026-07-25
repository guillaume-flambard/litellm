"""Shared content-to-text helper for compression guardrails (headroom, compresr).

Compression services only transform plain-string message content: every
transform in the service pipeline gates on ``isinstance(content, str)`` and
silently skips the OpenAI list-of-parts shape. Guardrails that send messages
to such a service collapse text-bearing part lists to strings with this
helper; how the rewritten text is written back is each guardrail's own
policy.
"""


def content_to_text(content: object) -> str:
    """Collapse a message ``content`` (str or list-of-parts) to plain text.

    For the multimodal list shape, joins ``{type: "text", text: ...}`` parts
    with blank-line separators; non-text parts are ignored.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n\n".join(parts)
    return ""
