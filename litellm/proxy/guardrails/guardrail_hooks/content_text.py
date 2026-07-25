"""Shared content<->text helpers for compression guardrails (headroom, compresr).

Compression services only transform plain-string message content: every
transform in the service pipeline gates on ``isinstance(content, str)`` and
silently skips the OpenAI list-of-parts shape. Guardrails that send messages
to such a service must collapse text-bearing part lists to strings on the way
out and write the returned text back into the original shape on the way in,
so non-text parts (images, audio) and part-level fields (``cache_control``)
survive compression.
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


def replace_text_in_content(content: object, new_text: str) -> object:
    """Write ``new_text`` back into a ``content`` value, preserving shape.

    ``str`` content is replaced directly. For list-of-parts content the first
    text part carries ``new_text``, later text parts are dropped, and
    non-text parts (images, audio, files) pass through untouched.
    """
    if isinstance(content, str):
        return new_text
    if isinstance(content, list):
        out: list[object] = []
        replaced = False
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                if not replaced:
                    out.append({**part, "text": new_text})
                    replaced = True
                continue
            out.append(part)
        if not replaced:
            out.insert(0, {"type": "text", "text": new_text})
        return out
    return new_text
