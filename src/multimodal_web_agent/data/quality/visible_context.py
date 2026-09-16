from __future__ import annotations

from typing import Any, Iterable, Mapping


def _message_value(message: Any, key: str, default: str = "") -> str:
    if isinstance(message, Mapping):
        return str(message.get(key, default))
    return str(getattr(message, key, default))


def build_visible_text_context(
    *,
    question: str,
    history_messages: Iterable[Any],
) -> str:
    """Return only text already exposed to the agent at the current state."""
    parts = [str(question).strip()]
    for message in history_messages:
        role = _message_value(message, "role").casefold()
        content = _message_value(message, "content").strip()
        if not content:
            continue
        if role in {"assistant", "tool"}:
            parts.append(content)
    return "\n".join(part for part in parts if part)


def visible_information_text(history_messages: Iterable[Any]) -> str:
    return "\n".join(
        _message_value(message, "content").strip()
        for message in history_messages
        if _message_value(message, "role").casefold() == "tool"
        and _message_value(message, "content").strip()
    )
