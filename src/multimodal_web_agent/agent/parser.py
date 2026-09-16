"""Strict, full-string parser for the frozen Protocol-SFT action grammar."""

from __future__ import annotations

import re
from typing import Optional, Tuple

from .protocol_errors import ProtocolError
from .schema import ActionType, ParsedAction


_REASON_OPEN = "<reason>"
_REASON_CLOSE = "</reason>"
_ACTION_OPENS = ("<search>", "<text_search>", "<answer>")
_KNOWN_TAG_RE = re.compile(
    r"</?(?:reason|search|img|text_search|answer|information)(?:\s[^>]*)?>",
    re.IGNORECASE,
)
_ANY_TAG_RE = re.compile(r"<[^>]*>")


def _invalid(raw: str, error: ProtocolError) -> ParsedAction:
    return ParsedAction(False, None, None, None, error, raw)


def _tag_count(text: str, tag: str) -> int:
    return text.count(tag)


def _has_action_marker(text: str) -> bool:
    return any(
        marker in text
        for marker in (
            "<search>",
            "</search>",
            "<img>",
            "</img>",
            "<text_search>",
            "</text_search>",
            "<answer>",
            "</answer>",
        )
    )


def _extract_single(
    text: str,
    opening: str,
    closing: str,
) -> Optional[Tuple[str, int, int]]:
    if text.count(opening) != 1 or text.count(closing) != 1:
        return None
    start = text.index(opening)
    content_start = start + len(opening)
    end = text.index(closing, content_start)
    return text[content_start:end], start, end + len(closing)


def parse_action(raw_text: str) -> ParsedAction:
    """Parse exactly one reason followed by exactly one frozen action.

    Whitespace around and between the two elements is accepted. Any non-whitespace
    text outside those elements is rejected. The image-search syntax intentionally
    follows the frozen, XML-like ``<search><img></search>`` spelling.
    """

    raw = "" if raw_text is None else str(raw_text)
    text = raw.strip()
    if not text:
        return _invalid(raw, ProtocolError.EMPTY_OUTPUT)

    lowered = text.lower()
    if "<information" in lowered or "</information" in lowered:
        return _invalid(raw, ProtocolError.FORGED_INFORMATION)

    reason_open_count = _tag_count(text, _REASON_OPEN)
    reason_close_count = _tag_count(text, _REASON_CLOSE)
    if reason_open_count > 1 or reason_close_count > 1:
        return _invalid(raw, ProtocolError.MULTIPLE_REASONS)
    if reason_open_count == 0 and reason_close_count == 0:
        if "reason" in lowered and ("<" in text or ">" in text):
            return _invalid(raw, ProtocolError.INCOMPLETE_XML)
        return _invalid(raw, ProtocolError.MISSING_REASON)
    if reason_open_count != 1 or reason_close_count != 1:
        return _invalid(raw, ProtocolError.INCOMPLETE_XML)

    reason_data = _extract_single(text, _REASON_OPEN, _REASON_CLOSE)
    if reason_data is None:
        return _invalid(raw, ProtocolError.INCOMPLETE_XML)
    reason, reason_start, reason_end = reason_data
    if reason_start != 0:
        if text[:reason_start].strip():
            return _invalid(raw, ProtocolError.EXTRA_TEXT)
    if _has_action_marker(reason):
        return _invalid(raw, ProtocolError.NESTED_ACTION)
    if _ANY_TAG_RE.search(reason) or "<" in reason or ">" in reason:
        return _invalid(raw, ProtocolError.INVALID_XML)
    reason = reason.strip()
    if not reason:
        return _invalid(raw, ProtocolError.EMPTY_REASON)

    suffix = text[reason_end:]
    action_count = sum(suffix.count(marker) for marker in _ACTION_OPENS)
    action_count += sum(text[:reason_start].count(marker) for marker in _ACTION_OPENS)
    if action_count > 1:
        action_pairs = (
            ("<search>", "</search>"),
            ("<text_search>", "</text_search>"),
            ("<answer>", "</answer>"),
        )
        for opening, closing in action_pairs:
            start = suffix.find(opening)
            end = suffix.find(closing, start + len(opening)) if start >= 0 else -1
            if start >= 0 and end >= 0:
                interior = suffix[start + len(opening):end]
                if any(marker in interior for marker in _ACTION_OPENS):
                    return _invalid(raw, ProtocolError.NESTED_ACTION)
        return _invalid(raw, ProtocolError.MULTIPLE_ACTIONS)
    if action_count == 0:
        if any(name in suffix for name in ("search", "answer")) and "<" in suffix:
            return _invalid(raw, ProtocolError.INCOMPLETE_XML)
        if _ANY_TAG_RE.search(suffix):
            return _invalid(raw, ProtocolError.UNKNOWN_ACTION)
        return _invalid(raw, ProtocolError.UNKNOWN_ACTION)

    action_text = suffix.strip()
    action_type: Optional[ActionType] = None
    content: Optional[str] = None

    if "<search>" in suffix:
        action_type = ActionType.IMAGE_SEARCH
        if suffix.count("<search>") != 1 or suffix.count("</search>") != 1:
            return _invalid(raw, ProtocolError.INCOMPLETE_XML)
        if "<img>" not in suffix:
            return _invalid(raw, ProtocolError.INVALID_XML)
        if action_text != "<search><img></search>":
            remainder = (
                action_text.replace("<search>", "", 1)
                .replace("<img>", "", 1)
                .replace("</search>", "", 1)
            )
            if _has_action_marker(remainder):
                return _invalid(raw, ProtocolError.NESTED_ACTION)
            return _invalid(raw, ProtocolError.EXTRA_TEXT)
        content = "img"
    elif "<text_search>" in suffix:
        action_type = ActionType.TEXT_SEARCH
        extracted = _extract_single(suffix, "<text_search>", "</text_search>")
        if extracted is None:
            return _invalid(raw, ProtocolError.INCOMPLETE_XML)
        query, start, end = extracted
        if _has_action_marker(query):
            return _invalid(raw, ProtocolError.NESTED_ACTION)
        if _ANY_TAG_RE.search(query) or "<" in query or ">" in query:
            return _invalid(raw, ProtocolError.INVALID_XML)
        if suffix[:start].strip() or suffix[end:].strip():
            return _invalid(raw, ProtocolError.EXTRA_TEXT)
        query = query.strip()
        if not query:
            return _invalid(raw, ProtocolError.EMPTY_QUERY)
        content = query
    elif "<answer>" in suffix:
        action_type = ActionType.ANSWER
        extracted = _extract_single(suffix, "<answer>", "</answer>")
        if extracted is None:
            return _invalid(raw, ProtocolError.INCOMPLETE_XML)
        answer, start, end = extracted
        if _has_action_marker(answer):
            return _invalid(raw, ProtocolError.NESTED_ACTION)
        if _ANY_TAG_RE.search(answer) or "<" in answer or ">" in answer:
            return _invalid(raw, ProtocolError.INVALID_XML)
        if suffix[:start].strip() or suffix[end:].strip():
            return _invalid(raw, ProtocolError.EXTRA_TEXT)
        answer = answer.strip()
        if not answer:
            return _invalid(raw, ProtocolError.EMPTY_ANSWER)
        content = answer

    if action_type is None:
        return _invalid(raw, ProtocolError.UNKNOWN_ACTION)

    # Reject malformed or unknown tags that were not consumed by the exact grammar.
    consumed = text[:reason_start] + text[reason_end:]
    if action_type != ActionType.IMAGE_SEARCH:
        unknown_tags = [
            tag.group(0)
            for tag in _ANY_TAG_RE.finditer(consumed)
            if tag.group(0)
            not in {
                "<text_search>",
                "</text_search>",
                "<answer>",
                "</answer>",
            }
        ]
        if unknown_tags:
            return _invalid(raw, ProtocolError.INVALID_XML)

    return ParsedAction(True, action_type, reason, content, None, raw)


__all__ = ["parse_action"]
