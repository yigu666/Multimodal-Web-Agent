from __future__ import annotations

import re
import string
import unicodedata
from typing import List, Optional, Sequence

from .answer_normalizer import normalize_answer
from .relation_mapper import RelationMatch


_XML_RE = re.compile(r"<[^>]*>")
_TOKEN_RE = re.compile(r"[^\W_]+(?:['’-][^\W_]+)?", re.UNICODE)
_LEADING_QUESTION_WORDS = {
    "what", "which", "who", "whom", "whose", "where", "when", "why", "how",
}
_VISUAL_FILLERS = {
    "this", "that",
}
_QUESTION_FUNCTION_WORDS = _LEADING_QUESTION_WORDS | {
    "am", "are", "can", "could", "did", "do", "does", "is", "was", "were", "would",
}


def _strip_xml_like_markup(value: str) -> str:
    without_tags = _XML_RE.sub(" ", unicodedata.normalize("NFKC", value))
    return without_tags.replace("<", " ").replace(">", " ")


def query_tokens(text: str) -> List[str]:
    return _TOKEN_RE.findall(unicodedata.normalize("NFKC", text))


def build_bootstrap_query(
    question: str,
    *,
    visible_context: Optional[str] = None,
) -> str:
    """Build a deterministic query from currently visible model input only."""

    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be non-empty")

    question_parts = query_tokens(_strip_xml_like_markup(question))
    visible_parts = (
        query_tokens(_strip_xml_like_markup(visible_context))
        if visible_context and visible_context.strip()
        else []
    )
    parts = []
    seen = set()
    for token in question_parts + visible_parts:
        lowered = token.casefold()
        if lowered in _QUESTION_FUNCTION_WORDS or lowered in _VISUAL_FILLERS:
            continue
        if lowered in seen:
            continue
        seen.add(lowered)
        parts.append(lowered)

    if len(parts) < 3:
        parts = []
        seen.clear()
        for token in question_parts:
            lowered = token.casefold()
            if lowered in _LEADING_QUESTION_WORDS or lowered in seen:
                continue
            seen.add(lowered)
            parts.append(lowered)
    if len(parts) < 3:
        parts.extend(["factual", "information"][: 3 - len(parts)])
    parts = parts[:32]
    query = " ".join(parts).strip()
    if not query or _XML_RE.search(query):
        raise ValueError("could not build a valid query")
    return query


def build_image_context_query(
    question: str,
    identified_entity: str,
    relation: RelationMatch,
) -> str:
    """Build a query only from the question relation and a visible image entity."""

    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be non-empty")
    if not relation.matched or not relation.query_prefix.strip():
        raise ValueError("question relation is not mapped")
    if not isinstance(identified_entity, str) or not identified_entity.strip():
        raise ValueError("identified entity must be non-empty")
    if _XML_RE.search(identified_entity) or "<" in identified_entity or ">" in identified_entity:
        raise ValueError("identified entity contains XML-like markup")
    entity = " ".join(query_tokens(identified_entity))
    prefix = " ".join(query_tokens(relation.query_prefix)).casefold()
    query = (prefix + " " + entity).strip()
    errors = query_contract_errors(query, 3, 32)
    if errors:
        raise ValueError("invalid image-context query: %s" % ", ".join(errors))
    return query


def contains_answer_leak(
    query: str,
    accepted_answers: Sequence[str],
) -> bool:
    def leak_normalize(value: str) -> str:
        spaced = "".join(" " if character in string.punctuation else character for character in value)
        return normalize_answer(spaced)

    query_normalized = leak_normalize(query)
    normalized_query = " %s " % query_normalized
    compact_query = query_normalized.replace(" ", "")
    for answer in accepted_answers:
        normalized_answer = leak_normalize(answer)
        if not normalized_answer:
            continue
        if " %s " % normalized_answer in normalized_query:
            return True
        if any(ord(character) > 127 for character in normalized_answer):
            if normalized_answer in compact_query:
                return True
    return False


def query_contract_errors(query: str, min_tokens: int = 3, max_tokens: int = 32) -> List[str]:
    errors = []
    count = len(query_tokens(query))
    if not query.strip():
        errors.append("empty_query")
    if _XML_RE.search(query):
        errors.append("query_contains_xml")
    if count < min_tokens:
        errors.append("query_too_short")
    if count > max_tokens:
        errors.append("query_too_long")
    return errors
