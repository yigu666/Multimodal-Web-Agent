from __future__ import annotations

import re
from typing import Any, Iterable, Mapping, Sequence

from .evidence_reachability import (
    answer_evidence_reachable,
    evidence_document_ids,
)
from .generic_terms import (
    GENERIC_QUERY_TERMS,
    QUESTION_WORDS,
    VISUAL_CATEGORY_TERMS,
    VISUAL_REFERENCE_TERMS,
    content_tokens,
    normalize_text,
    text_tokens,
)
from .rejection import SharedRejectionReason, reason_values
from .schema import ActionType, ActionValidation, VisibleEntity


def _question_core_tokens(question: str) -> tuple[str, ...]:
    return tuple(
        token
        for token in text_tokens(question)
        if token not in QUESTION_WORDS
    )


def compute_query_question_copy_ratio(question: str, query: str) -> float:
    query_tokens = text_tokens(query)
    if not query_tokens:
        return 0.0
    question_tokens = set(_question_core_tokens(question))
    return sum(token in question_tokens for token in query_tokens) / len(
        query_tokens
    )


def _visible_entities(
    query: str,
    visible_text_context: str,
) -> tuple[VisibleEntity, ...]:
    anchors = _retrieval_anchor_tokens(
        query=query,
        visible_text_context=visible_text_context,
    )
    entities = []
    for token in sorted(anchors):
        entities.append(
            VisibleEntity(
                value=token,
                provenance="visible_named_or_unique_entity",
                source_span=token,
                visible_in_current_text_state=True,
            )
        )
    return tuple(entities)


def is_unresolvable_visual_reference(
    query: str,
    visible_text_context: str,
) -> bool:
    tokens = set(text_tokens(query))
    if not tokens:
        return True
    substantive = set(content_tokens(query))
    visible = set(text_tokens(visible_text_context))
    anchors = _retrieval_anchor_tokens(
        query=query,
        visible_text_context=visible_text_context,
    )
    if _has_unresolved_visual_subject(
        query, has_retrieval_anchor=bool(anchors)
    ):
        return True
    has_visible_substantive = bool(substantive & visible)
    visual_count = len(tokens & VISUAL_REFERENCE_TERMS)
    return visual_count > 0 and not has_visible_substantive


def _query_answer_leaks(
    query: str,
    aliases: Sequence[str],
) -> bool:
    normalized = " %s " % normalize_text(query)
    return any(
        alias.strip()
        and (" %s " % normalize_text(alias)) in normalized
        for alias in aliases
    )


def _same_context_only(
    text_results: Iterable[Any] | None,
    source_context_document_ids: Iterable[str],
) -> bool:
    result_ids = set(evidence_document_ids(text_results))
    context_ids = {str(value) for value in source_context_document_ids}
    return bool(result_ids) and result_ids.issubset(context_ids)


def validate_text_search_action(
    *,
    query: str,
    visible_text_context: str,
    question: str,
    accepted_answer_aliases: tuple[str, ...],
    text_results: list[dict] | None,
    require_evidence_reachability: bool = True,
    maximum_question_copy_ratio_without_entity: float = 0.85,
    source_context_document_ids: Iterable[str] = (),
) -> ActionValidation:
    reasons = []
    query = str(query).strip()
    if not query:
        reasons.append(SharedRejectionReason.GENERIC_QUERY)
    if "<" in query or ">" in query:
        reasons.append(SharedRejectionReason.STRICT_PARSER_INVALID)
    if _query_answer_leaks(query, accepted_answer_aliases):
        reasons.append(SharedRejectionReason.QUERY_ANSWER_LEAK)

    query_tokens = set(text_tokens(query))
    visible_tokens = set(text_tokens(visible_text_context))
    substantive = set(content_tokens(query))
    entities = _visible_entities(query, visible_text_context)
    if substantive and not (substantive & visible_tokens):
        reasons.append(SharedRejectionReason.ENTITY_NOT_VISIBLE)
    if not entities:
        reasons.append(SharedRejectionReason.MISSING_VISIBLE_ENTITY)
    unresolved_visual = is_unresolvable_visual_reference(
        query, visible_text_context
    )
    if (
        not substantive
        or query_tokens.issubset(GENERIC_QUERY_TERMS)
        or (not entities and unresolved_visual)
    ):
        reasons.append(SharedRejectionReason.GENERIC_QUERY)
    if unresolved_visual:
        reasons.append(SharedRejectionReason.UNRESOLVABLE_VISUAL_REFERENCE)

    copy_ratio = compute_query_question_copy_ratio(question, query)
    if (
        copy_ratio > maximum_question_copy_ratio_without_entity
        and not entities
    ):
        reasons.append(SharedRejectionReason.QUESTION_COPY_QUERY)

    if require_evidence_reachability:
        if not text_results or not answer_evidence_reachable(
            accepted_answer_aliases, text_results
        ):
            reasons.append(SharedRejectionReason.UNREACHABLE_EVIDENCE)
        elif _same_context_only(
            text_results, source_context_document_ids
        ):
            reasons.append(
                SharedRejectionReason.SAME_CONTEXT_EVIDENCE_ONLY
            )

    return ActionValidation(
        action_type=ActionType.TEXT_SEARCH.value,
        executable=not reasons,
        reasons=reason_values(reasons),
        visible_entities=entities,
        evidence_document_ids=evidence_document_ids(text_results),
    )
RAW_TOKEN_RE = re.compile(r"[^\W_]+(?:['’-][^\W_]+)*", re.UNICODE)
DEICTIC_TERMS = {"this", "that", "these", "those"}
IMAGE_CONTAINER_TERMS = {"image", "photo", "picture"}
IDENTIFICATION_TERMS = {
    "breed", "company", "identify", "identification", "logo", "name",
    "represented", "symbol", "type",
}
NON_ENTITY_CAPITALIZED_TERMS = GENERIC_QUERY_TERMS | {
    "question", "information", "search", "results", "wikipedia",
}


def _proper_entity_tokens(value: str) -> set[str]:
    tokens = RAW_TOKEN_RE.findall(str(value))
    values = set()
    for token in tokens:
        normalized = normalize_text(token)
        if not normalized or normalized in NON_ENTITY_CAPITALIZED_TERMS:
            continue
        if token.isupper() and len(token) >= 2:
            values.add(normalized)
        elif token[:1].isupper() and any(
            character.islower() for character in token[1:]
        ):
            values.add(normalized)
        elif any(character.isdigit() for character in token) and (
            any(character.isalpha() for character in token)
            or not (
                token.isdigit()
                and len(token) == 4
                and 1800 <= int(token) <= 2100
            )
        ):
            values.add(normalized)
    return values


def _quoted_entity_tokens(value: str) -> set[str]:
    values = set()
    for match in re.finditer(r"[\"“”'‘’]([^\"“”'‘’]+)[\"“”'‘’]", str(value)):
        values.update(content_tokens(match.group(1)))
    return values


def _retrieval_anchor_tokens(
    *,
    query: str,
    visible_text_context: str,
) -> set[str]:
    query_tokens = set(text_tokens(query))
    visible_tokens = set(text_tokens(visible_text_context))
    explicit = (
        _proper_entity_tokens(visible_text_context)
        | _quoted_entity_tokens(visible_text_context)
    )
    return query_tokens & visible_tokens & explicit


def infer_query_anchor_provenance(
    *,
    query: str,
    question: str,
    information: str = "",
) -> Mapping[str, Any] | None:
    """Return a visible named/unique query anchor without using future data."""
    query_normalized = normalize_text(query)
    if not query_normalized:
        return None
    for source, visible in (("question", question), ("information", information)):
        query_anchors = (
            _proper_entity_tokens(query) | _quoted_entity_tokens(query)
        )
        visible_anchors = (
            _proper_entity_tokens(visible) | _quoted_entity_tokens(visible)
        )
        shared = query_anchors & visible_anchors
        if not shared:
            continue
        raw_tokens = RAW_TOKEN_RE.findall(str(query))
        named_phrases = []
        current_phrase = []
        for token in raw_tokens:
            is_named_shape = (
                token[:1].isupper()
                or token.isupper()
                or any(character.isdigit() for character in token)
            )
            if is_named_shape:
                current_phrase.append(token)
            elif current_phrase:
                named_phrases.append(" ".join(current_phrase))
                current_phrase = []
        if current_phrase:
            named_phrases.append(" ".join(current_phrase))
        visible_normalized = " %s " % normalize_text(visible)
        visible_phrases = [
            phrase
            for phrase in named_phrases
            if (" %s " % normalize_text(phrase)) in visible_normalized
            and set(text_tokens(phrase)) & shared
        ]
        runs = []
        current = []
        for token in raw_tokens:
            if normalize_text(token) in shared:
                current.append(token)
            elif current:
                runs.append(current)
                current = []
        if current:
            runs.append(current)
        chosen = (
            max(visible_phrases, key=lambda value: len(text_tokens(value)))
            if visible_phrases
            else (" ".join(max(runs, key=len)) if runs else sorted(shared)[0])
        )
        normalized = normalize_text(chosen)
        quoted_phrases = {
            normalize_text(match.group(1))
            for match in re.finditer(
                r"[\"“”'‘’]([^\"“”'‘’]+)[\"“”'‘’]", str(query)
            )
        }
        if normalized in quoted_phrases:
            anchor_type = "quoted_entity"
        elif any(character.isdigit() for character in chosen):
            anchor_type = "numeric_identifier"
        elif chosen.isupper() and len(chosen) >= 2:
            anchor_type = "uppercase_acronym"
        else:
            anchor_type = "named_entity"
        return {
            "query_anchor": chosen,
            "query_anchor_type": anchor_type,
            "query_anchor_provenance": {
                "source": source,
                "source_span": chosen,
                "visible_in_current_text_state": True,
            },
        }
    return None


def _has_unresolved_visual_subject(
    query: str,
    *,
    has_retrieval_anchor: bool,
) -> bool:
    tokens = set(text_tokens(query))
    if has_retrieval_anchor:
        return False
    if tokens & DEICTIC_TERMS:
        return True
    if tokens & IMAGE_CONTAINER_TERMS:
        return True
    if tokens & IDENTIFICATION_TERMS and (
        tokens & (VISUAL_REFERENCE_TERMS | VISUAL_CATEGORY_TERMS)
    ):
        return True
    normalized = " %s " % normalize_text(query)
    patterns = (
        " name of ",
        " breed of ",
        " breed is ",
        " type of ",
        " company represented by ",
        " represented by ",
    )
    return any(pattern in normalized for pattern in patterns)
