from __future__ import annotations

from collections import Counter
import re
from typing import Any, Mapping, Sequence

from multimodal_web_agent.evaluation.unified_agent.answer_metrics import (
    normalize_answer,
)

from .answer_equivalence import deterministic_alias_equivalence


_RELATIONS = {
    "time": re.compile(r"\b(when|what year|date|how old|centur(?:y|ies))\b", re.I),
    "place": re.compile(r"\b(where|which country|which continent|located)\b", re.I),
    "person": re.compile(r"\b(who|whose|founder|founded|designed|architect)\b", re.I),
    "quantity": re.compile(r"\b(how many|how much|number|percent|percentage)\b", re.I),
    "diet": re.compile(r"\b(eat|diet|food|feed)\b", re.I),
    "material": re.compile(r"\b(material|made of|constructed from)\b", re.I),
    "purpose": re.compile(r"\b(used for|use of|purpose|known for|depict)\b", re.I),
    "type": re.compile(r"\b(what type|what kind|classification|species)\b", re.I),
    "attribute": re.compile(
        r"\b(what does|what is|which group|conditions|preparation|relative|"
        r"damaged|closed|opened|started|finished)\b",
        re.I,
    ),
}
_ENTITY_QUESTION = re.compile(
    r"\b(what|who)\s+(?:is|are)\s+(?:this|that|the object|the person)\b|"
    r"\bidentify\b|\bwhat is the name of (?:this|the pictured)\b",
    re.I,
)
_ENTITY_PATTERNS = (
    re.compile(
        r"identifies the query image as\s+(.+?)\.\s+Official",
        re.I,
    ),
    re.compile(r"^([^\n|]{2,100}?)\s+-\s+(?:Wikipedia|Wikimedia Commons)\b", re.I),
)


def question_relation(question: str) -> str:
    for name, pattern in _RELATIONS.items():
        if pattern.search(question or ""):
            return name
    return "unknown"


def question_asks_for_entity(question: str) -> bool:
    return bool(_ENTITY_QUESTION.search(question or ""))


def extract_identified_entities(retrieved: Sequence[Mapping[str, Any]]) -> list[str]:
    values = []
    seen = set()
    for call in retrieved:
        if call.get("tool") != "image_search" or call.get("status") != "success":
            continue
        for result in call.get("results", []):
            text = str(result.get("text", ""))
            for pattern in _ENTITY_PATTERNS:
                match = pattern.search(text)
                if not match:
                    continue
                entity = match.group(1).strip()
                if entity.casefold().startswith("file:"):
                    entity = entity[5:].strip()
                normalized = normalize_answer(entity)
                if normalized and normalized not in seen:
                    seen.add(normalized)
                    values.append(entity)
                break
    return values


def entity_copy_failure(
    *,
    question: str,
    accepted_answers: Sequence[str],
    final_answer: str | None,
    retrieved: Sequence[Mapping[str, Any]],
) -> tuple[bool, list[str]]:
    if not final_answer or question_asks_for_entity(question):
        return False, []
    entities = extract_identified_entities(retrieved)
    for entity in entities:
        answer_matches_entity = (
            normalize_answer(final_answer) == normalize_answer(entity)
            or deterministic_alias_equivalence(final_answer, entity)
        )
        accepted_is_entity = any(
            normalize_answer(alias) == normalize_answer(entity)
            or deterministic_alias_equivalence(alias, entity)
            for alias in accepted_answers
        )
        if answer_matches_entity and not accepted_is_entity:
            return True, [entity]
    return False, []


def answer_span_copy_failure(
    final_answer: str | None,
    *,
    final_correct: bool,
    retrieved: Sequence[Mapping[str, Any]],
) -> bool:
    normalized = normalize_answer(final_answer or "")
    if final_correct or not normalized:
        return False
    return any(
        normalized in normalize_answer(str(result.get("text", "")))
        for call in retrieved if call.get("status") == "success"
        for result in call.get("results", [])
    )


def repeat_search_failure(retrieved: Sequence[Mapping[str, Any]]) -> bool:
    signatures = []
    for call in retrieved:
        query = call.get("query") or {}
        identity = (
            query.get("text") if query.get("kind") == "text"
            else query.get("image_sha256")
        )
        signatures.append((call.get("tool"), normalize_answer(str(identity or ""))))
    return len(signatures) != len(set(signatures))


def query_quality(
    query: str,
    *,
    question: str,
    prior_entities: Sequence[str] = (),
) -> dict[str, Any]:
    raw = str(query or "").strip()
    normalized = normalize_answer(raw)
    tokens = normalized.split()
    empty = not raw
    ellipsis = raw in {"...", "…"}
    punctuation_only = bool(raw) and not any(char.isalnum() for char in raw)
    too_short = len(tokens) < 2
    question_copy = bool(normalized) and normalized == normalize_answer(question)
    generic = bool(
        len(tokens) <= 3
        and not any(char.isdigit() for char in raw)
        and not any(char.isupper() for char in raw[1:])
    )
    entity_missing = False
    if prior_entities:
        query_tokens = set(tokens)
        entity_missing = not any(
            query_tokens & set(normalize_answer(entity).split())
            for entity in prior_entities
        )
    relation = question_relation(question)
    relation_terms = {
        "time": {"when", "year", "date", "century"},
        "place": {"where", "country", "continent", "location", "located"},
        "person": {"who", "founder", "founded", "designer", "architect"},
        "quantity": {"many", "much", "number", "percent", "percentage"},
        "diet": {"eat", "diet", "food", "feed"},
        "material": {"material", "made", "constructed"},
        "purpose": {"use", "used", "purpose", "known"},
        "type": {"type", "kind", "classification", "species"},
    }.get(relation, set())
    relation_missing = bool(relation_terms and not (set(tokens) & relation_terms))
    hard = empty or ellipsis or punctuation_only or too_short
    return {
        "empty_query": empty,
        "ellipsis_query": ellipsis,
        "punctuation_only_query": punctuation_only,
        "too_short_query": too_short,
        "question_copy": question_copy,
        "generic_query": generic,
        "entity_missing": entity_missing,
        "relation_missing": relation_missing,
        "query_quality_failure": hard or generic or entity_missing or relation_missing,
        "hard_failure": hard,
        "heuristic_failure": generic or entity_missing or relation_missing,
        "relation_type": relation,
        "rule_types": {
            "empty_query": "deterministic",
            "ellipsis_query": "deterministic",
            "punctuation_only_query": "deterministic",
            "too_short_query": "deterministic",
            "question_copy": "deterministic",
            "generic_query": "heuristic",
            "entity_missing": "heuristic",
            "relation_missing": "heuristic",
        },
    }


def summarize_query_audit(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    fields = (
        "empty_query", "ellipsis_query", "punctuation_only_query",
        "too_short_query", "question_copy", "generic_query",
        "entity_missing", "relation_missing",
    )
    return {
        "text_search_attempt_count": len(rows),
        **{"%s_count" % field: sum(bool(row[field]) for row in rows) for field in fields},
        "successful_execution_count": sum(
            row.get("status") == "success" for row in rows
        ),
        "evidence_hit_count": sum(bool(row.get("evidence_hit_any")) for row in rows),
        "failure_type_counts": dict(Counter(
            failure
            for row in rows for failure in row.get("failure_types", [])
        )),
    }
