from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
import json
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

from multimodal_web_agent.agent import parse_action
from multimodal_web_agent.evaluation.unified_agent.answer_metrics import (
    normalize_answer,
)

from .answer_equivalence import evaluate_answer_v2
from .diagnostics import (
    answer_span_copy_failure,
    entity_copy_failure,
    extract_identified_entities,
    query_quality,
    question_relation,
    repeat_search_failure,
    summarize_query_audit,
)
from .evidence import (
    FrozenEvidenceStore,
    aggregate_evidence_matches,
    evidence_match_layers,
)
from .labels import source_split
from .schema import EvidenceAuditRecord


def _call_matches(
    call: dict[str, Any],
    accepted_answers: Sequence[str],
    question: str,
) -> None:
    for result in call.get("results", []):
        result["answer_match"] = evidence_match_layers(
            result["text"], accepted_answers, question=question
        )
    combined = aggregate_evidence_matches(
        [call], accepted_answers, question=question
    )
    call["answer_match"] = combined


def audit_episode(
    *,
    model: str,
    example: Mapping[str, Any],
    episode: Mapping[str, Any],
    store: FrozenEvidenceStore,
) -> dict[str, Any]:
    if episode["eval_id"] != example["eval_id"]:
        raise ValueError("episode/example identity mismatch")
    answer_v2 = evaluate_answer_v2(
        episode.get("final_answer"),
        example["answer_aliases"],
        question=example["question"],
        count_semantic_as_strict=False,
    )
    if int(episode["normalized_em"]) != answer_v2.em_v1:
        raise RuntimeError("stored EM differs from v1 evaluator replay")
    if abs(float(episode["token_f1"]) - answer_v2.token_f1_v1) > 1e-12:
        raise RuntimeError("stored Token F1 differs from v1 evaluator replay")

    calls: list[dict[str, Any]] = []
    prior_entities: list[str] = []
    turns = list(episode["turns"])
    for index, turn in enumerate(turns):
        action = turn.get("action_type")
        if action not in {"image_search", "text_search"}:
            continue
        parsed = parse_action(str(turn.get("generated_text", "")))
        query = str(parsed.content or "") if action == "text_search" else ""
        status = "success" if turn.get("tool_executed") else str(
            turn.get("parse_error") or "not_executed"
        )
        base: dict[str, Any]
        if status == "success":
            base = (
                store.image_search(example)
                if action == "image_search" else store.text_search(query)
            )
        else:
            base = {
                "tool": action,
                "query": (
                    {"kind": "text", "text": query}
                    if action == "text_search" else {
                        "kind": "query_image",
                        "image_sha256": example["image_sha256"],
                        "source_data_id": example["source_data_id"],
                    }
                ),
                "information": None,
                "results": [],
            }
        base.update({
            "turn": int(turn["turn_index"]),
            "status": status,
            "generated_action": turn.get("generated_text"),
            "next_action": (
                turns[index + 1].get("action_type")
                if index + 1 < len(turns) else None
            ),
            "next_generated_text": (
                turns[index + 1].get("generated_text")
                if index + 1 < len(turns) else None
            ),
        })
        _call_matches(base, list(example["answer_aliases"]), str(example["question"]))
        if action == "text_search":
            base["query_quality"] = query_quality(
                query,
                question=str(example["question"]),
                prior_entities=prior_entities,
            )
        calls.append(base)
        if action == "image_search" and status == "success":
            prior_entities.extend(extract_identified_entities([base]))

    evidence = aggregate_evidence_matches(
        calls,
        example["answer_aliases"],
        question=str(example["question"]),
    )
    finished = bool(
        episode.get("final_answer") is not None
        and turns
        and turns[-1].get("action_type") == "answer"
    )
    correct_v1 = bool(answer_v2.em_v1)
    correct_v2 = bool(answer_v2.em_v2_strict)
    entity_failure, copied_entities = entity_copy_failure(
        question=str(example["question"]),
        accepted_answers=list(example["answer_aliases"]),
        final_answer=episode.get("final_answer"),
        retrieved=calls,
    )
    span_failure = answer_span_copy_failure(
        episode.get("final_answer"),
        final_correct=correct_v2,
        retrieved=calls,
    )
    repeat_failure = repeat_search_failure(calls)
    termination = bool(evidence["evidence_hit_any"] and not finished)
    relation_failure = bool(
        evidence["evidence_hit_any"]
        and finished
        and not correct_v2
        and question_relation(str(example["question"])) != "unknown"
    )
    query_failure = any(
        bool(call.get("query_quality", {}).get("query_quality_failure"))
        for call in calls
    )
    notes = []
    if entity_failure:
        notes.append(
            "entity_copy_failure (heuristic): final answer matches identified "
            "image entity while the question asks for an attribute; entities=%s"
            % json.dumps(copied_entities, ensure_ascii=False)
        )
    if relation_failure:
        notes.append(
            "relation_extraction_failure (heuristic): returned evidence contains "
            "an accepted answer but the attribute answer is wrong"
        )
    if span_failure:
        notes.append(
            "answer_span_copy_failure (heuristic): wrong final answer is copied "
            "from returned evidence"
        )
    if termination:
        notes.append(
            "termination_failure (deterministic): answer evidence was returned "
            "but no final answer was completed"
        )
    if repeat_failure:
        notes.append(
            "repeat_search_failure (deterministic): identical tool/query signature repeated"
        )
    if query_failure:
        notes.append("query_quality_failure: see per-query deterministic/heuristic flags")
    successful_calls = [call for call in calls if call["status"] == "success"]
    if not successful_calls:
        quadrant = "not_applicable_no_successful_search"
    elif evidence["evidence_hit_any"]:
        if not finished:
            quadrant = "E_evidence_hit_no_final_answer"
        elif correct_v1:
            quadrant = "A_evidence_hit_final_correct"
        else:
            quadrant = "B_evidence_hit_final_wrong"
    elif not finished:
        quadrant = "H_no_evidence_hit_no_final_answer"
    elif correct_v1:
        quadrant = "C_no_evidence_hit_final_correct"
    else:
        quadrant = "D_no_evidence_hit_final_wrong"
    record = EvidenceAuditRecord(
        episode_id=str(episode["eval_id"]),
        model=model,
        task_type=str(episode["task_type"]),
        search_required=bool(episode["search_required"]),
        source_dataset=str(example["source_dataset"]),
        source_split=source_split(example),
        question=str(example["question"]),
        accepted_answers=list(example["answer_aliases"]),
        final_answer=episode.get("final_answer"),
        protocol_valid=bool(episode["episode_protocol_valid"]),
        finished_with_answer=finished,
        em_v1=answer_v2.em_v1,
        token_f1_v1=answer_v2.token_f1_v1,
        em_v2_strict=answer_v2.em_v2_strict,
        token_f1_v2=answer_v2.token_f1_v2,
        semantic_equivalence_v2=answer_v2.semantic_equivalence_v2,
        numeric_equivalence_v2=answer_v2.numeric_equivalence_v2,
        unit_equivalence_v2=answer_v2.unit_equivalence_v2,
        alias_equivalence_v2=answer_v2.alias_equivalence_v2,
        search_attempt_count=len(calls),
        successful_search_count=len(successful_calls),
        image_search_count=sum(call["tool"] == "image_search" for call in successful_calls),
        text_search_count=sum(call["tool"] == "text_search" for call in successful_calls),
        tool_budget_exceeded=any(call["status"] == "tool_budget_exceeded" for call in calls),
        tool_execution_failure=bool(episode["tool_execution_failure"]),
        retrieved_information=calls,
        information_total_chars=sum(len(str(call.get("information") or "")) for call in successful_calls),
        information_truncated=any(
            result.get("truncated") is True
            for call in successful_calls for result in call.get("results", [])
        ),
        information_truncation_unknown=any(
            result.get("truncated") is None
            for call in successful_calls for result in call.get("results", [])
        ),
        **evidence,
        evidence_quadrant_v1=quadrant,
        evidence_available_final_correct=bool(evidence["evidence_hit_any"] and finished and correct_v1),
        evidence_available_final_wrong=bool(evidence["evidence_hit_any"] and finished and not correct_v1),
        evidence_available_no_answer=bool(evidence["evidence_hit_any"] and not finished),
        evidence_absent_final_correct=bool(not evidence["evidence_hit_any"] and finished and correct_v1),
        evidence_absent_final_wrong=bool(not evidence["evidence_hit_any"] and (not finished or not correct_v1)),
        evidence_available_final_correct_v2=bool(evidence["evidence_hit_any"] and finished and correct_v2),
        evidence_available_final_wrong_v2=bool(evidence["evidence_hit_any"] and finished and not correct_v2),
        evidence_absent_final_correct_v2=bool(not evidence["evidence_hit_any"] and finished and correct_v2),
        evidence_absent_final_wrong_v2=bool(not evidence["evidence_hit_any"] and (not finished or not correct_v2)),
        entity_copy_failure=entity_failure,
        relation_extraction_failure=relation_failure,
        answer_span_copy_failure=span_failure,
        termination_failure=termination,
        repeat_search_failure=repeat_failure,
        query_quality_failure=query_failure,
        first_action=str(turns[0].get("action_type") or "none") if turns else "none",
        action_sequence=[str(turn.get("action_type") or "none") for turn in turns],
        final_answer_normalized=normalize_answer(episode.get("final_answer") or ""),
        diagnostic_rule_types={
            "entity_copy_failure": "heuristic",
            "relation_extraction_failure": "heuristic",
            "answer_span_copy_failure": "heuristic",
            "termination_failure": "deterministic",
            "repeat_search_failure": "deterministic",
            "query_quality_failure": "mixed; see query_quality.rule_types",
        },
        diagnostic_notes=notes,
    )
    return record.to_dict()


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _utilization_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    searched = [row for row in rows if int(row["successful_search_count"]) > 0]
    hit = [row for row in searched if row["evidence_hit_any"]]
    hit_answer = [row for row in hit if row["finished_with_answer"]]
    hit_correct = [row for row in hit_answer if row["em_v1"]]
    hit_wrong = [row for row in hit_answer if not row["em_v1"]]
    hit_no_answer = [row for row in hit if not row["finished_with_answer"]]
    absent_correct = [
        row for row in searched
        if not row["evidence_hit_any"] and row["finished_with_answer"] and row["em_v1"]
    ]
    hit_correct_v2 = [row for row in hit_answer if row["em_v2_strict"]]
    return {
        "episode_count": len(rows),
        "successful_search_episode_count": len(searched),
        "evidence_hit_count": len(hit),
        "evidence_hit_and_correct_count": len(hit_correct),
        "evidence_hit_but_wrong_count": len(hit_wrong),
        "evidence_hit_but_no_answer_count": len(hit_no_answer),
        "no_evidence_but_correct_count": len(absent_correct),
        "evidence_hit_rate": _ratio(len(hit), len(searched)),
        "evidence_hit_and_correct_rate": _ratio(len(hit_correct), len(searched)),
        "evidence_hit_but_wrong_rate": _ratio(len(hit_wrong), len(searched)),
        "evidence_hit_but_no_answer_rate": _ratio(len(hit_no_answer), len(searched)),
        "no_evidence_but_correct_rate": _ratio(len(absent_correct), len(searched)),
        "evidence_utilization_rate": _ratio(len(hit_correct), len(hit_answer)),
        "evidence_utilization_rate_v2": _ratio(len(hit_correct_v2), len(hit_answer)),
        "answer_completion_given_evidence_rate": _ratio(len(hit_answer), len(hit)),
        "entity_copy_failure_count": sum(bool(row["entity_copy_failure"]) for row in rows),
        "entity_copy_failure_rate": _ratio(
            sum(bool(row["entity_copy_failure"]) for row in rows), len(rows)
        ),
        "termination_failure_count": sum(bool(row["termination_failure"]) for row in rows),
        "repeat_search_failure_count": sum(bool(row["repeat_search_failure"]) for row in rows),
        "evidence_hit_then_none_count": sum(
            row["evidence_hit_any"] and not row["finished_with_answer"] for row in rows
        ),
        "evidence_hit_then_repeat_search_count": sum(
            row["evidence_hit_any"] and row["repeat_search_failure"] for row in rows
        ),
        "evidence_hit_then_budget_exceeded_count": sum(
            row["evidence_hit_any"] and row["tool_budget_exceeded"] for row in rows
        ),
        "evidence_hit_finish_failure_rate": _ratio(len(hit_no_answer), len(hit)),
        "protocol_valid_no_answer_count": sum(
            row["protocol_valid"] and not row["finished_with_answer"] for row in rows
        ),
        "evidence_quadrant_counts": dict(Counter(
            row["evidence_quadrant_v1"] for row in rows
        )),
    }


def evidence_metrics(records: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    output = {}
    for model, rows in records.items():
        dimensions: dict[str, dict[str, Any]] = {}
        dimensions["overall"] = {"overall": _utilization_metrics(rows)}
        for field in ("task_type", "source_dataset", "first_action"):
            grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
            for row in rows:
                grouped[str(row[field])].append(row)
            dimensions[field] = {
                key: _utilization_metrics(value) for key, value in sorted(grouped.items())
            }
        call_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            call_groups[str(row["successful_search_count"])].append(row)
        dimensions["number_of_tool_calls"] = {
            key: _utilization_metrics(value)
            for key, value in sorted(call_groups.items())
        }
        tool_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            tools = sorted({
                call["tool"] for call in row["retrieved_information"]
                if call["status"] == "success"
            })
            tool_groups["+".join(tools) if tools else "no_tool"].append(row)
        dimensions["search_tool"] = {
            key: _utilization_metrics(value) for key, value in sorted(tool_groups.items())
        }
        output[model] = dimensions
    return {
        "schema_version": "unified-agent-eval-v1-2-evidence-metrics-v1",
        "correctness_basis": {
            "primary": "em_v1",
            "secondary": "em_v2_strict",
        },
        "models": output,
    }


def text_query_audit(records: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    rows = []
    for model in ("sft", "grpo"):
        for episode in records[model]:
            prior_entities = []
            for call in episode["retrieved_information"]:
                if call["tool"] == "image_search" and call["status"] == "success":
                    prior_entities.extend(extract_identified_entities([call]))
                if call["tool"] != "text_search":
                    continue
                quality = dict(call.get("query_quality") or query_quality(
                    str(call.get("query", {}).get("text", "")),
                    question=episode["question"],
                    prior_entities=prior_entities,
                ))
                failure_types = [
                    field for field in (
                        "empty_query", "ellipsis_query", "punctuation_only_query",
                        "too_short_query", "question_copy", "generic_query",
                        "entity_missing", "relation_missing",
                    ) if quality[field]
                ]
                rows.append({
                    "model": model,
                    "episode_id": episode["episode_id"],
                    "question": episode["question"],
                    "identified_image_entities": list(prior_entities),
                    "query": call.get("query", {}).get("text", ""),
                    "status": call["status"],
                    "top_5": [result["text"] for result in call.get("results", [])],
                    "evidence_hit_any": bool(call["answer_match"]["evidence_hit_any"]),
                    "final_answer": episode["final_answer"],
                    "failure_types": failure_types,
                    **quality,
                })
    return {
        "schema_version": "unified-agent-eval-v1-2-text-query-audit-v1",
        "models": {
            model: summarize_query_audit([row for row in rows if row["model"] == model])
            for model in ("sft", "grpo")
        },
        "queries": rows,
        "online_search_used": False,
    }


def image_truncation_audit(
    records: Mapping[str, Sequence[Mapping[str, Any]]]
) -> dict[str, Any]:
    calls = []
    for model, episodes in records.items():
        for episode in episodes:
            for call in episode["retrieved_information"]:
                if call["tool"] != "image_search" or call["status"] != "success":
                    continue
                full_text = "\n".join(
                    str(result.get("full_source_text") or "")
                    for result in call["results"]
                )
                returned_text = "\n".join(
                    str(result.get("text") or "") for result in call["results"]
                )
                full = evidence_match_layers(
                    full_text, episode["accepted_answers"], question=episode["question"]
                )
                returned = evidence_match_layers(
                    returned_text, episode["accepted_answers"], question=episode["question"]
                )
                calls.append({
                    "model": model,
                    "episode_id": episode["episode_id"],
                    "question_relation": question_relation(episode["question"]),
                    "source_dataset": episode["source_dataset"],
                    "task_type": episode["task_type"],
                    "full_source_evidence_hit": full["evidence_hit_any"],
                    "returned_information_evidence_hit": returned["evidence_hit_any"],
                    "answer_lost_by_truncation": bool(
                        full["evidence_hit_any"] and not returned["evidence_hit_any"]
                    ),
                    "record_truncated": any(result["truncated"] for result in call["results"]),
                    "full_source_available": all(
                        result.get("full_source_text") is not None for result in call["results"]
                    ),
                })
    def summary(values: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        available = [row for row in values if row["full_source_available"]]
        losses = sum(row["answer_lost_by_truncation"] for row in available)
        full_hits = sum(row["full_source_evidence_hit"] for row in available)
        return {
            "call_count": len(values),
            "full_source_available_count": len(available),
            "full_source_evidence_hit_count": full_hits,
            "returned_information_evidence_hit_count": sum(
                row["returned_information_evidence_hit"] for row in available
            ),
            "answer_lost_by_truncation_count": losses,
            "answer_lost_by_truncation_rate_given_full_hit": _ratio(losses, full_hits),
            "full_source_truncation_audit_unavailable": len(available) != len(values),
        }
    dimensions = {"overall": {"overall": summary(calls)}}
    for field in ("model", "question_relation", "source_dataset", "task_type"):
        grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in calls:
            grouped[str(row[field])].append(row)
        dimensions[field] = {
            key: summary(value) for key, value in sorted(grouped.items())
        }
    dimensions["model"] = {
        model: dimensions["model"].get(model, summary([]))
        for model in ("raw", "sft", "grpo")
    }
    return {
        "schema_version": "unified-agent-eval-v1-2-image-truncation-audit-v1",
        "record_max_chars": 512,
        "environment_modified": False,
        "dimensions": dimensions,
        "calls": calls,
    }


def answer_evaluator_metrics(
    records: Mapping[str, Sequence[Mapping[str, Any]]]
) -> dict[str, Any]:
    models = {}
    for model, rows in records.items():
        count = len(rows)
        models[model] = {
            "episode_count": count,
            "em_v1": mean(float(row["em_v1"]) for row in rows),
            "token_f1_v1": mean(float(row["token_f1_v1"]) for row in rows),
            "em_v2_strict": mean(float(row["em_v2_strict"]) for row in rows),
            "token_f1_v2": mean(float(row["token_f1_v2"]) for row in rows),
            "semantic_equivalence_count": sum(row["semantic_equivalence_v2"] for row in rows),
            "numeric_equivalence_count": sum(row["numeric_equivalence_v2"] for row in rows),
            "unit_equivalence_count": sum(row["unit_equivalence_v2"] for row in rows),
            "alias_equivalence_count": sum(row["alias_equivalence_v2"] for row in rows),
            "v1_wrong_v2_strict_correct_count": sum(
                not row["em_v1"] and row["em_v2_strict"] for row in rows
            ),
        }
    return {
        "schema_version": "unified-answer-evaluator-v2-metrics-v1",
        "strict_semantic_enabled": False,
        "models": models,
    }
