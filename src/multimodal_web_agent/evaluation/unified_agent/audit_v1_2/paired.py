from __future__ import annotations

from collections import Counter
import hashlib
import json
from typing import Any, Mapping, Sequence

from multimodal_web_agent.evaluation.unified_agent.answer_metrics import (
    normalize_answer,
)


def _tool_choice(row: Mapping[str, Any]) -> str:
    retrieved = row.get("retrieved_information") or []
    if not retrieved:
        return "no_tool"
    return str(retrieved[0]["tool"]).replace("_search", "")


def _trace_signature(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    signature = []
    for call in row.get("retrieved_information") or []:
        query = call.get("query") or {}
        information = str(call.get("information") or "")
        signature.append({
            "tool": call.get("tool"),
            "query": query,
            "status": call.get("status"),
            "information_sha256": (
                hashlib.sha256(information.encode("utf-8")).hexdigest()
                if information else None
            ),
        })
    return signature


def _transition(left: bool, right: bool) -> str:
    return "%s_to_%s" % (
        "correct" if left else "wrong",
        "correct" if right else "wrong",
    )


def build_paired_behavior_audit(
    sft_rows: Sequence[Mapping[str, Any]],
    grpo_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    sft = {str(row["episode_id"]): row for row in sft_rows}
    grpo = {str(row["episode_id"]): row for row in grpo_rows}
    if set(sft) != set(grpo) or len(sft) != 200:
        raise ValueError("SFT/GRPO paired episode IDs differ")
    correctness_v1: Counter[str] = Counter()
    correctness_v2: Counter[str] = Counter()
    search: Counter[str] = Counter()
    tool_choice: Counter[str] = Counter()
    cases = []
    same_search_trace = same_actions = same_query = same_info = 0
    same_final = 0
    for episode_id in sorted(sft):
        left = sft[episode_id]
        right = grpo[episode_id]
        correctness_v1[_transition(bool(left["em_v1"]), bool(right["em_v1"]))] += 1
        correctness_v2[_transition(
            bool(left["em_v2_strict"]), bool(right["em_v2_strict"])
        )] += 1
        left_search = int(left["successful_search_count"]) > 0
        right_search = int(right["successful_search_count"]) > 0
        search["%s_to_%s" % (
            "search" if left_search else "no_search",
            "search" if right_search else "no_search",
        )] += 1
        left_tool = _tool_choice(left)
        right_tool = _tool_choice(right)
        tool_choice["%s_to_%s" % (left_tool, right_tool)] += 1
        left_trace = _trace_signature(left)
        right_trace = _trace_signature(right)
        trace_same = left_trace == right_trace
        actions_same = left.get("action_sequence") == right.get("action_sequence")
        queries_same = [item["query"] for item in left_trace] == [
            item["query"] for item in right_trace
        ]
        information_same = [item["information_sha256"] for item in left_trace] == [
            item["information_sha256"] for item in right_trace
        ]
        final_same = normalize_answer(left.get("final_answer") or "") == normalize_answer(
            right.get("final_answer") or ""
        )
        same_search_trace += trace_same
        same_actions += actions_same
        same_query += queries_same
        same_info += information_same
        same_final += final_same
        categories = []
        if not left["em_v1"] and right["em_v1"]:
            categories.append("sft_wrong_to_grpo_correct")
        if left["em_v1"] and not right["em_v1"]:
            categories.append("sft_correct_to_grpo_wrong")
        if not left["evidence_hit_any"] and right["evidence_hit_any"]:
            categories.append("sft_no_evidence_to_grpo_evidence_hit")
        if left["evidence_hit_any"] and not right["evidence_hit_any"]:
            categories.append("sft_evidence_hit_to_grpo_no_evidence")
        if not left["finished_with_answer"] and right["finished_with_answer"]:
            categories.append("sft_no_answer_to_grpo_answer")
        if left["finished_with_answer"] and not right["finished_with_answer"]:
            categories.append("sft_answer_to_grpo_no_answer")
        if categories:
            cases.append({
                "episode_id": episode_id,
                "question": left["question"],
                "accepted_answers": left["accepted_answers"],
                "categories": categories,
                "sft": {
                    "answer": left["final_answer"],
                    "em_v1": left["em_v1"],
                    "em_v2_strict": left["em_v2_strict"],
                    "search_trace": left_trace,
                    "retrieved_information": left.get("retrieved_information") or [],
                    "evidence_hit_any": left["evidence_hit_any"],
                    "finished_with_answer": left["finished_with_answer"],
                },
                "grpo": {
                    "answer": right["final_answer"],
                    "em_v1": right["em_v1"],
                    "em_v2_strict": right["em_v2_strict"],
                    "search_trace": right_trace,
                    "retrieved_information": right.get("retrieved_information") or [],
                    "evidence_hit_any": right["evidence_hit_any"],
                    "finished_with_answer": right["finished_with_answer"],
                },
            })
    for key in (
        "correct_to_correct", "correct_to_wrong", "wrong_to_correct", "wrong_to_wrong"
    ):
        correctness_v1[key] += 0
        correctness_v2[key] += 0
    for key in (
        "no_search_to_no_search", "no_search_to_search",
        "search_to_no_search", "search_to_search",
    ):
        search[key] += 0
    for left_tool in ("image", "text", "no_tool"):
        for right_tool in ("image", "text", "no_tool"):
            tool_choice[f"{left_tool}_to_{right_tool}"] += 0
    count = len(sft)
    return {
        "schema_version": "unified-agent-eval-v1-2-paired-behavior-v1",
        "paired_episode_count": count,
        "answer_correctness_v1": dict(sorted(correctness_v1.items())),
        "answer_correctness_v2": dict(sorted(correctness_v2.items())),
        "search_behavior": dict(sorted(search.items())),
        "tool_choice": dict(sorted(tool_choice.items())),
        "retrieval_trace_identity": {
            "same_search_trace_count": same_search_trace,
            "different_search_trace_count": count - same_search_trace,
            "same_action_sequence_count": same_actions,
            "different_action_sequence_count": count - same_actions,
            "same_query_sequence_count": same_query,
            "different_query_sequence_count": count - same_query,
            "same_returned_information_count": same_info,
            "different_returned_information_count": count - same_info,
            "same_final_answer_count": same_final,
            "different_final_answer_count": count - same_final,
        },
        "effective_change_case_count": len(cases),
        "cases": cases,
    }
