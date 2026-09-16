from __future__ import annotations

from collections import Counter
from statistics import mean
from typing import Any, Mapping, Sequence

from .schema import TASK_TYPES


def search_decision_metrics(
    episodes: Sequence[Mapping[str, Any]],
) -> dict[str, float | int]:
    tp = fp = fn = tn = 0
    for row in episodes:
        gold = bool(row["search_required"])
        predicted = int(row["tool_call_count"]) > 0
        if gold and predicted:
            tp += 1
        elif not gold and predicted:
            fp += 1
        elif gold and not predicted:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )
    return {
        "search_decision_f1": f1,
        "search_precision": precision,
        "search_recall": recall,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
    }


def aggregate_agent_metrics(
    episodes: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not episodes:
        raise ValueError("episode metrics require at least one episode")
    count = len(episodes)
    search = search_decision_metrics(episodes)
    main = {
        "episode_count": count,
        "normalized_em": mean(float(row["normalized_em"]) for row in episodes),
        "token_f1": mean(float(row["token_f1"]) for row in episodes),
        "agent_success_at_budget": mean(
            bool(row["agent_success_at_budget"]) for row in episodes
        ),
        "search_decision_f1": search["search_decision_f1"],
        "avg_tool_calls": mean(
            int(row["tool_call_count"]) for row in episodes
        ),
        "protocol_valid": mean(
            bool(row["episode_protocol_valid"]) for row in episodes
        ),
    }
    by_type = {}
    for task_type in TASK_TYPES:
        rows = [row for row in episodes if row["task_type"] == task_type]
        by_type[task_type] = {
            "n": len(rows),
            "em": (
                mean(float(row["normalized_em"]) for row in rows)
                if rows else "not_applicable"
            ),
        }
    diagnostics = {
        **search,
        "over_search_rate": search["false_positive"] / count,
        "under_search_rate": search["false_negative"] / count,
        "malformed_episode_rate": mean(
            not bool(row["episode_protocol_valid"]) for row in episodes
        ),
        "tool_failure_rate": mean(
            bool(row["tool_execution_failure"]) for row in episodes
        ),
        "max_turn_exhaustion_rate": mean(
            bool(row["max_turn_exhausted"]) for row in episodes
        ),
        "image_search_call_count": sum(
            int(row["image_search_call_count"]) for row in episodes
        ),
        "text_search_call_count": sum(
            int(row["text_search_call_count"]) for row in episodes
        ),
    }
    return {**main, "task_type_em": by_type}, diagnostics
