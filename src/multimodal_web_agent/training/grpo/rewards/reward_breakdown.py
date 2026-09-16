from __future__ import annotations

from collections import defaultdict
import math
import statistics
from typing import Any, Mapping, Sequence


def token_advantage_stats(values) -> dict[str, Any]:
    if values is None:
        return {
            "status": "not_verifiable_from_persisted_rollouts",
            "min": None, "max": None, "mean": None, "std": None,
            "nonzero_tokens": None, "policy_token_count": None,
        }
    return dict(values)


def group_summary(breakdowns: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in breakdowns:
        groups[str(row.get("prompt_id", ""))].append(row)
    output = []
    for prompt_id, rows in sorted(groups.items()):
        terminal = [float(row["terminal_reward"]) for row in rows]
        text = [
            float(action["local_utility"])
            for row in rows for action in row.get("search_actions", [])
            if action.get("tool") == "text_search" and action.get("executed")
        ]
        image = [
            float(action["local_utility"])
            for row in rows for action in row.get("search_actions", [])
            if action.get("tool") == "image_search" and action.get("executed")
        ]
        token_stds = [
            float(row["token_advantage_stats"]["std"])
            for row in rows
            if isinstance(row.get("token_advantage_stats", {}).get("std"), (int, float))
        ]
        output.append({
            "prompt_id": prompt_id,
            "trajectory_count": len(rows),
            "terminal_reward_mean": statistics.mean(terminal),
            "terminal_reward_std": statistics.pstdev(terminal) if len(terminal) > 1 else 0.0,
            "terminal_zero_variance": len(set(round(value, 12) for value in terminal)) <= 1,
            "text_local_count": len(text),
            "text_local_mean": statistics.mean(text) if text else 0.0,
            "text_local_std": statistics.pstdev(text) if len(text) > 1 else 0.0,
            "image_local_count": len(image),
            "image_local_mean": statistics.mean(image) if image else 0.0,
            "image_local_std": statistics.pstdev(image) if len(image) > 1 else 0.0,
            "token_advantage_std": statistics.mean(token_stds) if token_stds else None,
        })
    return output


def aggregate_metrics(
    breakdowns: Sequence[Mapping[str, Any]],
    group_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    denominator = max(len(breakdowns), 1)
    group_denominator = max(len(group_rows), 1)
    text = [
        float(action["local_utility"])
        for row in breakdowns for action in row.get("search_actions", [])
        if action.get("tool") == "text_search" and action.get("executed")
    ]
    image = [
        float(action["local_utility"])
        for row in breakdowns for action in row.get("search_actions", [])
        if action.get("tool") == "image_search" and action.get("executed")
    ]
    values = [
        float(value)
        for row in breakdowns
        for value in (
            row.get("answer_score", 0.0), row.get("evidence_use_score", 0.0),
            row.get("missed_evidence", 0.0), row.get("terminal_reward", 0.0),
        )
    ] + text + image
    return {
        "zero_variance_terminal_group_ratio": sum(bool(row["terminal_zero_variance"]) for row in group_rows) / group_denominator,
        "nonzero_local_text_group_ratio": sum(float(row["text_local_std"]) > 1e-12 for row in group_rows) / group_denominator,
        "nonzero_local_image_group_ratio": sum(float(row["image_local_std"]) > 1e-12 for row in group_rows) / group_denominator,
        "mean_answer_score": sum(float(row["answer_score"]) for row in breakdowns) / denominator,
        "mean_text_query_utility": sum(text) / len(text) if text else 0.0,
        "mean_image_retrieval_utility": sum(image) / len(image) if image else 0.0,
        "mean_evidence_use": sum(float(row["evidence_use_score"]) for row in breakdowns) / denominator,
        "mean_missed_evidence": sum(float(row["missed_evidence"]) for row in breakdowns) / denominator,
        "wrong_span_copy_rate": sum(bool(row["wrong_span_copy"]) for row in breakdowns) / denominator,
        "missing_answer_rate": sum(not bool(row["finished_with_answer"]) for row in breakdowns) / denominator,
        "all_numeric_values_finite": all(math.isfinite(value) for value in values),
    }
