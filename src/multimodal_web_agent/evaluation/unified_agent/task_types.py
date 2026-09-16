from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

from .schema import TASK_TYPES
from .source_adapters.base import SourceCandidate


DEV_PER_TYPE = 50
TEST_PER_TYPE = 200
TOTAL_PER_TYPE = DEV_PER_TYPE + TEST_PER_TYPE


def _rank(seed: int, key: str) -> str:
    return hashlib.sha256(("%d:%s" % (seed, key)).encode("utf-8")).hexdigest()


def balanced_split(
    candidates: Sequence[SourceCandidate],
    approvals: Mapping[str, Mapping[str, Any]],
    *,
    seed: int,
) -> tuple[list[tuple[SourceCandidate, Mapping[str, Any]]], list[
    tuple[SourceCandidate, Mapping[str, Any]]
], dict[str, int]]:
    grouped = {task_type: [] for task_type in TASK_TYPES}
    for candidate in candidates:
        approval = approvals.get(candidate.candidate_key)
        if not approval:
            continue
        if approval.get("reviewed") is not True or approval.get("approved") is not True:
            continue
        task_type = str(approval.get("task_type", ""))
        if task_type not in grouped:
            continue
        grouped[task_type].append((candidate, approval))
    shortfall = {
        task_type: max(0, TOTAL_PER_TYPE - len(rows))
        for task_type, rows in grouped.items()
    }
    if any(shortfall.values()):
        return [], [], shortfall
    dev = []
    test = []
    for task_type in TASK_TYPES:
        rows = sorted(
            grouped[task_type],
            key=lambda item: _rank(seed, item[0].candidate_key),
        )[:TOTAL_PER_TYPE]
        dev.extend(rows[:DEV_PER_TYPE])
        test.extend(rows[DEV_PER_TYPE:])
    if len({row[0].candidate_key for row in dev + test}) != 1000:
        raise ValueError("balanced split contains duplicate candidates")
    return dev, test, shortfall


def efficiency_subset(
    examples: Sequence[Any],
    *,
    per_type: int = 25,
) -> list[Any]:
    selected = []
    for task_type in TASK_TYPES:
        rows = sorted(
            (row for row in examples if row.task_type == task_type),
            key=lambda row: row.eval_id,
        )
        if len(rows) < per_type:
            raise ValueError("efficiency subset task type shortfall")
        selected.extend(rows[:per_type])
    return selected
