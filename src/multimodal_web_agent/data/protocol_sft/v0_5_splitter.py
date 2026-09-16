from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence


def _stable_key(seed: int, candidate: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        ("%d:%s" % (seed, candidate["candidate_id"])).encode("utf-8")
    ).hexdigest()


def _take_exact_weight(
    rows: Sequence[Mapping[str, Any]],
    *,
    target: int,
) -> set[str]:
    solutions: dict[int, tuple[str, ...]] = {0: ()}
    for row in rows:
        candidate_id = str(row["candidate_id"])
        weight = len(row["trajectory"]["steps"])
        for total, chosen in sorted(
            list(solutions.items()), reverse=True
        ):
            updated = total + weight
            if updated > target or updated in solutions:
                continue
            solutions[updated] = (*chosen, candidate_id)
        if target in solutions:
            return set(solutions[target])
    raise ValueError("cannot allocate exact state-action split weight %d" % target)


def assign_v05_group_aware_splits(
    candidates: Sequence[Mapping[str, Any]],
    *,
    seed: int,
) -> dict[str, str]:
    rows = sorted(candidates, key=lambda row: _stable_key(seed, row))
    dimensions = (
        "source_data_id",
        "entity_group_id",
        "near_duplicate_group_id",
    )
    for field in dimensions:
        values = [str(row[field]) for row in rows]
        if len(values) != len(set(values)):
            raise ValueError(
                "v0.5 selection must have unique %s before splitting" % field
            )
    if sum(len(row["trajectory"]["steps"]) for row in rows) != 1000:
        raise ValueError("v0.5 split input must contain 1000 state-actions")
    test_ids = _take_exact_weight(rows, target=100)
    remaining = [
        row for row in rows if str(row["candidate_id"]) not in test_ids
    ]
    dev_ids = _take_exact_weight(remaining, target=100)
    return {
        str(row["candidate_id"]): (
            "test"
            if str(row["candidate_id"]) in test_ids
            else "dev"
            if str(row["candidate_id"]) in dev_ids
            else "train"
        )
        for row in rows
    }
