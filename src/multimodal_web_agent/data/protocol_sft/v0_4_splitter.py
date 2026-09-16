from __future__ import annotations

from collections import defaultdict
from collections import Counter
import hashlib
from typing import Any, Mapping, Sequence


def _stable_key(seed: int, value: str) -> str:
    return hashlib.sha256(
        ("%d:%s" % (seed, value)).encode("utf-8")
    ).hexdigest()


def assign_group_aware_splits(
    candidates: Sequence[Mapping[str, Any]],
    *,
    seed: int,
) -> dict[str, str]:
    rows = list(candidates)
    parent = list(range(len(rows)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for field in (
        "source_data_id",
        "entity_group_id",
        "near_duplicate_group_id",
    ):
        first_by_value = {}
        for index, row in enumerate(rows):
            value = str(row[field])
            if value in first_by_value:
                union(index, first_by_value[value])
            else:
                first_by_value[value] = index

    components = defaultdict(list)
    for index, row in enumerate(rows):
        components[find(index)].append(row)

    targets = {}
    route_counts = Counter(str(row["route"]) for row in rows)
    for route, count in route_counts.items():
        if count % 10:
            raise ValueError(
                "route count must be a multiple of ten: %s=%d"
                % (route, count)
            )
        targets[route] = {
            "train": count * 8 // 10,
            "dev": count // 10,
            "test": count // 10,
        }

    ordered_components = sorted(
        components.values(),
        key=lambda group: (
            -len(group),
            _stable_key(
                seed,
                min(str(row["candidate_id"]) for row in group),
            ),
        ),
    )
    used = {
        route: {"train": 0, "dev": 0, "test": 0}
        for route in route_counts
    }
    assignments = {}
    for group in ordered_components:
        contribution = Counter(str(row["route"]) for row in group)
        feasible = [
            split
            for split in ("train", "dev", "test")
            if all(
                used[route][split] + amount
                <= targets[route][split]
                for route, amount in contribution.items()
            )
        ]
        if not feasible:
            raise ValueError(
                "group-aware 80/10/10 allocation is infeasible for group %s"
                % min(str(row["candidate_id"]) for row in group)
            )
        split = max(
            feasible,
            key=lambda candidate_split: (
                sum(
                    targets[route][candidate_split]
                    - used[route][candidate_split]
                    for route in contribution
                ),
                {"train": 2, "dev": 1, "test": 0}[candidate_split],
            ),
        )
        for route, amount in contribution.items():
            used[route][split] += amount
        for row in group:
            assignments[str(row["candidate_id"])] = split

    if used != targets:
        raise ValueError(
            "group-aware split could not reach exact route quotas: "
            "used=%r targets=%r" % (used, targets)
        )
    return assignments


def group_split_leak_report(
    candidates: Sequence[Mapping[str, Any]],
    assignments: Mapping[str, str],
) -> dict[str, int]:
    dimensions = {
        "source_split_leak_count": "source_data_id",
        "entity_group_split_leak_count": "entity_group_id",
        "near_duplicate_split_leak_count": "near_duplicate_group_id",
        "trajectory_split_leak_count": "candidate_id",
    }
    report = {}
    for metric, field in dimensions.items():
        seen = defaultdict(set)
        for row in candidates:
            seen[str(row[field])].add(
                assignments[str(row["candidate_id"])]
            )
        report[metric] = sum(len(splits) > 1 for splits in seen.values())
    return report


def state_action_counts_by_split(
    candidates: Sequence[Mapping[str, Any]],
    assignments: Mapping[str, str],
) -> dict[str, int]:
    counts = {"train": 0, "dev": 0, "test": 0}
    for row in candidates:
        counts[assignments[str(row["candidate_id"])]] += len(
            row["trajectory"]["steps"]
        )
    return counts
