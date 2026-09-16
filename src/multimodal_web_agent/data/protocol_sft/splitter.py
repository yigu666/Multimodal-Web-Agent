from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Dict, List, Mapping, Sequence

from .schema import RouteType, Trajectory


FULL_V0_2_ROUTE_SPLIT_QUOTAS: Dict[str, Dict[str, int]] = {
    RouteType.DIRECT_ANSWER.value: {"train": 160, "dev": 21, "test": 21},
    RouteType.IMAGE_SEARCH_ANSWER.value: {"train": 160, "dev": 20, "test": 20},
    RouteType.TEXT_SEARCH_ANSWER.value: {"train": 52, "dev": 6, "test": 6},
    RouteType.IMAGE_TEXT_SEARCH_ANSWER.value: {"train": 72, "dev": 9, "test": 9},
}


FULL_V0_3_ROUTE_SPLIT_QUOTAS: Dict[str, Dict[str, int]] = {
    RouteType.DIRECT_ANSWER.value: {"train": 192, "dev": 27, "test": 27},
    RouteType.IMAGE_SEARCH_ANSWER.value: {"train": 234, "dev": 29, "test": 29},
    RouteType.TEXT_SEARCH_ANSWER.value: {"train": 52, "dev": 6, "test": 6},
    RouteType.IMAGE_TEXT_SEARCH_ANSWER.value: {"train": 12, "dev": 1, "test": 1},
}


def _stable_key(seed: int, value: str) -> str:
    return hashlib.sha256(("%d:%s" % (seed, value)).encode("utf-8")).hexdigest()


def _one_step_quotas(targets: Mapping[str, int], one_step_count: int) -> Dict[str, int]:
    names = ("dev", "test", "train")
    total = sum(targets.values())
    candidates = []
    for dev_count in range(min(one_step_count, targets["dev"]) + 1):
        if (targets["dev"] - dev_count) % 2:
            continue
        for test_count in range(min(one_step_count - dev_count, targets["test"]) + 1):
            if (targets["test"] - test_count) % 2:
                continue
            train_count = one_step_count - dev_count - test_count
            if train_count < 0 or train_count > targets["train"]:
                continue
            if (targets["train"] - train_count) % 2:
                continue
            values = {"dev": dev_count, "test": test_count, "train": train_count}
            score = sum(
                abs(values[name] - one_step_count * targets[name] / (total or 1))
                for name in names
            )
            candidates.append((score, dev_count, test_count, train_count, values))
    if not candidates:
        raise ValueError("cannot distribute one-step and two-step groups to exact split targets")
    return min(candidates, key=lambda item: item[:4])[-1]


def assign_splits(
    trajectories: Sequence[Trajectory],
    targets: Mapping[str, int],
    seed: int,
) -> Dict[str, str]:
    if set(targets) != {"train", "dev", "test"}:
        raise ValueError("split targets must define train, dev, and test")
    if any(value < 0 for value in targets.values()):
        raise ValueError("split targets cannot be negative")

    by_source: Dict[str, List[Trajectory]] = defaultdict(list)
    trajectory_ids = set()
    for trajectory in trajectories:
        if trajectory.trajectory_id in trajectory_ids:
            raise ValueError("duplicate trajectory_id: %s" % trajectory.trajectory_id)
        trajectory_ids.add(trajectory.trajectory_id)
        by_source[trajectory.source_data_id].append(trajectory)
    for source_data_id, group in by_source.items():
        routes = {trajectory.route for trajectory in group}
        if len(routes) != 1 or len(group) != 1:
            raise ValueError(
                "source_data_id %s appears in multiple trajectories or routes" % source_data_id
            )

    groups = sorted(
        by_source.items(),
        key=lambda item: (_stable_key(seed, item[0]), item[0]),
    )
    total_weight = sum(len(item.steps) for _, group in groups for item in group)
    if total_weight != sum(targets.values()):
        raise ValueError(
            "split target total %d does not match %d examples"
            % (sum(targets.values()), total_weight)
        )

    weighted_groups = [
        (source_data_id, group, sum(len(item.steps) for item in group))
        for source_data_id, group in groups
    ]
    if any(weight not in {1, 2} for _, _, weight in weighted_groups):
        raise ValueError("Protocol-SFT v0 splitter supports only one-step or two-step groups")
    one_step_groups = [(source, group) for source, group, weight in weighted_groups if weight == 1]
    two_step_groups = [(source, group) for source, group, weight in weighted_groups if weight == 2]
    one_step_quotas = _one_step_quotas(targets, len(one_step_groups))
    two_step_quotas = {
        name: (targets[name] - one_step_quotas[name]) // 2
        for name in targets
    }
    if sum(two_step_quotas.values()) != len(two_step_groups):
        raise ValueError("two-step group count does not fit exact split targets")

    source_to_split: Dict[str, str] = {}
    one_offset = 0
    two_offset = 0
    for split_name in ("dev", "test", "train"):
        selected_one = one_step_groups[
            one_offset:one_offset + one_step_quotas[split_name]
        ]
        selected_two = two_step_groups[
            two_offset:two_offset + two_step_quotas[split_name]
        ]
        for source_data_id, _ in selected_one + selected_two:
            source_to_split[source_data_id] = split_name
        one_offset += one_step_quotas[split_name]
        two_offset += two_step_quotas[split_name]

    return {
        trajectory.trajectory_id: source_to_split[trajectory.source_data_id]
        for trajectory in trajectories
    }


def proportional_state_action_targets(
    trajectories: Sequence[Trajectory],
) -> Dict[str, int]:
    """Create feasible 80/10/10 targets without splitting trajectory groups."""

    one_step_count = sum(len(trajectory.steps) == 1 for trajectory in trajectories)
    two_step_count = sum(len(trajectory.steps) == 2 for trajectory in trajectories)
    if one_step_count + two_step_count != len(trajectories):
        raise ValueError("only one-step and two-step trajectories are supported")

    def allocate(count: int) -> Dict[str, int]:
        base = {
            "train": count * 8 // 10,
            "dev": count // 10,
            "test": count // 10,
        }
        remainder = count - sum(base.values())
        fractional_order = sorted(
            ("train", "dev", "test"),
            key=lambda name: (
                -((count * {"train": 0.8, "dev": 0.1, "test": 0.1}[name]) - base[name]),
                {"train": 0, "dev": 1, "test": 2}[name],
            ),
        )
        for index in range(remainder):
            base[fractional_order[index]] += 1
        return base

    one_groups = allocate(one_step_count)
    two_groups = allocate(two_step_count)
    return {
        name: one_groups[name] + 2 * two_groups[name]
        for name in ("train", "dev", "test")
    }


def proportional_route_split_quotas(
    trajectories: Sequence[Trajectory],
) -> Dict[str, Dict[str, int]]:
    """Allocate each route at trajectory level using deterministic 80/10/10 quotas."""

    counts: Dict[str, int] = defaultdict(int)
    for trajectory in trajectories:
        if len(trajectory.steps) not in {1, 2, 3}:
            raise ValueError("only one-step, two-step, and three-step routes are supported")
        counts[trajectory.route] += 1

    def allocate(count: int) -> Dict[str, int]:
        exact = {"train": count * 0.8, "dev": count * 0.1, "test": count * 0.1}
        result = {name: int(value) for name, value in exact.items()}
        remainder = count - sum(result.values())
        order = sorted(
            result,
            key=lambda name: (
                -(exact[name] - result[name]),
                {"train": 0, "dev": 1, "test": 2}[name],
            ),
        )
        for index in range(remainder):
            result[order[index]] += 1
        return result

    return {route: allocate(count) for route, count in sorted(counts.items())}


def assign_splits_by_route_quotas(
    trajectories: Sequence[Trajectory],
    route_quotas: Mapping[str, Mapping[str, int]],
    seed: int,
) -> Dict[str, str]:
    """Assign whole trajectories to exact per-route quotas."""

    if any(
        set(quotas) != {"train", "dev", "test"}
        for quotas in route_quotas.values()
    ):
        raise ValueError("every route quota must define train, dev, and test")
    by_route: Dict[str, List[Trajectory]] = defaultdict(list)
    source_ids = set()
    trajectory_ids = set()
    for trajectory in trajectories:
        if trajectory.source_data_id in source_ids:
            raise ValueError("source_data_id appears in multiple trajectories")
        if trajectory.trajectory_id in trajectory_ids:
            raise ValueError("duplicate trajectory_id")
        source_ids.add(trajectory.source_data_id)
        trajectory_ids.add(trajectory.trajectory_id)
        by_route[trajectory.route].append(trajectory)
    if set(by_route) != set(route_quotas):
        raise ValueError("route quota keys do not match trajectory routes")

    assignments: Dict[str, str] = {}
    for route, trajectories_for_route in sorted(by_route.items()):
        quotas = route_quotas[route]
        if sum(quotas.values()) != len(trajectories_for_route):
            raise ValueError("route quota total does not match %s" % route)
        ordered = sorted(
            trajectories_for_route,
            key=lambda trajectory: (
                _stable_key(seed, "%s:%s" % (route, trajectory.source_data_id)),
                trajectory.source_data_id,
            ),
        )
        offset = 0
        for split in ("dev", "test", "train"):
            selected = ordered[offset:offset + int(quotas[split])]
            for trajectory in selected:
                assignments[trajectory.trajectory_id] = split
            offset += int(quotas[split])
    return assignments


def state_action_targets_from_assignments(
    trajectories: Sequence[Trajectory],
    assignments: Mapping[str, str],
) -> Dict[str, int]:
    targets = {"train": 0, "dev": 0, "test": 0}
    for trajectory in trajectories:
        targets[assignments[trajectory.trajectory_id]] += len(trajectory.steps)
    return targets
