from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence


SEARCH_TASK_TYPES = (
    "visual_search_required",
    "text_search_required",
    "mixed_search_required",
)


@dataclass(frozen=True)
class JointCapacityResult:
    total_assignable: int
    visual_assigned: int
    text_assigned: int
    mixed_assigned: int
    target_per_type: int
    full_quota_satisfied: bool
    visual_shortfall: int
    text_shortfall: int
    mixed_shortfall: int
    total_shortfall: int
    maximum_equal_per_type: int
    maximum_equal_four_way_size: int
    minimum_assigned_type_count: int
    assignment_spread: int
    assignments: dict[str, str]
    raw_max_flow_assignments: dict[str, str]
    raw_max_flow_counts: dict[str, int]
    unassigned_candidate_ids: tuple[str, ...]
    bottleneck_summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class _Dinic:
    def __init__(self, node_count: int):
        self.graph: list[list[list[int]]] = [[] for _ in range(node_count)]

    def add_edge(
        self, source: int, target: int, capacity: int
    ) -> list[int]:
        forward = [target, capacity, 0]
        reverse = [source, 0, 0]
        forward[2] = len(self.graph[target])
        reverse[2] = len(self.graph[source])
        self.graph[source].append(forward)
        self.graph[target].append(reverse)
        return forward

    def maximum_flow(self, source: int, sink: int) -> int:
        total = 0
        while True:
            level = [-1] * len(self.graph)
            level[source] = 0
            queue = [source]
            for node in queue:
                for target, capacity, _ in self.graph[node]:
                    if capacity and level[target] < 0:
                        level[target] = level[node] + 1
                        queue.append(target)
            if level[sink] < 0:
                return total
            cursor = [0] * len(self.graph)

            def send(node: int, amount: int) -> int:
                if node == sink:
                    return amount
                while cursor[node] < len(self.graph[node]):
                    edge = self.graph[node][cursor[node]]
                    target, capacity, reverse_index = edge
                    if capacity and level[target] == level[node] + 1:
                        pushed = send(target, min(amount, capacity))
                        if pushed:
                            edge[1] -= pushed
                            self.graph[target][reverse_index][1] += pushed
                            return pushed
                    cursor[node] += 1
                return 0

            while True:
                pushed = send(source, 10**9)
                if not pushed:
                    break
                total += pushed


def _normalized_eligibility(
    eligibility: Mapping[str, Sequence[str]],
) -> dict[str, tuple[str, ...]]:
    result = {}
    for candidate_id, raw_types in eligibility.items():
        candidate = str(candidate_id).strip()
        if not candidate:
            raise ValueError("capacity candidate ID cannot be empty")
        if candidate in result:
            raise ValueError("duplicate capacity candidate ID")
        values = tuple(
            task_type
            for task_type in SEARCH_TASK_TYPES
            if task_type in set(raw_types)
        )
        if values:
            result[candidate] = values
    return result


def _solve(
    eligibility: Mapping[str, tuple[str, ...]],
    quotas: Mapping[str, int],
    *,
    initial_assignments: Mapping[str, str] | None = None,
) -> tuple[int, dict[str, str]]:
    candidate_ids = sorted(
        eligibility,
        key=lambda key: (len(eligibility[key]), key),
    )
    source = 0
    candidate_offset = 1
    type_offset = candidate_offset + len(candidate_ids)
    sink = type_offset + len(SEARCH_TASK_TYPES)
    flow = _Dinic(sink + 1)
    type_nodes = {
        task_type: type_offset + index
        for index, task_type in enumerate(SEARCH_TASK_TYPES)
    }
    candidate_nodes = {}
    for index, candidate_id in enumerate(candidate_ids):
        node = candidate_offset + index
        candidate_nodes[candidate_id] = node
        flow.add_edge(source, node, 1)
        for task_type in eligibility[candidate_id]:
            flow.add_edge(node, type_nodes[task_type], 1)
    for task_type in SEARCH_TASK_TYPES:
        quota = int(quotas[task_type])
        if quota < 0:
            raise ValueError("capacity quota cannot be negative")
        flow.add_edge(type_nodes[task_type], sink, quota)

    initial = dict(initial_assignments or {})
    initial_counts = {task_type: 0 for task_type in SEARCH_TASK_TYPES}

    def push_existing_edge(node: int, target: int) -> None:
        for edge in flow.graph[node]:
            if edge[0] == target and edge[1] > 0:
                edge[1] -= 1
                flow.graph[target][edge[2]][1] += 1
                return
        raise ValueError("initial capacity assignment is not feasible")

    for candidate_id, task_type in initial.items():
        if (
            candidate_id not in candidate_nodes
            or task_type not in eligibility[candidate_id]
        ):
            raise ValueError("initial assignment violates eligibility")
        initial_counts[task_type] += 1
        if initial_counts[task_type] > int(quotas[task_type]):
            raise ValueError("initial assignment exceeds type quota")
        node = candidate_nodes[candidate_id]
        push_existing_edge(source, node)
        push_existing_edge(node, type_nodes[task_type])
        push_existing_edge(type_nodes[task_type], sink)

    total = len(initial) + flow.maximum_flow(source, sink)
    assignments = {}
    for index, candidate_id in enumerate(candidate_ids):
        node = candidate_offset + index
        for target, capacity, _ in flow.graph[node]:
            for task_type, type_node in type_nodes.items():
                if target == type_node and capacity == 0:
                    assignments[candidate_id] = task_type
                    break
            if candidate_id in assignments:
                break
    return total, assignments


def _solve_bounded(
    eligibility: Mapping[str, tuple[str, ...]],
    *,
    lower_quotas: Mapping[str, int],
    upper_quotas: Mapping[str, int],
    required_total: int,
) -> tuple[bool, dict[str, str]]:
    """Solve an exact-cardinality assignment with per-type lower bounds.

    This is a standard lower-bound circulation reduction.  It is used after
    the maximum cardinality is known, so balancing can never reduce the first
    (maximum-total) objective.
    """
    candidate_ids = sorted(
        eligibility,
        key=lambda key: (len(eligibility[key]), key),
    )
    source = 0
    candidate_offset = 1
    type_offset = candidate_offset + len(candidate_ids)
    sink = type_offset + len(SEARCH_TASK_TYPES)
    super_source = sink + 1
    super_sink = sink + 2
    flow = _Dinic(super_sink + 1)
    demands = [0] * (super_sink + 1)
    type_nodes = {
        task_type: type_offset + index
        for index, task_type in enumerate(SEARCH_TASK_TYPES)
    }
    candidate_edges: dict[tuple[str, str], tuple[list[int], int]] = {}

    def bounded_edge(
        start: int, end: int, lower: int, upper: int
    ) -> list[int]:
        if lower < 0 or upper < lower:
            raise ValueError("invalid lower/upper capacity bound")
        edge = flow.add_edge(start, end, upper - lower)
        demands[start] -= lower
        demands[end] += lower
        return edge

    for index, candidate_id in enumerate(candidate_ids):
        node = candidate_offset + index
        bounded_edge(source, node, 0, 1)
        for task_type in eligibility[candidate_id]:
            edge = bounded_edge(node, type_nodes[task_type], 0, 1)
            candidate_edges[(candidate_id, task_type)] = (edge, 1)
    for task_type in SEARCH_TASK_TYPES:
        bounded_edge(
            type_nodes[task_type],
            sink,
            int(lower_quotas[task_type]),
            int(upper_quotas[task_type]),
        )
    bounded_edge(sink, source, required_total, required_total)

    demand_total = 0
    for node, demand in enumerate(demands[:super_source]):
        if demand > 0:
            flow.add_edge(super_source, node, demand)
            demand_total += demand
        elif demand < 0:
            flow.add_edge(node, super_sink, -demand)
    if flow.maximum_flow(super_source, super_sink) != demand_total:
        return False, {}

    assignments: dict[str, str] = {}
    for candidate_id in candidate_ids:
        for task_type in SEARCH_TASK_TYPES:
            record = candidate_edges.get((candidate_id, task_type))
            if record is None:
                continue
            edge, original_capacity = record
            if original_capacity - edge[1] == 1:
                assignments[candidate_id] = task_type
                break
    if len(assignments) != required_total:
        raise RuntimeError("bounded circulation assignment extraction failed")
    return True, assignments


def _maximum_equal_per_type(
    eligibility: Mapping[str, tuple[str, ...]],
) -> int:
    individual = [
        sum(task_type in values for values in eligibility.values())
        for task_type in SEARCH_TASK_TYPES
    ]
    upper = min(
        [len(eligibility) // len(SEARCH_TASK_TYPES), *individual],
        default=0,
    )
    lower = 0
    while lower < upper:
        middle = (lower + upper + 1) // 2
        quotas = {task_type: middle for task_type in SEARCH_TASK_TYPES}
        total, _ = _solve(eligibility, quotas)
        if total == middle * len(SEARCH_TASK_TYPES):
            lower = middle
        else:
            upper = middle - 1
    return lower


def solve_joint_capacity(
    eligibility: Mapping[str, Sequence[str]],
    *,
    target_per_type: int = 250,
) -> JointCapacityResult:
    normalized = _normalized_eligibility(eligibility)
    quotas = {
        task_type: int(target_per_type)
        for task_type in SEARCH_TASK_TYPES
    }
    maximum_total, raw_assignments = _solve(normalized, quotas)
    maximum_equal = min(
        _maximum_equal_per_type(normalized),
        target_per_type,
    )

    # Objective 2: among maximum-total solutions, maximize the minimum type
    # count.  Feasibility is monotone in the lower bound, so binary search is
    # exact and does not depend on an arbitrary initial maximum-flow result.
    lower = 0
    upper = min(target_per_type, maximum_total // 3)
    best_minimum = 0
    while lower <= upper:
        middle = (lower + upper) // 2
        feasible, _ = _solve_bounded(
            normalized,
            lower_quotas={
                task_type: middle for task_type in SEARCH_TASK_TYPES
            },
            upper_quotas=quotas,
            required_total=maximum_total,
        )
        if feasible:
            best_minimum = middle
            lower = middle + 1
        else:
            upper = middle - 1

    # Objective 3: with total and minimum fixed, minimize max-min spread.
    lower = 0
    upper = target_per_type - best_minimum
    best_spread = upper
    assignments: dict[str, str] = {}
    while lower <= upper:
        middle = (lower + upper) // 2
        feasible, candidate_assignments = _solve_bounded(
            normalized,
            lower_quotas={
                task_type: best_minimum
                for task_type in SEARCH_TASK_TYPES
            },
            upper_quotas={
                task_type: best_minimum + middle
                for task_type in SEARCH_TASK_TYPES
            },
            required_total=maximum_total,
        )
        if feasible:
            best_spread = middle
            assignments = candidate_assignments
            upper = middle - 1
        else:
            lower = middle + 1
    # Objective 4 is deterministic: candidate IDs and task types are sorted
    # before graph construction, and Dinic traverses edges in insertion order.
    total = len(assignments)
    if total != maximum_total:
        raise RuntimeError(
            "failed to recover a balanced maximum-cardinality assignment"
        )
    counts = {
        task_type: sum(value == task_type for value in assignments.values())
        for task_type in SEARCH_TASK_TYPES
    }
    raw_counts = {
        task_type: sum(
            value == task_type for value in raw_assignments.values()
        )
        for task_type in SEARCH_TASK_TYPES
    }
    shortfalls = {
        task_type: max(0, target_per_type - counts[task_type])
        for task_type in SEARCH_TASK_TYPES
    }
    subset_bottlenecks = {}
    for mask in range(1, 1 << len(SEARCH_TASK_TYPES)):
        subset = tuple(
            task_type
            for index, task_type in enumerate(SEARCH_TASK_TYPES)
            if mask & (1 << index)
        )
        candidate_count = sum(
            any(task_type in values for task_type in subset)
            for values in normalized.values()
        )
        demand = target_per_type * len(subset)
        subset_bottlenecks["+".join(subset)] = {
            "eligible_candidate_union_count": candidate_count,
            "target_demand": demand,
            "remaining_shortfall": max(0, demand - candidate_count),
        }
    unassigned = tuple(sorted(set(normalized) - set(assignments)))
    return JointCapacityResult(
        total_assignable=total,
        visual_assigned=counts["visual_search_required"],
        text_assigned=counts["text_search_required"],
        mixed_assigned=counts["mixed_search_required"],
        target_per_type=target_per_type,
        full_quota_satisfied=all(
            counts[task_type] == target_per_type
            for task_type in SEARCH_TASK_TYPES
        ),
        visual_shortfall=shortfalls["visual_search_required"],
        text_shortfall=shortfalls["text_search_required"],
        mixed_shortfall=shortfalls["mixed_search_required"],
        total_shortfall=sum(shortfalls.values()),
        maximum_equal_per_type=maximum_equal,
        maximum_equal_four_way_size=maximum_equal * 4,
        minimum_assigned_type_count=min(counts.values(), default=0),
        assignment_spread=(
            max(counts.values(), default=0)
            - min(counts.values(), default=0)
        ),
        assignments=assignments,
        raw_max_flow_assignments=raw_assignments,
        raw_max_flow_counts=raw_counts,
        unassigned_candidate_ids=unassigned,
        bottleneck_summary={
            "search_eligible_union_count": len(normalized),
            "individual_eligible_counts": {
                task_type: sum(
                    task_type in values for values in normalized.values()
                )
                for task_type in SEARCH_TASK_TYPES
            },
            "subset_capacity": subset_bottlenecks,
        },
    )
