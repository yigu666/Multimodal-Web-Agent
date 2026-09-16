from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
from typing import Any, Mapping, Sequence

from .capacity_assignment import SEARCH_TASK_TYPES, solve_joint_capacity
from .source_adapters.base import SourceCandidate


RELEASE_TASK_TYPES = ("search_free", *SEARCH_TASK_TYPES)


def _stable_rank(seed: int, *values: object) -> str:
    payload = ":".join([str(seed), *(str(value) for value in values)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _dimension(candidate: SourceCandidate, name: str) -> str:
    metadata = candidate.source_metadata
    if name == "source":
        return candidate.source_dataset
    if name == "question_type":
        return str(
            metadata.get("question_type")
            or metadata.get("knowledge_source")
            or "unknown"
        )
    if name == "entity":
        return str(
            metadata.get("entity_id")
            or metadata.get("entity")
            or metadata.get("wikipedia_title")
            or candidate.source_data_id
        )
    raise ValueError("unknown release-selection diversity dimension")


def _diversity_order(
    candidates: Sequence[SourceCandidate],
    *,
    seed: int,
) -> list[SourceCandidate]:
    dimensions = ("source", "question_type", "entity")
    frequencies = {
        name: Counter(_dimension(candidate, name) for candidate in candidates)
        for name in dimensions
    }
    return sorted(
        candidates,
        key=lambda candidate: (
            frequencies["source"][_dimension(candidate, "source")],
            frequencies["question_type"][
                _dimension(candidate, "question_type")
            ],
            frequencies["entity"][_dimension(candidate, "entity")],
            _stable_rank(seed, candidate.candidate_key),
            candidate.candidate_key,
        ),
    )


def select_balanced_release(
    candidates: Sequence[SourceCandidate],
    eligibility: Mapping[str, Sequence[str]],
    *,
    target_per_type: int,
    seed: int,
    minimum_reserve_per_type: int = 5,
) -> dict[str, Any]:
    """Select a balanced release and disjoint reserves deterministically.

    The maximum-cardinality solver first assigns the complete currently
    achievable search pool.  The release takes the diversity-ranked prefix
    of each assignment group and preserves every remaining assignment as a
    same-type reserve.  Search-free rows are drawn only from candidates not
    used by any search assignment, so one candidate can never have two roles.
    """
    candidate_map = {item.candidate_key: item for item in candidates}
    if len(candidate_map) != len(candidates):
        raise ValueError("release candidate IDs must be unique")
    missing = set(eligibility) - set(candidate_map)
    if missing:
        raise ValueError("eligibility references unknown release candidates")

    # Use the same candidate IDs and exact solver ordering as the audited v1
    # capacity report. Diversity affects which assigned rows enter the release,
    # but cannot silently change the audited maximum assignment itself.
    joint = solve_joint_capacity(
        eligibility, target_per_type=max(1, len(candidates))
    )
    assignments = dict(joint.assignments)
    grouped = {task_type: [] for task_type in SEARCH_TASK_TYPES}
    for candidate_id, task_type in assignments.items():
        grouped[task_type].append(candidate_map[candidate_id])
    for task_type in SEARCH_TASK_TYPES:
        grouped[task_type] = _diversity_order(
            grouped[task_type], seed=seed
        )
        required = target_per_type + minimum_reserve_per_type
        if len(grouped[task_type]) < required:
            raise ValueError(
                "%s capacity %d is below release+reserve requirement %d"
                % (task_type, len(grouped[task_type]), required)
            )

    selected = {
        task_type: grouped[task_type][:target_per_type]
        for task_type in SEARCH_TASK_TYPES
    }
    reserves = {
        task_type: grouped[task_type][target_per_type:]
        for task_type in SEARCH_TASK_TYPES
    }
    search_assignment_ids = set(assignments)
    search_free_pool = _diversity_order(
        [
            candidate for candidate in candidates
            if candidate.candidate_key not in search_assignment_ids
        ],
        seed=seed,
    )
    required_search_free = target_per_type + minimum_reserve_per_type
    if len(search_free_pool) < required_search_free:
        raise ValueError(
            "search_free capacity %d is below release+reserve requirement %d"
            % (len(search_free_pool), required_search_free)
        )
    selected["search_free"] = search_free_pool[:target_per_type]
    reserves["search_free"] = search_free_pool[
        target_per_type:required_search_free
    ]

    selected_ids = {
        candidate.candidate_key
        for rows in selected.values() for candidate in rows
    }
    reserve_ids = {
        candidate.candidate_key
        for rows in reserves.values() for candidate in rows
    }
    if selected_ids & reserve_ids:
        raise RuntimeError("release selection overlaps reserve candidates")
    if len(selected_ids) != target_per_type * len(RELEASE_TASK_TYPES):
        raise RuntimeError("release selection is not exactly four-way balanced")

    return {
        "selected": selected,
        "reserves": reserves,
        "maximum_assignment": assignments,
        "capacity": {
            "search_free": len(search_free_pool),
            "visual_search_required": len(
                grouped["visual_search_required"]
            ),
            "text_search_required": len(
                grouped["text_search_required"]
            ),
            "mixed_search_required": len(
                grouped["mixed_search_required"]
            ),
            "maximum_joint_search_assignment": len(assignments),
        },
    }


def deterministic_stratified_split(
    candidates: Sequence[SourceCandidate],
    *,
    dev_count: int,
    seed: int,
) -> tuple[list[SourceCandidate], list[SourceCandidate]]:
    """Select Dev by deterministic round-robin over source/type/entity cells."""
    if dev_count < 0 or dev_count > len(candidates):
        raise ValueError("invalid deterministic Dev count")
    cells: dict[tuple[str, str, str], list[SourceCandidate]] = defaultdict(list)
    for candidate in candidates:
        key = (
            _dimension(candidate, "source"),
            _dimension(candidate, "question_type"),
            _dimension(candidate, "entity"),
        )
        cells[key].append(candidate)
    for key, rows in cells.items():
        cells[key] = sorted(
            rows,
            key=lambda item: (
                _stable_rank(seed, "split", item.candidate_key),
                item.candidate_key,
            ),
        )
    ordered_cells = sorted(
        cells,
        key=lambda key: (
            len(cells[key]),
            _stable_rank(seed, "cell", *key),
            key,
        ),
    )
    dev = []
    cursor = 0
    while len(dev) < dev_count:
        progressed = False
        for key in ordered_cells:
            if cursor < len(cells[key]):
                dev.append(cells[key][cursor])
                progressed = True
                if len(dev) == dev_count:
                    break
        if not progressed:
            raise RuntimeError("deterministic split exhausted candidates")
        cursor += 1
    dev_ids = {item.candidate_key for item in dev}
    test = sorted(
        (item for item in candidates if item.candidate_key not in dev_ids),
        key=lambda item: (
            _stable_rank(seed, "test", item.candidate_key),
            item.candidate_key,
        ),
    )
    return dev, test
