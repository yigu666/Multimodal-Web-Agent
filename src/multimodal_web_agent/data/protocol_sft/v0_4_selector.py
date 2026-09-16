from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Iterable, Mapping, Sequence


ROUTE_STEPS = {
    "direct_answer": 1,
    "image_search_answer": 2,
    "text_search_answer": 2,
    "image_text_search_answer": 3,
}


class V04SelectionShortfall(RuntimeError):
    pass


@dataclass(frozen=True)
class RouteRange:
    minimum: int
    preferred: int
    maximum: int


@dataclass(frozen=True)
class RoutePlan:
    direct_answer: int
    image_search_answer: int
    text_search_answer: int
    image_text_search_answer: int

    @property
    def state_action_examples(self) -> int:
        return (
            self.direct_answer
            + 2 * self.image_search_answer
            + 2 * self.text_search_answer
            + 3 * self.image_text_search_answer
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "direct_answer": self.direct_answer,
            "image_search_answer": self.image_search_answer,
            "text_search_answer": self.text_search_answer,
            "image_text_search_answer": self.image_text_search_answer,
        }


def _largest_ten_at_most(value: int) -> int:
    return max(0, int(value) // 10 * 10)


def choose_dynamic_route_plan(
    availability: Mapping[str, int],
    *,
    state_action_target: int = 1000,
    direct_range: RouteRange = RouteRange(280, 310, 340),
    image_range: RouteRange = RouteRange(250, 315, 380),
    text_range: RouteRange = RouteRange(0, 60, 80),
    image_text_range: RouteRange = RouteRange(0, 20, 30),
) -> RoutePlan:
    text_cap = _largest_ten_at_most(
        min(
            int(availability.get("text_search_answer", 0)),
            text_range.preferred,
            text_range.maximum,
        )
    )
    image_text_cap = _largest_ten_at_most(
        min(
            int(availability.get("image_text_search_answer", 0)),
            image_text_range.preferred,
            image_text_range.maximum,
        )
    )
    candidates = []
    for text_count in range(text_cap, text_range.minimum - 1, -10):
        for image_text_count in range(
            image_text_cap, image_text_range.minimum - 1, -10
        ):
            for direct_count in range(
                direct_range.minimum, direct_range.maximum + 1, 10
            ):
                remainder = (
                    state_action_target
                    - direct_count
                    - 2 * text_count
                    - 3 * image_text_count
                )
                if remainder < 0 or remainder % 2:
                    continue
                image_count = remainder // 2
                if image_count % 10:
                    continue
                if not image_range.minimum <= image_count <= image_range.maximum:
                    continue
                plan = RoutePlan(
                    direct_answer=direct_count,
                    image_search_answer=image_count,
                    text_search_answer=text_count,
                    image_text_search_answer=image_text_count,
                )
                if any(
                    plan.to_dict()[route] > int(availability.get(route, 0))
                    for route in ROUTE_STEPS
                ):
                    continue
                score = (
                    -text_count,
                    -image_text_count,
                    abs(direct_count - direct_range.preferred),
                    abs(direct_count - image_count),
                    direct_count,
                )
                candidates.append((score, plan))
    if not candidates:
        raise V04SelectionShortfall(
            "no dynamic D/I/T/M plan can satisfy the strict candidate "
            "availability and state-action target"
        )
    return min(candidates, key=lambda item: item[0])[1]


def _stable_key(seed: int, candidate: Mapping[str, Any]) -> str:
    value = "%d:%s" % (seed, candidate["candidate_id"])
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _group_values(candidate: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(candidate["source_data_id"]),
        str(candidate["entity_group_id"]),
        str(candidate["near_duplicate_group_id"]),
    )


def unique_group_availability(
    candidates: Sequence[Mapping[str, Any]],
    *,
    excluded_source_ids: Iterable[str] = (),
    excluded_entity_group_ids: Iterable[str] = (),
) -> dict[str, int]:
    excluded_sources = set(excluded_source_ids)
    excluded_entities = set(excluded_entity_group_ids)
    groups = {route: set() for route in ROUTE_STEPS}
    for candidate in candidates:
        route = str(candidate.get("route", ""))
        if route not in groups:
            continue
        source, entity, near = _group_values(candidate)
        if source in excluded_sources or entity in excluded_entities:
            continue
        groups[route].add((source, entity, near))
    return {route: len(values) for route, values in groups.items()}


def select_candidates(
    candidates: Sequence[Mapping[str, Any]],
    *,
    plan: RoutePlan,
    seed: int,
    excluded_source_ids: Iterable[str] = (),
    excluded_entity_group_ids: Iterable[str] = (),
) -> list[dict[str, Any]]:
    excluded_sources = set(excluded_source_ids)
    excluded_entities = set(excluded_entity_group_ids)
    used_sources = set()
    used_entities = set()
    used_near = set()
    selected = []
    targets = plan.to_dict()
    selection_order = (
        "text_search_answer",
        "image_text_search_answer",
        "direct_answer",
        "image_search_answer",
    )
    for route in selection_order:
        ordered = sorted(
            (
                candidate
                for candidate in candidates
                if candidate.get("route") == route
            ),
            key=lambda row: (_stable_key(seed, row), row["candidate_id"]),
        )
        route_selected = 0
        for candidate in ordered:
            source, entity, near = _group_values(candidate)
            if (
                source in excluded_sources
                or entity in excluded_entities
                or source in used_sources
                or entity in used_entities
                or near in used_near
            ):
                continue
            selected.append(dict(candidate))
            used_sources.add(source)
            used_entities.add(entity)
            used_near.add(near)
            route_selected += 1
            if route_selected == targets[route]:
                break
        if route_selected != targets[route]:
            raise V04SelectionShortfall(
                "%s strict unique-group shortfall: selected=%d target=%d"
                % (route, route_selected, targets[route])
            )
    if sum(
        ROUTE_STEPS[str(candidate["route"])] for candidate in selected
    ) != plan.state_action_examples:
        raise AssertionError("selected state-action count is inconsistent")
    return selected
