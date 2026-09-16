from __future__ import annotations

from typing import Mapping

from .v0_4_selector import RoutePlan, RouteRange, V04SelectionShortfall


def choose_v05_route_plan(
    availability: Mapping[str, int],
    *,
    state_action_target: int = 1000,
    direct_range: RouteRange = RouteRange(200, 310, 500),
    image_range: RouteRange = RouteRange(200, 290, 400),
    text_range: RouteRange = RouteRange(0, 40, 80),
    image_text_range: RouteRange = RouteRange(0, 10, 30),
) -> RoutePlan:
    caps = {
        "direct_answer": min(
            direct_range.maximum, int(availability.get("direct_answer", 0))
        ),
        "image_search_answer": min(
            image_range.maximum,
            int(availability.get("image_search_answer", 0)),
        ),
        "text_search_answer": min(
            text_range.maximum,
            int(availability.get("text_search_answer", 0)),
        ),
        "image_text_search_answer": min(
            image_text_range.maximum,
            int(availability.get("image_text_search_answer", 0)),
        ),
    }
    candidates = []
    for text_count in range(
        text_range.minimum, caps["text_search_answer"] + 1
    ):
        for image_text_count in range(
            image_text_range.minimum,
            caps["image_text_search_answer"] + 1,
        ):
            for direct_count in range(
                direct_range.minimum, caps["direct_answer"] + 1
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
                if not image_range.minimum <= image_count <= caps[
                    "image_search_answer"
                ]:
                    continue
                plan = RoutePlan(
                    direct_answer=direct_count,
                    image_search_answer=image_count,
                    text_search_answer=text_count,
                    image_text_search_answer=image_text_count,
                )
                score = (
                    abs(text_count - min(
                        text_range.preferred,
                        caps["text_search_answer"],
                    )),
                    abs(image_text_count - min(
                        image_text_range.preferred,
                        caps["image_text_search_answer"],
                    )),
                    abs(direct_count - direct_range.preferred),
                    abs(image_count - image_range.preferred),
                )
                candidates.append((score, plan))
    if not candidates:
        raise V04SelectionShortfall(
            "no strict v0.5 D/I/T/M plan can reach the state-action target"
        )
    return min(candidates, key=lambda item: item[0])[1]
