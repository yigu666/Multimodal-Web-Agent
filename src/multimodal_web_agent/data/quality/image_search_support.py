from __future__ import annotations

from typing import Any, Mapping, Sequence

from .evidence_reachability import alias_in_text
from .rejection import SharedRejectionReason, reason_values
from .schema import ActionType, ActionValidation, VisibleEntity


def _cache_titles(image_cache_entry: Any) -> tuple[str, ...]:
    if image_cache_entry is None:
        return ()
    if hasattr(image_cache_entry, "usable_image_results"):
        return tuple(
            str(title)
            for _index, title, _descriptor
            in image_cache_entry.usable_image_results
        )
    if isinstance(image_cache_entry, Mapping):
        raw = image_cache_entry.get(
            "titles",
            image_cache_entry.get("tool_returned_web_title_list", ()),
        )
        return tuple(str(value).strip() for value in raw or () if str(value).strip())
    return ()


def validate_image_search_action(
    *,
    question: str,
    image_cache_entry: Any | None,
    image_exists: bool = True,
    answer_aliases: Sequence[str] = (),
    require_answer_support: bool = False,
) -> ActionValidation:
    reasons = []
    if not image_exists:
        reasons.append(SharedRejectionReason.EMPTY_INFORMATION)
    titles = _cache_titles(image_cache_entry)
    if image_cache_entry is None:
        reasons.append(SharedRejectionReason.IMAGE_CACHE_MISS)
    elif not titles:
        reasons.append(SharedRejectionReason.EMPTY_INFORMATION)
    if require_answer_support and not alias_in_text(
        tuple(answer_aliases), "\n".join(titles)
    ):
        reasons.append(SharedRejectionReason.ANSWER_UNSUPPORTED)
    entities = (
        (
            VisibleEntity(
                value=titles[0],
                provenance="image_search_information",
                source_span=titles[0],
                visible_in_current_text_state=False,
            ),
        )
        if titles
        else ()
    )
    return ActionValidation(
        action_type=ActionType.IMAGE_SEARCH.value,
        executable=not reasons,
        reasons=reason_values(reasons),
        visible_entities=entities,
    )


def image_cache_titles(image_cache_entry: Any | None) -> tuple[str, ...]:
    return _cache_titles(image_cache_entry)
