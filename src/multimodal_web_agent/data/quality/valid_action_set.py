from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

from .answer_support import validate_answer_action
from .image_search_support import validate_image_search_action
from .query_executability import validate_text_search_action
from .schema import ActionType, ActionValidation
from .visible_context import (
    build_visible_text_context,
    visible_information_text,
)


def compute_action_validations(
    candidate_state: Mapping[str, Any],
) -> Dict[str, ActionValidation]:
    question = str(candidate_state.get("question", ""))
    state = list(candidate_state.get("state", ()))
    transition = str(
        candidate_state.get(
            "state_type", candidate_state.get("transition", "")
        )
    )
    aliases = tuple(
        str(value) for value in candidate_state.get("answer_aliases", ())
    )
    validations: Dict[str, ActionValidation] = {}
    if transition.startswith("initial_"):
        validations[ActionType.DIRECT_ANSWER.value] = validate_answer_action(
            state_type="initial_to_direct_answer",
            answer_aliases=aliases,
            visible_information="",
            source_category=str(candidate_state.get("source_category", "")),
            question=question,
            historically_audited_direct=bool(
                candidate_state.get("historically_audited_direct", False)
            ),
        )
        validations[ActionType.IMAGE_SEARCH.value] = (
            validate_image_search_action(
                question=question,
                image_cache_entry=candidate_state.get("image_cache_entry"),
                image_exists=bool(candidate_state.get("image_exists", True)),
            )
        )
    if transition == "image_information_to_answer":
        validations[ActionType.DIRECT_ANSWER.value] = validate_answer_action(
            state_type=transition,
            answer_aliases=aliases,
            visible_information=visible_information_text(state),
            source_category=str(candidate_state.get("source_category", "")),
            question=question,
        )
    if transition == "text_information_to_answer":
        validations[ActionType.DIRECT_ANSWER.value] = validate_answer_action(
            state_type=transition,
            answer_aliases=aliases,
            visible_information=visible_information_text(state),
            source_category=str(candidate_state.get("source_category", "")),
            question=question,
        )
    target_query = str(candidate_state.get("target_query", ""))
    if target_query:
        validations[ActionType.TEXT_SEARCH.value] = (
            validate_text_search_action(
                query=target_query,
                visible_text_context=build_visible_text_context(
                    question=question,
                    history_messages=state,
                ),
                question=question,
                accepted_answer_aliases=aliases,
                text_results=candidate_state.get("text_results"),
                require_evidence_reachability=bool(
                    candidate_state.get(
                        "require_evidence_reachability", True
                    )
                ),
                maximum_question_copy_ratio_without_entity=float(
                    candidate_state.get(
                        "maximum_question_copy_ratio_without_entity", 0.85
                    )
                ),
                source_context_document_ids=candidate_state.get(
                    "source_context_document_ids", ()
                ),
            )
        )
    return validations


def compute_valid_action_set(
    candidate_state: Mapping[str, Any],
) -> Tuple[str, ...]:
    return tuple(
        action
        for action, validation in compute_action_validations(
            candidate_state
        ).items()
        if validation.executable
    )
