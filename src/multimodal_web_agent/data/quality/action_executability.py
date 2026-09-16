from __future__ import annotations

from typing import Any, Iterable, Mapping

from multimodal_web_agent.agent import ActionType as ProtocolActionType
from multimodal_web_agent.agent import parse_action

from .answer_support import validate_answer_action
from .image_search_support import validate_image_search_action
from .query_executability import validate_text_search_action
from .rejection import SharedRejectionReason, reason_values
from .schema import ActionValidation
from .visible_context import (
    build_visible_text_context,
    visible_information_text,
)


def _history_messages(state: Iterable[Any]) -> list[Any]:
    return list(state)


def _question_from_state(candidate: Mapping[str, Any]) -> str:
    return str(candidate.get("question", "")).strip()


def validate_target_action(
    candidate: Mapping[str, Any],
) -> ActionValidation:
    target = str(candidate.get("target", ""))
    parsed = parse_action(target)
    if not parsed.valid or parsed.action_type is None:
        return ActionValidation(
            action_type="",
            executable=False,
            reasons=(SharedRejectionReason.STRICT_PARSER_INVALID.value,),
        )
    state_type = str(candidate.get("state_type", candidate.get("transition", "")))
    question = _question_from_state(candidate)
    state = _history_messages(candidate.get("state", ()))
    visible_context = build_visible_text_context(
        question=question,
        history_messages=state,
    )
    aliases = tuple(str(value) for value in candidate.get("answer_aliases", ()))
    if parsed.action_type == ProtocolActionType.ANSWER:
        return validate_answer_action(
            state_type=state_type,
            answer_aliases=aliases,
            visible_information=visible_information_text(state),
            source_category=str(candidate.get("source_category", "")),
            question=question,
            historically_audited_direct=bool(
                candidate.get("historically_audited_direct", False)
            ),
        )
    if parsed.action_type == ProtocolActionType.IMAGE_SEARCH:
        return validate_image_search_action(
            question=question,
            image_cache_entry=candidate.get("image_cache_entry"),
            image_exists=bool(candidate.get("image_exists", True)),
        )
    if parsed.action_type == ProtocolActionType.TEXT_SEARCH:
        return validate_text_search_action(
            query=parsed.content,
            visible_text_context=visible_context,
            question=question,
            accepted_answer_aliases=aliases,
            text_results=candidate.get("text_results"),
            require_evidence_reachability=bool(
                candidate.get("require_evidence_reachability", True)
            ),
            maximum_question_copy_ratio_without_entity=float(
                candidate.get(
                    "maximum_question_copy_ratio_without_entity", 0.85
                )
            ),
            source_context_document_ids=candidate.get(
                "source_context_document_ids", ()
            ),
        )
    return ActionValidation(
        action_type="",
        executable=False,
        reasons=reason_values([SharedRejectionReason.NO_VALID_ACTION]),
    )
