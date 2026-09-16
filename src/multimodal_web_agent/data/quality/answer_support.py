from __future__ import annotations

from .evidence_reachability import alias_in_text
from .rejection import SharedRejectionReason, reason_values
from .schema import ActionType, ActionValidation


EXTERNAL_RELATION_TERMS = {
    "founded", "founded by", "designed", "designed by", "named after",
    "opened", "officially open", "year", "date", "born", "died",
    "located", "headquarters", "award", "prize", "sentenced", "contract",
}


def _requires_external_fact(question: str) -> bool:
    normalized = " ".join(str(question).casefold().split())
    return any(term in normalized for term in EXTERNAL_RELATION_TERMS)


def validate_answer_action(
    *,
    state_type: str,
    answer_aliases: tuple[str, ...],
    visible_information: str,
    source_category: str,
    question: str = "",
    historically_audited_direct: bool = False,
) -> ActionValidation:
    reasons = []
    aliases = tuple(alias for alias in answer_aliases if alias.strip())
    if not aliases:
        reasons.append(SharedRejectionReason.ANSWER_UNSUPPORTED)
    if state_type == "initial_to_direct_answer":
        if (
            source_category != "search_free"
            and not historically_audited_direct
        ):
            reasons.append(SharedRejectionReason.ANSWER_UNSUPPORTED)
        if _requires_external_fact(question):
            reasons.append(SharedRejectionReason.ANSWER_UNSUPPORTED)
    elif state_type in {
        "image_information_to_answer",
        "text_information_to_answer",
    }:
        if not alias_in_text(aliases, visible_information):
            reasons.append(SharedRejectionReason.ANSWER_UNSUPPORTED)
    else:
        reasons.append(SharedRejectionReason.ANSWER_UNSUPPORTED)
    return ActionValidation(
        action_type=ActionType.DIRECT_ANSWER.value,
        executable=not reasons,
        reasons=reason_values(reasons),
    )
