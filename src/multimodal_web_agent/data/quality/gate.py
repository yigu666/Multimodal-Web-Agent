from __future__ import annotations

import re
from typing import Any, Dict, Iterable, Mapping, Sequence

from multimodal_web_agent.agent import ActionType as ProtocolActionType
from multimodal_web_agent.agent import parse_action

from .action_executability import validate_target_action
from .duplicate_grouper import (
    entity_group_id,
    near_duplicate_group_id,
)
from .evidence_reachability import alias_in_text
from .image_search_support import image_cache_titles
from .rejection import SharedRejectionReason, reason_values
from .schema import GateDecision, GateResult
from .stage_policy import SFTCanonicalActionPolicy
from .valid_action_set import compute_action_validations
from .visible_context import build_visible_text_context
from .visible_context import visible_information_text


DOCUMENT_RE = re.compile(r"\[cache_document=([^\]]+)\]")


def _message_dict(message: Any) -> Dict[str, Any]:
    if isinstance(message, Mapping):
        return dict(message)
    return {
        "role": str(getattr(message, "role", "")),
        "content": str(getattr(message, "content", "")),
        "trainable": bool(getattr(message, "trainable", False)),
    }


def _text_results_from_state(state: Iterable[Any]) -> list[dict]:
    results = []
    for message in state:
        value = _message_dict(message)
        if value["role"] != "tool":
            continue
        content = value["content"]
        if "[Text Search Results]" not in content:
            continue
        for line in content.splitlines():
            stripped = line.strip()
            if not stripped or not stripped[:1].isdigit():
                continue
            match = DOCUMENT_RE.search(stripped)
            document_id = match.group(1) if match else ""
            text = DOCUMENT_RE.sub("", stripped)
            text = re.sub(r"^\d+\.\s*", "", text).strip()
            results.append(
                {"document_id": document_id, "title": text}
            )
    return results


def _future_text_results(
    steps: Sequence[Mapping[str, Any]], step_index: int
) -> list[dict]:
    for later in steps[step_index + 1:]:
        results = _text_results_from_state(later.get("state", ()))
        if results:
            return results
    return []


def _future_text_provenance(
    steps: Sequence[Mapping[str, Any]], step_index: int
) -> dict[str, Any]:
    for later in steps[step_index + 1:]:
        provenance = dict(later.get("information_provenance") or {})
        if provenance.get("backend") or provenance.get("document_ids"):
            return provenance
    return {}


def _protocol_action_name(target: str) -> str:
    parsed = parse_action(target)
    if not parsed.valid or parsed.action_type is None:
        return ""
    return {
        ProtocolActionType.ANSWER: "answer",
        ProtocolActionType.IMAGE_SEARCH: "image_search",
        ProtocolActionType.TEXT_SEARCH: "text_search",
    }[parsed.action_type]


class SharedExecutabilityGate:
    schema_version = "shared-executability-gate-v0.2"

    def __init__(
        self,
        *,
        maximum_question_copy_ratio_without_entity: float = 0.85,
        require_evidence_reachability: bool = True,
    ):
        self.maximum_question_copy_ratio_without_entity = float(
            maximum_question_copy_ratio_without_entity
        )
        self.require_evidence_reachability = bool(
            require_evidence_reachability
        )
        self.sft_policy = SFTCanonicalActionPolicy()

    def evaluate_trajectory(
        self,
        trajectory: Mapping[str, Any],
        *,
        image_cache_entry: Any | None,
        fingerprint: Mapping[str, Any] | None = None,
        historically_audited_direct: bool = False,
    ) -> Dict[str, Any]:
        source_data_id = str(trajectory["source_data_id"])
        question = str(trajectory["question"])
        source = dict(trajectory.get("source", {}))
        aliases = tuple(str(x) for x in trajectory.get("accepted_answers", ()))
        steps = [dict(step) for step in trajectory.get("steps", ())]
        titles = image_cache_titles(image_cache_entry)
        entity_group = entity_group_id(titles=titles, question=question)
        fingerprint = dict(fingerprint or {})
        near_group = str(
            fingerprint.get("near_duplicate_group_id")
            or near_duplicate_group_id(source_data_id=source_data_id)
        )
        state_results = []
        all_reasons = []
        all_valid = []

        for index, step in enumerate(steps):
            parsed = parse_action(str(step.get("target", "")))
            target_query = (
                parsed.content
                if parsed.valid
                and parsed.action_type == ProtocolActionType.TEXT_SEARCH
                else ""
            )
            text_results = (
                _future_text_results(steps, index)
                if target_query
                else _text_results_from_state(step.get("state", ()))
            )
            provenance = dict(step.get("information_provenance") or {})
            candidate = {
                "source_data_id": source_data_id,
                "question": question,
                "state": [
                    _message_dict(message)
                    for message in step.get("state", ())
                ],
                "state_type": str(step.get("transition", "")),
                "transition": str(step.get("transition", "")),
                "target": str(step.get("target", "")),
                "target_query": target_query,
                "answer_aliases": aliases,
                "source_category": str(source.get("category", "")),
                "historically_audited_direct": historically_audited_direct,
                "image_cache_entry": image_cache_entry,
                "image_exists": bool(step.get("image_refs")),
                "text_results": text_results,
                "source_context_document_ids": provenance.get(
                    "document_ids", ()
                ),
                "require_evidence_reachability":
                    self.require_evidence_reachability,
                "maximum_question_copy_ratio_without_entity":
                    self.maximum_question_copy_ratio_without_entity,
            }
            validations = compute_action_validations(candidate)
            valid_actions = tuple(
                action
                for action, validation in validations.items()
                if validation.executable
            )
            original_action = _protocol_action_name(candidate["target"])
            target_validation = validate_target_action(candidate)
            policy = self.sft_policy.select(
                original_action=original_action,
                valid_action_set=valid_actions,
                evidence_chain_complete=target_validation.executable,
            )
            reasons = list(target_validation.reasons) + list(policy.reasons)
            future_provenance = (
                _future_text_provenance(steps, index)
                if target_query
                else {}
            )
            if (
                target_query
                and future_provenance.get(
                    "evidence_hit_leave_one_source_out"
                )
                is False
            ):
                reasons.append(
                    SharedRejectionReason.UNREACHABLE_EVIDENCE.value
                )
            if (
                candidate["state_type"]
                == "image_information_to_text_search"
                and alias_in_text(
                    aliases,
                    visible_information_text(candidate["state"]),
                )
            ):
                reasons.append(
                    SharedRejectionReason.ANSWER_ALREADY_VISIBLE.value
                )
            if not valid_actions:
                reasons.append(SharedRejectionReason.NO_VALID_ACTION.value)
            all_reasons.extend(reasons)
            all_valid.extend(valid_actions)
            state_result = GateResult(
                source_data_id=source_data_id,
                state_type=candidate["state_type"],
                decision=(
                    GateDecision.ACCEPT if not reasons
                    else GateDecision.REJECT
                ),
                valid_actions=valid_actions,
                invalid_action_reasons={
                    action: validation.reasons
                    for action, validation in validations.items()
                    if not validation.executable
                },
                rejection_reasons=reason_values(reasons),
                entity_group_id=entity_group,
                near_duplicate_group_id=near_group,
                metadata={
                    "original_action": original_action,
                    "target_query": target_query,
                    "visible_text_context": build_visible_text_context(
                        question=question,
                        history_messages=candidate["state"],
                    ),
                    "evidence_document_ids": list(
                        target_validation.evidence_document_ids
                    ),
                    "visible_entities": [
                        entity.to_dict()
                        for entity in target_validation.visible_entities
                    ],
                    "evidence_hit_leave_one_source_out": (
                        future_provenance.get(
                            "evidence_hit_leave_one_source_out"
                        )
                    ),
                },
            )
            state_results.append(state_result.to_dict())

        rejection_reasons = reason_values(all_reasons)
        return {
            "shared_executability_gate_schema": self.schema_version,
            "source_data_id": source_data_id,
            "route": str(trajectory.get("route", "")),
            "decision": (
                GateDecision.REJECT.value
                if rejection_reasons
                else GateDecision.ACCEPT.value
            ),
            "rejection_reasons": list(rejection_reasons),
            "valid_action_set_by_state": {
                result["state_type"]: result["valid_actions"]
                for result in state_results
            },
            "canonical_sft_action_by_state": {
                result["state_type"]: result["metadata"]["original_action"]
                for result in state_results
                if result["decision"] == GateDecision.ACCEPT.value
            },
            "state_validations": state_results,
            "entity_group_id": entity_group,
            "near_duplicate_group_id": near_group,
            "source_hash": fingerprint.get("source_hash", ""),
            "image_dhash": fingerprint.get("image_dhash", ""),
            "image_dimensions": list(
                fingerprint.get("image_dimensions", ())
            ),
            "cache_hash": str(
                getattr(image_cache_entry, "cache_file_sha256", "")
            ),
            "evidence_document_ids": sorted(
                {
                    document_id
                    for result in state_results
                    for document_id in result["metadata"].get(
                        "evidence_document_ids", ()
                    )
                }
            ),
            "visible_entity_provenance": [
                entity
                for validation in state_results
                for entity in validation["metadata"].get(
                    "visible_entities", ()
                )
            ],
            "repair_mode": "reject_only",
        }
