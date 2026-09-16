from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .rejection import SharedRejectionReason


@dataclass(frozen=True)
class StagePolicyResult:
    accepted: bool
    canonical_action: str
    reasons: tuple[str, ...]
    metadata: dict[str, Any]


class SFTCanonicalActionPolicy:
    repair_mode = "reject_only"

    def select(
        self,
        *,
        original_action: str,
        valid_action_set: Sequence[str],
        evidence_chain_complete: bool,
    ) -> StagePolicyResult:
        reasons = []
        if original_action not in valid_action_set:
            reasons.append(
                SharedRejectionReason.CONFLICTING_ROUTE_LABEL.value
            )
        if not evidence_chain_complete:
            reasons.append(
                SharedRejectionReason.UNREACHABLE_EVIDENCE.value
            )
        return StagePolicyResult(
            accepted=not reasons,
            canonical_action=original_action if not reasons else "",
            reasons=tuple(sorted(set(reasons))),
            metadata={
                "repair_mode": self.repair_mode,
                "automatic_route_conversion": False,
                "automatic_query_rewrite": False,
            },
        )


class GRPOExecutabilityPolicy:
    def metadata(
        self,
        *,
        source_data_id: str,
        state_id: str,
        valid_action_set: Sequence[str],
        blocked_actions: Mapping[str, Sequence[str]],
        evidence_reachable: bool,
    ) -> dict[str, Any]:
        valid = tuple(valid_action_set)
        return {
            "source_data_id": source_data_id,
            "state_id": state_id,
            "valid_action_set": list(valid),
            "blocked_actions": {
                key: list(value) for key, value in blocked_actions.items()
            },
            "no_valid_action": not valid,
            "evidence_reachable": bool(evidence_reachable),
        }

    def accepts(self, metadata: Mapping[str, Any]) -> bool:
        return (
            not metadata.get("no_valid_action", True)
            and bool(metadata.get("evidence_reachable", False))
        )


class EvaluationExecutabilityPolicy:
    def decide(
        self,
        *,
        valid_action_set: Sequence[str],
        evidence_reachable: bool,
        split_leak: bool,
        strictly_verifiable: bool = True,
    ) -> str:
        if split_leak or not valid_action_set or not evidence_reachable:
            return "reject"
        if not strictly_verifiable:
            return "quarantine"
        return "main_eval"
