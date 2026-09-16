from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, Tuple


class GateDecision(str, Enum):
    ACCEPT = "accept"
    REJECT = "reject"
    QUARANTINE = "quarantine"


class ActionType(str, Enum):
    DIRECT_ANSWER = "answer"
    IMAGE_SEARCH = "image_search"
    TEXT_SEARCH = "text_search"


@dataclass(frozen=True)
class VisibleEntity:
    value: str
    provenance: str
    source_span: str
    visible_in_current_text_state: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ActionValidation:
    action_type: str
    executable: bool
    reasons: Tuple[str, ...] = ()
    visible_entities: Tuple[VisibleEntity, ...] = ()
    evidence_document_ids: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GateResult:
    source_data_id: str
    state_type: str
    decision: GateDecision
    valid_actions: Tuple[str, ...]
    invalid_action_reasons: Dict[str, Tuple[str, ...]]
    rejection_reasons: Tuple[str, ...]
    entity_group_id: str
    near_duplicate_group_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["decision"] = self.decision.value
        return value
