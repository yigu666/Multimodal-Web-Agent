from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if hasattr(value, "detach"):
        return _json_value(value.detach().cpu().tolist())
    if hasattr(value, "tolist"):
        return _json_value(value.tolist())
    return str(value)


@dataclass(frozen=True)
class PromptPoolItem:
    prompt_uid: str
    source_data_id: str
    data_id: str
    source_split: str
    image_ref: dict[str, Any]
    image_sha256: str
    question: str
    ground_truth: str
    candidate_answers: list[str]
    category: str
    image_cache_key: str
    valid_action_set: list[str]
    entity_group_id: str
    near_duplicate_group_id: str
    source_group_id: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        return _json_value(value)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PromptPoolItem":
        raw = dict(value)
        raw.setdefault("source_data_id", raw.get("data_id", ""))
        raw.setdefault("data_id", raw.get("source_data_id", ""))
        raw.setdefault("source_split", "FVQA Train")
        raw.setdefault("image_ref", {})
        raw.setdefault("candidate_answers", [])
        raw.setdefault("valid_action_set", [])
        raw.setdefault("metadata", {})
        return cls(
            prompt_uid=str(raw["prompt_uid"]),
            source_data_id=str(raw["source_data_id"]),
            data_id=str(raw["data_id"]),
            source_split=str(raw["source_split"]),
            image_ref=dict(raw["image_ref"]),
            image_sha256=str(raw.get("image_sha256", "")),
            question=str(raw["question"]),
            ground_truth=str(raw["ground_truth"]),
            candidate_answers=[str(x) for x in raw["candidate_answers"]],
            category=str(raw["category"]),
            image_cache_key=str(raw.get("image_cache_key", raw["data_id"])),
            valid_action_set=[str(x) for x in raw["valid_action_set"]],
            entity_group_id=str(raw.get("entity_group_id", "")),
            near_duplicate_group_id=str(raw.get("near_duplicate_group_id", "")),
            source_group_id=str(raw.get("source_group_id", raw["data_id"])),
            metadata=dict(raw["metadata"]),
        )


@dataclass
class VisualInputRecord:
    image_sha256: str
    pixel_values: Any = None
    image_grid_thw: Any = None

    def to_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))


@dataclass
class RolloutRecord:
    prompt_uid: str
    rollout_uid: str
    data_id: str
    rollout_index: int
    generation_seed: int
    full_input_ids: Any = None
    attention_mask: Any = None
    position_ids: Any = None
    response_ids: Any = None
    policy_action_mask: Any = None
    information_mask: Any = None
    old_log_probs: Any = None
    base_policy_log_probs: Any = None
    behavior_policy_log_probs: Any = None
    exploration_trace: list[dict[str, Any]] = field(default_factory=list)
    exploration_metadata: dict[str, Any] = field(default_factory=dict)
    image_sha256: str = ""
    pixel_values: Any = None
    image_grid_thw: Any = None
    assistant_turn_spans: list[Any] = field(default_factory=list)
    information_spans: list[Any] = field(default_factory=list)
    generated_token_ids_by_turn: list[Any] = field(default_factory=list)
    actions: list[dict[str, Any]] = field(default_factory=list)
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    answer_text: str = ""
    answer_em: float = 0.0
    answer_f1: float = 0.0
    category: str = ""
    search_count: int = 0
    used_image_search: bool = False
    used_text_search: bool = False
    duplicate_query: bool = False
    protocol_valid: bool = False
    protocol_error: str | None = None
    terminal_reason: str = ""
    reward_components: dict[str, Any] = field(default_factory=dict)
    reward_total: float = 0.0
    base_model_hash: str = ""
    adapter_hash: str = ""
    processor_hash: str = ""
    chat_template_hash: str = ""
    generation_config_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RolloutRecord":
        names = set(cls.__dataclass_fields__)
        raw = {key: value[key] for key in names if key in value}
        return cls(**raw)

    @property
    def prompt_group_id(self) -> str:
        return self.prompt_uid


def validate_rollout_group(records: Sequence[RolloutRecord], group_size: int = 4) -> None:
    if len(records) != group_size:
        raise ValueError(f"expected group_size={group_size}, got {len(records)}")
    prompt_ids = {record.prompt_uid for record in records}
    if len(prompt_ids) != 1:
        raise ValueError("rollout group contains multiple prompt_uid values")
    if len({record.rollout_uid for record in records}) != len(records):
        raise ValueError("rollout_uid values must be unique")
    if len({record.generation_seed for record in records}) != len(records):
        raise ValueError("generation_seed values must be unique")
