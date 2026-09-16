from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import re
from typing import Any, Mapping


SCHEMA_VERSION = "unified-agent-eval-v1"
V1_1_SCHEMA_VERSION = "unified-agent-eval-v1-1"
EMBARGO_SCHEMA_VERSION = "unified-agent-eval-v1-test-embargo"
V1_1_EMBARGO_SCHEMA_VERSION = "unified-agent-eval-v1-1-test-embargo"
REGISTRY_SCHEMA_VERSION = "unified-agent-eval-v1-model-registry"
V1_1_REGISTRY_SCHEMA_VERSION = "unified-agent-eval-v1-1-model-registry"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class UnifiedTaskType(str, Enum):
    SEARCH_FREE = "search_free"
    VISUAL_SEARCH_REQUIRED = "visual_search_required"
    TEXT_SEARCH_REQUIRED = "text_search_required"
    MIXED_SEARCH_REQUIRED = "mixed_search_required"


TASK_TYPES = tuple(item.value for item in UnifiedTaskType)


@dataclass(frozen=True)
class UnifiedEvalExample:
    eval_id: str
    source_dataset: str
    source_data_id: str
    question: str
    image_path: str
    image_sha256: str
    answer_aliases: tuple[str, ...]
    task_type: str
    search_required: bool
    maximum_agent_turns: int = 3
    maximum_tool_calls: int = 2
    maximum_image_search_calls: int = 1
    maximum_text_search_calls: int = 1
    source_metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema_version not in {SCHEMA_VERSION, V1_1_SCHEMA_VERSION}:
            raise ValueError("Unified Eval schema version mismatch")
        if not self.eval_id or not self.source_dataset or not self.source_data_id:
            raise ValueError("evaluation identifiers must be non-empty")
        if not self.question.strip() or not self.image_path.strip():
            raise ValueError("question and image path must be non-empty")
        if not SHA256_RE.fullmatch(self.image_sha256):
            raise ValueError("image_sha256 must be a lowercase SHA256")
        if not self.answer_aliases or any(
            not str(alias).strip() for alias in self.answer_aliases
        ):
            raise ValueError("answer aliases must be non-empty")
        task_type = UnifiedTaskType(self.task_type)
        expected_search = task_type != UnifiedTaskType.SEARCH_FREE
        if self.search_required is not expected_search:
            raise ValueError("search_required disagrees with task_type")
        budgets = (
            self.maximum_agent_turns,
            self.maximum_tool_calls,
            self.maximum_image_search_calls,
            self.maximum_text_search_calls,
        )
        if budgets != (3, 2, 1, 1):
            raise ValueError("Unified Eval budgets are frozen at 3/2/1/1")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "UnifiedEvalExample":
        item = cls(
            eval_id=str(value["eval_id"]),
            source_dataset=str(value["source_dataset"]),
            source_data_id=str(value["source_data_id"]),
            question=str(value["question"]),
            image_path=str(value["image_path"]),
            image_sha256=str(value["image_sha256"]),
            answer_aliases=tuple(str(x) for x in value["answer_aliases"]),
            task_type=str(value["task_type"]),
            search_required=bool(value["search_required"]),
            maximum_agent_turns=int(value.get("maximum_agent_turns", 3)),
            maximum_tool_calls=int(value.get("maximum_tool_calls", 2)),
            maximum_image_search_calls=int(
                value.get("maximum_image_search_calls", 1)
            ),
            maximum_text_search_calls=int(
                value.get("maximum_text_search_calls", 1)
            ),
            source_metadata=dict(value.get("source_metadata", {})),
            schema_version=str(value.get("schema_version", SCHEMA_VERSION)),
        )
        item.validate()
        return item


@dataclass(frozen=True)
class EvaluatedModel:
    model_id: str
    stage: str
    base_model_path: str
    adapter_path: str | None
    model_tree_sha256: str
    adapter_tree_sha256: str | None
    parameter_count: int
    base_parameter_count: int = 0
    adapter_parameter_count: int = 0

    def validate(self) -> None:
        if self.model_id not in {"raw", "sft", "reward_v21", "stage2"}:
            raise ValueError("unsupported Unified Eval model_id")
        expected_stage = {
            "raw": "raw",
            "sft": "protocol_format_sft",
            "reward_v21": "reward_v21_full",
            "stage2": "stage2_continued_grpo",
        }[self.model_id]
        if self.stage != expected_stage:
            raise ValueError("model stage/model_id mismatch")
        if not self.base_model_path:
            raise ValueError("base_model_path is required")
        if not SHA256_RE.fullmatch(self.model_tree_sha256):
            raise ValueError("model tree fingerprint is invalid")
        if self.model_id == "raw":
            if self.adapter_path is not None or self.adapter_tree_sha256 is not None:
                raise ValueError("Raw model must not register an Adapter")
        elif bool(self.adapter_path) != bool(self.adapter_tree_sha256):
            raise ValueError(
                "Adapter path and Adapter fingerprint must be registered together"
            )
        elif self.model_id in {"sft", "reward_v21", "stage2"} and not self.adapter_path:
            raise ValueError("Adapter model requires an Adapter fingerprint")
        elif self.adapter_tree_sha256 is not None and not SHA256_RE.fullmatch(
            self.adapter_tree_sha256
        ):
            raise ValueError("optional Adapter fingerprint is invalid")
        if self.parameter_count <= 0:
            raise ValueError("parameter_count must be positive")
        if self.base_parameter_count < 0 or self.adapter_parameter_count < 0:
            raise ValueError("parameter component counts cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvaluatedModel":
        model = cls(
            model_id=str(value["model_id"]),
            stage=str(value["stage"]),
            base_model_path=str(value["base_model_path"]),
            adapter_path=(
                str(value["adapter_path"])
                if value.get("adapter_path") is not None else None
            ),
            model_tree_sha256=str(value["model_tree_sha256"]),
            adapter_tree_sha256=(
                str(value["adapter_tree_sha256"])
                if value.get("adapter_tree_sha256") is not None else None
            ),
            parameter_count=int(value["parameter_count"]),
            base_parameter_count=int(value.get("base_parameter_count", 0)),
            adapter_parameter_count=int(
                value.get("adapter_parameter_count", 0)
            ),
        )
        model.validate()
        return model
