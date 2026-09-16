from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import yaml


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


@dataclass(frozen=True)
class RunConfig:
    name: str
    mode: str
    seed: int
    output_dir: Path
    overwrite_output_dir: bool = False


@dataclass(frozen=True)
class DataConfig:
    train_file: Path
    dev_file: Path
    test_file: Optional[Path] = None
    schema_version: str = "protocol-sft-v0.3"
    manifest_file: Optional[Path] = None
    audit_file: Optional[Path] = None
    hashes_file: Optional[Path] = None
    master_pool_manifest_file: Optional[Path] = None
    source_parquet: Optional[Path] = None
    smoke_examples_per_transition: int = 2
    expected_smoke_examples: int = 12
    expected_train_count: Optional[int] = None
    expected_dev_count: Optional[int] = None
    expected_test_count: Optional[int] = None
    max_seq_len: int = 1536
    pixel_budget: int = 200704
    allow_test_access: bool = True
    allow_test_during_training: bool = True
    test_embargoed: bool = False


@dataclass(frozen=True)
class ModelConfig:
    path: Path
    local_files_only: bool = True
    load_in_4bit: bool = True
    compute_dtype: str = "float16"
    quant_type: str = "nf4"
    double_quant: bool = True
    freeze_visual: bool = True
    gradient_checkpointing: bool = True
    use_reentrant: bool = False


@dataclass(frozen=True)
class LoRAConfig:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    target_suffixes: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )
    excluded_name_fragments: tuple[str, ...] = (
        "visual", "vision", "merger", "image",
    )


@dataclass(frozen=True)
class SamplingConfig:
    mode: str = "natural"
    epoch_size: Optional[int] = None
    balanced_fraction: float = 0.0
    seed: int = 20260722
    transition_quotas: Mapping[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class LossWeightConfig:
    enabled: bool = False
    reason: float = 1.0
    image_search_action: float = 1.0
    text_search_open_close_tags: float = 1.0
    text_query_payload: float = 1.0
    answer_open_close_tags: float = 1.0
    answer_payload: float = 1.0
    assistant_end: float = 1.0


@dataclass(frozen=True)
class LossConfig:
    mode: str = "standard_current_turn"
    weighted_target_loss: bool = False


@dataclass(frozen=True)
class InitialRouteAuditConfig:
    approval_file: Optional[Path] = None


@dataclass(frozen=True)
class TrainingConfig:
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    max_steps: Optional[int] = None
    num_train_epochs: int = 3
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.05
    lr_scheduler_type: str = "cosine"
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    logging_steps: int = 10
    save_steps: int = 0
    evaluation_strategy: str = "epoch"
    save_strategy: str = "epoch"
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    fp16: bool = True
    bf16: bool = False
    optim: str = "paged_adamw_8bit"
    dataloader_num_workers: int = 0
    remove_unused_columns: bool = False
    report_to: tuple[str, ...] = ()


@dataclass(frozen=True)
class SFTConfig:
    project_root: Path
    objective: str
    run: RunConfig
    data: DataConfig
    model: ModelConfig
    lora: LoRAConfig
    sampling: SamplingConfig
    loss: LossConfig
    loss_weights: LossWeightConfig
    initial_route_audit: InitialRouteAuditConfig
    training: TrainingConfig
    config_path: Optional[Path] = None

    @property
    def is_smoke(self) -> bool:
        return self.run.mode == "smoke"

    @property
    def is_full(self) -> bool:
        return self.run.mode == "full"

    @property
    def is_v0_4(self) -> bool:
        return self.data.schema_version == "protocol-sft-v0.4"

    @property
    def is_format(self) -> bool:
        return (
            self.objective == "protocol_format"
            or self.data.schema_version == "protocol-format-sft-v1"
        )

    @property
    def config_hash(self) -> str:
        return sha256_json(self.to_dict(relative=False))

    def to_dict(self, *, relative: bool = False) -> Dict[str, Any]:
        def convert(value: Any) -> Any:
            if isinstance(value, Path):
                if relative:
                    try:
                        return str(value.relative_to(self.project_root)).replace("\\", "/")
                    except ValueError:
                        pass
                return str(value).replace("\\", "/")
            if isinstance(value, tuple):
                return [convert(item) for item in value]
            if isinstance(value, Mapping):
                return {str(key): convert(item) for key, item in value.items()}
            if hasattr(value, "__dataclass_fields__"):
                return {key: convert(item) for key, item in asdict(value).items()}
            return value
        return convert(self)

    def validate(self) -> None:
        if self.run.mode not in {"smoke", "full"}:
            raise ValueError("run.mode must be smoke or full")
        if self.model.compute_dtype != "float16" or self.training.bf16:
            raise ValueError("Protocol-SFT requires FP16 and forbids BF16")
        if not self.training.fp16:
            raise ValueError("Protocol-SFT requires fp16=true")
        if not self.model.load_in_4bit or self.model.quant_type.casefold() != "nf4":
            raise ValueError("Protocol-SFT requires NF4 four-bit model loading")
        if not self.model.local_files_only:
            raise ValueError("model loading must be local_files_only")
        if self.training.remove_unused_columns:
            raise ValueError("remove_unused_columns must be false for multimodal batches")
        if not self.model.freeze_visual:
            raise ValueError("visual modules must be frozen")
        if self.data.max_seq_len <= 0 or self.data.pixel_budget <= 0:
            raise ValueError("sequence and pixel budgets must be positive")
        if self.loss.mode != "standard_current_turn":
            raise ValueError("Protocol-SFT requires standard_current_turn loss")
        if self.loss.weighted_target_loss != self.loss_weights.enabled:
            raise ValueError(
                "loss.weighted_target_loss and loss_weights must agree"
            )
        if self.is_v0_4:
            if self.loss.weighted_target_loss or self.loss_weights.enabled:
                raise ValueError("Protocol-SFT v0.4 forbids weighted Target loss")
            if self.data.manifest_file is None or self.data.audit_file is None:
                raise ValueError("v0.4 requires manifest_file and audit_file")
            if self.data.hashes_file is None:
                raise ValueError("v0.4 requires hashes_file")
            if self.data.allow_test_access:
                raise ValueError("v0.4 training configs must forbid Test access")
            if self.is_full and (
                self.data.allow_test_during_training
                or not self.data.test_embargoed
            ):
                raise ValueError("v0.4 Full must keep Test embargoed")
        if self.is_format:
            if self.objective != "protocol_format":
                raise ValueError("Format SFT requires objective=protocol_format")
            if self.data.schema_version != "protocol-format-sft-v1":
                raise ValueError("Format SFT schema mismatch")
            if self.data.manifest_file is None or self.data.audit_file is None:
                raise ValueError("Format SFT requires manifest_file and audit_file")
            if self.data.hashes_file is None:
                raise ValueError("Format SFT requires hashes_file")
            if self.data.test_file is not None:
                raise ValueError("Format SFT must not configure a Test split")
            if self.data.allow_test_access or self.data.allow_test_during_training:
                raise ValueError("Format SFT forbids Test access")
            if self.loss.weighted_target_loss or self.loss_weights.enabled:
                raise ValueError("Format SFT forbids weighted Target loss")
        if self.is_smoke and self.sampling.mode == "natural":
            if self.data.smoke_examples_per_transition != 2:
                raise ValueError("Smoke must select two examples per transition")
            expected_steps = (
                10 if self.is_format else (12 if self.is_v0_4 else 4)
            )
            if self.training.max_steps != expected_steps:
                raise ValueError(
                    "Smoke must run exactly %d optimizer steps"
                    % expected_steps
                )
            if self.is_v0_4 and self.data.expected_smoke_examples != 12:
                raise ValueError("v0.4 Smoke must contain exactly 12 examples")
            if self.is_format and self.data.expected_smoke_examples != 10:
                raise ValueError("Format Smoke must contain exactly 10 examples")
        if self.is_full:
            if self.is_format:
                if (
                    self.data.expected_train_count != 900
                    or self.data.expected_dev_count != 100
                    or self.data.expected_test_count is not None
                ):
                    raise ValueError(
                        "Format Full expects Train=900, Dev=100 and no Test"
                    )
            else:
                if self.data.expected_train_count != 800 or self.data.expected_dev_count != 100:
                    raise ValueError("Full expects 800 train and 100 dev examples")
                if self.data.expected_test_count != 100:
                    raise ValueError("Full expects 100 test examples")
            if self.training.num_train_epochs != 3:
                raise ValueError("Full must run three epochs")
        if self.lora.r <= 0 or self.lora.alpha <= 0:
            raise ValueError("LoRA rank and alpha must be positive")
        if self.lora.bias != "none" or self.lora.task_type != "CAUSAL_LM":
            raise ValueError("Protocol-SFT LoRA requires bias=none and task_type=CAUSAL_LM")
        if self.sampling.mode not in {
            "natural", "mixed_transition", "initial_router_focused",
            "format_exposure_balance",
        }:
            raise ValueError(
                "sampling.mode must be natural, mixed_transition or "
                "initial_router_focused"
            )
        if self.sampling.mode == "format_exposure_balance":
            expected = {
                "initial_to_direct_answer": 200,
                "initial_to_image_search": 200,
                "image_information_to_answer": 200,
                "image_information_to_text_search": 200,
                "text_information_to_answer": 200,
            }
            if not self.is_format or not self.is_full:
                raise ValueError(
                    "format_exposure_balance is only valid for Format Full"
                )
            if self.sampling.epoch_size != 1000:
                raise ValueError("Format Full epoch_size must be 1000")
            if dict(self.sampling.transition_quotas) != expected:
                raise ValueError("Format Full transition quotas are not frozen")
        if self.sampling.mode == "mixed_transition":
            if self.sampling.epoch_size is None or self.sampling.epoch_size <= 0:
                raise ValueError("mixed_transition sampling requires a positive epoch_size")
            if not 0.0 <= self.sampling.balanced_fraction <= 1.0:
                raise ValueError("sampling.balanced_fraction must be between zero and one")
            balanced_count = round(
                self.sampling.epoch_size * self.sampling.balanced_fraction
            )
            if balanced_count % 6:
                raise ValueError("balanced sample count must be divisible by six")
            if self.is_smoke:
                if (
                    self.sampling.epoch_size != 60
                    or self.sampling.balanced_fraction != 1.0
                    or self.training.max_steps != 12
                ):
                    raise ValueError(
                        "Rebalanced Smoke requires 60 balanced samples and 12 steps"
                    )
            if self.is_full and (
                self.sampling.epoch_size != 1000
                or self.sampling.balanced_fraction != 0.60
            ):
                raise ValueError(
                    "Rebalanced Full requires epoch_size=1000 and balanced_fraction=0.60"
                )
        if self.sampling.mode == "initial_router_focused":
            expected_smoke = {
                "initial_to_direct_answer": 8,
                "initial_to_image_search": 8,
                "initial_to_text_search": 8,
                "image_information_to_answer": 4,
                "image_information_to_text_search": 4,
                "text_information_to_answer": 4,
            }
            expected_full = {
                "initial_to_direct_answer": 300,
                "initial_to_image_search": 300,
                "initial_to_text_search": 300,
                "image_information_to_answer": 100,
                "image_information_to_text_search": 100,
                "text_information_to_answer": 100,
            }
            expected = expected_smoke if self.is_smoke else expected_full
            if dict(self.sampling.transition_quotas) != expected:
                raise ValueError(
                    "initial_router_focused transition quotas differ from "
                    "the frozen Router-fix contract"
                )
            if self.sampling.epoch_size != sum(expected.values()):
                raise ValueError(
                    "Router-fix epoch_size must equal transition quota sum"
                )
            if self.is_smoke and self.training.max_steps != 36:
                raise ValueError("Router-fix Smoke requires 36 steps")
            if self.initial_route_audit.approval_file is None:
                raise ValueError(
                    "Router-fix requires an Initial Route approval file"
                )
            if not self.loss_weights.enabled:
                raise ValueError("Router-fix requires weighted Target loss")
        if self.loss_weights.enabled:
            for name in (
                "reason", "image_search_action",
                "text_search_open_close_tags", "text_query_payload",
                "answer_open_close_tags", "answer_payload",
                "assistant_end",
            ):
                value = float(getattr(self.loss_weights, name))
                if not math.isfinite(value) or not value > 0:
                    raise ValueError(
                        f"loss_weights.{name} must be finite and positive"
                    )


def _section(raw: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = raw.get(name, {})
    if not isinstance(value, Mapping):
        raise ValueError("config section %s must be a mapping" % name)
    return value


def load_config(path: Path, project_root: Optional[Path] = None) -> SFTConfig:
    path = Path(path).resolve()
    root = Path(project_root or path.parents[2]).resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    run = _section(raw, "run")
    data = _section(raw, "data")
    model = _section(raw, "model")
    lora = _section(raw, "lora")
    sampling = _section(raw, "sampling")
    loss = _section(raw, "loss")
    loss_weights = _section(raw, "loss_weights")
    initial_route_audit = _section(raw, "initial_route_audit")
    training = _section(raw, "training")
    eval_strategy = training.get("evaluation_strategy", training.get("eval_strategy", "epoch"))
    config = SFTConfig(
        project_root=root,
        objective=str(raw.get("objective", "protocol_sft")),
        config_path=path,
        run=RunConfig(
            name=str(run.get("name", path.stem)),
            mode=str(run.get("mode", "smoke")),
            seed=int(run.get("seed", 20260722)),
            output_dir=_resolve(root, str(run.get("output_dir", "outputs/protocol_sft"))),
            overwrite_output_dir=bool(run.get("overwrite_output_dir", False)),
        ),
        data=DataConfig(
            train_file=_resolve(root, str(data.get("train_file", "data/processed/protocol_sft_v0_3/train.jsonl"))),
            dev_file=_resolve(root, str(data.get("dev_file", "data/processed/protocol_sft_v0_3/dev.jsonl"))),
            test_file=_resolve(root, str(data["test_file"])) if data.get("test_file") else None,
            schema_version=str(
                data.get("schema_version", "protocol-sft-v0.3")
            ),
            manifest_file=(
                _resolve(root, str(data["manifest_file"]))
                if data.get("manifest_file")
                else None
            ),
            audit_file=(
                _resolve(root, str(data["audit_file"]))
                if data.get("audit_file")
                else None
            ),
            hashes_file=(
                _resolve(root, str(data["hashes_file"]))
                if data.get("hashes_file")
                else None
            ),
            master_pool_manifest_file=(
                _resolve(root, str(data["master_pool_manifest_file"]))
                if data.get("master_pool_manifest_file")
                else None
            ),
            source_parquet=_resolve(root, str(data["source_parquet"])) if data.get("source_parquet") else None,
            smoke_examples_per_transition=int(data.get("smoke_examples_per_transition", 2)),
            expected_smoke_examples=int(
                data.get("expected_smoke_examples", 12)
            ),
            expected_train_count=int(data["expected_train_count"]) if data.get("expected_train_count") is not None else None,
            expected_dev_count=int(data["expected_dev_count"]) if data.get("expected_dev_count") is not None else None,
            expected_test_count=int(data["expected_test_count"]) if data.get("expected_test_count") is not None else None,
            max_seq_len=int(data.get("max_seq_len", 1536)),
            pixel_budget=int(data.get("pixel_budget", 200704)),
            allow_test_access=bool(data.get("allow_test_access", True)),
            allow_test_during_training=bool(
                data.get("allow_test_during_training", True)
            ),
            test_embargoed=bool(data.get("test_embargoed", False)),
        ),
        model=ModelConfig(
            path=_resolve(root, str(model.get("path", "models/Qwen2.5-VL-3B-Instruct"))),
            local_files_only=bool(model.get("local_files_only", True)),
            load_in_4bit=bool(model.get("load_in_4bit", True)),
            compute_dtype=str(model.get("compute_dtype", "float16")),
            quant_type=str(model.get("quant_type", "nf4")),
            double_quant=bool(model.get("double_quant", True)),
            freeze_visual=bool(model.get("freeze_visual", True)),
            gradient_checkpointing=bool(model.get("gradient_checkpointing", True)),
            use_reentrant=bool(model.get("use_reentrant", False)),
        ),
        lora=LoRAConfig(
            r=int(lora.get("r", 16)),
            alpha=int(lora.get("alpha", 32)),
            dropout=float(lora.get("dropout", 0.05)),
            bias=str(lora.get("bias", "none")),
            task_type=str(lora.get("task_type", "CAUSAL_LM")),
            target_suffixes=tuple(str(item) for item in lora.get("target_suffixes", LoRAConfig.target_suffixes)),
            excluded_name_fragments=tuple(str(item).lower() for item in lora.get("excluded_name_fragments", LoRAConfig.excluded_name_fragments)),
        ),
        sampling=SamplingConfig(
            mode=str(sampling.get("mode", "natural")),
            epoch_size=(
                int(sampling["epoch_size"])
                if sampling.get("epoch_size") is not None
                else None
            ),
            balanced_fraction=float(sampling.get("balanced_fraction", 0.0)),
            seed=int(sampling.get("seed", run.get("seed", 20260722))),
            transition_quotas={
                str(key): int(value)
                for key, value in _section(
                    sampling, "transition_quotas"
                ).items()
            },
        ),
        loss=LossConfig(
            mode=str(loss.get("mode", "standard_current_turn")),
            weighted_target_loss=bool(
                loss.get("weighted_target_loss", bool(loss_weights))
            ),
        ),
        loss_weights=LossWeightConfig(
            enabled=bool(loss_weights),
            reason=float(loss_weights.get("reason", 1.0)),
            image_search_action=float(
                loss_weights.get("image_search_action", 1.0)
            ),
            text_search_open_close_tags=float(
                loss_weights.get("text_search_open_close_tags", 1.0)
            ),
            text_query_payload=float(
                loss_weights.get("text_query_payload", 1.0)
            ),
            answer_open_close_tags=float(
                loss_weights.get("answer_open_close_tags", 1.0)
            ),
            answer_payload=float(
                loss_weights.get("answer_payload", 1.0)
            ),
            assistant_end=float(
                loss_weights.get("assistant_end", 1.0)
            ),
        ),
        initial_route_audit=InitialRouteAuditConfig(
            approval_file=(
                _resolve(root, str(initial_route_audit["approval_file"]))
                if initial_route_audit.get("approval_file")
                else None
            ),
        ),
        training=TrainingConfig(
            per_device_train_batch_size=int(training.get("per_device_train_batch_size", 1)),
            per_device_eval_batch_size=int(training.get("per_device_eval_batch_size", 1)),
            gradient_accumulation_steps=int(training.get("gradient_accumulation_steps", 1)),
            max_steps=int(training["max_steps"]) if training.get("max_steps") is not None else None,
            num_train_epochs=int(training.get("num_train_epochs", 3)),
            learning_rate=float(training.get("learning_rate", 2e-4)),
            warmup_ratio=float(training.get("warmup_ratio", 0.05)),
            lr_scheduler_type=str(training.get("lr_scheduler_type", "cosine")),
            weight_decay=float(training.get("weight_decay", 0.0)),
            max_grad_norm=float(training.get("max_grad_norm", 1.0)),
            logging_steps=int(training.get("logging_steps", 10)),
            save_steps=int(training.get("save_steps", 0)),
            evaluation_strategy=str(eval_strategy),
            save_strategy=str(training.get("save_strategy", "epoch")),
            save_total_limit=int(training.get("save_total_limit", 3)),
            load_best_model_at_end=bool(training.get("load_best_model_at_end", True)),
            metric_for_best_model=str(training.get("metric_for_best_model", "eval_loss")),
            greater_is_better=bool(training.get("greater_is_better", False)),
            fp16=bool(training.get("fp16", True)),
            bf16=bool(training.get("bf16", False)),
            optim=str(training.get("optim", "paged_adamw_8bit")),
            dataloader_num_workers=int(training.get("dataloader_num_workers", 0)),
            remove_unused_columns=bool(training.get("remove_unused_columns", False)),
            report_to=tuple(str(item) for item in training.get("report_to", [])),
        ),
    )
    config.validate()
    return config
