from __future__ import annotations

import gc
import json
import math
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from multimodal_web_agent.data.protocol_sft.schema import StateActionExample

from .audit import audit_mask_examples, audit_target_weight_examples
from .checkpoint import (
    changed_parameter_count,
    data_manifest_hash,
    load_adapter_into_model,
    save_adapter_checkpoint,
    snapshot_trainable_parameters,
    validate_resume_checkpoint,
)
from .checkpoint_selection import (
    CheckpointCandidate,
    NoNonCollapsedCheckpointError,
    materialize_selection,
    select_checkpoint,
)
from .format_checkpoint_selection import (
    materialize_format_selection,
    select_format_checkpoint,
)
from .format_metrics import (
    evaluate_format_generation_records,
    format_gate_report,
)
from .collator import ProtocolSFTCollator
from .config import SFTConfig, sha256_file, sha256_json
from .dataset import (
    ProtocolSFTDataset,
    load_protocol_splits,
    load_split,
    select_format_smoke_subset,
    select_smoke_subset,
)
from .generation import generate_records
from .masking import TokenizedExample, tokenize_current_turn
from .metrics import check_format_gates, evaluate_generation_records
from .model_factory import (
    environment_versions,
    load_processor,
    load_qwen_base,
    load_qwen_lora,
)
from .renderer import ProtocolRenderer
from .sampler import (
    EXPECTED_TRAIN_TRANSITION_COUNTS,
    EXPECTED_V0_4_TRAIN_TRANSITION_COUNTS,
    DeterministicInitialRouterSampler,
    DeterministicFormatExposureSampler,
    DeterministicMixedTransitionSampler,
    InitialRouterSamplingConfig,
    TransitionSamplingConfig,
)
from .test_embargo import initialize_embargo
from .v0_4_contract import (
    experiment_data_provenance,
    validate_v0_4_training_contract,
)
from .route_audit import validate_initial_route_approval
from .weighted_loss import weighted_causal_lm_loss


def _json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _data_manifest_path(config: SFTConfig) -> Path:
    if config.data.manifest_file is not None:
        return config.data.manifest_file
    return (
        config.project_root
        / "data"
        / "manifests"
        / "protocol_sft_v0_3_manifest.json"
    )


def _validate_format_data_contract(
    config: SFTConfig,
) -> tuple[Dict[str, Any], Dict[str, List[StateActionExample]]]:
    if config.data.manifest_file is None or config.data.audit_file is None:
        raise ValueError("Format data contract files are missing")
    manifest = json.loads(
        config.data.manifest_file.read_text(encoding="utf-8")
    )
    audit = json.loads(config.data.audit_file.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "protocol-format-sft-v1":
        raise ValueError("Format manifest schema mismatch")
    if audit.get("dataset_schema") != "protocol-format-sft-v1":
        raise ValueError("Format audit schema mismatch")
    if audit.get("passed") is not True:
        raise ValueError("Format data audit has not passed")
    if audit.get("test_file_present") is not False:
        raise ValueError("Format data view must not contain Test")
    if config.data.test_file is not None or config.data.allow_test_access:
        raise ValueError("Format training attempted to configure Test access")
    splits = {
        "train": load_split(
            config.data.train_file, config.data.expected_train_count
        ),
        "dev": load_split(
            config.data.dev_file, config.data.expected_dev_count
        ),
    }
    return {
        "dataset_schema": manifest["schema_version"],
        "dataset_audit_passed": True,
        "test_accessed": False,
        "state_action_examples": audit.get("state_action_examples"),
        "split_counts": audit.get("split_counts"),
    }, splits


class TokenizedListDataset(Dataset):
    def __init__(self, values: Sequence[TokenizedExample]):
        self.values = list(values)

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index: int) -> TokenizedExample:
        return self.values[index]


def _load_images(config: SFTConfig, examples: Sequence[StateActionExample]) -> List[Any]:
    dataset = ProtocolSFTDataset(examples, source_parquet=config.data.source_parquet)
    return [dataset[index].image for index in range(len(dataset))]


def prepare_smoke_subset(config: SFTConfig, train_examples: Sequence[StateActionExample]) -> List[StateActionExample]:
    selected = (
        select_format_smoke_subset(
            train_examples, config.data.smoke_examples_per_transition
        )
        if config.is_format
        else select_smoke_subset(
            train_examples, config.data.smoke_examples_per_transition
        )
    )
    subset_path = config.run.output_dir / "smoke_subset.jsonl"
    subset_path.parent.mkdir(parents=True, exist_ok=True)
    with subset_path.open("w", encoding="utf-8", newline="\n") as handle:
        for example in selected:
            handle.write(json.dumps(example.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
    _json_write(
        config.run.output_dir / "smoke_subset_manifest.json",
        {
            "schema_version": config.data.schema_version,
            "sample_ids": [example.example_id for example in selected],
            "examples_checked": len(selected),
            "examples_per_transition": config.data.smoke_examples_per_transition,
            "transitions": {transition: 2 for transition in sorted({example.transition for example in selected})},
        },
    )
    return selected


def _tokenize_examples(
    processor: Any,
    renderer: ProtocolRenderer,
    examples: Sequence[StateActionExample],
    images: Sequence[Any],
    max_seq_len: int,
    loss_weights: Any = None,
) -> List[TokenizedExample]:
    return [
        tokenize_current_turn(
            processor,
            renderer,
            example,
            image,
            max_seq_len=max_seq_len,
            loss_weights=loss_weights,
        )
        for example, image in zip(examples, images)
    ]


def _move_batch(batch: Mapping[str, Any], device: torch.device) -> Dict[str, Any]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
        if key not in {"metadata", "token_weights", "segment_ids"}
    }


def _optimizer(model: Any, config: SFTConfig) -> Any:
    if config.training.optim == "paged_adamw_8bit":
        try:
            import bitsandbytes as bnb
        except ImportError as exc:
            raise RuntimeError("Full/Smoke QLoRA requires bitsandbytes paged_adamw_8bit") from exc
        return bnb.optim.PagedAdamW8bit(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
    return torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )


def _scheduler(optimizer: Any, config: SFTConfig, total_steps: int) -> Any:
    if str(config.training.lr_scheduler_type).casefold() in {"constant", "constant_with_warmup"}:
        warmup_steps = int(total_steps * config.training.warmup_ratio)
        warmup_steps = max(0, min(warmup_steps, total_steps - 1))

        def constant_scale(step: int) -> float:
            if str(config.training.lr_scheduler_type).casefold() == "constant_with_warmup" and warmup_steps:
                return max(1e-8, min(1.0, step / warmup_steps))
            return 1.0

        return torch.optim.lr_scheduler.LambdaLR(optimizer, constant_scale)
    warmup_steps = int(total_steps * config.training.warmup_ratio)
    warmup_steps = max(0, min(warmup_steps, total_steps - 1))

    def scale(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return max(1e-8, step / warmup_steps)
        remaining = max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, (step - warmup_steps) / remaining))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def _gradient_norm(model: Any) -> float:
    values = [parameter.grad.detach() for parameter in model.parameters() if parameter.grad is not None]
    if not values:
        return 0.0
    return float(torch.norm(torch.stack([torch.norm(value.float()) for value in values])).item())


@torch.no_grad()
def _evaluate_loss(model: Any, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    losses = []
    for batch in loader:
        outputs = model(**_move_batch(batch, device))
        loss = outputs.loss if hasattr(outputs, "loss") else outputs["loss"]
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite evaluation loss")
        losses.append(float(loss.detach().cpu()))
    model.train()
    return sum(losses) / len(losses) if losses else float("inf")


def _copy_checkpoint(
    source: Path,
    destination: Path,
    *,
    include_optimizer: bool = True,
) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(
        source,
        destination,
        ignore=(
            None
            if include_optimizer
            else shutil.ignore_patterns("optimizer.pt")
        ),
    )


def _prepare_output_dir(config: SFTConfig, resume: Optional[Path]) -> Path:
    output = config.run.output_dir.resolve()
    allowed_root = (config.project_root / "outputs").resolve()
    try:
        output.relative_to(allowed_root)
    except ValueError as exc:
        raise ValueError(
            "training output_dir must stay inside the project outputs directory"
        ) from exc
    existing_names = (
        {item.name for item in output.iterdir()}
        if output.exists()
        else set()
    )
    if (
        existing_names
        and config.run.overwrite_output_dir
        and resume is None
    ):
        shutil.rmtree(output)
        existing_names = set()
    allowed_preflight_files = {
        "mask_audit.json",
        "pretrain_mask_audit.json",
        "target_weight_audit.json",
        # Cold-start V1 writes its read-only audit and boundary contracts before
        # invoking the shared trainer.  They are immutable preflight artifacts,
        # not stale checkpoints and must survive training initialization.
        "audit",
        "contracts",
        "comparison",
        "reports",
        "eval",
        "training",
        # Detached cold-start launch writes this log before Python enters the
        # trainer; it is not a checkpoint or stale training artifact.
        "formal_training.stdout.log",
    }
    if (
        existing_names
        and not existing_names.issubset(allowed_preflight_files)
        and not config.run.overwrite_output_dir
        and resume is None
    ):
        raise FileExistsError("output directory is non-empty: %s" % output)
    output.mkdir(parents=True, exist_ok=True)
    return output


def _manifest_hash(config: SFTConfig) -> str:
    path = _data_manifest_path(config)
    if not path.is_file():
        raise FileNotFoundError("Protocol-SFT Manifest is required: %s" % path)
    return data_manifest_hash(path)


def _git_metadata(root: Path) -> Dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip())
    except (OSError, subprocess.CalledProcessError):
        commit = None
        dirty = None
    return {"git_commit": commit, "git_dirty": dirty}


def _model_config_hash(config: SFTConfig, model: Any) -> str:
    path = config.model.path / "config.json"
    if path.is_file():
        return sha256_file(path)
    value = (
        model.config.to_dict()
        if hasattr(model, "config") and hasattr(model.config, "to_dict")
        else {}
    )
    return sha256_json(value)


def _processor_config_hash(config: SFTConfig, processor: Any) -> str:
    names = (
        "preprocessor_config.json",
        "processor_config.json",
        "tokenizer_config.json",
        "chat_template.json",
    )
    files = {
        name: sha256_file(config.model.path / name)
        for name in names
        if (config.model.path / name).is_file()
    }
    if files:
        return sha256_json(files)
    value: Dict[str, Any] = {
        "processor_init_kwargs": getattr(processor, "init_kwargs", {}),
        "tokenizer_init_kwargs": getattr(
            getattr(processor, "tokenizer", None),
            "init_kwargs",
            {},
        ),
    }
    return sha256_json(value)


def train_from_config(config: SFTConfig, resume_from_checkpoint: Optional[Path] = None) -> Dict[str, Any]:
    config.validate()
    if not config.model.gradient_checkpointing:
        raise ValueError("Smoke/Full training requires gradient checkpointing")
    output = _prepare_output_dir(config, resume_from_checkpoint)
    torch.manual_seed(config.run.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.run.seed)
    routerfix_sampling = (
        config.sampling.mode == "initial_router_focused"
    )
    v0_4_contract: Dict[str, Any] = {}
    format_contract: Dict[str, Any] = {}
    if config.is_v0_4:
        v0_4_contract, splits = validate_v0_4_training_contract(config)
        _json_write(output / "data_contract.json", v0_4_contract)
    elif config.is_format:
        format_contract, splits = _validate_format_data_contract(config)
        _json_write(output / "data_contract.json", format_contract)
    elif routerfix_sampling:
        # Test is embargoed until the separate internal-regression script.
        splits = {
            "train": load_split(
                config.data.train_file,
                config.data.expected_train_count,
            ),
            "dev": load_split(
                config.data.dev_file,
                config.data.expected_dev_count,
            ),
        }
    else:
        splits = load_protocol_splits(config.data)
    train_examples = splits["train"]
    mixed_sampling = config.sampling.mode == "mixed_transition"
    format_sampling = config.sampling.mode == "format_exposure_balance"
    indexed_sampling = mixed_sampling or routerfix_sampling or format_sampling
    weighted_loss_enabled = bool(config.loss_weights.enabled)
    if config.is_v0_4 and weighted_loss_enabled:
        raise ValueError("Protocol-SFT v0.4 forbids weighted Target loss")
    training_test_embargo = bool(
        config.is_v0_4 or routerfix_sampling or config.is_format
    )
    manifest_hash = _manifest_hash(config)
    approval: Dict[str, Any] = {}
    approval_hash: Optional[str] = None
    if routerfix_sampling:
        expected_reviewed_count = sum(
            example.transition == "initial_to_text_search"
            for split in ("train", "dev")
            for example in splits[split]
        )
        approval_path = config.initial_route_audit.approval_file
        if approval_path is None:
            raise ValueError("Router-fix approval path is missing")
        approval = validate_initial_route_approval(
            approval_path,
            dataset_manifest_hash=manifest_hash,
            expected_reviewed_count=expected_reviewed_count,
        )
        approval_hash = sha256_file(approval_path)
    sampling_epochs = []
    expected_train_transitions = (
        EXPECTED_V0_4_TRAIN_TRANSITION_COUNTS
        if config.is_v0_4
        else EXPECTED_TRAIN_TRANSITION_COUNTS
    )
    if mixed_sampling:
        sampler = DeterministicMixedTransitionSampler(
            train_examples,
            TransitionSamplingConfig(
                epoch_size=int(config.sampling.epoch_size or 0),
                balanced_fraction=config.sampling.balanced_fraction,
                seed=config.sampling.seed,
            ),
            expected_counts=expected_train_transitions,
        )
        schedule_count = 1 if config.is_smoke else config.training.num_train_epochs
        sampling_epochs = [
            sampler.sample_epoch(epoch_index)
            for epoch_index in range(schedule_count)
        ]
    elif routerfix_sampling:
        sampler = DeterministicInitialRouterSampler(
            train_examples,
            InitialRouterSamplingConfig(
                epoch_size=int(config.sampling.epoch_size or 0),
                transition_quotas=config.sampling.transition_quotas,
                seed=config.sampling.seed,
            ),
            expected_counts=expected_train_transitions,
        )
        schedule_count = (
            1 if config.is_smoke
            else config.training.num_train_epochs
        )
        sampling_epochs = [
            sampler.sample_epoch(epoch_index)
            for epoch_index in range(schedule_count)
        ]
    elif format_sampling:
        sampler = DeterministicFormatExposureSampler(
            train_examples,
            InitialRouterSamplingConfig(
                epoch_size=int(config.sampling.epoch_size or 0),
                transition_quotas=config.sampling.transition_quotas,
                seed=config.sampling.seed,
            ),
        )
        sampling_epochs = [
            sampler.sample_epoch(epoch_index)
            for epoch_index in range(config.training.num_train_epochs)
        ]
    if config.is_smoke and indexed_sampling:
        smoke_examples = [
            train_examples[index] for index in sampling_epochs[0].indices
        ]
    elif config.is_smoke:
        smoke_examples = prepare_smoke_subset(config, train_examples)
    else:
        smoke_examples = train_examples
    if config.is_smoke:
        active_splits = {"train": smoke_examples}
    else:
        active_splits = {
            name: splits[name] for name in ("train", "dev")
        }
    resume_contract: Dict[str, Any] = {}
    if resume_from_checkpoint is not None:
        resume_contract = validate_resume_checkpoint(resume_from_checkpoint, config, manifest_hash)
    # Processor-only preflight keeps the Full mask audit ahead of any GPU model
    # allocation and blocks direct train-script invocations just like the
    # server wrapper does.
    processor = load_processor(config)
    renderer = ProtocolRenderer(processor)
    images_by_split = {
        split: _load_images(config, examples) for split, examples in active_splits.items()
    }
    mask_reports = {
        split: audit_mask_examples(
            processor, renderer, examples, images_by_split[split], max_seq_len=config.data.max_seq_len
        )
        for split, examples in active_splits.items()
    }
    mask_report = {
        "examples_checked": sum(report["examples_checked"] for report in mask_reports.values()),
        "train_examples_checked": int(
            mask_reports.get("train", {}).get("examples_checked", 0)
        ),
        "dev_examples_checked": int(
            mask_reports.get("dev", {}).get("examples_checked", 0)
        ),
        "test_examples_checked": int(
            mask_reports.get("test", {}).get("examples_checked", 0)
        ),
        "mask_failure_count": sum(report["mask_failure_count"] for report in mask_reports.values()),
        "target_truncation_count": sum(report["target_truncation_count"] for report in mask_reports.values()),
        "sequence_overflow_count": sum(report["sequence_overflow_count"] for report in mask_reports.values()),
        "empty_target_count": sum(report["empty_target_count"] for report in mask_reports.values()),
        "prefix_mismatch_count": sum(report["prefix_mismatch_count"] for report in mask_reports.values()),
        "history_active_label_count": sum(report["history_active_label_count"] for report in mask_reports.values()),
        "information_active_label_count": sum(report["information_active_label_count"] for report in mask_reports.values()),
        "image_prefix_active_label_count": sum(report["image_prefix_active_label_count"] for report in mask_reports.values()),
        "splits": mask_reports,
    }
    expected_audit_count = (
        int(config.sampling.epoch_size or 0)
        if config.is_smoke and indexed_sampling
        else (
            (10 if config.is_format else 12)
            if config.is_smoke
            else (
                1000
                if config.is_format
                else (900 if training_test_embargo else 1000)
            )
        )
    )
    if mask_report["examples_checked"] != expected_audit_count:
        raise RuntimeError(
            "mask audit checked %d examples; expected %d"
            % (mask_report["examples_checked"], expected_audit_count)
        )
    _json_write(output / ("mask_audit.json" if config.is_smoke else "pretrain_mask_audit.json"), mask_report)
    if indexed_sampling:
        _json_write(
            output / "sampling_audit.json",
            {
                "mode": config.sampling.mode,
                "epochs": [
                    dict(sampled.audit) for sampled in sampling_epochs
                ],
            },
        )
    if any(
        mask_report[key]
        for key in (
            "mask_failure_count", "target_truncation_count", "sequence_overflow_count",
            "empty_target_count", "prefix_mismatch_count", "history_active_label_count",
            "information_active_label_count", "image_prefix_active_label_count",
        )
    ):
        raise RuntimeError("mask audit failed; training is blocked")
    target_weight_report: Dict[str, Any] = {
        "examples_checked": 0,
        "target_weight_alignment_failure_count": 0,
        "alignment_method_counts": {},
        "splits": {},
    }
    if weighted_loss_enabled:
        target_weight_reports = {
            split: audit_target_weight_examples(
                processor,
                renderer,
                examples,
                images_by_split[split],
                max_seq_len=config.data.max_seq_len,
                loss_weights=config.loss_weights,
            )
            for split, examples in active_splits.items()
        }
        target_weight_report = {
            "examples_checked": sum(
                report["examples_checked"]
                for report in target_weight_reports.values()
            ),
            "target_weight_alignment_failure_count": sum(
                report["target_weight_alignment_failure_count"]
                for report in target_weight_reports.values()
            ),
            "alignment_method_counts": {
                method: sum(
                    report["alignment_method_counts"].get(method, 0)
                    for report in target_weight_reports.values()
                )
                for method in sorted({
                    method
                    for report in target_weight_reports.values()
                    for method in report["alignment_method_counts"]
                })
            },
            "splits": target_weight_reports,
        }
        _json_write(
            output / "target_weight_audit.json",
            target_weight_report,
        )
        if target_weight_report[
            "target_weight_alignment_failure_count"
        ]:
            raise RuntimeError(
                "Target weight alignment audit failed; training is blocked"
            )
    tokenized_by_split = {
        split: _tokenize_examples(
            processor,
            renderer,
            examples,
            images_by_split[split],
            config.data.max_seq_len,
            config.loss_weights if weighted_loss_enabled else None,
        )
        for split, examples in active_splits.items()
    }
    model, loaded_processor, model_audit = load_qwen_lora(
        config,
        processor=processor,
        adapter_path=resume_from_checkpoint,
    )
    if loaded_processor is not processor:
        raise RuntimeError("training must reuse the Processor used for mask audit")
    environment = environment_versions()
    data_hash_manifest = (
        config.data.hashes_file
        if config.data.hashes_file is not None
        else (
            config.project_root
            / "data"
            / "manifests"
            / "protocol_sft_v0_3_files.sha256"
        )
    )
    provenance = (
        experiment_data_provenance(v0_4_contract)
        if config.is_v0_4
        else {}
    )
    training_manifest = {
        **provenance,
        **_git_metadata(config.project_root),
        "mode": config.run.mode,
        "objective": (
            "protocol_format_only" if config.is_format else config.objective
        ),
        "base_model_path": str(config.model.path).replace("\\", "/"),
        "base_model_config_hash": _model_config_hash(config, model),
        "processor_config_hash": _processor_config_hash(config, processor),
        "chat_template_hash": renderer.chat_template_sha256,
        "renderer": renderer.manifest_metadata(),
        "data_manifest_hash": manifest_hash,
        "data_files_sha256_manifest_hash": (
            sha256_file(data_hash_manifest)
            if data_hash_manifest.is_file()
            else None
        ),
        "data_file_hashes": {
            split: sha256_file(path)
            for split, path in {
                "train": config.data.train_file,
                "dev": config.data.dev_file,
                "test": config.data.test_file,
            }.items()
            if (
                path is not None
                and path.is_file()
                and (not training_test_embargo or split != "test")
            )
        },
        "config_hash": config.config_hash,
        "sampling_mode": config.sampling.mode,
        "sampling_epoch_size": config.sampling.epoch_size,
        "sampling_balanced_fraction": config.sampling.balanced_fraction,
        "sampling_seed": config.sampling.seed,
        "sampler_quotas": dict(config.sampling.transition_quotas),
        "sampler_purpose": (
            "format_exposure_balance" if config.is_format else None
        ),
        "sampler_is_policy_distribution": (
            False if config.is_format else None
        ),
        "sampling_epochs": [
            dict(sampled.audit) for sampled in sampling_epochs
        ],
        "loss_weight_config": config.to_dict(relative=False)[
            "loss_weights"
        ],
        "loss": config.to_dict(relative=False)["loss"],
        "initial_route_approval_hash": approval_hash,
        "initial_route_approval": approval,
        "test_split_embargoed_during_training": training_test_embargo,
        "test_accessed": False if training_test_embargo else None,
        "learns_protocol_syntax": True if config.is_format else None,
        "learns_action_serialization": True if config.is_format else None,
        "learns_optimal_routing": False if config.is_format else None,
        "learns_search_quality": False if config.is_format else None,
        "learns_answer_quality": False if config.is_format else None,
        "policy_metrics_are_diagnostic_only": (
            True if config.is_format else None
        ),
        "weighted_loss_enabled": weighted_loss_enabled,
        "test_evaluation_performed": False if config.is_format else None,
        "lora_config": config.to_dict(relative=False)["lora"],
        "training_config_hash": sha256_json(
            config.to_dict(relative=False)["training"]
        ),
        "dataset_unchanged": not config.is_v0_4,
        "protocol_sft_v0_3_hash_unchanged": not config.is_v0_4,
        "lora_module_names": list(model_audit.matched_lora_module_names),
        "model_audit": model_audit.to_dict(),
        "environment": environment,
        "seed": config.run.seed,
    }
    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is None:
        raise RuntimeError("processor tokenizer must define pad_token_id")
    collator = ProtocolSFTCollator(int(pad_token_id))
    def make_train_loader(epoch_index: int) -> DataLoader:
        values = tokenized_by_split["train"]
        if indexed_sampling and config.is_full:
            values = [
                values[index]
                for index in sampling_epochs[epoch_index].indices
            ]
        return DataLoader(
            TokenizedListDataset(values),
            batch_size=config.training.per_device_train_batch_size,
            shuffle=False if indexed_sampling or config.is_smoke else True,
            generator=torch.Generator().manual_seed(
                config.run.seed + epoch_index
            ),
            collate_fn=collator,
            num_workers=config.training.dataloader_num_workers,
        )

    train_loader = make_train_loader(0)
    eval_loader = None
    if "dev" in tokenized_by_split:
        eval_loader = DataLoader(
            TokenizedListDataset(tokenized_by_split["dev"]),
            batch_size=config.training.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=config.training.dataloader_num_workers,
        )
    device = next(model.parameters()).device
    optimizer = _optimizer(model, config)
    updates_per_epoch = max(
        1,
        math.ceil(len(train_loader) / config.training.gradient_accumulation_steps),
    )
    planned_steps = int(config.training.max_steps or (updates_per_epoch * (1 if config.is_smoke else config.training.num_train_epochs)))
    scheduler = _scheduler(optimizer, config, planned_steps)
    if resume_from_checkpoint is not None:
        optimizer_state = Path(resume_from_checkpoint) / "optimizer.pt"
        if optimizer_state.is_file():
            saved_state = torch.load(optimizer_state, map_location="cpu", weights_only=False)
            if isinstance(saved_state, dict) and "optimizer" in saved_state:
                optimizer.load_state_dict(saved_state["optimizer"])
                if "scheduler" in saved_state:
                    scheduler.load_state_dict(saved_state["scheduler"])
            else:
                optimizer.load_state_dict(saved_state)
    before = snapshot_trainable_parameters(model)
    steps = 0
    optimizer_steps = int(resume_contract.get("optimizer_steps", 0))
    records: List[Dict[str, Any]] = [
        dict(item) for item in resume_contract.get("records", [])
    ]
    epoch_summaries: List[Dict[str, Any]] = [
        dict(item) for item in resume_contract.get("epoch_summaries", [])
    ]
    protocol_candidates: List[CheckpointCandidate] = []
    best_eval = float(resume_contract.get("best_eval_loss", float("inf")))
    if (
        config.is_full
        and not indexed_sampling
        and resume_from_checkpoint is not None
        and not (output / "best_adapter").is_dir()
    ):
        checkpoint_eval = resume_contract.get("eval_loss")
        if checkpoint_eval is None or float(checkpoint_eval) != best_eval:
            raise ValueError(
                "best_adapter is missing and the resume checkpoint is not the recorded best"
            )
        _copy_checkpoint(
            Path(resume_from_checkpoint),
            output / "best_adapter",
            include_optimizer=False,
        )
    total_epochs = 1 if config.is_smoke else config.training.num_train_epochs
    first_epoch = int(resume_contract.get("epoch", 0)) + 1 if resume_contract else 1
    if first_epoch > total_epochs:
        raise ValueError("resume checkpoint has already completed the configured training")
    if indexed_sampling and config.is_full and first_epoch > 1:
        for completed_epoch in range(1, first_epoch):
            checkpoint = output / ("checkpoint-epoch-%d" % completed_epoch)
            metrics_path = checkpoint / (
                "dev_metrics.json"
                if routerfix_sampling or config.is_v0_4 or config.is_format
                else "dev_protocol_metrics.json"
            )
            contract_path = checkpoint / "checkpoint_contract.json"
            if not metrics_path.is_file() or not contract_path.is_file():
                raise ValueError(
                    "resume requires prior Dev protocol metrics for every epoch"
                )
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            contract = json.loads(contract_path.read_text(encoding="utf-8"))
            protocol_candidates.append(
                CheckpointCandidate(
                    name=checkpoint.name,
                    path=checkpoint,
                    metrics=metrics,
                    eval_loss=float(contract.get("eval_loss", float("inf"))),
                    split="dev",
                )
            )
    start = time.perf_counter()
    for epoch in range(first_epoch, total_epochs + 1):
        train_loader = make_train_loader(epoch - 1)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_losses: List[float] = []
        epoch_unweighted_losses: List[float] = []
        logging_window_stats: List[Dict[str, Any]] = []
        epoch_grad_norms_before: List[float] = []
        epoch_grad_norms_after: List[float] = []
        epoch_start = time.perf_counter()
        for batch_index, batch in enumerate(train_loader):
            outputs = model(**_move_batch(batch, device))
            if weighted_loss_enabled:
                logits = (
                    outputs.logits
                    if hasattr(outputs, "logits")
                    else outputs["logits"]
                )
                loss, batch_loss_stats = weighted_causal_lm_loss(
                    logits,
                    batch["labels"].to(device),
                    batch["token_weights"].to(device),
                    batch["segment_ids"].to(device),
                )
            else:
                loss = (
                    outputs.loss
                    if hasattr(outputs, "loss")
                    else outputs["loss"]
                )
                loss_value = float(loss.detach().cpu())
                batch_loss_stats = {
                    "total_weighted_loss": loss_value,
                    "unweighted_target_loss": loss_value,
                }
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite training loss")
            epoch_losses.append(float(loss.detach().cpu()))
            epoch_unweighted_losses.append(
                float(batch_loss_stats["unweighted_target_loss"])
            )
            logging_window_stats.append(dict(batch_loss_stats))
            (loss / config.training.gradient_accumulation_steps).backward()
            steps += 1
            should_step = (
                steps % config.training.gradient_accumulation_steps == 0
                or batch_index == len(train_loader) - 1
            )
            if should_step:
                grad_norm_before = float(torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad],
                    config.training.max_grad_norm,
                ).detach().cpu())
                grad_norm_after = _gradient_norm(model)
                if not math.isfinite(grad_norm_before) or not math.isfinite(
                    grad_norm_after
                ):
                    raise RuntimeError("non-finite gradient norm")
                epoch_grad_norms_before.append(grad_norm_before)
                epoch_grad_norms_after.append(grad_norm_after)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
                window_record: Dict[str, Any] = {
                    "epoch": epoch,
                    "step": optimizer_steps,
                    "train_loss": float(loss.detach().cpu()),
                    "gradient_norm_before_clipping": grad_norm_before,
                    "gradient_norm_after_clipping": grad_norm_after,
                    "learning_rate": float(scheduler.get_last_lr()[0]),
                }
                if weighted_loss_enabled:
                    loss_fields = (
                        "total_weighted_loss", "unweighted_target_loss",
                        "reason_loss", "action_tag_loss",
                        "query_payload_loss", "answer_payload_loss",
                    )
                    count_fields = (
                        "reason_active_token_count",
                        "action_active_token_count",
                        "query_active_token_count",
                        "answer_active_token_count",
                    )
                    for key in loss_fields:
                        window_record[key] = sum(
                            float(row.get(key, 0.0))
                            for row in logging_window_stats
                        ) / len(logging_window_stats)
                    for key in count_fields:
                        window_record[key] = sum(
                            int(row.get(key, 0))
                            for row in logging_window_stats
                        )
                records.append(window_record)
                logging_window_stats = []
                if config.is_smoke and optimizer_steps >= int(config.training.max_steps or 4):
                    break
        eval_loss = _evaluate_loss(model, eval_loader, device) if eval_loader is not None else None
        if eval_loss is not None:
            records[-1]["eval_loss"] = eval_loss
        candidate_best = min(best_eval, eval_loss) if eval_loss is not None else best_eval
        epoch_summary = {
            "epoch": epoch,
            "train_loss": (
                sum(epoch_losses) / len(epoch_losses)
                if epoch_losses
                else None
            ),
            "weighted_loss": (
                sum(epoch_losses) / len(epoch_losses)
                if epoch_losses else None
            ),
            "unweighted_loss": (
                sum(epoch_unweighted_losses)
                / len(epoch_unweighted_losses)
                if epoch_unweighted_losses else None
            ),
            "eval_loss": eval_loss,
            "learning_rate": float(scheduler.get_last_lr()[0]),
            "gradient_norm_before_clipping": (
                sum(epoch_grad_norms_before) / len(epoch_grad_norms_before)
                if epoch_grad_norms_before
                else None
            ),
            "gradient_norm_after_clipping": (
                sum(epoch_grad_norms_after) / len(epoch_grad_norms_after)
                if epoch_grad_norms_after
                else None
            ),
            "optimizer_steps": optimizer_steps,
            "gpu_max_memory_allocated": (
                int(torch.cuda.max_memory_allocated())
                if torch.cuda.is_available()
                else 0
            ),
            "gpu_max_memory_reserved": (
                int(torch.cuda.max_memory_reserved())
                if torch.cuda.is_available()
                else 0
            ),
            "wall_clock_seconds": round(time.perf_counter() - epoch_start, 3),
        }
        if indexed_sampling:
            epoch_summary["sampling_audit"] = dict(
                sampling_epochs[epoch - 1].audit
            )
        epoch_summaries.append(epoch_summary)
        training_manifest["epoch_training_summaries"] = epoch_summaries
        checkpoint_name = "checkpoint-final" if config.is_smoke else "checkpoint-epoch-%d" % epoch
        checkpoint = save_adapter_checkpoint(
            model, processor, output / checkpoint_name,
            config=config,
            data_manifest_hash=manifest_hash,
            model_audit=model_audit.to_dict(),
            training_manifest=training_manifest,
            extra={
                "epoch": epoch,
                "optimizer_steps": optimizer_steps,
                "best_eval_loss": candidate_best,
                "eval_loss": eval_loss,
                "records": records,
                "epoch_summaries": epoch_summaries,
            },
        )
        torch.save(
            {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()},
            checkpoint / "optimizer.pt",
        )
        if indexed_sampling and config.is_full:
            model.eval()
            previous_use_cache = bool(model.config.use_cache)
            model.config.use_cache = True
            dev_records = generate_records(
                model,
                processor,
                renderer,
                splits["dev"],
                images_by_split["dev"],
                max_new_tokens=96,
            )
            if config.is_format:
                dev_metrics = evaluate_format_generation_records(
                    dev_records,
                    target_truncation_count=int(
                        mask_reports["dev"]["target_truncation_count"]
                    ),
                )
            else:
                dev_metrics = evaluate_generation_records(
                    dev_records,
                    tokenizer=processor.tokenizer,
                )
                dev_metrics["target_truncation_count"] = int(
                    mask_reports["dev"]["target_truncation_count"]
                )
            dev_metrics["evaluation_split"] = "dev"
            dev_metrics["eval_loss"] = eval_loss
            epoch_summary["dev_generation_metrics"] = dev_metrics
            training_manifest["epoch_training_summaries"] = epoch_summaries
            with (checkpoint / "dev_predictions.jsonl").open(
                "w", encoding="utf-8", newline="\n"
            ) as handle:
                for record in dev_records:
                    handle.write(
                        json.dumps(
                            record.to_dict(),
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                        + "\n"
                    )
            _json_write(
                checkpoint / (
                    "dev_metrics.json"
                    if routerfix_sampling or config.is_v0_4 or config.is_format
                    else "dev_protocol_metrics.json"
                ),
                dev_metrics,
            )
            _json_write(
                checkpoint / "train_manifest.json",
                training_manifest,
            )
            protocol_candidates.append(
                CheckpointCandidate(
                    name=checkpoint_name,
                    path=checkpoint,
                    metrics=dev_metrics,
                    eval_loss=float(
                        eval_loss if eval_loss is not None else float("inf")
                    ),
                    split="dev",
                )
            )
            model.config.use_cache = previous_use_cache
            model.train()
        if eval_loss is not None and eval_loss < best_eval:
            best_eval = eval_loss
            if config.is_full and not indexed_sampling:
                _copy_checkpoint(
                    checkpoint,
                    output / "best_adapter",
                    include_optimizer=False,
                )
        if config.is_smoke and optimizer_steps >= int(config.training.max_steps or 4):
            break
    if not config.is_smoke:
        last_checkpoint = output / ("checkpoint-epoch-%d" % total_epochs)
        if not last_checkpoint.is_dir():
            raise RuntimeError("final epoch checkpoint was not saved")
        _copy_checkpoint(
            last_checkpoint,
            output / "final_adapter",
            include_optimizer=False,
        )
        if indexed_sampling:
            if config.is_format:
                selected, selection_report = select_format_checkpoint(
                    protocol_candidates
                )
                materialize_format_selection(
                    selected,
                    selection_report,
                    output,
                    copy_adapter=selection_report[
                        "any_epoch_format_gate_passed"
                    ],
                )
                gate = format_gate_report(selected.metrics)
                gate.update({
                    "evaluation_split": "dev",
                    "selected_checkpoint": selected.name,
                    "any_epoch_format_gate_passed": selection_report[
                        "any_epoch_format_gate_passed"
                    ],
                    "status": (
                        "PROTOCOL_FORMAT_SFT_V1_PASS"
                        if selection_report["any_epoch_format_gate_passed"]
                        else "PROTOCOL_FORMAT_SFT_V1_FAIL"
                    ),
                })
                _json_write(output / "format_gate.json", gate)
                if not selection_report["any_epoch_format_gate_passed"]:
                    raise RuntimeError(
                        "PROTOCOL_FORMAT_SFT_V1_FAIL: no Epoch passed Format Gate"
                    )
            else:
                try:
                    selected, selection_report = select_checkpoint(
                        protocol_candidates,
                        require_non_collapsed=(
                            routerfix_sampling or config.is_v0_4
                        ),
                    )
                except NoNonCollapsedCheckpointError as exc:
                    _json_write(
                        output / "checkpoint_selection_failure.json",
                        {
                            "error": str(exc),
                            "selection_split": "dev",
                            "all_candidates_routing_collapsed": True,
                        },
                    )
                    raise
                materialize_selection(selected, selection_report, output)
            if (routerfix_sampling or config.is_v0_4) and not config.is_format:
                selected_gates = check_format_gates(selected.metrics)
                dev_gate_passed = bool(
                    all(selected_gates.values())
                    and not selected.metrics.get(
                        "routing_collapsed", True
                    )
                    and float(
                        selected.metrics.get(
                            "minimum_initial_transition_recall", 0.0
                        )
                    ) > 0.0
                )
                _json_write(
                    output / "dev_protocol_gate.json",
                    {
                        "evaluation_split": "dev",
                        "selected_checkpoint": selected.name,
                        "protocol_gates": selected_gates,
                        "routing_collapsed": bool(
                            selected.metrics.get(
                                "routing_collapsed", True
                            )
                        ),
                        "minimum_initial_transition_recall": (
                            selected.metrics.get(
                                "minimum_initial_transition_recall", 0.0
                            )
                        ),
                        "passed": dev_gate_passed,
                        "status": (
                            "PROTOCOL_SFT_V0_4_DEV_GATE_PASS"
                            if dev_gate_passed
                            else "PROTOCOL_SFT_V0_4_DEV_GATE_FAIL"
                        ),
                        "test_metrics_used": False,
                    },
                )
                if config.is_v0_4:
                    selection_path = output / "checkpoint_selection.json"
                    selection_value = json.loads(
                        selection_path.read_text(encoding="utf-8")
                    )
                    selection_value["dev_gate_passed"] = dev_gate_passed
                    selection_value["test_metrics_used"] = False
                    _json_write(selection_path, selection_value)
                    initialize_embargo(
                        output / "test_embargo_state.json"
                    )
        elif not (output / "best_adapter").is_dir():
            raise RuntimeError("best_adapter was not preserved during Full training")
    delta = changed_parameter_count(before, model)
    train_metrics = {
        "optimizer_steps": optimizer_steps,
        "epochs_completed": total_epochs if not config.is_smoke else 1,
        "records": records,
        "epochs": epoch_summaries,
        "lora_parameter_delta": delta,
        "weighted_loss_enabled": weighted_loss_enabled,
        "target_weight_alignment_failure_count": target_weight_report[
            "target_weight_alignment_failure_count"
        ],
        "wall_clock_seconds": round(time.perf_counter() - start, 3),
        "gpu_max_memory_allocated": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
        "gpu_max_memory_reserved": int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else 0,
    }
    _json_write(output / "train_metrics.json", train_metrics)
    _json_write(output / "environment.json", environment)
    _json_write(output / "train_manifest.json", training_manifest)
    output_config = config.to_dict(relative=True)
    (output / "resolved_config.yaml").write_text(
        yaml.safe_dump(
            output_config,
            allow_unicode=True,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    _json_write(output / "model_audit.json", model_audit.to_dict())
    _json_write(output / "trainer_state.json", {"optimizer_steps": optimizer_steps, "epoch": total_epochs, "resumed_from": str(resume_from_checkpoint) if resume_from_checkpoint else None})
    if config.is_smoke:
        checkpoint_reloaded = False
        reload_generation_succeeded = False
        parsed_reload_records: List[Any] = []
        expected_transition_coverage = 5 if config.is_format else 6
        sampler_six_transition_coverage = (
            len(set(example.transition for example in smoke_examples))
            == expected_transition_coverage
        )
        try:
            del optimizer, scheduler, outputs, loss, model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            base_model, reload_processor = load_qwen_base(config)
            base_model.config.use_cache = True
            generation_config = getattr(base_model, "generation_config", None)
            if generation_config is not None:
                generation_config.temperature = None
                generation_config.top_p = None
                generation_config.top_k = None
            reloaded_model = load_adapter_into_model(base_model, output / "checkpoint-final")
            reloaded_renderer = ProtocolRenderer(reload_processor)
            ProtocolRenderer.assert_compatible_metadata(
                reloaded_renderer.manifest_metadata(),
                renderer.manifest_metadata(),
            )
            if indexed_sampling or config.is_v0_4 or config.is_format:
                first_by_transition: Dict[str, tuple[StateActionExample, Any]] = {}
                for example, image in zip(
                    smoke_examples, images_by_split["train"]
                ):
                    first_by_transition.setdefault(
                        example.transition, (example, image)
                    )
                reload_examples = [
                    first_by_transition[name][0]
                    for name in sorted(first_by_transition)
                ]
                reload_images = [
                    first_by_transition[name][1]
                    for name in sorted(first_by_transition)
                ]
            else:
                reload_examples = smoke_examples[:2]
                reload_images = images_by_split["train"][:2]
            reload_records = generate_records(
                reloaded_model,
                reload_processor,
                reloaded_renderer,
                reload_examples,
                reload_images,
                max_new_tokens=96,
            )
            from multimodal_web_agent.agent import parse_action
            expected_generation_count = (
                (
                    5
                    if config.is_format
                    else (6 if indexed_sampling or config.is_v0_4 else 2)
                )
            )
            parsed_reload_records = [
                parse_action(record.generated_text)
                for record in reload_records
            ]
            reload_generation_succeeded = (
                len(reload_records) == expected_generation_count
                and len(parsed_reload_records) == expected_generation_count
            )
            checkpoint_reloaded = True
            with (output / "reload_generation.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
                for record in reload_records:
                    handle.write(json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
            del reloaded_model, base_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:
            _json_write(output / "reload_generation_error.json", {"error": repr(exc)})
        _json_write(
            output / "smoke_pass.json",
            {
                "passed": bool(
                    optimizer_steps == int(config.training.max_steps or 0)
                    and delta > 0
                    and all(math.isfinite(float(row["train_loss"])) for row in records)
                    and all(
                        math.isfinite(
                            float(
                                row.get(
                                    "unweighted_target_loss",
                                    row["train_loss"],
                                )
                            )
                        )
                        for row in records
                    )
                    and all(
                        math.isfinite(float(row["gradient_norm_before_clipping"]))
                        and math.isfinite(float(row["gradient_norm_after_clipping"]))
                        for row in records
                    )
                    and model_audit.gradient_checkpointing
                    and model_audit.matched_lora_module_count > 0
                    and model_audit.trainable_visual_parameter_count == 0
                    and model_audit.trainable_base_parameter_count == 0
                    and mask_report["mask_failure_count"] == 0
                    and target_weight_report[
                        "target_weight_alignment_failure_count"
                    ] == 0
                    and (
                        not config.is_v0_4
                        or (
                            v0_4_contract.get(
                                "dataset_audit_passed"
                            ) is True
                            and len(smoke_examples) == 12
                            and not weighted_loss_enabled
                            and config.loss.mode
                            == "standard_current_turn"
                        )
                    )
                    and (
                        not config.is_format
                        or (
                            format_contract.get("dataset_audit_passed") is True
                            and len(smoke_examples) == 10
                            and not weighted_loss_enabled
                            and config.loss.mode == "standard_current_turn"
                        )
                    )
                    and (
                        not routerfix_sampling
                        or weighted_loss_enabled
                    )
                    and sampler_six_transition_coverage
                    and (output / "checkpoint-final").is_dir()
                    and checkpoint_reloaded
                    and reload_generation_succeeded
                ),
                "mode": "smoke",
                "dataset_schema": config.data.schema_version,
                "dataset_manifest_hash": manifest_hash,
                "dataset_audit_passed": (
                    format_contract.get("dataset_audit_passed")
                    if config.is_format
                    else (
                        v0_4_contract.get("dataset_audit_passed")
                        if config.is_v0_4 else None
                    )
                ),
                "test_accessed": False if training_test_embargo else None,
                "examples_checked": len(smoke_examples),
                "transition_coverage_count": len(
                    {example.transition for example in smoke_examples}
                ),
                "config_hash": config.config_hash,
                "finite_loss": all(math.isfinite(float(row["train_loss"])) for row in records),
                "finite_unweighted_loss": all(
                    math.isfinite(
                        float(
                            row.get(
                                "unweighted_target_loss",
                                row["train_loss"],
                            )
                        )
                    )
                    for row in records
                ),
                "finite_gradients": all(
                    math.isfinite(float(row["gradient_norm_before_clipping"]))
                    and math.isfinite(float(row["gradient_norm_after_clipping"]))
                    for row in records
                ),
                "optimizer_steps": optimizer_steps,
                "sampling_mode": config.sampling.mode,
                "weighted_loss_enabled": weighted_loss_enabled,
                "initial_route_approval_hash": approval_hash,
                "target_weight_alignment_failure_count": (
                    target_weight_report[
                        "target_weight_alignment_failure_count"
                    ]
                ),
                "sampler_six_transition_coverage": (
                    sampler_six_transition_coverage
                ),
                "sampler_transition_coverage": sampler_six_transition_coverage,
                "lora_parameter_delta_positive": delta > 0,
                "gradient_checkpointing": model_audit.gradient_checkpointing,
                "matched_lora_module_count": model_audit.matched_lora_module_count,
                "trainable_visual_parameter_count": model_audit.trainable_visual_parameter_count,
                "trainable_base_parameter_count": model_audit.trainable_base_parameter_count,
                "mask_failures": mask_report["mask_failure_count"],
                "checkpoint_saved": (output / "checkpoint-final").is_dir(),
                "checkpoint_reloaded": checkpoint_reloaded,
                "reload_generation_succeeded": reload_generation_succeeded,
                "reload_strict_parser_processed_count": (
                    len(parsed_reload_records)
                    if reload_generation_succeeded
                    else 0
                ),
                "reload_strict_parser_valid_count": sum(
                    bool(item.valid) for item in parsed_reload_records
                ),
            },
        )
    return {"train_metrics": train_metrics, "mask_audit": mask_report, "model_audit": model_audit.to_dict()}
