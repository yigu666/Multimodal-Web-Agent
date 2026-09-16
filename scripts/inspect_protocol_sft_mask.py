#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from multimodal_web_agent.training.sft.audit import audit_mask_examples  # noqa: E402
from multimodal_web_agent.training.sft.config import load_config  # noqa: E402
from multimodal_web_agent.training.sft.dataset import ProtocolSFTDataset, load_protocol_splits, load_split, select_format_smoke_subset, select_smoke_subset  # noqa: E402
from multimodal_web_agent.training.sft.model_factory import load_processor  # noqa: E402
from multimodal_web_agent.training.sft.renderer import ProtocolRenderer  # noqa: E402
from multimodal_web_agent.training.sft.sampler import (  # noqa: E402
    EXPECTED_TRAIN_TRANSITION_COUNTS,
    EXPECTED_V0_4_TRAIN_TRANSITION_COUNTS,
    DeterministicInitialRouterSampler,
    DeterministicMixedTransitionSampler,
    InitialRouterSamplingConfig,
    TransitionSamplingConfig,
)
from multimodal_web_agent.training.sft.v0_4_contract import (  # noqa: E402
    validate_v0_4_training_contract,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit Protocol-SFT current-turn masks")
    parser.add_argument("--project-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--config", required=True)
    parser.add_argument("--all-splits", action="store_true")
    parser.add_argument("--exclude-test", action="store_true")
    args = parser.parse_args()
    config = load_config((args.project_root / args.config).resolve(), args.project_root)
    if config.is_v0_4:
        _, splits = validate_v0_4_training_contract(config)
    elif config.sampling.mode == "initial_router_focused":
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
    if args.exclude_test:
        splits.pop("test", None)
    sampling_audit = None
    expected_train_counts = (
        EXPECTED_V0_4_TRAIN_TRANSITION_COUNTS
        if config.is_v0_4
        else EXPECTED_TRAIN_TRANSITION_COUNTS
    )
    if config.is_smoke and not args.all_splits:
        if config.sampling.mode == "mixed_transition":
            sampler = DeterministicMixedTransitionSampler(
                splits["train"],
                TransitionSamplingConfig(
                    epoch_size=int(config.sampling.epoch_size or 0),
                    balanced_fraction=config.sampling.balanced_fraction,
                    seed=config.sampling.seed,
                ),
                expected_counts=expected_train_counts,
            )
            sampled = sampler.sample_epoch(0)
            sampling_audit = dict(sampled.audit)
            splits = {
                "train": [
                    splits["train"][index] for index in sampled.indices
                ]
            }
        elif config.sampling.mode == "initial_router_focused":
            sampler = DeterministicInitialRouterSampler(
                splits["train"],
                InitialRouterSamplingConfig(
                    epoch_size=int(config.sampling.epoch_size or 0),
                    transition_quotas=config.sampling.transition_quotas,
                    seed=config.sampling.seed,
                ),
                expected_counts=expected_train_counts,
            )
            sampled = sampler.sample_epoch(0)
            sampling_audit = dict(sampled.audit)
            splits = {
                "train": [
                    splits["train"][index] for index in sampled.indices
                ]
            }
        else:
            selector = (
                select_format_smoke_subset
                if config.is_format else select_smoke_subset
            )
            splits = {"train": selector(splits["train"], config.data.smoke_examples_per_transition)}
    processor = load_processor(config)
    renderer = ProtocolRenderer(processor)
    reports = {}
    for split, examples in splits.items():
        dataset = ProtocolSFTDataset(
            examples,
            source_parquet=config.data.source_parquet,
        )
        images = [dataset[index].image for index in range(len(dataset))]
        reports[split] = audit_mask_examples(
            processor, renderer, examples, images, max_seq_len=config.data.max_seq_len
        )
    report = {
        "examples_checked": sum(item["examples_checked"] for item in reports.values()),
        "train_examples_checked": int(
            reports.get("train", {}).get("examples_checked", 0)
        ),
        "dev_examples_checked": int(
            reports.get("dev", {}).get("examples_checked", 0)
        ),
        "test_examples_checked": int(
            reports.get("test", {}).get("examples_checked", 0)
        ),
        "mask_failure_count": sum(item["mask_failure_count"] for item in reports.values()),
        "target_truncation_count": sum(item["target_truncation_count"] for item in reports.values()),
        "sequence_overflow_count": sum(item["sequence_overflow_count"] for item in reports.values()),
        "empty_target_count": sum(item["empty_target_count"] for item in reports.values()),
        "prefix_mismatch_count": sum(item["prefix_mismatch_count"] for item in reports.values()),
        "history_active_label_count": sum(item["history_active_label_count"] for item in reports.values()),
        "information_active_label_count": sum(item["information_active_label_count"] for item in reports.values()),
        "image_prefix_active_label_count": sum(item["image_prefix_active_label_count"] for item in reports.values()),
        "renderer": renderer.manifest_metadata(),
        "sampling_audit": sampling_audit,
        "splits": reports,
    }
    expected_count = (
        int(config.sampling.epoch_size or 0)
        if config.is_smoke
        and not args.all_splits
        and config.sampling.mode in {
            "mixed_transition", "initial_router_focused"
        }
        else (
            (10 if config.is_format else 12)
            if config.is_smoke and not args.all_splits else 1000
        )
    )
    if (
        config.is_full
        and (
            config.sampling.mode == "initial_router_focused"
            or config.is_v0_4
            or args.exclude_test
        )
        and not config.is_format
    ):
        expected_count = 900
    if config.is_v0_4 and args.all_splits:
        expected_count = 900
    if report["examples_checked"] != expected_count:
        report["mask_failure_count"] += 1
        report.setdefault("contract_failures", []).append(
            "examples_checked=%d, expected=%d"
            % (report["examples_checked"], expected_count)
        )
    output = config.run.output_dir / ("mask_audit.json" if config.is_smoke and not args.all_splits else "pretrain_mask_audit.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    blocked = (
        "mask_failure_count", "target_truncation_count", "sequence_overflow_count",
        "empty_target_count", "prefix_mismatch_count", "history_active_label_count",
        "information_active_label_count", "image_prefix_active_label_count",
    )
    passed = all(int(report[key]) == 0 for key in blocked)
    if config.is_smoke and not passed:
        (config.run.output_dir / "smoke_pass.json").write_text(
            json.dumps(
                {
                    "passed": False,
                    "mode": "smoke",
                    "dataset_schema": config.data.schema_version,
                    "mask_failures": report["mask_failure_count"],
                    "error": "Smoke mask audit failed",
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
