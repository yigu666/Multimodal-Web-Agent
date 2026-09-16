#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from multimodal_web_agent.evaluation.unified_agent.stage2_frozen_dev import (  # noqa: E402
    assert_paired,
    audit_metrics,
    evaluate_checkpoint,
    load_frozen_baseline,
    validate_eval_config,
    write_checksums,
    write_json,
)
from multimodal_web_agent.training.grpo.stage2_contract import (  # noqa: E402
    resolve_v21_selected_checkpoint,
    sha256_file,
    sha256_tree,
)


READY = "GRPO_STAGE2_STEP0_FROZEN_DEV_READY"


def _show(value: Any) -> str:
    return "N/A" if value is None else f"{float(value):.6f}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=Path("configs/evaluation/stage2_short_frozen_dev.yaml"),
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    root = ROOT.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    validate_eval_config(root, config)
    output = args.output_dir or Path(str(config["step0_output"]))
    output = output if output.is_absolute() else root / output
    if output.exists():
        raise FileExistsError(f"refusing to overwrite Stage-2 Step-0: {output}")
    output.mkdir(parents=True)

    checkpoint, v21_manifest = resolve_v21_selected_checkpoint(root)
    configured_manifest = (root / str(config["v21_manifest"])).resolve()
    if configured_manifest != (
        root / "outputs/grpo_reward_v21_full/run_manifest.json"
    ).resolve():
        raise RuntimeError("Step-0 v2.1 manifest path differs from contract")
    historical = load_frozen_baseline(root, config["v21_baseline"])
    current = evaluate_checkpoint(
        project_root=root,
        config=config,
        adapter=checkpoint,
        output_dir=output / "dev/stage2",
        progress_label="STAGE2_STEP0_DEV",
    )
    pairing = assert_paired(historical, current)
    write_json(output / "paired_input_audit.json", pairing)
    dataset = root / str(config["dataset"])
    _, historical_metrics = audit_metrics(
        project_root=root,
        dataset=dataset,
        episodes=historical,
        output=output / "evidence_audit_historical_v21.jsonl",
    )
    expected_historical = dict(config["historical_v21_expected"])
    historical_tolerance = float(config["historical_metric_tolerance"])
    for key, expected in expected_historical.items():
        if abs(float(historical_metrics[key]) - float(expected)) > historical_tolerance:
            raise RuntimeError(
                f"historical Reward v2.1 metric no longer reproduces: {key}"
            )
    _, step0_metrics = audit_metrics(
        project_root=root,
        dataset=dataset,
        episodes=current,
        output=output / "evidence_audit_stage2_step0.jsonl",
    )
    tolerances = dict(config["step0_tolerance"])
    comparisons = {}
    for key, tolerance in tolerances.items():
        delta = float(step0_metrics[key]) - float(historical_metrics[key])
        comparisons[key] = {
            "historical": float(historical_metrics[key]),
            "step0": float(step0_metrics[key]),
            "delta": delta,
            "absolute_tolerance": float(tolerance),
            "passed": abs(delta) <= float(tolerance),
        }
    reproduced = all(row["passed"] for row in comparisons.values())
    comparison = {
        "schema_version": "grpo-stage2-step0-comparison-v1",
        "historical_v21": historical_metrics,
        "stage2_step0": step0_metrics,
        "reproduction_checks": comparisons,
        "reproduction_passed": reproduced,
    }
    write_json(output / "comparison.json", comparison)
    lines = [
        "# Stage-2 Step-0 Frozen Dev reproduction", "",
        "| Metric | Historical Reward v2.1 | Stage-2 Step-0 | Delta | Pass |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, row in comparisons.items():
        lines.append(
            f"| {key} | {_show(row['historical'])} | {_show(row['step0'])} | "
            f"{float(row['delta']):+.6f} | {row['passed']} |"
        )
    lines.extend([
        "",
        f"- Initialization: `{checkpoint}`",
        f"- Checkpoint SHA256 tree: `{sha256_tree(checkpoint)}`",
        f"- Reproduction passed: `{reproduced}`",
        "- Training performed: `false`",
        "- Frozen Test accessed: `false`",
        "",
        READY if reproduced else "GRPO_STAGE2_STEP0_REPRODUCTION_FAILED",
        "",
    ])
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    manifest = {
        "schema_version": "grpo-stage2-step0-frozen-dev-run-v1",
        "marker": READY if reproduced else "GRPO_STAGE2_STEP0_REPRODUCTION_FAILED",
        "reproduction_passed": reproduced,
        "stage2_init_checkpoint": str(checkpoint),
        "stage2_init_checkpoint_tree_sha256": sha256_tree(checkpoint),
        "v21_full_manifest_sha256": sha256_file(configured_manifest),
        "v21_selected_checkpoint_from_manifest": v21_manifest["selected_checkpoint"],
        "historical_v21_metrics": historical_metrics,
        "step0_metrics": step0_metrics,
        "episode_count": 200,
        "paired_input_audit_passed": True,
        "training_performed": False,
        "dynamic_internet_accessed": False,
        "frozen_test_accessed": False,
        "config_sha256": sha256_file(config_path),
    }
    write_json(output / "run_manifest.json", manifest)
    write_checksums(output)
    if not reproduced:
        raise RuntimeError("Stage-2 Step-0 did not reproduce Reward v2.1")
    print(READY)
    print("FROZEN_TEST_NOT_ACCESSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
