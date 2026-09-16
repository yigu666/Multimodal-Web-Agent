#!/usr/bin/env python3
"""Freeze inputs for the successful 64-update Stage2 S2-A continuation."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import yaml


ROOT = Path(__file__).resolve().parents[1]


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_hash(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise RuntimeError(f"empty checkpoint: {path}")
    for item in files:
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(file_hash(item)))
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=Path("configs/grpo/stage2_v21_continue_short_256.yaml"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("outputs/grpo_stage2_contract"),
    )
    args = parser.parse_args()
    config_path = (ROOT / args.config).resolve()
    output = (ROOT / args.output_dir).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("strategy") != "S2_A" or config.get("mode") != "short":
        raise ValueError("the public Stage2 contract supports S2-A short only")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite: {output}")

    stage1_manifest_path = ROOT / "outputs/grpo_reward_v21_full/run_manifest.json"
    stage1 = json.loads(stage1_manifest_path.read_text(encoding="utf-8"))
    if stage1.get("marker") != "GRPO_REWARD_V21_FULL_COMPLETE":
        raise RuntimeError("Reward-v2.1 full run is not complete")
    checkpoint = (ROOT / stage1["selected_checkpoint"]).resolve()
    step0_path = ROOT / config["step0_evaluation"] / "run_manifest.json"
    step0 = json.loads(step0_path.read_text(encoding="utf-8"))
    if step0.get("marker") != "GRPO_STAGE2_STEP0_FROZEN_DEV_READY":
        raise RuntimeError("Stage2 step-0 evaluation is not ready")
    if step0.get("reproduction_passed") is not True:
        raise RuntimeError("Stage2 step-0 did not reproduce Reward-v2.1")

    reward_path = (ROOT / config["reward_config"]).resolve()
    frozen_files = sorted({
        config_path,
        reward_path,
        stage1_manifest_path.resolve(),
        step0_path.resolve(),
        ROOT / "data/raw/fvqa/fvqa_train.parquet",
        ROOT / "data/raw/fvqa/fvqa_train_image_search_results_cache.pkl",
        ROOT / "data/processed/grpo_prompt_pool_v1/train.jsonl",
        ROOT / "data/processed/grpo_reward_v2/text_corpus_answer_coverage.jsonl",
        ROOT / "data/processed/grpo_reward_v2/question_only_baseline.jsonl",
        *(path.resolve() for path in (ROOT / "src/multimodal_web_agent").rglob("*.py")),
    })
    missing = [path for path in frozen_files if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing frozen inputs: " + ", ".join(map(str, missing)))
    frozen = {
        path.relative_to(ROOT).as_posix(): file_hash(path)
        for path in frozen_files
    }
    manifest = {
        "schema_version": "grpo-stage2-s2a-public-contract-v1",
        "marker": "GRPO_STAGE2_CONTRACT_READY",
        "passed": True,
        "formal_route": "S2_A_SHORT",
        "stage2_init_checkpoint": str(checkpoint),
        "stage2_init_checkpoint_tree_sha256": tree_hash(checkpoint),
        "stage1_manifest": stage1_manifest_path.relative_to(ROOT).as_posix(),
        "stage1_manifest_sha256": file_hash(stage1_manifest_path),
        "v21_learning_rate": 5.0e-7,
        "stage2_learning_rate": 2.5e-7,
        "optimizer_state_loaded": False,
        "scheduler_state_loaded": False,
        "config_sha256": {
            config_path.relative_to(ROOT).as_posix(): file_hash(config_path),
            reward_path.relative_to(ROOT).as_posix(): file_hash(reward_path),
        },
        "source_sha256": {
            key: value for key, value in frozen.items() if key.startswith("src/")
        },
        "frozen_input_sha256": {
            key: value for key, value in frozen.items() if not key.startswith("src/")
        },
        "training_performed": False,
        "frozen_test_accessed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=output.name + ".tmp.", dir=output.parent))
    try:
        (temporary / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, output)
    except Exception:
        import shutil
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print("GRPO_STAGE2_CONTRACT_READY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
