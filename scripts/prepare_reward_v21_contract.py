#!/usr/bin/env python3
"""Freeze the successful Reward-v2.1 inputs into a one-run contract."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable

import yaml


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise RuntimeError(f"empty artifact tree: {path}")
    for item in files:
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(item)))
    return digest.hexdigest()


def frozen_file(path: Path) -> tuple[str, dict[str, Any]]:
    resolved = path.resolve()
    relative = resolved.relative_to(ROOT.resolve()).as_posix()
    stat = resolved.stat()
    return relative, {
        "sha256": sha256_file(resolved),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def require_files(paths: Iterable[Path]) -> list[Path]:
    values = sorted({path.resolve() for path in paths})
    missing = [path for path in values if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing frozen inputs: " + ", ".join(map(str, missing)))
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training-config",
        type=Path,
        default=Path("configs/grpo/reward_v21_full_server.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/grpo_reward_v21_full_contract"),
    )
    args = parser.parse_args()
    training_path = (ROOT / args.training_config).resolve()
    output = (ROOT / args.output_dir).resolve()
    training = yaml.safe_load(training_path.read_text(encoding="utf-8"))
    if training.get("experiment_id") != "reward_v21":
        raise ValueError("this public contract generator supports Reward-v2.1 only")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite: {output}")

    reward_path = (ROOT / training["reward_config"]).resolve()
    reward = yaml.safe_load(reward_path.read_text(encoding="utf-8"))
    if reward["reward"].get("version") != "v2.1":
        raise ValueError("Reward-v2.1 config required")
    if reward["frozen_contract"].get("full_training_allowed") is not True:
        raise ValueError("full training gate is closed")

    smoke_dir = (ROOT / training["smoke_baseline"]).resolve()
    smoke_manifest = json.loads((smoke_dir / "run_manifest.json").read_text(encoding="utf-8"))
    if smoke_manifest.get("status") != "passed":
        raise RuntimeError("the prerequisite exploration smoke did not pass")
    smoke = {
        "path": smoke_dir.relative_to(ROOT).as_posix(),
        "manifest_sha256": sha256_file(smoke_dir / "run_manifest.json"),
        "summary_sha256": sha256_file(smoke_dir / "smoke_summary.json"),
        "files_sha256": sha256_file(smoke_dir / "files.sha256"),
    }

    paths = reward["paths"]
    fixed = [
        training_path,
        reward_path,
        ROOT / "configs/grpo/common_v1_server.yaml",
        ROOT / paths["prompt_pool"],
        ROOT / "data/processed/grpo_prompt_pool_v1/manifest.json",
        ROOT / "data/processed/grpo_prompt_pool_v1/audit.json",
        ROOT / paths["image_search_cache"],
        ROOT / "data/raw/fvqa/fvqa_train.parquet",
        ROOT / paths["coverage_cache"],
        ROOT / paths["coverage_manifest"],
        ROOT / paths["question_baseline_cache"],
        ROOT / paths["question_baseline_manifest"],
        *(ROOT / "src/multimodal_web_agent").rglob("*.py"),
        *(ROOT / "scripts").glob("*.py"),
        *(smoke_dir / name for name in (
            "run_manifest.json", "smoke_summary.json",
            "behavior_logprob_audit.json", "files.sha256",
        )),
    ]
    adapter = (ROOT / training["initialization"]["adapter_path"]).resolve()
    frozen_paths = require_files([*fixed, *(p for p in adapter.rglob("*") if p.is_file())])
    artifacts = dict(frozen_file(path) for path in frozen_paths)

    common = yaml.safe_load((ROOT / "configs/grpo/common_v1_server.yaml").read_text(encoding="utf-8"))
    base_model = Path(common["model"]["path"])
    if not base_model.is_absolute():
        base_model = ROOT / base_model
    coverage_manifest = json.loads((ROOT / paths["coverage_manifest"]).read_text(encoding="utf-8"))
    baseline_manifest = json.loads((ROOT / paths["question_baseline_manifest"]).read_text(encoding="utf-8"))
    manifest = {
        "schema_version": "grpo-reward-v21-public-execution-contract-v1",
        "marker": "GRPO_REWARD_V21_FULL_CONTRACT_READY",
        "experiment_id": "reward_v21",
        "full_execution_authorized": True,
        "maximum_successful_runs": 1,
        "automatic_start": False,
        "automatic_retry": False,
        "initialization": "frozen_sft_adapter",
        "reward_v0_checkpoint_used": False,
        "reward_v2_checkpoint_used": False,
        "sft_adapter_path": adapter.relative_to(ROOT).as_posix(),
        "sft_adapter_sha256": sha256_tree(adapter),
        "base_model_path": str(base_model),
        "base_model_sha256": sha256_tree(base_model),
        "scale": training["scale"],
        "smoke_baseline": smoke,
        "frozen_artifacts": artifacts,
        "text_corpus_sha256": coverage_manifest["text_corpus_sha256"],
        "question_baseline_manifest_sha256": sha256_file(ROOT / paths["question_baseline_manifest"]),
        "coverage_manifest_sha256": sha256_file(ROOT / paths["coverage_manifest"]),
        "baseline_cache_schema": baseline_manifest.get("schema_version"),
        "frozen_dev_access_during_training": False,
        "unified_frozen_test_accessed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=output.name + ".tmp.", dir=output.parent))
    try:
        (temporary / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (temporary / "frozen_artifacts.json").write_text(
            json.dumps(artifacts, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, output)
    except Exception:
        import shutil
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print("GRPO_REWARD_V21_FULL_CONTRACT_READY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
