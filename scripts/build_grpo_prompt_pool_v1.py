#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from multimodal_web_agent.training.grpo.prompt_pool import build_prompt_pool_v1


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the exact GRPO Prompt Pool v1 contract")
    parser.add_argument("--source", type=Path, default=ROOT / "data/raw/fvqa/fvqa_train.parquet")
    parser.add_argument("--cache", type=Path, default=ROOT / "data/raw/fvqa/fvqa_train_image_search_results_cache.pkl")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/processed/grpo_prompt_pool_v1")
    parser.add_argument("--manifest-dir", type=Path, default=ROOT / "data/manifests")
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--allow-short", action="store_true", help="fixture-only mode; never use for the formal server build")
    args = parser.parse_args()
    # Keep this list deliberately narrow.  Only the frozen Protocol Format
    # SFT split payloads define historical membership; diagnostics and model
    # prediction JSONL files must never be interpreted as data usage.
    historical = [
        ROOT / "data/processed/protocol_sft_v0_4",
        ROOT / "data/processed/protocol_sft_v0_5",
        ROOT / "data/processed/protocol_format_sft_v1",
    ]
    manifest = build_prompt_pool_v1(
        source_path=args.source, cache_path=args.cache, output_dir=args.output_dir,
        manifest_dir=args.manifest_dir, seed=args.seed, historical_paths=historical,
        strict=not args.allow_short,
    )
    if any(
        int(value) != 0
        for pair in manifest["group_isolation"].values()
        for value in pair.values()
    ):
        raise RuntimeError("group isolation is non-zero; refusing READY")
    if not all(int(value) == 1 for value in manifest["pool_pass"].values()):
        raise RuntimeError("pool count contract failed; refusing READY")
    print("GRPO_PROMPT_POOL_V1_READY")
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
