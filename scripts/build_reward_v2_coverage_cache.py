from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from multimodal_web_agent.data.protocol_sft.cache_reader import ImageSearchCache
from multimodal_web_agent.data.protocol_sft.text_retriever import BootstrapTextRetriever
from multimodal_web_agent.training.grpo.rewards.config import load_reward_v2_config
from multimodal_web_agent.training.grpo.rewards.coverage_cache import (
    BASELINE_SCHEMA,
    COVERAGE_SCHEMA,
    build_coverage_rows,
    build_question_baseline_rows,
    canonical_corpus_sha256,
    file_sha256,
    validate_cache_pair,
)
from multimodal_web_agent.training.grpo.rewards.evidence_support import MATCHER_VERSION


def _resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        for row in rows
    )


def _publish_exact(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"refusing to overwrite a different Reward v2 cache: {path}")
        return
    handle, temporary = tempfile.mkstemp(prefix=path.name + ".tmp.", dir=str(path.parent))
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable_not_git_worktree"


def build(config_path: Path) -> dict[str, Any]:
    config = load_reward_v2_config(config_path)
    prompt_path = _resolve(config.paths.prompt_pool)
    cache_path = _resolve(config.paths.image_search_cache)
    prompts = _read_jsonl(prompt_path)
    cache = ImageSearchCache.load(cache_path, label="fvqa_train_official_cache")
    retriever = BootstrapTextRetriever.from_image_cache(cache)
    corpus_sha = canonical_corpus_sha256(retriever.documents)
    coverage_rows = build_coverage_rows(
        prompts, retriever.documents, text_corpus_sha256=corpus_sha
    )
    baseline_rows = build_question_baseline_rows(
        prompts, retriever, text_corpus_sha256=corpus_sha, top_k=3
    )
    validate_cache_pair(coverage_rows, baseline_rows)
    coverage_path = _resolve(config.paths.coverage_cache)
    baseline_path = _resolve(config.paths.question_baseline_cache)
    _publish_exact(coverage_path, _jsonl_bytes(coverage_rows))
    _publish_exact(baseline_path, _jsonl_bytes(baseline_rows))
    common_path = _resolve(config.paths.environment_manifest)
    common = __import__("yaml").safe_load(common_path.read_text(encoding="utf-8")) or {}
    environment = common.get("environment", {})
    expected_environment = {
        "image_top_k": 3,
        "text_top_k": 3,
        "image_context_policy": "canonical_cached_top3",
    }
    if {key: environment.get(key) for key in expected_environment} != expected_environment:
        raise RuntimeError("formal GRPO environment differs from frozen top-3 contract")
    shared = {
        "reward_version": "hierarchical_grounded_search_v2",
        "git_commit": _git_commit(),
        "config_sha256": file_sha256(config_path),
        "sft_adapter_path": config.paths.sft_adapter,
        "sft_adapter_sha256": str(
            config.raw["frozen_contract"]["sft_adapter_sha256"]
        ),
        "prompt_pool_path": str(prompt_path),
        "prompt_pool_sha256": file_sha256(prompt_path),
        "image_search_cache_path": str(cache_path),
        "image_search_cache_sha256": file_sha256(cache_path),
        "text_corpus_sha256": corpus_sha,
        "document_count": len(retriever.documents),
        "prompt_count": len(prompts),
        "matcher_version": MATCHER_VERSION,
        "environment_manifest_path": str(common_path),
        "environment_manifest_sha256": file_sha256(common_path),
        "top_k": 3,
        "tie_breaking": "score_desc_document_id_asc",
        "model_inference_rerun": False,
        "training_performed": False,
        "unified_frozen_test_accessed": False,
    }
    coverage_manifest = {
        "schema_version": COVERAGE_SCHEMA,
        **shared,
        "coverage_cache_sha256": file_sha256(coverage_path),
        "corpus_has_answer_count": sum(bool(row["corpus_has_answer"]) for row in coverage_rows),
    }
    baseline_manifest = {
        "schema_version": BASELINE_SCHEMA,
        **shared,
        "question_baseline_cache_sha256": file_sha256(baseline_path),
    }
    _publish_exact(_resolve(config.paths.coverage_manifest), _json_bytes(coverage_manifest))
    _publish_exact(_resolve(config.paths.question_baseline_manifest), _json_bytes(baseline_manifest))
    return {
        "coverage_manifest": coverage_manifest,
        "question_baseline_manifest": baseline_manifest,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.config if args.config.is_absolute() else ROOT / args.config)
    print("HIERARCHICAL_GROUNDED_SEARCH_REWARD_V2_CACHE_READY")
    print({
        "prompt_count": result["coverage_manifest"]["prompt_count"],
        "document_count": result["coverage_manifest"]["document_count"],
        "corpus_has_answer_count": result["coverage_manifest"]["corpus_has_answer_count"],
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
