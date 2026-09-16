from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

from multimodal_web_agent.evaluation.unified_agent.audit_v1_2.audit import (
    audit_episode,
)
from multimodal_web_agent.evaluation.unified_agent.audit_v1_2.evidence import (
    FrozenEvidenceStore,
    read_jsonl,
)
from multimodal_web_agent.evaluation.unified_agent.embargo import (
    assert_dev_authorized,
)
from multimodal_web_agent.evaluation.unified_agent.evaluation import (
    clear_runtime_memory,
    evaluate_loaded_model,
    load_runtime_model,
    read_examples,
)
from multimodal_web_agent.evaluation.unified_agent.fingerprints import (
    model_fingerprint,
    register_model,
    sha256_file,
    sha256_tree,
)
from multimodal_web_agent.training.grpo.rewards.text_query_utility import (
    rank_sensitive_utility,
)


METRIC_KEYS = (
    "overall_em_v1",
    "overall_token_f1_v1",
    "search_required_em_v1",
    "search_required_token_f1_v1",
    "protocol_valid_rate",
    "missing_answer_rate",
    "zero_tool_call_rate",
    "avg_tool_calls",
    "image_search_count",
    "text_search_count",
    "evidence_utilization_rate",
    "evidence_hit_but_wrong_rate",
    "wrong_span_copy_rate",
    "text_search_query_utility",
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def write_checksums(root: Path) -> None:
    files = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.name != "files.sha256"
    )
    (root / "files.sha256").write_text(
        "".join(
            f"{sha256_file(path)}  {path.relative_to(root).as_posix()}\n"
            for path in files
        ),
        encoding="utf-8",
    )


def verify_checksums(root: Path, manifest: Path) -> dict[str, str]:
    verified: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(None, 1)
        relative = relative.strip().lstrip("*")
        path = root / relative
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"Frozen Dev checksum mismatch: {relative}")
        verified[relative] = actual
    return verified


def validate_eval_config(project_root: Path, config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != "grpo-stage2-frozen-dev-v1":
        raise RuntimeError("Stage-2 Frozen Dev config schema mismatch")
    if config.get("frozen_test_access") is not False:
        raise RuntimeError("Stage-2 Frozen Test embargo was relaxed")
    assert_dev_authorized(project_root / str(config["test_embargo"]))
    environment_manifest = (
        project_root / str(config["environment_dir"])
        / "environment_manifest.json"
    )
    if sha256_file(environment_manifest) != str(
        config["environment_manifest_sha256"]
    ):
        raise RuntimeError("Stage-2 Frozen Dev environment changed")


def load_frozen_baseline(
    project_root: Path, spec: Mapping[str, Any], *, verify: bool = True,
) -> list[dict[str, Any]]:
    root = project_root / str(spec["checksum_root"])
    if verify:
        verify_checksums(root, project_root / str(spec["checksum_manifest"]))
    path = project_root / str(spec["path"])
    manifest = json.loads(
        (path / "run_manifest.json").read_text(encoding="utf-8")
    )
    expected_model_id = str(spec["expected_model_id"])
    if manifest.get("model_id") != expected_model_id:
        raise RuntimeError("Reward v2.1 Frozen Dev model identity differs")
    if manifest.get("split") != "dev" or manifest.get("episode_count") != 200:
        raise RuntimeError("Reward v2.1 baseline is not Frozen Dev 200")
    if manifest.get("training_performed") is not False:
        raise RuntimeError("Reward v2.1 baseline evaluation performed training")
    if manifest.get("test_accessed") is not False:
        raise RuntimeError("Reward v2.1 baseline accessed Frozen Test")
    rows = read_jsonl(path / "episodes.jsonl")
    if len(rows) != 200:
        raise RuntimeError("Reward v2.1 baseline episode count differs")
    return rows


def assert_paired(
    reference: Sequence[Mapping[str, Any]],
    candidate: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    keys = ("eval_id", "initial_prompt_sha256")
    mismatches = {
        key: sum(
            str(left.get(key)) != str(right.get(key))
            for left, right in zip(reference, candidate)
        ) + abs(len(reference) - len(candidate))
        for key in keys
    }
    if any(mismatches.values()):
        raise RuntimeError(f"Stage-2 Frozen Dev pairing mismatch: {mismatches}")
    return {
        "episode_count": len(reference),
        "mismatches": mismatches,
        "passed": True,
        "frozen_test_accessed": False,
    }


def _query_utility(rows: Sequence[Mapping[str, Any]]) -> tuple[float, int]:
    values: list[float] = []
    for row in rows:
        for call in row["retrieved_information"]:
            if call["tool"] != "text_search" or call["status"] != "success":
                continue
            utility = rank_sensitive_utility(
                row["accepted_answers"],
                [str(result["text"]) for result in call.get("results", [])],
                question=str(row["question"]),
            )
            values.append(float(utility.value))
    return (mean(values) if values else 0.0), len(values)


def metrics(
    episodes: Sequence[Mapping[str, Any]],
    audited: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(episodes) != 200 or len(audited) != 200:
        raise RuntimeError("Stage-2 metrics require Frozen Dev 200")
    search_required = [row for row in audited if row["search_required"]]
    searched = [row for row in audited if int(row["successful_search_count"]) > 0]
    hit = [row for row in searched if row["evidence_hit_any"]]
    hit_answer = [row for row in hit if row["finished_with_answer"]]
    hit_correct = [row for row in hit_answer if row["em_v1"]]
    hit_wrong = [row for row in hit_answer if not row["em_v1"]]
    query_utility, query_count = _query_utility(audited)
    avg_tool_calls = mean(int(row["tool_call_count"]) for row in episodes)
    return {
        "episode_count": 200,
        "overall_em_v1": mean(float(row["em_v1"]) for row in audited),
        "overall_token_f1_v1": mean(float(row["token_f1_v1"]) for row in audited),
        "search_required_em_v1": mean(
            float(row["em_v1"]) for row in search_required
        ),
        "search_required_token_f1_v1": mean(
            float(row["token_f1_v1"]) for row in search_required
        ),
        "protocol_valid_rate": mean(bool(row["protocol_valid"]) for row in audited),
        "missing_answer_rate": mean(
            not bool(row["finished_with_answer"]) for row in audited
        ),
        "zero_tool_call_rate": mean(
            int(row["tool_call_count"]) == 0 for row in episodes
        ),
        "avg_tool_calls": avg_tool_calls,
        "image_search_count": sum(
            int(row["image_search_call_count"]) for row in episodes
        ),
        "text_search_count": sum(
            int(row["text_search_call_count"]) for row in episodes
        ),
        "evidence_utilization_rate": (
            len(hit_correct) / len(hit_answer) if hit_answer else None
        ),
        "evidence_hit_but_wrong_rate": (
            len(hit_wrong) / len(searched) if searched else None
        ),
        "wrong_span_copy_rate": mean(
            bool(row["answer_span_copy_failure"]) for row in audited
        ),
        "text_search_query_utility": query_utility,
        "text_search_query_count": query_count,
        "diagnostic_metrics_are_nonblocking": True,
    }


def audit_metrics(
    *, project_root: Path, dataset: Path,
    episodes: Sequence[Mapping[str, Any]], output: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    examples = [
        json.loads(line)
        for line in dataset.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(examples) != 200:
        raise RuntimeError("Stage-2 Frozen Dev dataset count differs from 200")
    store = FrozenEvidenceStore(project_root)
    audited = [
        audit_episode(
            model="stage2", example=example, episode=episode, store=store
        )
        for example, episode in zip(examples, episodes)
    ]
    write_jsonl(output, audited)
    return audited, metrics(episodes, audited)


def evaluate_checkpoint(
    *, project_root: Path, config: Mapping[str, Any], adapter: Path,
    output_dir: Path, progress_label: str,
) -> list[dict[str, Any]]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite Stage-2 evaluation: {output_dir}")
    registry_path = output_dir.parent.parent / "model_registry.json"
    register_model(
        registry_path,
        model_fingerprint(
            model_id="stage2", stage="stage2_continued_grpo",
            base_model_path=Path(str(config["base_model_path"])),
            adapter_path=adapter,
        ),
    )
    runner_config = {
        "model_registry": registry_path.relative_to(project_root).as_posix(),
        "training_config": str(config["training_config"]),
        "environment_dir": str(config["environment_dir"]),
        "dataset_dir": str(config["dataset_dir"]),
        "budgets": dict(config["budgets"]),
        "generation": dict(config["generation"]),
    }
    models_config = {"models": {"stage2": {
        "enabled": True,
        "stage": "stage2_continued_grpo",
        "base_model_path": str(config["base_model_path"]),
        "adapter_path": adapter.as_posix(),
    }}}
    examples = read_examples(project_root / str(config["dataset"]), 200)
    model, processor, _ = load_runtime_model(
        project_root, runner_config, models_config, "stage2"
    )
    try:
        result = evaluate_loaded_model(
            project_root=project_root,
            runner_config=runner_config,
            model_id="stage2",
            model=model,
            processor=processor,
            examples=examples,
            split="dev",
            output_dir=output_dir,
        )
    finally:
        del model
        del processor
        clear_runtime_memory()
        print(f"GPU_MEMORY_CLEARED_AFTER_{progress_label}", flush=True)
    return list(result["rows"])
