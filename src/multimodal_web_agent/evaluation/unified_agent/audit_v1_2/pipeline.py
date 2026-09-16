from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

from multimodal_web_agent.evaluation.unified_agent.fingerprints import (
    sha256_file,
)

from .audit import (
    answer_evaluator_metrics,
    audit_episode,
    evidence_metrics,
    image_truncation_audit,
    text_query_audit,
)
from .evidence import FrozenEvidenceStore, read_jsonl
from .labels import task_label_provenance
from .paired import build_paired_behavior_audit
from .reporting import (
    answer_comparison_markdown,
    evidence_casebook_markdown,
    evidence_metrics_markdown,
    final_report_markdown,
    frozen_test_semantics_markdown,
    image_truncation_markdown,
    paired_behavior_markdown,
    paired_casebook_markdown,
    reward_design_markdown,
    statistical_tests_markdown,
    task_label_markdown,
    text_casebook_markdown,
    text_query_markdown,
)
from .schema import AUDIT_RUN_SCHEMA, EVIDENCE_AUDIT_SCHEMA
from .statistics import paired_statistical_audit


OUTPUT_NAMES = (
    "run_manifest.json",
    "source_hashes.json",
    "evidence_audit_raw.jsonl",
    "evidence_audit_sft.jsonl",
    "evidence_audit_grpo.jsonl",
    "evidence_metrics.json",
    "evidence_metrics.md",
    "answer_evaluator_v2_metrics.json",
    "answer_evaluator_v2_comparison.md",
    "paired_behavior_audit.json",
    "paired_behavior_audit.md",
    "paired_casebook.md",
    "evidence_utilization_casebook.md",
    "text_search_query_audit.json",
    "text_search_query_audit.md",
    "text_search_casebook.md",
    "image_truncation_audit.json",
    "image_truncation_audit.md",
    "task_type_label_audit.json",
    "task_type_label_audit.md",
    "task_type_suspect_cases.jsonl",
    "frozen_test_access_semantics.md",
    "statistical_tests.json",
    "statistical_tests.md",
    "reward_v2_design_inputs.md",
    "report.md",
    "files.sha256",
)
EXPECTED_DEV_SHA256 = (
    "70b7295458a71bc8a64885aae06c29935cf94acd1f27eea66e655fa75648ddaf"
)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, values: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"
            for value in values
        ),
        encoding="utf-8",
    )


def _read_checksums(path: Path) -> dict[str, str]:
    values = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            raise RuntimeError(
                f"invalid checksum line {line_number} in {path.as_posix()}"
            )
        digest, relative = parts
        relative = relative.lstrip("*").strip().replace("\\", "/")
        if relative in values:
            raise RuntimeError(f"duplicate checksum path: {relative}")
        values[relative] = digest
    return values


def _verify_relative_file(root: Path, relative: str, expected: str) -> str:
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise RuntimeError("checksum path escapes root") from exc
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"checksum mismatch: {relative}")
    return actual


def _verify_v1_1_outputs(project_root: Path) -> dict[str, Any]:
    output_root = project_root / "outputs/unified_agent_eval_v1_1"
    checksum_path = output_root / "files.sha256"
    checksums = _read_checksums(checksum_path)
    verified = {
        relative: _verify_relative_file(output_root, relative, digest)
        for relative, digest in sorted(checksums.items())
    }
    required = {
        f"dev/{model}/{name}"
        for model in ("raw", "sft", "grpo")
        for name in ("episodes.jsonl", "metrics.json", "run_manifest.json")
    } | {"dev/paired_input_audit.json"}
    if not required <= set(verified):
        raise RuntimeError("v1.1 checksum manifest misses required Dev artifacts")
    return {
        "checksum_manifest": checksum_path.relative_to(project_root).as_posix(),
        "checksum_manifest_sha256": sha256_file(checksum_path),
        "verified_file_count": len(verified),
        "verified": verified,
    }


def _verify_processed_inputs(
    project_root: Path,
    examples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    data_root = project_root / "data/processed/unified_agent_eval_v1_1"
    fixed = {
        "dev.jsonl": EXPECTED_DEV_SHA256,
    }
    verified = {
        relative: _verify_relative_file(data_root, relative, expected)
        for relative, expected in fixed.items()
    }
    environment_root = data_root / "environment"
    environment_checksums = _read_checksums(
        environment_root / "environment_files.sha256"
    )
    verified_environment = {
        f"environment/{relative}": _verify_relative_file(
            environment_root, relative, expected
        )
        for relative, expected in sorted(environment_checksums.items())
    }
    verified.update(verified_environment)
    evidence_pairs = [
        (
            str(row["evidence_path"]).replace("\\", "/"),
            sha256_file(data_root / str(row["evidence_path"])),
        )
        for row in examples
    ]
    evidence_pairs.sort()
    evidence_digest = hashlib.sha256("".join(
        f"{digest}  {relative}\n" for relative, digest in evidence_pairs
    ).encode("utf-8")).hexdigest()
    return {
        "validation_method": (
            "fixed frozen Dev fingerprint; environment_files.sha256 "
            "restricted to environment root; per-evidence embedded content_sha256"
        ),
        "selected_verified_file_count": len(verified) + len(evidence_pairs),
        "unified_frozen_partition_files_verified": 0,
        "dev_and_environment_only": True,
        "verified": verified,
        "referenced_evidence_file_count": len(evidence_pairs),
        "referenced_evidence_checksum_list_sha256": evidence_digest,
    }


def _load_and_validate_episodes(
    project_root: Path,
    examples: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    expected_ids = [str(row["eval_id"]) for row in examples]
    output_root = project_root / "outputs/unified_agent_eval_v1_1/dev"
    models = {}
    for model in ("raw", "sft", "grpo"):
        directory = output_root / model
        rows = read_jsonl(directory / "episodes.jsonl")
        manifest = json.loads(
            (directory / "run_manifest.json").read_text(encoding="utf-8")
        )
        if len(rows) != 200 or [row["eval_id"] for row in rows] != expected_ids:
            raise RuntimeError(f"{model} episodes do not match frozen Dev order")
        if any(row["model_id"] != model for row in rows):
            raise RuntimeError(f"{model} episode model identity mismatch")
        if manifest.get("episode_count") != 200 or manifest.get("split") != "dev":
            raise RuntimeError(f"{model} run manifest is not Frozen Dev 200")
        if manifest.get("test_accessed") is not False:
            raise PermissionError(f"{model} manifest records frozen-partition access")
        if manifest.get("training_performed") is not False:
            raise PermissionError(f"{model} evaluation manifest records training")
        if manifest.get("dynamic_internet_accessed") is not False:
            raise PermissionError(f"{model} evaluation accessed dynamic internet")
        models[model] = rows
    paired = json.loads(
        (output_root / "paired_input_audit.json").read_text(encoding="utf-8")
    )
    if paired.get("passed") is not True or paired.get("test_accessed") is not False:
        raise RuntimeError("v1.1 paired input contract is not valid")
    return models


def _validate_existing_trace_if_available(
    project_root: Path,
    records: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    path = project_root / (
        "outputs/analysis/unified_agent_eval_v1_1_retrieval_traces/"
        "retrieval_traces.jsonl"
    )
    if not path.is_file():
        return {
            "available": False,
            "validated": False,
            "reason": "optional prior reconstruction trace is absent",
        }
    expected = {
        (model, str(row["episode_id"])): row
        for model in ("sft", "grpo") for row in records[model]
    }
    rows = read_jsonl(path)
    compared_calls = 0
    for trace in rows:
        key = (str(trace["model_id"]), str(trace["eval_id"]))
        if key not in expected:
            raise RuntimeError("prior reconstruction has unexpected episode")
        actual_calls = expected[key]["retrieved_information"]
        prior_calls = trace["searches"]
        if len(actual_calls) != len(prior_calls):
            raise RuntimeError("prior reconstruction search count mismatch")
        for actual, prior in zip(actual_calls, prior_calls):
            prior_query = dict(prior.get("query") or {})
            if actual["tool"] == "text_search":
                query_equal = actual["query"].get("text") == prior_query.get("text")
            else:
                query_equal = (
                    actual["query"].get("image_sha256")
                    == prior_query.get("image_sha256")
                    and actual["query"].get("source_data_id")
                    == prior_query.get("source_data_id")
                )
            if not query_equal or actual["status"] != prior["status"]:
                raise RuntimeError("prior reconstruction action/query mismatch")
            if (actual.get("information") or None) != (
                prior.get("retrieved_information") or None
            ):
                raise RuntimeError("prior reconstruction Information mismatch")
            compared_calls += 1
    return {
        "available": True,
        "validated": True,
        "path": path.relative_to(project_root).as_posix(),
        "sha256": sha256_file(path),
        "episode_trace_count": len(rows),
        "compared_call_count": compared_calls,
        "information_exact_match": True,
    }


def _manifest(
    source_hashes: Mapping[str, Any],
    records: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    return {
        "schema_version": AUDIT_RUN_SCHEMA,
        "audit_name": "Unified Agent Eval v1.2 Evidence Utilization and Evaluation Audit",
        "input_benchmark": "unified_agent_eval_v1_1",
        "input_split": "dev",
        "models": ["raw", "sft", "grpo"],
        "episode_count_per_model": {
            model: len(rows) for model, rows in records.items()
        },
        "evidence_audit_schema": EVIDENCE_AUDIT_SCHEMA,
        "environment_manifest_sha256": source_hashes["environment"][
            "environment_manifest_sha256"
        ],
        "source_hash_validation_passed": True,
        "existing_trace_validation": source_hashes["existing_trace_validation"],
        "model_inference_rerun": False,
        "training_performed": False,
        "reward_modified": False,
        "environment_modified": False,
        "prompt_pool_modified": False,
        "online_access": False,
        "unified_frozen_test_accessed": False,
        "checkpoint_selected": False,
        "atomic_publish": True,
    }


def _write_hashes(output: Path) -> None:
    paths = [
        path for path in sorted(output.iterdir())
        if path.is_file() and path.name != "files.sha256"
    ]
    (output / "files.sha256").write_text("".join(
        f"{sha256_file(path)}  {path.name}\n" for path in paths
    ), encoding="utf-8")


def run_audit(project_root: Path, output_dir: Path) -> dict[str, Any]:
    project_root = Path(project_root).resolve()
    output_dir = Path(output_dir)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    output_dir = output_dir.resolve()
    expected_parent = (project_root / "outputs").resolve()
    try:
        output_dir.relative_to(expected_parent)
    except ValueError as exc:
        raise ValueError("audit output must stay under project outputs") from exc
    if output_dir.exists():
        raise FileExistsError(f"audit output already exists: {output_dir}")
    temporary = output_dir.with_name(f".{output_dir.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"audit temporary output exists: {temporary}")
    temporary.mkdir(parents=True)
    try:
        output_hashes = _verify_v1_1_outputs(project_root)
        data_root = project_root / "data/processed/unified_agent_eval_v1_1"
        examples = read_jsonl(data_root / "dev.jsonl")
        if len(examples) != 200:
            raise RuntimeError("frozen Dev does not contain 200 examples")
        processed_hashes = _verify_processed_inputs(project_root, examples)
        episodes = _load_and_validate_episodes(project_root, examples)
        by_id = {str(row["eval_id"]): row for row in examples}
        store = FrozenEvidenceStore(project_root)
        records: dict[str, list[dict[str, Any]]] = {}
        for model in ("raw", "sft", "grpo"):
            records[model] = [
                audit_episode(
                    model=model,
                    example=by_id[str(episode["eval_id"])],
                    episode=episode,
                    store=store,
                )
                for episode in episodes[model]
            ]
        existing_trace = _validate_existing_trace_if_available(
            project_root, records
        )
        source_hashes = {
            "schema_version": "unified-agent-eval-v1-2-source-hashes-v1",
            "v1_1_outputs": output_hashes,
            "processed_inputs": processed_hashes,
            "environment": store.source_hashes(),
            "existing_trace_validation": existing_trace,
            "online_access": False,
            "unified_frozen_test_accessed": False,
        }
        metrics = evidence_metrics(records)
        answer_metrics = answer_evaluator_metrics(records)
        paired = build_paired_behavior_audit(records["sft"], records["grpo"])
        query_audit = text_query_audit(records)
        truncation = image_truncation_audit(records)
        assignments = [
            {
                "eval_id": row["eval_id"],
                "task_type": row["task_type"],
                "selection_seed": 20260731,
                "split": "dev",
                "assignment_reconstructed_from_frozen_dev": True,
            }
            for row in examples
        ]
        label_audit, suspects = task_label_provenance(examples, assignments)
        tests = paired_statistical_audit(records["sft"], records["grpo"])

        for model, rows in records.items():
            _write_jsonl(temporary / f"evidence_audit_{model}.jsonl", rows)
        _write_json(temporary / "evidence_metrics.json", metrics)
        (temporary / "evidence_metrics.md").write_text(
            evidence_metrics_markdown(metrics), encoding="utf-8"
        )
        _write_json(temporary / "answer_evaluator_v2_metrics.json", answer_metrics)
        (temporary / "answer_evaluator_v2_comparison.md").write_text(
            answer_comparison_markdown(answer_metrics), encoding="utf-8"
        )
        _write_json(temporary / "paired_behavior_audit.json", paired)
        (temporary / "paired_behavior_audit.md").write_text(
            paired_behavior_markdown(paired), encoding="utf-8"
        )
        (temporary / "paired_casebook.md").write_text(
            paired_casebook_markdown(paired), encoding="utf-8"
        )
        (temporary / "evidence_utilization_casebook.md").write_text(
            evidence_casebook_markdown(records), encoding="utf-8"
        )
        _write_json(temporary / "text_search_query_audit.json", query_audit)
        (temporary / "text_search_query_audit.md").write_text(
            text_query_markdown(query_audit), encoding="utf-8"
        )
        (temporary / "text_search_casebook.md").write_text(
            text_casebook_markdown(query_audit), encoding="utf-8"
        )
        _write_json(temporary / "image_truncation_audit.json", truncation)
        (temporary / "image_truncation_audit.md").write_text(
            image_truncation_markdown(truncation), encoding="utf-8"
        )
        _write_json(temporary / "task_type_label_audit.json", label_audit)
        (temporary / "task_type_label_audit.md").write_text(
            task_label_markdown(label_audit), encoding="utf-8"
        )
        _write_jsonl(temporary / "task_type_suspect_cases.jsonl", suspects)
        (temporary / "frozen_test_access_semantics.md").write_text(
            frozen_test_semantics_markdown(label_audit), encoding="utf-8"
        )
        _write_json(temporary / "statistical_tests.json", tests)
        (temporary / "statistical_tests.md").write_text(
            statistical_tests_markdown(tests), encoding="utf-8"
        )
        (temporary / "reward_v2_design_inputs.md").write_text(
            reward_design_markdown(
                records, metrics, query_audit, truncation, label_audit
            ), encoding="utf-8"
        )
        (temporary / "report.md").write_text(
            final_report_markdown(
                records, metrics, answer_metrics, paired, query_audit,
                truncation, label_audit, tests,
            ), encoding="utf-8"
        )
        _write_json(temporary / "source_hashes.json", source_hashes)
        run_manifest = _manifest(source_hashes, records)
        _write_json(temporary / "run_manifest.json", run_manifest)
        _write_hashes(temporary)
        if {path.name for path in temporary.iterdir()} != set(OUTPUT_NAMES):
            raise RuntimeError("audit output file contract mismatch")
        os.replace(temporary, output_dir)
        return run_manifest
    except Exception as exc:
        if temporary.exists():
            _write_json(temporary / "failure.json", {
                "failed_at": datetime.now(timezone.utc).isoformat(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "model_inference_rerun": False,
                "training_performed": False,
                "environment_modified": False,
                "unified_frozen_test_accessed": False,
            })
            failed = output_dir.with_name(
                f"{output_dir.name}.failed."
                f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
            )
            if failed.exists():
                shutil.rmtree(temporary)
            else:
                os.replace(temporary, failed)
        raise
