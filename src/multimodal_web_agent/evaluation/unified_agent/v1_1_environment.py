from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

from PIL import Image

from .answer_metrics import answer_reachable
from .environment import (
    FrozenToolEnvironment,
    build_frozen_environment,
)
from .fingerprints import sha256_file
from .leakage import audit_internal_duplicates, audit_leakage
from .source_adapters.base import SourceCandidate
from .v1_1_release import (
    RELEASE_TASK_TYPES,
    SCHEMA_VERSION,
    V11ReleaseError,
    _counts,
    _json,
    _jsonl,
    _load_approved_candidates,
    _paths,
    _read_jsonl,
    _resolve,
    _write_external_hashes,
    _write_tree_hashes,
)
from .data_builder import load_history_references


def _candidate_from_row(dataset: Path, row: Mapping[str, Any]) -> SourceCandidate:
    image_path = dataset / str(row["image_path"])
    evidence_path = dataset / str(row["evidence_path"])
    if not image_path.is_file() or not evidence_path.is_file():
        raise FileNotFoundError("release image/evidence is missing")
    raw = image_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != row["image_sha256"]:
        raise V11ReleaseError("release image SHA256 mismatch")
    with Image.open(image_path) as image:
        image.verify()
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    expected_candidate_id = "%s:%s" % (
        row["source_dataset"], row["source_data_id"]
    )
    if evidence.get("candidate_id") != expected_candidate_id:
        raise V11ReleaseError("release evidence candidate ID mismatch")
    fingerprint_payload = dict(evidence)
    declared_hash = fingerprint_payload.pop("content_sha256", None)
    actual_hash = hashlib.sha256(json.dumps(
        fingerprint_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    if actual_hash != declared_hash:
        raise V11ReleaseError("release evidence SHA256 mismatch")
    candidate = SourceCandidate(
        source_dataset=str(row["source_dataset"]),
        source_data_id=str(row["source_data_id"]),
        question=str(row["question"]),
        image_bytes=raw,
        image_extension=image_path.suffix or ".jpg",
        answer_aliases=tuple(str(value) for value in row["answer_aliases"]),
        # The release task is a final exact-assignment decision. It is not
        # necessarily equal to the source adapter's original suggested route,
        # which is part of the provenance candidate_sha256.
        suggested_task_type=None,
        image_search_records=tuple(
            str(value) for value in evidence["image_search_records"]
        ),
        text_corpus_records=tuple(
            str(value) for value in evidence["text_corpus_records"]
        ),
        source_metadata=dict(row.get("source_metadata") or {}),
    )
    candidate.validate()
    candidate_sha256 = str(row.get("candidate_sha256") or "")
    if (
        len(candidate_sha256) != 64
        or any(value not in "0123456789abcdef" for value in candidate_sha256)
    ):
        raise V11ReleaseError("release provenance fingerprint is invalid")
    return candidate


def _reachable(candidate: SourceCandidate, task_type: str) -> bool:
    if task_type == "search_free":
        return True
    if task_type == "visual_search_required":
        records = candidate.image_search_records[:5]
    elif task_type == "text_search_required":
        records = candidate.text_corpus_records
    elif task_type == "mixed_search_required":
        records = (
            candidate.image_search_records[:5]
            + candidate.text_corpus_records
        )
    else:
        raise V11ReleaseError("invalid release task type")
    return answer_reachable(candidate.answer_aliases, records)


def _validated(
    dataset: Path,
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[SourceCandidate], list[tuple[int, str]]]:
    candidates = []
    failures = []
    for index, row in enumerate(rows):
        try:
            candidate = _candidate_from_row(dataset, row)
            if not _reachable(candidate, str(row["task_type"])):
                raise V11ReleaseError("assigned answer is unreachable")
            candidates.append(candidate)
        except Exception as exc:
            candidates.append(None)  # type: ignore[arg-type]
            failures.append((index, "%s:%s" % (
                type(exc).__name__, str(exc)
            )))
    return candidates, failures


def _replace_failures(
    dataset: Path,
    selected_rows: list[dict[str, Any]],
    reserve_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    _, failures = _validated(dataset, selected_rows)
    replacements = []
    for index, reason in failures:
        failed = selected_rows[index]
        replacement_index = None
        for candidate_index, reserve in enumerate(reserve_rows):
            if reserve["task_type"] != failed["task_type"]:
                continue
            try:
                candidate = _candidate_from_row(dataset, reserve)
            except Exception:
                continue
            if _reachable(candidate, str(reserve["task_type"])):
                replacement_index = candidate_index
                break
        if replacement_index is None:
            raise V11ReleaseError(
                "no valid same-type reserve for %s" % failed["eval_id"]
            )
        reserve = reserve_rows.pop(replacement_index)
        selected_rows[index] = {
            **reserve,
            "eval_id": failed["eval_id"],
            "reserve_only": False,
            "replacement_allowed_before_freeze": False,
            "replacement_allowed_after_freeze": False,
        }
        replacements.append({
            "eval_id": failed["eval_id"],
            "failed_candidate_sha256": failed["candidate_sha256"],
            "replacement_candidate_sha256": reserve["candidate_sha256"],
            "task_type": failed["task_type"],
            "reason": reason,
        })
    return selected_rows, reserve_rows, replacements


def _environment_tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "environment_files.sha256":
            continue
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _verify_relative_hashes(root: Path, checksum_path: Path) -> None:
    for line_number, line in enumerate(
        checksum_path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            raise V11ReleaseError(
                "invalid environment checksum line %d" % line_number
            )
        expected, relative = parts
        path = (root / relative.lstrip("*").strip()).resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError as exc:
            raise V11ReleaseError(
                "environment checksum escapes its root"
            ) from exc
        if not path.is_file() or sha256_file(path) != expected:
            raise V11ReleaseError(
                "environment checksum mismatch: %s" % relative
            )


def _prune_unreferenced_release_assets(
    dataset: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    referenced_images = {str(row["image_path"]) for row in rows}
    referenced_evidence = {str(row["evidence_path"]) for row in rows}
    for path in (dataset / "images").iterdir():
        relative = path.relative_to(dataset).as_posix()
        if path.is_file() and relative not in referenced_images:
            path.unlink()
    for path in (dataset / "evidence").iterdir():
        relative = path.relative_to(dataset).as_posix()
        if path.is_file() and relative not in referenced_evidence:
            path.unlink()


def freeze_v1_1_environment(
    root: Path,
    data_config: Mapping[str, Any],
    environment_config: Mapping[str, Any],
) -> dict[str, Any]:
    root = Path(root).resolve()
    paths = _paths(root, data_config)
    if not paths.output.is_dir():
        raise FileNotFoundError(paths.output)
    manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
    if manifest.get("environment_frozen") is True:
        raise V11ReleaseError("v1.1 environment is already frozen")
    embargo_before = paths.embargo.read_bytes()
    temporary = paths.output.with_name(
        ".unified_agent_eval_v1_1.freeze-%d" % os.getpid()
    )
    previous = paths.output.with_name(
        ".unified_agent_eval_v1_1.pre-freeze-%d" % os.getpid()
    )
    if temporary.exists() or previous.exists():
        raise FileExistsError("v1.1 freeze temporary path exists")
    shutil.copytree(paths.output, temporary)
    try:
        dev = _read_jsonl(temporary / "dev.jsonl")
        test = _read_jsonl(temporary / "test.jsonl")
        reserves = _read_jsonl(temporary / "reserve_candidates.jsonl")
        selected = dev + test
        selected, reserves, replacements = _replace_failures(
            temporary, selected, reserves
        )
        dev = [row for row in selected if ":dev:" in row["eval_id"]]
        test = [row for row in selected if ":test:" in row["eval_id"]]
        if len(dev) != 200 or len(test) != 700:
            raise V11ReleaseError("replacement changed split cardinality")
        if _counts(dev) != {
            task_type: 50 for task_type in RELEASE_TASK_TYPES
        } or _counts(test) != {
            task_type: 175 for task_type in RELEASE_TASK_TYPES
        }:
            raise V11ReleaseError("replacement changed four-way balance")
        _jsonl(temporary / "dev.jsonl", dev)
        _jsonl(temporary / "test.jsonl", test)
        _jsonl(temporary / "reserve_candidates.jsonl", reserves)
        _prune_unreferenced_release_assets(
            temporary, [*dev, *test, *reserves]
        )

        candidates, failures = _validated(temporary, dev + test)
        if failures:
            raise V11ReleaseError(
                "environment validation still fails after replacement"
            )
        import yaml
        history_raw = yaml.safe_load(
            _resolve(root, data_config["history_data_config"]).read_text(
                encoding="utf-8"
            )
        )
        references, statuses = load_history_references(root, history_raw)
        leakage = audit_leakage(candidates, references)
        duplicates = audit_internal_duplicates(candidates)
        if not leakage["hard_gate_passed"]:
            raise V11ReleaseError("post-replacement leakage audit failed")
        if duplicates["hard_reject_candidates"]:
            raise V11ReleaseError("post-replacement duplicate audit failed")

        environment = temporary / "environment"
        if environment.exists():
            raise V11ReleaseError("temporary environment unexpectedly exists")
        environment_manifest = build_frozen_environment(
            environment,
            candidates,
            image_search_top_k=int(
                environment_config["image_search"]["top_k"]
            ),
            text_search_top_k=int(
                environment_config["text_search"]["top_k"]
            ),
        )
        environment_manifest.update({
            "release_schema_version": (
                "unified-agent-eval-v1-1-frozen-environment"
            ),
            "benchmark_version": "unified_agent_eval_v1_1",
            "build_timestamp": datetime.now(timezone.utc).isoformat(),
            "build_code_version": SCHEMA_VERSION,
            "formatter_hash": hashlib.sha256(
                environment_manifest["image_search"]["formatter"].encode(
                    "utf-8"
                )
            ).hexdigest(),
            "replacement_count": len(replacements),
            "replacement_disabled_after_freeze": True,
            "test_evaluated": False,
        })
        _json(environment / "environment_manifest.json", environment_manifest)
        verification = temporary / ".environment_verification"
        build_frozen_environment(
            verification,
            candidates,
            image_search_top_k=int(
                environment_config["image_search"]["top_k"]
            ),
            text_search_top_k=int(
                environment_config["text_search"]["top_k"]
            ),
        )
        # Compare the deterministic files before adding release timestamps.
        first_core = {
            "image": sha256_file(environment / "image_search/results.jsonl"),
            "corpus": sha256_file(environment / "text_corpus/documents.jsonl"),
            "index": sha256_file(
                environment / "text_index/index_manifest.json"
            ),
        }
        second_core = {
            "image": sha256_file(
                verification / "image_search/results.jsonl"
            ),
            "corpus": sha256_file(
                verification / "text_corpus/documents.jsonl"
            ),
            "index": sha256_file(
                verification / "text_index/index_manifest.json"
            ),
        }
        if first_core != second_core:
            raise V11ReleaseError("frozen environment is not deterministic")
        shutil.rmtree(verification)
        frozen = FrozenToolEnvironment(
            environment,
            image_search_top_k=int(
                environment_config["image_search"]["top_k"]
            ),
            text_search_top_k=int(
                environment_config["text_search"]["top_k"]
            ),
        )
        if frozen._documents:
            query = frozen._documents[0].text
            if frozen.text_search(query) != frozen.text_search(query):
                raise V11ReleaseError("text environment query is unstable")
        environment_files = [
            path for path in sorted(environment.rglob("*"))
            if path.is_file() and path.name != "environment_files.sha256"
        ]
        (environment / "environment_files.sha256").write_text(
            "".join(
                "%s  %s\n" % (
                    sha256_file(path),
                    path.relative_to(environment).as_posix(),
                )
                for path in environment_files
            ),
            encoding="utf-8",
        )
        manifest.update({
            "environment_frozen": True,
            "environment_frozen_at": datetime.now(timezone.utc).isoformat(),
            "environment_tree_sha256": _environment_tree_hash(environment),
            "reserve_replacement_count": len(replacements),
            "reserve_replacement_disabled": True,
            "test_embargo_opened": False,
        })
        audit = json.loads((temporary / "audit.json").read_text(
            encoding="utf-8"
        ))
        audit.update({
            "environment_frozen": True,
            "environment_examples_checked": 900,
            "environment_reachable_count": 900,
            "environment_deterministic_count": 900,
            "image_files_valid": 900,
            "reserve_replacements": replacements,
            "post_replacement_leakage": leakage,
            "post_replacement_duplicates": duplicates,
            "test_accessed": False,
        })
        _json(temporary / "manifest.json", manifest)
        _json(temporary / "audit.json", audit)
        _json(temporary / "environment_reachability.json", {
            "schema_version": (
                "unified-agent-eval-v1-1-environment-reachability"
            ),
            "examples_checked": 900,
            "images_valid": 900,
            "evidence_valid": 900,
            "answers_reachable": 900,
            "deterministic_records": 900,
            "replacement_count": len(replacements),
            "passed": True,
        })
        _write_tree_hashes(temporary)
        os.replace(paths.output, previous)
        try:
            os.replace(temporary, paths.output)
        except Exception:
            os.replace(previous, paths.output)
            raise
        shutil.rmtree(previous)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        if previous.exists() and not paths.output.exists():
            os.replace(previous, paths.output)
        raise

    _json(paths.manifest, manifest)
    _json(paths.audit, audit)
    selected_rows = dev + test
    _json(paths.reserved_ids, {
        "schema_version": "unified-agent-eval-v1-1-reserved-ids",
        "reserved_source_ids": sorted({
            row["source_data_id"] for row in selected_rows
        }),
    })
    paths.reserved_images.write_text(
        "".join(
            "%s  %s:%s\n" % (
                row["image_sha256"],
                row["source_dataset"],
                row["source_data_id"],
            )
            for row in sorted(
                selected_rows,
                key=lambda item: (
                    item["image_sha256"], item["source_data_id"]
                ),
            )
        ),
        encoding="utf-8",
    )
    if paths.embargo.read_bytes() != embargo_before:
        raise V11ReleaseError("Test embargo changed during environment freeze")
    _write_external_hashes(root, paths)
    return {
        "manifest": environment_manifest,
        "reachability": {
            "examples_checked": 900,
            "reachable": 900,
            "deterministic": 900,
        },
        "replacement_count": len(replacements),
        "test_embargo_opened": False,
    }


def verify_v1_1_environment(
    root: Path,
    data_config: Mapping[str, Any],
    environment_config: Mapping[str, Any],
) -> dict[str, Any]:
    root = Path(root).resolve()
    paths = _paths(root, data_config)
    manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
    embargo = json.loads(paths.embargo.read_text(encoding="utf-8"))
    if manifest.get("environment_frozen") is not True:
        raise V11ReleaseError("v1.1 environment is not frozen")
    if embargo.get("opened") is not False or embargo.get(
        "evaluation_count"
    ) != 0:
        raise V11ReleaseError("v1.1 Test embargo is not closed")
    environment = FrozenToolEnvironment(
        _resolve(root, environment_config["environment_dir"]),
        image_search_top_k=int(
            environment_config["image_search"]["top_k"]
        ),
        text_search_top_k=int(
            environment_config["text_search"]["top_k"]
        ),
    )
    _verify_relative_hashes(
        environment.root,
        environment.root / "environment_files.sha256",
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "environment_root": environment.root.as_posix(),
        "environment_frozen": True,
        "test_embargo_opened": False,
        "passed": True,
    }
