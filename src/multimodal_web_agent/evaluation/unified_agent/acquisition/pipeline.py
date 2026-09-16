from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

import yaml

from ..source_expansion import build_source_expansion
from .archive_importer import import_archive
from .discovery import discover_source
from .downloader import download_public_source
from .errors import ArchiveRequiredError, SourceInvalidError
from .normalizer import normalize_source
from .provenance import sha256_file
from .registry import load_registry
from .schema import ACQUISITION_SCHEMA, AcquisitionSource


def _path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def ensure_acquisition_directories(
    root: Path, config: Mapping[str, Any]
) -> None:
    for value in config["paths"].values():
        _path(root, value).mkdir(parents=True, exist_ok=True)
    incoming = _path(root, config["paths"]["incoming_root"])
    for name in ("infoseek", "mmsearch", "other"):
        (incoming / name).mkdir(parents=True, exist_ok=True)


def _tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def _merge_roots(
    roots: Sequence[Path],
    *,
    source_name: str,
    extracted_parent: Path,
) -> tuple[Path, str]:
    components = [
        _tree_hash(root) if root.is_dir() else sha256_file(root)
        for root in roots
    ]
    digest = hashlib.sha256(
        "\n".join(sorted(components)).encode("utf-8")
    ).hexdigest()
    target = extracted_parent / source_name / ("combined-" + digest)
    if target.is_dir():
        return target, digest
    temporary = target.with_name(".combined-%s.tmp-%d" % (
        digest, os.getpid()
    ))
    temporary.mkdir(parents=True)
    try:
        for index, source_root in enumerate(roots):
            for path in sorted(
                item for item in source_root.rglob("*")
                if item.is_file()
            ):
                relative = path.relative_to(source_root)
                destination = temporary / relative
                if destination.exists():
                    if sha256_file(destination) != sha256_file(path):
                        destination = (
                            temporary / ("_package_%d" % index) / relative
                        )
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not destination.exists():
                    shutil.copy2(path, destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, target)
    except Exception:
        if temporary.is_dir():
            shutil.rmtree(temporary)
        raise
    return target, digest


def _incoming_paths(
    root: Path, source: AcquisitionSource
) -> list[Path]:
    paths = []
    for pattern in source.incoming_archive_globs:
        pattern_path = Path(pattern)
        if pattern_path.is_absolute():
            parent = pattern_path.parent
            paths.extend(parent.glob(pattern_path.name))
        else:
            paths.extend(root.glob(pattern))
    return sorted({
        path.resolve() for path in paths
        if path.is_file() or path.is_dir()
    })


def _incoming_target(
    root: Path,
    config: Mapping[str, Any],
    source: AcquisitionSource,
) -> Path:
    if source.incoming_archive_globs:
        pattern = Path(source.incoming_archive_globs[0])
        parent = pattern.parent
        return parent if parent.is_absolute() else root / parent
    return (
        _path(root, config["paths"]["incoming_root"]) / source.name
    )


def acquire_and_normalize_sources(
    root: Path,
    config: Mapping[str, Any],
    sources: Sequence[AcquisitionSource],
    *,
    allow_download: bool,
    allow_archive_import: bool,
) -> dict[str, Any]:
    paths = config["paths"]
    downloaded = _path(root, paths["downloaded_root"])
    extracted = _path(root, paths["extracted_root"])
    normalized = _path(root, paths["normalized_root"])
    manifests = _path(root, paths["manifest_root"])
    logs = _path(root, paths["acquisition_logs_root"])
    reports = []
    archive_required = []
    ready_sources = []
    for source in sources:
        if not source.enabled:
            continue
        if source.managed_by:
            reports.append({
                "source_name": source.name,
                "source_status": "delegated",
                "managed_by": source.managed_by,
            })
            continue
        packages: list[Path] = []
        download_required = False
        if allow_download and source.allow_network_download:
            try:
                packages.extend(download_public_source(
                    source, downloaded, logs
                ))
            except ArchiveRequiredError:
                download_required = True
        packages.extend(_incoming_paths(root, source))
        existing_downloads = downloaded / source.name
        if existing_downloads.is_dir():
            packages.extend(
                path for path in existing_downloads.iterdir()
                if path.is_file() or path.is_dir()
            )
        packages = sorted(set(packages))
        if not packages:
            archive_required.append({
                "source_name": source.name,
                "official_dataset_name": source.official_dataset_name,
                "official_source_reference": (
                    source.official_source_reference
                ),
                "required_complete_archive_type": (
                    "complete official Held-out/Test archive"
                ),
                "incoming_target_directory": str(
                    _incoming_target(root, config, source)
                ),
            })
            reports.append({
                "source_name": source.name,
                "source_status": "archive_required",
                "download_channel_unavailable": download_required,
            })
            continue
        if not allow_archive_import:
            reports.append({
                "source_name": source.name,
                "source_status": "downloaded_or_incoming",
                "package_count": len(packages),
            })
            continue
        roots = []
        for package in packages:
            roots.append(
                package if package.is_dir()
                else import_archive(
                    package,
                    source_name=source.name,
                    extracted_root=extracted,
                )
            )
        merged, archive_hash = _merge_roots(
            roots,
            source_name=source.name,
            extracted_parent=extracted,
        )
        discovery = discover_source(merged)
        report = normalize_source(
            source,
            extracted_root=merged,
            archive_sha256=archive_hash,
            discovery=discovery,
            normalized_parent=normalized,
        )
        reports.append(report)
        if report["source_status"] == "ready":
            ready_sources.append(source)
        source_manifest = normalized / source.name / "source_manifest.json"
        manifests.mkdir(parents=True, exist_ok=True)
        shutil.copy2(
            source_manifest, manifests / ("%s.json" % source.name)
        )
        (manifests / ("%s.sha256" % source.name)).write_text(
            "%s  %s\n" % (
                sha256_file(source_manifest),
                source_manifest.name,
            ),
            encoding="utf-8",
        )
    result = {
        "schema_version": "unified-eval-acquisition-sources-v1",
        "sources": reports,
        "archive_required": archive_required,
        "ready_source_names": [item.name for item in ready_sources],
    }
    (logs / "acquisition_sources.json").write_text(
        json.dumps(
            result, ensure_ascii=False, indent=2, sort_keys=True
        ) + "\n",
        encoding="utf-8",
    )
    return result


def _expansion_config(
    root: Path,
    config: Mapping[str, Any],
    registry: Sequence[AcquisitionSource],
    ready_names: set[str],
) -> dict[str, Any]:
    sources: dict[str, Any] = {
        "existing_candidates": {
            "enabled": True,
            "adapter": "existing_unified_eval_candidates",
            "data_config": config["history_data_config"],
        }
    }
    normalized_root = _path(
        root, config["paths"]["normalized_root"]
    )
    for source in registry:
        if source.name not in ready_names:
            continue
        sources[source.name] = {
            "enabled": True,
            "adapter": (
                source.source_plugin
                if source.source_plugin in {"infoseek", "mmsearch"}
                else "generic_heldout"
            ),
            "root_dir": str(normalized_root / source.name),
            "manifest_file": "source_manifest.json",
            "evidence_mode": "text",
            "field_map": {
                "source_id": "source_data_id",
                "question": "question",
                "answer_aliases": "answer_aliases",
                "query_image": "query_image_path",
                "image_evidence": "image_search_records",
                "text_evidence": "text_corpus_records",
                "evidence": "offline_evidence_records",
                "eligible_task_types": "eligible_task_types",
                "source_split": "source_split",
            },
        }
    target = int(config["capacity"]["target_visual"])
    return {
        "schema_version": "unified-agent-eval-v1-source-expansion-v2",
        "targets": {
            "search_free": 250,
            "visual_search_required": target,
            "text_search_required": int(
                config["capacity"]["target_text"]
            ),
            "mixed_search_required": int(
                config["capacity"]["target_mixed"]
            ),
        },
        "history_data_config": config["history_data_config"],
        "sources": sources,
        "capacity": {
            "solver": "exact_max_flow",
            "require_full_joint_quota": True,
            "generate_review_package_only_if_capacity_passes": True,
        },
        "boundaries": {
            "publish_processed_dataset": False,
            "create_formal_approval": False,
            "freeze_environment": False,
            "run_model_evaluation": False,
            "open_test_embargo": False,
        },
        "acquisition_mode": True,
        "output": {
            "staging_dir": (
                "data/staging/"
                "unified_agent_eval_v1_acquisition_next"
            ),
        },
    }


def _publish_staging(root: Path, next_staging: Path) -> Path:
    current = root / "data/staging/unified_agent_eval_v1"
    history = root / "outputs/unified_agent_eval_v1_acquisition/history"
    history.mkdir(parents=True, exist_ok=True)
    if current.is_dir():
        if _tree_hash(current) == _tree_hash(next_staging):
            shutil.rmtree(next_staging)
            return current
        checksum = current / "files.sha256"
        suffix = (
            sha256_file(checksum)[:16]
            if checksum.is_file() else _tree_hash(current)[:16]
        )
        backup = history / ("staging-" + suffix)
        if backup.exists():
            raise SourceInvalidError(
                "staging history already exists: %s" % backup
            )
        os.replace(current, backup)
    os.replace(next_staging, current)
    return current


def run_acquisition(
    root: Path,
    acquisition_config_path: Path,
    registry_path: Path,
    *,
    allow_download: bool = True,
    allow_archive_import: bool = True,
    run_capacity: bool = True,
) -> dict[str, Any]:
    root = Path(root).resolve()
    config = yaml.safe_load(
        Path(acquisition_config_path).read_text(encoding="utf-8")
    )
    if config.get("schema_version") != ACQUISITION_SCHEMA:
        raise SourceInvalidError("acquisition config schema mismatch")
    if any(config["automation"].get(name) is not False for name in (
        "require_manual_sample_filtering",
        "require_manual_manifest_editing",
        "require_manual_evidence_creation",
        "require_manual_approval",
    )):
        raise SourceInvalidError("manual acquisition dependency is enabled")
    ensure_acquisition_directories(root, config)
    registry = load_registry(registry_path)
    source_result = acquire_and_normalize_sources(
        root,
        config,
        registry,
        allow_download=allow_download,
        allow_archive_import=allow_archive_import,
    )
    capacity = None
    if run_capacity and source_result["ready_source_names"]:
        expansion = _expansion_config(
            root,
            config,
            registry,
            set(source_result["ready_source_names"]),
        )
        next_staging = _path(
            root, expansion["output"]["staging_dir"]
        )
        if next_staging.exists():
            raise SourceInvalidError(
                "acquisition-next staging already exists"
            )
        build = build_source_expansion(root, expansion)
        _publish_staging(root, next_staging)
        capacity = build["capacity"]
    elif run_capacity:
        current_capacity = (
            root / "data/staging/unified_agent_eval_v1/"
            "joint_capacity.json"
        )
        if current_capacity.is_file():
            capacity = json.loads(
                current_capacity.read_text(encoding="utf-8")
            )
    return {
        "sources": source_result,
        "capacity": capacity,
        "capacity_gate_passed": bool(
            capacity and capacity.get("capacity_gate_passed")
        ),
    }
