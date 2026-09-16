from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


MANIFEST_SCHEMA = "unified-eval-heldout-source-v1"
AUTOMATIC_MANIFEST_SCHEMA = "unified-eval-heldout-source-v2"
READINESS_STATUSES = {"ready", "partial", "unavailable", "invalid"}


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence):
        return [str(item) for item in value]
    return []


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_checksums(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise ValueError(
                "invalid checksums.sha256 line %d" % line_number
            )
        relative = parts[1].lstrip("*").strip().replace("\\", "/")
        if not relative:
            raise ValueError(
                "empty checksums.sha256 path on line %d" % line_number
            )
        result[relative] = parts[0].casefold()
    return result


def check_source_readiness(
    project_root: Path,
    source_name: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a standard Held-out Source Package without importing it."""
    project_root = Path(project_root).resolve()
    enabled = config.get("enabled") is True
    root_value = config.get("root_dir")
    root = (
        _resolve(project_root, str(root_value)).resolve()
        if root_value else None
    )
    base = {
        "schema_version": "unified-eval-source-readiness-v1",
        "source_name": source_name,
        "configured": root_value is not None,
        "enabled": enabled,
        "root_dir": str(root) if root is not None else None,
        "root_exists": bool(root and root.is_dir()),
        "manifest_exists": False,
        "license_verified": False,
        "annotation_files_found": 0,
        "image_files_found": 0,
        "evidence_files_found": 0,
        "hashes_verified": False,
        "source_status": "unavailable",
        "blocking_reasons": [],
        "manifest": None,
        "acquisition": None,
        "resolved": {},
    }
    if not enabled:
        base["blocking_reasons"] = ["source_disabled"]
        return base
    if root is None or not root.is_dir():
        base["blocking_reasons"] = ["source_root_missing"]
        return base

    manifest_name = str(
        config.get("manifest_file") or "source_manifest.json"
    )
    manifest_path = _resolve(root, manifest_name)
    base["manifest_exists"] = manifest_path.is_file()
    if not manifest_path.is_file():
        base["source_status"] = "invalid"
        base["blocking_reasons"] = ["source_manifest_missing"]
        return base
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        base["source_status"] = "invalid"
        base["blocking_reasons"] = [
            "source_manifest_unreadable:%s" % type(exc).__name__
        ]
        return base
    if not isinstance(manifest, Mapping):
        base["source_status"] = "invalid"
        base["blocking_reasons"] = ["source_manifest_not_object"]
        return base
    manifest = dict(manifest)
    base["manifest"] = manifest
    reasons: list[str] = []
    invalid: list[str] = []
    automatic_manifest = (
        manifest.get("schema_version") == AUTOMATIC_MANIFEST_SCHEMA
    )
    acquisition_name = str(
        manifest.get("acquisition_report_file")
        or "acquisition_report.json"
    )
    acquisition_path = _resolve(root, acquisition_name)
    if automatic_manifest and acquisition_path.is_file():
        try:
            acquisition = json.loads(
                acquisition_path.read_text(encoding="utf-8")
            )
            required_counts = (
                "raw_records", "accepted", "quarantined", "rejected"
            )
            if all(
                isinstance(acquisition.get(name), int)
                and acquisition[name] >= 0
                for name in required_counts
            ):
                base["acquisition"] = acquisition
            else:
                invalid.append(
                    "acquisition_report_count_schema_invalid"
                )
        except (OSError, json.JSONDecodeError):
            invalid.append("acquisition_report_unreadable")
    if manifest.get("schema_version") not in {
        MANIFEST_SCHEMA, AUTOMATIC_MANIFEST_SCHEMA
    }:
        invalid.append("source_manifest_schema_mismatch")
    if str(manifest.get("source_name") or "") != source_name:
        invalid.append("source_manifest_name_mismatch")
    required_fields = [
        "dataset_name",
        "source_split",
        "license_name",
        "license_file",
    ]
    if automatic_manifest:
        required_fields.append("official_source_reference")
        if manifest.get("created_by") != (
            "automatic_acquisition_pipeline"
        ):
            invalid.append("automatic_manifest_creator_mismatch")
        if manifest.get("license_verified") is not True:
            invalid.append("automatic_manifest_license_unverified")
    else:
        required_fields.extend([
            "dataset_version", "source_url", "downloaded_at",
        ])
    for field in required_fields:
        if not str(manifest.get(field) or "").strip():
            reasons.append("manifest_%s_missing" % field)
    if manifest.get("query_image_role") != "original_query_image":
        reasons.append("query_image_role_not_original")

    annotations = [
        _resolve(root, value)
        for value in _values(manifest.get("annotation_files"))
    ]
    image_roots = [
        _resolve(root, value)
        for value in _values(manifest.get("image_roots"))
    ]
    evidence_roots = [
        _resolve(root, value)
        for value in _values(manifest.get("evidence_roots"))
    ]
    missing_annotations = [
        path for path in annotations if not path.is_file()
    ]
    missing_image_roots = [
        path for path in image_roots if not path.is_dir()
    ]
    missing_evidence_roots = [
        path for path in evidence_roots if not path.is_dir()
    ]
    if not annotations:
        reasons.append("annotation_files_not_declared")
    if not image_roots:
        reasons.append("image_roots_not_declared")
    if not evidence_roots:
        reasons.append("evidence_roots_not_declared")
    if missing_annotations:
        invalid.append("declared_annotation_file_missing")
    if missing_image_roots:
        invalid.append("declared_image_root_missing")
    if missing_evidence_roots:
        invalid.append("declared_evidence_root_missing")

    license_path = (
        _resolve(root, str(manifest["license_file"]))
        if manifest.get("license_file") else None
    )
    if license_path is not None and not license_path.is_file():
        invalid.append("declared_license_file_missing")
    base["license_verified"] = bool(
        license_path and license_path.is_file()
    )
    existing_annotations = [
        path for path in annotations if path.is_file()
    ]
    image_files = sorted(
        path for directory in image_roots if directory.is_dir()
        for path in directory.rglob("*") if path.is_file()
    )
    evidence_files = sorted(
        path for directory in evidence_roots if directory.is_dir()
        for path in directory.rglob("*") if path.is_file()
    )
    base["annotation_files_found"] = len(existing_annotations)
    base["image_files_found"] = len(image_files)
    base["evidence_files_found"] = len(evidence_files)
    if not image_files:
        reasons.append("query_image_files_missing")
    if not evidence_files:
        reasons.append("evidence_files_missing")

    checksum_name = str(
        manifest.get("checksums_file") or "checksums.sha256"
    )
    checksum_path = _resolve(root, checksum_name)
    declared_files = [
        *existing_annotations,
        *image_files,
        *evidence_files,
    ]
    if license_path and license_path.is_file():
        declared_files.append(license_path)
    if not checksum_path.is_file():
        reasons.append("checksums_file_missing")
    else:
        try:
            checksums = _read_checksums(checksum_path)
            for path in declared_files:
                try:
                    relative = path.relative_to(root).as_posix()
                except ValueError:
                    invalid.append("declared_file_outside_source_root")
                    continue
                expected = checksums.get(relative)
                if expected is None:
                    invalid.append("checksum_entry_missing:%s" % relative)
                elif _sha256(path) != expected:
                    invalid.append("checksum_mismatch:%s" % relative)
            base["hashes_verified"] = not invalid
        except (OSError, UnicodeError, ValueError) as exc:
            invalid.append("checksums_unreadable:%s" % type(exc).__name__)

    base["resolved"] = {
        "manifest_file": str(manifest_path),
        "annotation_files": [str(path) for path in annotations],
        "image_roots": [str(path) for path in image_roots],
        "evidence_roots": [str(path) for path in evidence_roots],
        "license_file": str(license_path) if license_path else None,
        "checksums_file": str(checksum_path),
    }
    if invalid:
        base["source_status"] = "invalid"
        base["blocking_reasons"] = sorted(set(invalid + reasons))
    elif reasons:
        base["source_status"] = "partial"
        base["blocking_reasons"] = sorted(set(reasons))
    else:
        base["source_status"] = "ready"
        base["blocking_reasons"] = []
    if base["source_status"] not in READINESS_STATUSES:
        raise AssertionError("invalid readiness status")
    return base


def adapter_config_from_readiness(
    config: Mapping[str, Any],
    readiness: Mapping[str, Any],
) -> dict[str, Any]:
    """Merge verified manifest paths/metadata into explicit adapter config."""
    if readiness.get("source_status") != "ready":
        raise ValueError("only ready sources may enter an adapter")
    manifest = dict(readiness["manifest"])
    normalized = dict(config)
    normalized.update({
        "annotations": list(manifest["annotation_files"]),
        "image_roots": list(manifest["image_roots"]),
        "evidence_roots": list(manifest["evidence_roots"]),
        "license": manifest["license_name"],
        "license_file": manifest["license_file"],
        "source_url": manifest["source_url"],
        "source_split": manifest["source_split"],
        "source_manifest": manifest,
        "acquisition": readiness.get("acquisition"),
    })
    return normalized
