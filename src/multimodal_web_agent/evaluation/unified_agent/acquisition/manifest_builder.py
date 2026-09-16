from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .schema import NORMALIZED_MANIFEST_SCHEMA


def build_manifest(
    *,
    source_name: str,
    dataset_name: str | None,
    dataset_version: str | None,
    source_split: str | None,
    official_source_reference: str | None,
    license_metadata: Mapping[str, Any],
    archive_sha256: str,
    raw_record_count: int,
    normalized_root: Path,
    evidence_sources: Sequence[str],
) -> dict[str, Any]:
    return {
        "schema_version": NORMALIZED_MANIFEST_SCHEMA,
        "source_name": source_name,
        "dataset_name": dataset_name,
        "dataset_version": dataset_version,
        "source_split": source_split,
        "official_source_reference": official_source_reference,
        "source_url": official_source_reference,
        "license_name": license_metadata.get("license_name"),
        "license_verified": bool(
            license_metadata.get("license_verified")
        ),
        "license_file": (
            "license/%s"
            % Path(str(license_metadata["license_file"])).name
            if license_metadata.get("license_file") else None
        ),
        "archive_sha256": archive_sha256,
        "annotation_files": ["records.jsonl"],
        "image_roots": ["images"],
        "evidence_roots": ["evidence"],
        "evidence_sources": list(evidence_sources),
        "raw_record_count": raw_record_count,
        "query_image_role": "original_query_image",
        "acquisition_mode": "automatic_download_or_official_archive",
        "created_by": "automatic_acquisition_pipeline",
        "checksums_file": "files.sha256",
        "normalized_root": str(normalized_root),
    }


def write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    Path(path).write_text(
        json.dumps(
            dict(manifest), ensure_ascii=False, indent=2, sort_keys=True
        ) + "\n",
        encoding="utf-8",
    )
