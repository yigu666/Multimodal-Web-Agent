from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
from typing import Any, Mapping

from ..answer_metrics import answer_reachable
from .answer_aliases import extract_answer_aliases
from .checksum_builder import build_checksums
from .discovery import IMAGE_SUFFIXES
from .errors import SourceInvalidError
from .evidence_builder import build_official_evidence
from .license_metadata import detect_license
from .manifest_builder import build_manifest, write_manifest
from .provenance import sha256_file
from .query_image_resolver import resolve_query_image
from .schema import AcquisitionSource
from .sources import get_plugin


IMAGE_EVIDENCE_FIELDS = (
    "image_search_results",
    "image_search_records",
)
TEXT_EVIDENCE_FIELDS = (
    "offline_evidence_records",
    "evidence",
    "documents",
    "supporting_documents",
    "text_search_results",
    "page",
    "wiki",
)


def _records(row: Mapping[str, Any], fields: tuple[str, ...]) -> list[str]:
    result = []
    for field in fields:
        value = row.get(field)
        if value is None:
            continue
        values = value if isinstance(value, (list, tuple)) else [value]
        for item in values:
            if isinstance(item, Mapping):
                text = " | ".join(
                    str(item.get(key, "")).strip()
                    for key in ("title", "snippet", "text", "content")
                    if str(item.get(key, "")).strip()
                )
            else:
                text = str(item).strip()
            if text:
                result.append(text)
    return result


def _source_reference(
    source: AcquisitionSource,
    root: Path,
    discovery: Mapping[str, Any],
) -> str | None:
    if source.official_source_reference:
        return source.official_source_reference
    for value in discovery.get("readme_files", ()):
        text = (root / value).read_text(
            encoding="utf-8", errors="replace"
        )
        match = re.search(r"https://[^\s)>\"']+", text)
        if match:
            return match.group(0).rstrip(".,")
    return None


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(
                dict(row), ensure_ascii=False, sort_keys=True
            ) + "\n")


def normalize_source(
    source: AcquisitionSource,
    *,
    extracted_root: Path,
    archive_sha256: str,
    discovery: Mapping[str, Any],
    normalized_parent: Path,
) -> dict[str, Any]:
    output = Path(normalized_parent) / source.name
    if output.exists():
        report = output / "acquisition_report.json"
        if report.is_file():
            value = json.loads(report.read_text(encoding="utf-8"))
            if value.get("archive_sha256") == archive_sha256:
                return value
        raise SourceInvalidError(
            "normalized source exists with different provenance: %s"
            % source.name
        )
    temporary = output.with_name(
        ".%s.tmp-%d" % (source.name, os.getpid())
    )
    temporary.mkdir(parents=True)
    for name in ("images", "evidence", "license"):
        (temporary / name).mkdir()
    plugin = get_plugin(source.source_plugin)(source.preferred_splits)
    rows = plugin.load_rows(extracted_root, discovery)
    if not discovery.get("candidate_annotation_files") or not rows:
        shutil.rmtree(temporary)
        raise SourceInvalidError(
            "%s has no records matching its official schema" % source.name
        )
    image_files = [
        path for path in extracted_root.rglob("*")
        if path.is_file() and path.suffix.casefold() in IMAGE_SUFFIXES
    ]
    license_metadata = detect_license(extracted_root, discovery)
    reference = _source_reference(source, extracted_root, discovery)
    accepted = []
    rejected = []
    quarantined = []
    seen_ids = set()
    detected_splits = set()
    for annotation, row_index, row in rows:
        source_id = plugin.source_id(row)
        candidate_id = "%s:%s" % (source.name, source_id or row_index)
        reasons = []
        uncertainty = []
        question = plugin.question(row)
        aliases = extract_answer_aliases(row)
        if not source_id:
            reasons.append("missing_source_data_id")
        if source_id in seen_ids:
            reasons.append("duplicate_source_data_id")
        seen_ids.add(source_id)
        if not question:
            reasons.append("empty_question")
        if not aliases:
            reasons.append("empty_answer_aliases")
        try:
            image = resolve_query_image(
                row,
                source_root=extracted_root,
                image_files=image_files,
            )
        except (OSError, ValueError):
            image = None
            reasons.append("query_image_role_unverified")
        evidence = build_official_evidence(
            candidate_id, row, aliases
        )
        image_records = _records(row, IMAGE_EVIDENCE_FIELDS)
        text_records = _records(row, TEXT_EVIDENCE_FIELDS)
        eligible = []
        if aliases and answer_reachable(aliases, image_records):
            eligible.append("visual_search_required")
        if aliases and answer_reachable(aliases, text_records):
            eligible.append("text_search_required")
        if (
            image_records
            and text_records
            and aliases
            and answer_reachable(
                aliases, image_records + text_records
            )
        ):
            eligible.append("mixed_search_required")
        if not evidence["evidence_records"]:
            uncertainty.append("official_evidence_missing")
        elif not evidence["answer_reachable"]:
            uncertainty.append("official_evidence_answer_unreachable")
        if not license_metadata["license_verified"]:
            uncertainty.append("license_unverified")
        if not reference:
            uncertainty.append("source_provenance_unverified")
        if not eligible:
            uncertainty.append("task_eligibility_uncertain")
        split = plugin.split(row, annotation)
        if split:
            detected_splits.add(split)
        base = {
            "source_dataset": source.name,
            "source_split": split,
            "source_data_id": source_id,
            "question": question,
            "answer_aliases": list(aliases),
            "query_image_role_verified": bool(image),
            "offline_evidence_records": [
                item["text"] for item in evidence["evidence_records"]
            ],
            "image_search_records": image_records,
            "text_corpus_records": text_records,
            "answer_reachable": evidence["answer_reachable"],
            "license_verified": license_metadata["license_verified"],
            "source_provenance": {
                "official_source_reference": reference,
                "archive_sha256": archive_sha256,
                "annotation_file": annotation.relative_to(
                    extracted_root
                ).as_posix(),
                "annotation_row_index": row_index,
            },
            "eligible_task_types": eligible,
            "automatic_quality_decision": (
                "rejected" if reasons
                else "quarantined" if uncertainty
                else "accepted"
            ),
            "automatic_quality_reasons": sorted(
                set(reasons + uncertainty)
            ),
        }
        if image:
            extension = image["extension"]
            image_name = image["query_image_sha256"] + extension
            destination = temporary / "images" / image_name
            if not destination.exists():
                if image["bytes"] is not None:
                    destination.write_bytes(image["bytes"])
                else:
                    shutil.copy2(image["path"], destination)
            base.update({
                "query_image_path": "images/" + image_name,
                "query_image_sha256": image["query_image_sha256"],
                "query_image_role_source": image[
                    "query_image_role_source"
                ],
                "retrieval_result_images_excluded_from_input": True,
            })
        (temporary / "evidence" / (
            re.sub(r"[^A-Za-z0-9_.-]+", "_", source_id or str(row_index))
            + ".json"
        )).write_text(
            json.dumps(
                evidence, ensure_ascii=False, indent=2, sort_keys=True
            ) + "\n",
            encoding="utf-8",
        )
        if reasons:
            rejected.append(base)
        elif uncertainty:
            quarantined.append(base)
        else:
            accepted.append(base)
    if rejected and not accepted and not quarantined:
        shutil.rmtree(temporary)
        raise SourceInvalidError(
            "%s contains no structurally valid held-out records"
            % source.name
        )
    if license_metadata.get("license_file"):
        shutil.copy2(
            license_metadata["license_file"],
            temporary / "license"
            / Path(str(license_metadata["license_file"])).name,
        )
    _write_jsonl(temporary / "records.jsonl", accepted)
    _write_jsonl(temporary / "automatic_quarantine.jsonl", quarantined)
    _write_jsonl(temporary / "rejection_report.jsonl", rejected)
    split = (
        sorted(detected_splits)[0]
        if len(detected_splits) == 1
        else "multiple:" + ",".join(sorted(detected_splits))
        if detected_splits else "official_heldout_unspecified"
    )
    manifest = build_manifest(
        source_name=source.name,
        dataset_name=source.official_dataset_name,
        dataset_version=None,
        source_split=split,
        official_source_reference=reference,
        license_metadata=license_metadata,
        archive_sha256=archive_sha256,
        raw_record_count=len(rows),
        normalized_root=output,
        evidence_sources=sorted({
            item["source_type"]
            for row in accepted + quarantined
            for item in build_official_evidence(
                "%s:%s" % (source.name, row["source_data_id"]),
                row,
                row["answer_aliases"],
            )["evidence_records"]
        }),
    )
    write_manifest(temporary / "source_manifest.json", manifest)
    quality = {
        "schema_version": "unified-eval-automatic-quality-v1",
        "manual_review_required": False,
        "raw_records": len(rows),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "quarantined": len(quarantined),
        "accepted_candidate_ids": [
            "%s:%s" % (source.name, row["source_data_id"])
            for row in accepted
        ],
    }
    (temporary / "automatic_quality_report.json").write_text(
        json.dumps(quality, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = {
        "schema_version": "unified-eval-acquisition-report-v1",
        "source_name": source.name,
        "archive_sha256": archive_sha256,
        "source_status": (
            "ready" if accepted
            else "unavailable_for_formal_eval"
        ),
        **quality,
    }
    (temporary / "acquisition_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    build_checksums(temporary)
    output.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temporary, output)
    return report
