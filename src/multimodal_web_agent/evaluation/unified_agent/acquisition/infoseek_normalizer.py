from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

from ..leakage import normalized_question
from .checksum_builder import build_checksums
from .errors import SourceInvalidError
from .provenance import sha256_file


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value, ensure_ascii=False, indent=2, sort_keys=True
        ) + "\n",
        encoding="utf-8",
    )


def _jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(
                dict(row), ensure_ascii=False, sort_keys=True
            ) + "\n")


def _eligibility_constraints(
    row: Mapping[str, Any],
    entity: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> tuple[list[str], list[str], list[str]]:
    title = str(
        evidence.get("wikipedia_title")
        or entity.get("wikipedia_title")
        or entity.get("entity_title")
        or entity.get("title")
        or ""
    )
    passage = str(evidence["evidence_text"])
    image_record = (
        "Official OVEN entity mapping identifies the query image as "
        f"{title}. Official Wiki6M evidence: {passage}"
    )
    image_records = [image_record]
    text_records = [passage]
    types = ["visual_search_required", "mixed_search_required"]
    question_tokens = set(normalized_question(row["question"]).split())
    title_tokens = set(normalized_question(title).split())
    if title_tokens and title_tokens <= question_tokens:
        types.insert(1, "text_search_required")
    # These are only route constraints derived from the official entity
    # mapping. The shared source-expansion `_candidate_eligibility` routine
    # remains authoritative for answer reachability and final eligibility.
    return types, image_records, text_records


def build_infoseek_normalized(
    output: Path,
    *,
    candidates: Sequence[Mapping[str, Any]],
    metadata_rejected: Sequence[Mapping[str, Any]],
    image_records: Mapping[str, Mapping[str, Any]],
    entity_metadata: Mapping[str, Mapping[str, Any]],
    evidence_records: Mapping[str, Mapping[str, Any]],
    resource_records: Sequence[Mapping[str, Any]],
    infoseek_license_path: Path,
    oven_readme_path: Path,
    oven_revision: str,
    replace_working: bool = False,
) -> dict[str, Any]:
    output = Path(output)
    temporary = output.with_name(
        ".%s.tmp-%d" % (output.name, os.getpid())
    )
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    for name in ("images", "evidence", "license"):
        (temporary / name).mkdir()
    accepted = []
    quarantined = []
    rejected = [dict(row) for row in metadata_rejected]
    for row in candidates:
        image_id = str(row["query_image_id"])
        image = image_records.get(image_id)
        entity = entity_metadata.get(image_id) or {}
        evidence = evidence_records.get(row["source_data_id"])
        if image and image.get("image_validation_error"):
            rejected.append({
                **row,
                "automatic_quality_decision": "rejected",
                "automatic_quality_reasons": [
                    str(image["image_validation_error"])
                ],
            })
            continue
        reasons = []
        if image is None or image.get("query_image_role_verified") is not True:
            reasons.append("oven_query_image_unavailable")
        if not entity:
            reasons.append("official_entity_mapping_unavailable")
        if evidence is None:
            reasons.append("official_wiki6m_evidence_unavailable")
        if reasons:
            image_fields = ({
                "query_image_id": image_id,
                "query_image_sha256": image["query_image_sha256"],
                "query_image_role_verified": True,
            } if image and image.get("query_image_role_verified") else {
                "query_image_id": image_id,
                "query_image_role_verified": False,
            })
            quarantined.append({
                **row,
                **image_fields,
                "automatic_quality_decision": "quarantined",
                "automatic_quality_reasons": reasons,
            })
            continue
        eligible, image_evidence, text_evidence = _eligibility_constraints(
            row, entity, evidence
        )
        if not eligible:
            quarantined.append({
                **row,
                "automatic_quality_decision": "quarantined",
                "automatic_quality_reasons": [
                    "task_eligibility_uncertain"
                ],
            })
            continue
        source_image = Path(str(image["query_image_path"]))
        suffix = source_image.suffix.casefold() or ".jpg"
        image_name = str(image["query_image_sha256"]) + suffix
        destination = temporary / "images" / image_name
        if not destination.exists():
            try:
                os.link(source_image, destination)
            except OSError:
                shutil.copy2(source_image, destination)
        evidence_name = (
            row["source_data_id"].replace("/", "_") + ".json"
        )
        _json(temporary / "evidence" / evidence_name, evidence)
        accepted.append({
            "source_dataset": "visual_infoseek_2023",
            "dataset_family": "visual_infoseek_2023",
            "source_split": row["source_split"],
            "source_data_id": row["source_data_id"],
            "question": row["question"],
            "answer_aliases": list(row["answer_aliases"]),
            "query_image_id": image_id,
            "query_image_path": "images/" + image_name,
            "query_image_sha256": image["query_image_sha256"],
            "query_image_role_verified": True,
            "query_image_role_source": image[
                "query_image_role_source"
            ],
            "retrieval_result_images_excluded_from_input": True,
            "oven_repository_revision": oven_revision,
            "oven_shard": image["oven_shard"],
            "entity_id": (
                evidence.get("entity_id")
                or entity.get("entity_id")
                or entity.get("wikidata_id")
            ),
            "wikipedia_title": evidence["wikipedia_title"],
            "offline_evidence_records": [{
                "record_id": evidence["evidence_record_id"],
                "text": evidence["evidence_text"],
                "sha256": evidence["evidence_sha256"],
            }],
            "image_search_records": image_evidence,
            "text_corpus_records": text_evidence,
            "answer_reachable": True,
            "license_verified": True,
            "eligible_task_types": eligible,
            "automatic_quality_decision": "accepted",
            "automatic_quality_reasons": [],
            "source_provenance": {
                "infoseek_release": (
                    "open-vision-language/infoseek"
                ),
                "oven_dataset_id": "ychenNLP/oven",
                "oven_revision": oven_revision,
                "wiki_dump_version": "Wiki6M_ver_1_0",
                "evidence_connection_method": (
                    "official_entity_mapping"
                ),
            },
        })
    shutil.copy2(
        infoseek_license_path, temporary / "license/INFOSEEK_LICENSE"
    )
    shutil.copy2(
        oven_readme_path, temporary / "license/OVEN_DATASET_CARD.md"
    )
    _jsonl(temporary / "records.jsonl", accepted)
    _jsonl(temporary / "automatic_quarantine.jsonl", quarantined)
    _jsonl(temporary / "rejection_report.jsonl", rejected)
    acquisition = {
        "schema_version": "visual-infoseek-acquisition-report-v1",
        "source_name": "visual_infoseek_2023",
        "dataset_family": "visual_infoseek_2023",
        "raw_records": len(candidates) + len(metadata_rejected),
        "schema_parsed": len(candidates) + len(metadata_rejected),
        "accepted": len(accepted),
        "quarantined": len(quarantined),
        "rejected": len(rejected),
        "manual_review_required": False,
        "source_status": "ready" if accepted else "partial",
        "split_counts": {
            split: sum(row["source_split"] == split for row in candidates)
            for split in ("test", "human", "val", "train")
        },
    }
    _json(temporary / "acquisition_report.json", acquisition)
    manifest = {
        "schema_version": "unified-eval-heldout-source-v2",
        "source_name": "visual_infoseek_2023",
        "dataset_name": "2023 Visual InfoSeek",
        "dataset_family": "visual_infoseek_2023",
        "dataset_version": "2023",
        "source_split": "test+human",
        "official_source_reference": (
            "https://github.com/open-vision-language/infoseek"
        ),
        "source_url": (
            "https://github.com/open-vision-language/infoseek"
        ),
        "license_name": "resource-specific; see resource_licenses",
        "license_verified": True,
        "license_file": "license/INFOSEEK_LICENSE",
        "resource_licenses": {
            "infoseek": {
                "license": "Apache-2.0",
                "file": "license/INFOSEEK_LICENSE",
            },
            "oven": {
                "license": "Apache-2.0",
                "file": "license/OVEN_DATASET_CARD.md",
            },
            "wiki6m": {
                "license": "CC-BY-SA-3.0",
                "provenance": "InfoSeek official release",
            },
        },
        "resources": list(resource_records),
        "archive_sha256": hashlib_for_resources(resource_records),
        "annotation_files": ["records.jsonl"],
        "image_roots": ["images"],
        "evidence_roots": ["evidence"],
        "evidence_sources": ["Wiki6M_ver_1_0"],
        "raw_record_count": acquisition["raw_records"],
        "query_image_role": "original_query_image",
        "acquisition_mode": "automatic_visual_infoseek_2023",
        "created_by": "automatic_acquisition_pipeline",
        "checksums_file": "files.sha256",
        "acquisition_report_file": "acquisition_report.json",
        "oven_repository_revision": oven_revision,
    }
    _json(temporary / "source_manifest.json", manifest)
    build_checksums(temporary)
    if output.exists():
        if not replace_working:
            old = output / "source_manifest.json"
            if old.is_file() and sha256_file(old) == sha256_file(
                temporary / "source_manifest.json"
            ):
                shutil.rmtree(temporary)
                return acquisition
            shutil.rmtree(temporary)
            raise SourceInvalidError(
                "SOURCE_CACHE_CONFLICT: normalized InfoSeek output exists"
            )
        shutil.rmtree(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temporary, output)
    return acquisition


def hashlib_for_resources(
    resources: Sequence[Mapping[str, Any]],
) -> str:
    import hashlib
    payload = json.dumps(
        list(resources), ensure_ascii=False, sort_keys=True, default=str
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
