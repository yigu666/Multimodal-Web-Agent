from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

import yaml

from ..data_builder import load_history_references
from .checksum_builder import build_checksums
from .errors import SourceInvalidError
from .gcs_downloader import download_gcs_resource
from .infoseek_capacity_loop import (
    capacity_stop_status,
    evaluate_capacity,
    save_checkpoint,
    validate_baseline_capacity,
)
from .infoseek_metadata import (
    deterministic_stratified_order,
    join_qtype_metadata,
    load_infoseek_candidates,
    load_official_entity_metadata,
    load_oven_image_mapping,
    metadata_prefilter,
)
from .infoseek_normalizer import build_infoseek_normalized
from .infoseek_reserved import write_reserved_eval_manifests
from .infoseek_wikipedia_index import (
    build_wikipedia_index,
    connect_official_evidence,
)
from .oven_access import discover_hf_token, probe_oven_access
from .oven_shard_downloader import (
    check_disk_budget,
    download_oven_files,
)
from .oven_shard_planner import plan_oven_shards
from .pipeline import _publish_staging
from .selective_tar_extract import extract_selected_images


SCHEMA = "visual-infoseek-acquisition-v1"
IMPLEMENTATION_REVISION = (
    "visual-infoseek-acquisition-v2-tar-membership"
)
REQUIRED_OVEN_METADATA = (
    "README.md",
    "test_data/infoseek_human.jsonl",
    "test_data/infoseek_human_qtype.jsonl",
    "test_data/infoseek_test.jsonl",
    "test_data/infoseek_test_qtype.jsonl",
    "test_data/oven_entity_test.jsonl",
    "test_data/oven_human.jsonl",
    "test_data/oven_query_test.jsonl",
    "ovenid2impath.csv",
)


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _load_configuration(
    project_root: Path, config_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA:
        raise SourceInvalidError(
            "Visual InfoSeek acquisition config schema mismatch"
        )
    registry_path = _resolve(
        project_root, config["source_registry"]
    )
    registry = yaml.safe_load(
        registry_path.read_text(encoding="utf-8")
    )
    source = registry["sources"].get(config["source_key"])
    if not isinstance(source, Mapping):
        raise SourceInvalidError(
            "Visual InfoSeek registry source is missing"
        )
    source = dict(source)
    if (
        source.get("dataset_family") != "visual_infoseek_2023"
        or source.get("source_plugin") != "visual_infoseek"
        or source.get("images", {}).get("dataset_id") != "ychenNLP/oven"
    ):
        raise SourceInvalidError(
            "Visual InfoSeek official source identity mismatch"
        )
    if "train" not in source.get("forbidden_candidate_splits", ()):
        raise SourceInvalidError("InfoSeek Train must be forbidden")
    return config, source


def ensure_infoseek_directories(source_root: Path) -> dict[str, Path]:
    paths = {
        "raw_annotations": source_root / "raw/annotations",
        "raw_oven_metadata": source_root / "raw/oven_metadata",
        "raw_oven_shards": source_root / "raw/oven_shards",
        "raw_wikipedia": source_root / "raw/wikipedia",
        "downloads": source_root / "cache/downloads",
        "extracted_images": source_root / "cache/extracted_images",
        "evidence_index": source_root / "cache/evidence_index",
        "checkpoints": source_root / "cache/checkpoints",
        "normalized": source_root / "normalized",
        "logs": source_root / "logs",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _resource_record(
    *,
    provider: str,
    name: str,
    local_path: Path,
    report: Mapping[str, Any],
    license_name: str,
    official_release_page_commit: str | None = None,
) -> dict[str, Any]:
    filename = str(
        report.get("resource_filename")
        or report.get("filename")
        or local_path.name
    )
    dataset_id = report.get("dataset_id")
    revision = report.get("revision")
    official_url = report.get("official_resource_url")
    if not official_url and dataset_id and revision:
        official_url = (
            "https://huggingface.co/datasets/%s/resolve/%s/%s"
            % (dataset_id, revision, filename)
        )
    return {
        "resource_name": name,
        "resource_filename": filename,
        "provider": provider,
        "official_release_page_commit": official_release_page_commit,
        "official_resource_url": official_url,
        "dataset_id": dataset_id,
        "revision": revision,
        "license": license_name,
        "access_method": (
            "gated_huggingface"
            if provider == "oven_huggingface_snapshot"
            else "official_public_gcs"
        ),
        "local_path": str(local_path),
        "file_size": int(local_path.stat().st_size),
        "downloaded_size": int(local_path.stat().st_size),
        "sha256": (
            report.get("downloaded_sha256")
            or report.get("sha256")
        ),
        "downloaded_sha256": (
            report.get("downloaded_sha256")
            or report.get("sha256")
        ),
        "downloaded_at": report.get("downloaded_at"),
        "expected_content_type": report.get(
            "expected_content_type"
        ),
        "actual_content_type": report.get("actual_content_type"),
        "etag": report.get("etag"),
        "last_modified": report.get("last_modified"),
    }


def _baseline_image_ids(
    project_root: Path,
    history_config: Mapping[str, Any],
) -> set[str]:
    result = set()
    for source in history_config.get("history_sources", ()):
        path_value = source.get("path")
        if (
            not isinstance(path_value, (str, os.PathLike))
            or not str(path_value).strip()
        ):
            # Future training sources may be registered with a null path until
            # they exist. They remain visible to the unified leakage inventory
            # but cannot contribute image IDs yet.
            continue
        path = _resolve(project_root, path_value)
        if source.get("format") != "jsonl" or not path.is_file():
            continue
        for row in _read_jsonl(path):
            image_id = (
                row.get("query_image_id")
                or row.get("image_id")
                or (row.get("source_metadata") or {}).get(
                    "query_image_id"
                )
            )
            if image_id:
                result.add(str(image_id))
    staging = (
        project_root / "data/staging/unified_agent_eval_v1/"
        "candidates/post_dedup_candidates.jsonl"
    )
    for row in _read_jsonl(staging):
        image_id = (
            row.get("query_image_id")
            or (row.get("source_metadata") or {}).get("query_image_id")
        )
        if image_id:
            result.add(str(image_id))
    return result


def _evidence_for_candidates(
    candidates: Sequence[Mapping[str, Any]],
    entity_metadata: Mapping[str, Mapping[str, Any]],
    full_index: Path,
) -> dict[str, dict[str, Any]]:
    result = {}
    for row in candidates:
        entity = entity_metadata.get(row["query_image_id"]) or {}
        entity_id = str(
            entity.get("entity_id")
            or entity.get("wikidata_id")
            or entity.get("wikipedia_id")
            or ""
        ) or None
        title = str(
            entity.get("wikipedia_title")
            or entity.get("entity_title")
            or entity.get("title")
            or ""
        ) or None
        evidence = connect_official_evidence(
            full_index,
            candidate_id=(
                "visual_infoseek_2023:" + row["source_data_id"]
            ),
            entity_id=entity_id,
            wikipedia_title=title,
            answer_aliases=row["answer_aliases"],
        )
        if evidence:
            result[row["source_data_id"]] = evidence
    return result


def _publish_normalized(
    working: Path, final: Path
) -> None:
    working_manifest = working / "source_manifest.json"
    working_checksums = working / "files.sha256"
    if not working_manifest.is_file() or not working_checksums.is_file():
        raise SourceInvalidError(
            "normalized InfoSeek working snapshot is incomplete"
        )
    temporary = final.with_name("." + final.name + ".publish")
    previous = final.with_name("." + final.name + ".previous")
    # Recover automatically if a process stopped between the two atomic
    # directory renames. This cache is an incremental acquisition snapshot,
    # not the frozen/formal evaluation dataset.
    if previous.exists():
        final_complete = (
            (final / "source_manifest.json").is_file()
            and (final / "files.sha256").is_file()
        )
        if final_complete:
            shutil.rmtree(previous)
        else:
            if final.exists():
                shutil.rmtree(final)
            os.replace(previous, final)
    if temporary.exists():
        shutil.rmtree(temporary)
    if final.exists():
        old_checksums = final / "files.sha256"
        if (
            old_checksums.is_file()
            and old_checksums.read_bytes() == working_checksums.read_bytes()
        ):
            return
    shutil.copytree(working, temporary)
    if final.exists():
        os.replace(final, previous)
    try:
        os.replace(temporary, final)
    except Exception:
        if previous.exists() and not final.exists():
            os.replace(previous, final)
        raise
    if previous.exists():
        shutil.rmtree(previous)


def _existing_complete_result(
    project_root: Path,
    config: Mapping[str, Any],
    paths: Mapping[str, Path],
) -> dict[str, Any] | None:
    final = paths["normalized"]
    if not (final / "source_manifest.json").is_file():
        return None
    report_path = paths["logs"] / "final_report.json"
    if not report_path.is_file():
        return None
    previous_report = json.loads(
        report_path.read_text(encoding="utf-8")
    )
    if (
        previous_report.get("implementation_revision")
        != IMPLEMENTATION_REVISION
    ):
        return None
    outputs = (
        _resolve(project_root, config["paths"]["staging_root"]),
        _resolve(project_root, config["paths"]["unified_staging_root"]),
    )
    for output in outputs:
        if not output.exists():
            continue
        capacity_path = output / "joint_capacity.json"
        if capacity_path.is_file():
            capacity = json.loads(
                capacity_path.read_text(encoding="utf-8")
            )
            status = capacity_stop_status(
                capacity,
                formal_target_per_type=int(
                    config["capacity_stop"]["formal_target_per_type"]
                ),
                buffer_target_per_type=int(
                    config["capacity_stop"][
                        "acquisition_buffer_target_per_type"
                    ]
                ),
            )
            return {
                "capacity": capacity,
                "capacity_status": status,
                "resource_budget_reached": bool(
                    previous_report.get("resource_budget_reached")
                ),
                "implementation_revision": IMPLEMENTATION_REVISION,
                "reused": True,
            }
    return None


def run_visual_infoseek_acquisition(
    project_root: Path,
    config_path: Path,
) -> dict[str, Any]:
    project_root = Path(project_root).resolve()
    config, source = _load_configuration(project_root, config_path)
    source_root = _resolve(
        project_root, config["paths"]["source_root"]
    )
    paths = ensure_infoseek_directories(source_root)
    baseline_path = _resolve(
        project_root, config["paths"]["unified_staging_root"]
    ) / "joint_capacity.json"
    baseline = validate_baseline_capacity(baseline_path)
    existing = _existing_complete_result(
        project_root, config, paths
    )
    if existing:
        return {
            "schema_version": "visual-infoseek-acquisition-result-v1",
            "baseline": baseline,
            **existing,
            "formal_data_published": False,
            "frozen_environment_created": False,
            "raw_sft_eval_run": False,
            "test_embargo_opened": False,
        }

    download_config = config["download"]
    resource_records = []
    annotation_paths = []
    gcs_resources = source["annotations"]["resources"]
    for split in ("test", "human"):
        resource = gcs_resources[split]
        target = paths["raw_annotations"] / resource["resource_filename"]
        report = download_gcs_resource(
            resource,
            target,
            retries=int(download_config["retries"]),
            connect_timeout_seconds=int(
                download_config["connect_timeout_seconds"]
            ),
            read_timeout_seconds=int(
                download_config["read_timeout_seconds"]
            ),
            resume=bool(download_config["resume"]),
        )
        annotation_paths.append(target)
        resource_records.append(_resource_record(
            provider="official_gcs",
            name="infoseek_" + split,
            local_path=target,
            report=report,
            license_name="Apache-2.0",
            official_release_page_commit=source[
                "official_release_repository"
            ]["official_release_page_commit"],
        ))
    license_resource = source["license"]["infoseek_license"]
    infoseek_license = (
        paths["raw_annotations"] / license_resource["resource_filename"]
    )
    license_report = download_gcs_resource(
        license_resource,
        infoseek_license,
        retries=int(download_config["retries"]),
        connect_timeout_seconds=int(
            download_config["connect_timeout_seconds"]
        ),
        read_timeout_seconds=int(download_config["read_timeout_seconds"]),
        resume=bool(download_config["resume"]),
    )
    resource_records.append(_resource_record(
        provider="official_git_release",
        name="infoseek_license",
        local_path=infoseek_license,
        report=license_report,
        license_name="Apache-2.0",
        official_release_page_commit=source[
            "official_release_repository"
        ]["official_release_page_commit"],
    ))

    token, token_source = discover_hf_token()
    access = probe_oven_access(
        dataset_id=config["huggingface"]["dataset_id"],
        endpoints=config["huggingface"]["endpoints"],
        token=token,
        token_source=token_source,
    )
    oven_reports = download_oven_files(
        access,
        REQUIRED_OVEN_METADATA,
        paths["raw_oven_metadata"],
        token=token,
        retries=int(download_config["retries"]),
    )
    for report in oven_reports:
        local = paths["raw_oven_metadata"] / report["filename"]
        resource_records.append(_resource_record(
            provider="oven_huggingface_snapshot",
            name=report["filename"],
            local_path=local,
            report=report,
            license_name="Apache-2.0",
        ))

    candidates, schema_rejected = load_infoseek_candidates(
        annotation_paths,
        allow_val_as_eval_dev_only=bool(
            config["candidate_sampling"][
                "allow_val_as_eval_dev_only"
            ]
        ),
        supplemental_annotations=[
            paths["raw_oven_metadata"]
            / "test_data/infoseek_test.jsonl",
            paths["raw_oven_metadata"]
            / "test_data/infoseek_human.jsonl",
        ],
    )
    join_qtype_metadata(
        candidates,
        [
            paths["raw_oven_metadata"]
            / "test_data/infoseek_human_qtype.jsonl",
            paths["raw_oven_metadata"]
            / "test_data/infoseek_test_qtype.jsonl",
        ],
    )
    history_path = _resolve(
        project_root, config["history_data_config"]
    )
    history_config = yaml.safe_load(
        history_path.read_text(encoding="utf-8")
    )
    references, history_statuses = load_history_references(
        project_root, history_config
    )
    survivors, prefilter_rejected = metadata_prefilter(
        candidates,
        references,
        baseline_image_ids=_baseline_image_ids(
            project_root, history_config
        ),
    )
    ordered = deterministic_stratified_order(
        survivors,
        seed=int(config["candidate_sampling"]["seed"]),
        maximum_raw_candidates=int(
            config["candidate_sampling"]["maximum_raw_candidates"]
        ),
    )
    metadata_rejected = schema_rejected + prefilter_rejected
    _jsonl(
        paths["logs"] / "metadata_stage_rejected.jsonl",
        metadata_rejected,
    )
    _jsonl(
        paths["logs"] / "metadata_stage_survivors.jsonl",
        ordered,
    )

    image_ids = {row["query_image_id"] for row in ordered}
    entity_paths = [
        paths["raw_oven_metadata"] / name
        for name in REQUIRED_OVEN_METADATA
        if name.startswith("test_data/oven_")
    ]
    entity_metadata = load_official_entity_metadata(
        entity_paths, image_ids
    )
    required_wiki_entity_ids = {
        str(
            row.get("entity_id")
            or row.get("wikidata_id")
            or row.get("wikipedia_id")
            or ""
        ).strip()
        for row in entity_metadata.values()
    } - {""}
    required_wiki_titles = {
        str(
            row.get("wikipedia_title")
            or row.get("entity_title")
            or row.get("title")
            or ""
        ).strip()
        for row in entity_metadata.values()
    } - {""}

    wiki_reports = {}
    for name in ("title_only", "full"):
        resource = source["evidence"]["resources"][name]
        target = paths["raw_wikipedia"] / resource["resource_filename"]
        report = download_gcs_resource(
            resource,
            target,
            retries=int(download_config["retries"]),
            connect_timeout_seconds=int(
                download_config["connect_timeout_seconds"]
            ),
            read_timeout_seconds=int(
                download_config["read_timeout_seconds"]
            ),
            resume=bool(download_config["resume"]),
        )
        wiki_reports[name] = (target, report)
        resource_records.append(_resource_record(
            provider="official_gcs",
            name="wiki6m_" + name,
            local_path=target,
            report=report,
            license_name="CC-BY-SA-3.0",
            official_release_page_commit=source[
                "official_release_repository"
            ]["official_release_page_commit"],
        ))
    title_index = paths["evidence_index"] / "wiki6m_title.sqlite"
    full_index = paths["evidence_index"] / "wiki6m_full.sqlite"
    title_index_report = build_wikipedia_index(
        wiki_reports["title_only"][0],
        title_index,
        include_body=False,
        source_sha256=wiki_reports["title_only"][1][
            "downloaded_sha256"
        ],
        required_entity_ids=required_wiki_entity_ids,
        required_titles=required_wiki_titles,
    )
    full_index_report = build_wikipedia_index(
        wiki_reports["full"][0],
        full_index,
        include_body=True,
        source_sha256=wiki_reports["full"][1]["downloaded_sha256"],
        required_entity_ids=required_wiki_entity_ids,
        required_titles=required_wiki_titles,
    )
    _json(paths["logs"] / "wiki_index_report.json", {
        "title_only": title_index_report,
        "full": full_index_report,
    })

    image_mapping = load_oven_image_mapping(
        paths["raw_oven_metadata"] / "ovenid2impath.csv",
        image_ids,
    )
    plan = plan_oven_shards(
        ordered,
        image_mapping,
        access.files,
        maximum_shards=int(
            config["oven_images"]["maximum_oven_shards"]
        ),
        maximum_download_bytes=int(
            config["oven_images"]["maximum_download_bytes"]
        ),
        preferred_shards=sorted(
            str(row["path"])
            for row in access.files
            if (
                Path(str(row["path"])).name.casefold().startswith("shard")
                and (
                    paths["raw_oven_shards"] / str(row["path"])
                ).is_file()
            )
        ),
        allow_new_shard_downloads=bool(
            config["oven_images"].get(
                "allow_new_shard_downloads", False
            )
        ),
    )
    _json(paths["logs"] / "oven_shard_plan.json", plan)

    all_images: dict[str, dict[str, Any]] = {}
    processed: list[dict[str, Any]] = []
    evidence: dict[str, dict[str, Any]] = {}
    downloaded_shards = []
    downloaded_bytes = 0
    capacity = baseline
    capacity_status = None
    working = (
        paths["checkpoints"] / "working_normalized_v2_tar_membership"
    )
    batch_size = int(
        config["candidate_sampling"]["initial_batch_size"]
    )
    next_batch_size = int(
        config["candidate_sampling"]["next_batch_size"]
    )
    for shard_index, shard in enumerate(plan["required_shards"]):
        shard_file = next(
            row for row in access.files if row["path"] == shard
        )
        target = paths["raw_oven_shards"] / shard
        new_files = [] if target.is_file() else [shard_file]
        if new_files:
            required = check_disk_budget(
                paths["raw_oven_shards"],
                new_files,
                maximum_download_bytes=(
                    int(config["oven_images"]["maximum_download_bytes"])
                    - downloaded_bytes
                ),
                minimum_free_disk_bytes_after_download=int(
                    config["oven_images"][
                        "minimum_free_disk_bytes_after_download"
                    ]
                ),
            )
        else:
            required = 0
        shard_report = download_oven_files(
            access,
            [shard],
            paths["raw_oven_shards"],
            token=token,
            retries=int(download_config["retries"]),
        )[0]
        downloaded_bytes += required
        downloaded_shards.append(shard)
        resource_records.append(_resource_record(
            provider="oven_huggingface_snapshot",
            name=shard,
            local_path=target,
            report=shard_report,
            license_name="Apache-2.0",
        ))
        if plan.get("membership_mode") == "discover_from_tar_members":
            image_ids_to_scan = [
                row["query_image_id"]
                for row in ordered
                if (
                    row["query_image_id"] not in all_images
                    and row["query_image_id"]
                    in plan["member_by_image_id"]
                )
            ]
        else:
            image_ids_to_scan = plan["image_ids_by_shard"][shard]
        extracted = extract_selected_images(
            target,
            shard_name=shard,
            image_ids=image_ids_to_scan,
            member_by_image_id=plan["member_by_image_id"],
            output_root=paths["extracted_images"],
            oven_revision=access.revision,
        )
        all_images.update(extracted)
        shard_candidates = [
            row for row in ordered
            if row["query_image_id"] in extracted
        ]
        shard_candidates.sort(
            key=lambda row: ordered.index(row)
        )
        for offset in range(0, len(shard_candidates), batch_size):
            batch = shard_candidates[offset:offset + batch_size]
            known = {row["source_data_id"] for row in processed}
            processed.extend(
                row for row in batch
                if row["source_data_id"] not in known
            )
            evidence.update(_evidence_for_candidates(
                batch, entity_metadata, full_index
            ))
            build_infoseek_normalized(
                working,
                candidates=processed,
                metadata_rejected=metadata_rejected,
                image_records=all_images,
                entity_metadata=entity_metadata,
                evidence_records=evidence,
                resource_records=resource_records,
                infoseek_license_path=infoseek_license,
                oven_readme_path=(
                    paths["raw_oven_metadata"] / "README.md"
                ),
                oven_revision=access.revision,
                replace_working=True,
            )
            batch_fingerprint = hashlib.sha256(
                (
                    IMPLEMENTATION_REVISION
                    + "\n"
                    + access.revision
                    + "\n"
                    + "\n".join(
                        row["source_data_id"] for row in processed
                    )
                ).encode("utf-8")
            ).hexdigest()[:16]
            batch_output = paths["checkpoints"] / (
                "capacity_v2_%06d_%s"
                % (len(processed), batch_fingerprint)
            )
            if batch_output.is_dir():
                capacity = json.loads(
                    (batch_output / "joint_capacity.json").read_text(
                        encoding="utf-8"
                    )
                )
            else:
                result = evaluate_capacity(
                    project_root,
                    history_data_config=config["history_data_config"],
                    normalized_root=working,
                    output_root=batch_output,
                    target_per_type=int(
                        config["capacity_stop"][
                            "formal_target_per_type"
                        ]
                    ),
                )
                capacity = result["capacity"]
            capacity_status = capacity_stop_status(
                capacity,
                formal_target_per_type=int(
                    config["capacity_stop"]["formal_target_per_type"]
                ),
                buffer_target_per_type=int(
                    config["capacity_stop"][
                        "acquisition_buffer_target_per_type"
                    ]
                ),
            )
            save_checkpoint(
                paths["checkpoints"] / "capacity_loop.json",
                {
                    "processed_raw_candidates": len(processed),
                    "downloaded_oven_shards": downloaded_shards,
                    "downloaded_bytes": downloaded_bytes,
                    "capacity_status": capacity_status,
                    "joint_capacity": capacity,
                },
            )
            batch_size = next_batch_size
            if capacity_status:
                break
        if capacity_status:
            break

    if not working.is_dir():
        build_infoseek_normalized(
            working,
            candidates=[],
            metadata_rejected=metadata_rejected,
            image_records={},
            entity_metadata=entity_metadata,
            evidence_records={},
            resource_records=resource_records,
            infoseek_license_path=infoseek_license,
            oven_readme_path=paths["raw_oven_metadata"] / "README.md",
            oven_revision=access.revision,
            replace_working=True,
        )
    _publish_normalized(working, paths["normalized"])
    final_output = _resolve(
        project_root, config["paths"]["staging_root"]
    )
    revision_marker = (
        final_output / "visual_infoseek_acquisition_revision.json"
    )
    if final_output.exists():
        marker_value = (
            json.loads(revision_marker.read_text(encoding="utf-8"))
            if revision_marker.is_file() else {}
        )
        if (
            marker_value.get("implementation_revision")
            != IMPLEMENTATION_REVISION
        ):
            shutil.rmtree(final_output)
    if not final_output.exists():
        final_result = evaluate_capacity(
            project_root,
            history_data_config=config["history_data_config"],
            normalized_root=paths["normalized"],
            output_root=final_output,
            target_per_type=int(
                config["capacity_stop"]["formal_target_per_type"]
            ),
        )
        capacity = final_result["capacity"]
        _json(revision_marker, {
            "implementation_revision": IMPLEMENTATION_REVISION,
            "oven_revision": access.revision,
        })
    else:
        capacity = json.loads(
            (final_output / "joint_capacity.json").read_text(
                encoding="utf-8"
            )
        )
    capacity_status = capacity_stop_status(
        capacity,
        formal_target_per_type=int(
            config["capacity_stop"]["formal_target_per_type"]
        ),
        buffer_target_per_type=int(
            config["capacity_stop"][
                "acquisition_buffer_target_per_search_type"
            ]
        ) if "acquisition_buffer_target_per_search_type"
        in config["capacity_stop"] else int(
            config["capacity_stop"][
                "acquisition_buffer_target_per_type"
            ]
        ),
    )
    _publish_staging(project_root, final_output)
    accepted = _read_jsonl(paths["normalized"] / "records.jsonl")
    quarantined = _read_jsonl(
        paths["normalized"] / "automatic_quarantine.jsonl"
    )
    reserved = write_reserved_eval_manifests(
        accepted=accepted,
        quarantined=quarantined,
        ids_path=_resolve(project_root, config["paths"]["reserved_ids"]),
        images_path=_resolve(
            project_root, config["paths"]["reserved_images"]
        ),
    )
    resource_budget_reached = (
        capacity_status is None
        and (
            len(downloaded_shards)
            >= int(config["oven_images"]["maximum_oven_shards"])
            or downloaded_bytes
            >= int(config["oven_images"]["maximum_download_bytes"])
            or len(ordered)
            >= int(
                config["candidate_sampling"]["maximum_raw_candidates"]
            )
            or len(downloaded_shards) == len(plan["required_shards"])
        )
    )
    assignment = capacity["joint_assignment"]
    report = {
        "schema_version": "visual-infoseek-acquisition-result-v1",
        "implementation_revision": IMPLEMENTATION_REVISION,
        "dataset_family": "visual_infoseek_2023",
        "baseline": baseline,
        "history_sources": history_statuses,
        "processed_raw_candidates": len(processed),
        "sampled_metadata_candidates": len(ordered),
        "downloaded_oven_shards": downloaded_shards,
        "downloaded_bytes": downloaded_bytes,
        "oven_membership_mode": plan.get(
            "membership_mode", "explicit_mapping"
        ),
        "planned_shard_bytes": plan["estimated_download_bytes"],
        "new_shard_downloads_allowed": bool(
            plan.get("allow_new_shard_downloads", True)
        ),
        "accepted": len(accepted),
        "quarantined": len(quarantined),
        "metadata_rejected": len(metadata_rejected),
        "net_new_candidates": next((
            row["unified_filtering"]["net_new_candidates"]
            for row in json.loads(
                (
                    project_root
                    / "data/staging/unified_agent_eval_v1/"
                    "source_contribution.json"
                ).read_text(encoding="utf-8")
            )["sources"]
            if row["source_name"] == "visual_infoseek_2023"
        ), 0),
        "visual_assigned": assignment["visual_assigned"],
        "text_assigned": assignment["text_assigned"],
        "mixed_assigned": assignment["mixed_assigned"],
        "maximum_joint_search_assignment": assignment[
            "maximum_joint_search_assignment"
        ],
        "remaining_joint_shortfall": capacity[
            "joint_total_remaining_shortfall"
        ],
        "capacity_status": capacity_status,
        "capacity_gate_passed": bool(
            capacity.get("capacity_gate_passed")
        ),
        "resource_budget_reached": resource_budget_reached,
        "reserved": reserved,
        "formal_data_published": False,
        "frozen_environment_created": False,
        "raw_sft_eval_run": False,
        "test_embargo_opened": False,
    }
    _json(paths["logs"] / "final_report.json", report)
    return report
