from __future__ import annotations

from collections import defaultdict
from pathlib import PurePosixPath
import re
from typing import Any, Mapping, Sequence


SHARD_PATTERN = re.compile(r"(shard\d{2}\.tar)", re.IGNORECASE)


def shard_for_image_path(image_path: str) -> str | None:
    match = SHARD_PATTERN.search(str(image_path).replace("\\", "/"))
    return match.group(1).lower() if match else None


def plan_oven_shards(
    candidates: Sequence[Mapping[str, Any]],
    image_mapping: Mapping[str, Mapping[str, str]],
    repository_files: Sequence[Mapping[str, Any]],
    *,
    maximum_shards: int,
    maximum_download_bytes: int,
    preferred_shards: Sequence[str] = (),
    allow_new_shard_downloads: bool = True,
) -> dict[str, Any]:
    repo = {
        str(row["path"]): row
        for row in repository_files
        if SHARD_PATTERN.fullmatch(
            PurePosixPath(str(row["path"])).name
        )
    }
    invalid_sizes = sorted(
        path for path, row in repo.items()
        if int(row.get("size") or 0) <= 0
    )
    if invalid_sizes:
        raise ValueError(
            "OVEN shard size metadata is missing: %s"
            % ", ".join(invalid_sizes)
        )
    candidates_by_shard: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    image_ids_by_shard: dict[str, list[str]] = defaultdict(list)
    member_by_image_id = {}
    unresolved = []
    unknown_membership = []
    for row in candidates:
        image_id = str(row["query_image_id"])
        mapping = image_mapping.get(image_id)
        if not mapping:
            unresolved.append(image_id)
            continue
        member = str(mapping["image_path"]).replace("\\", "/")
        shard = (
            PurePosixPath(str(mapping.get("shard") or "")).name.casefold()
            or shard_for_image_path(member)
        )
        member_by_image_id[image_id] = member
        if not shard:
            unknown_membership.append(image_id)
            continue
        if shard not in repo:
            unresolved.append(image_id)
            continue
        candidates_by_shard[shard].append(row)
        image_ids_by_shard[shard].append(image_id)

    if unknown_membership:
        # The official HF snapshot does not publish an image-to-TAR index.
        # Select a deterministic shard set within the byte budget, then
        # discover membership from real TAR headers. Cached shards affect
        # scan order only, never the selected set.
        selected_set = []
        downloaded_bytes = 0
        for shard in sorted(repo):
            if len(selected_set) >= maximum_shards:
                break
            size = int(repo[shard]["size"])
            if downloaded_bytes + size > maximum_download_bytes:
                continue
            selected_set.append(shard)
            downloaded_bytes += size
        preferred = [
            PurePosixPath(value).name.casefold()
            for value in preferred_shards
        ]
        selected = [
            shard for shard in dict.fromkeys(preferred)
            if shard in selected_set
        ]
        if allow_new_shard_downloads:
            selected.extend(
                shard for shard in selected_set if shard not in selected
            )
        selected_bytes = sum(
            int(repo[shard]["size"]) for shard in selected
        )
        candidate_ids = sorted({
            str(row["query_image_id"])
            for row in candidates
            if str(row["query_image_id"]) in member_by_image_id
        })
        return {
            "schema_version": "oven-required-shard-plan-v2",
            "membership_mode": "discover_from_tar_members",
            "required_image_count": len(candidate_ids),
            "required_shards": selected,
            "selected_shard_set": selected_set,
            "available_cached_shards": [
                shard for shard in selected if shard in preferred
            ],
            "allow_new_shard_downloads": bool(
                allow_new_shard_downloads
            ),
            "image_ids_by_shard": {},
            "candidate_image_ids": candidate_ids,
            "member_by_image_id": {
                image_id: member_by_image_id[image_id]
                for image_id in candidate_ids
            },
            "estimated_download_bytes": selected_bytes,
            "maximum_shard_set_bytes": downloaded_bytes,
            "unresolved_image_ids": sorted(set(unresolved)),
            "selection_objective": (
                "cached_shards_only"
                if not allow_new_shard_downloads else
                "deterministic_shard_set_under_budget_then_cached_scan_order"
            ),
        }

    selected = []
    covered_strata: set[tuple[str, str, str]] = set()
    downloaded_bytes = 0
    remaining = set(candidates_by_shard)
    while remaining and len(selected) < maximum_shards:
        choices = []
        for shard in remaining:
            rows = candidates_by_shard[shard]
            size = int(repo[shard].get("size") or 0)
            strata = {
                (
                    str(row.get("source_split") or "unknown"),
                    str(row.get("question_type") or "unknown"),
                    str(row.get("entity_type") or "unknown"),
                )
                for row in rows
            }
            diversity_gain = len(strata - covered_strata)
            choices.append((
                -len(rows),
                -diversity_gain,
                size,
                shard,
                strata,
            ))
        _, _, size, shard, strata = min(choices)
        if downloaded_bytes + size > maximum_download_bytes:
            remaining.remove(shard)
            continue
        selected.append(shard)
        downloaded_bytes += size
        covered_strata.update(strata)
        remaining.remove(shard)

    selected_set = set(selected)
    planned_ids = {
        image_id
        for shard in selected
        for image_id in image_ids_by_shard[shard]
    }
    unresolved.extend(
        str(row["query_image_id"])
        for row in candidates
        if str(row["query_image_id"]) not in planned_ids
        and str(row["query_image_id"]) not in unresolved
    )
    return {
        "schema_version": "oven-required-shard-plan-v1",
        "required_image_count": len(planned_ids),
        "required_shards": selected,
        "image_ids_by_shard": {
            shard: sorted(set(image_ids_by_shard[shard]))
            for shard in selected
        },
        "member_by_image_id": {
            image_id: member_by_image_id[image_id]
            for image_id in sorted(planned_ids)
        },
        "estimated_download_bytes": downloaded_bytes,
        "unresolved_image_ids": sorted(set(unresolved)),
        "selection_objective": (
            "candidate_coverage_then_strata_diversity_then_download_bytes"
        ),
    }
