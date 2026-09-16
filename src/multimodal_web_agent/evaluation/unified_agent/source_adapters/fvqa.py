from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from multimodal_web_agent.data.protocol_sft.cache_reader import ImageSearchCache

from .base import (
    SourceCandidate,
    SourceScan,
    image_bytes,
    input_file_hashes,
    read_tabular,
    unique_aliases,
)


def _question(prompt: Any) -> str:
    if isinstance(prompt, str):
        return prompt.strip()
    if isinstance(prompt, list):
        users = [
            str(item.get("content", "")).strip()
            for item in prompt if isinstance(item, Mapping)
            and str(item.get("role", "")).casefold() == "user"
        ]
        if users:
            return users[-1]
    return ""


def _candidate_answers(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            return [value]
    return []


class FVQATestAdapter:
    source_name = "fvqa_test"

    def __init__(self, parquet_path: Path | None, cache_path: Path | None):
        self.parquet_path = Path(parquet_path) if parquet_path else None
        self.cache_path = Path(cache_path) if cache_path else None

    def scan(self) -> SourceScan:
        if self.parquet_path is None or not self.parquet_path.is_file():
            return SourceScan(self.source_name, False, 0, "FVQA Test unavailable")
        cache = (
            ImageSearchCache.load(self.cache_path)
            if self.cache_path is not None and self.cache_path.is_file()
            else None
        )
        rows = read_tabular(
            self.parquet_path,
            ["data_id", "prompt", "images", "reward_model", "data_source", "category"],
        )
        candidates = []
        rejected = []
        skipped_counts = {}
        for index, row in enumerate(rows):
            source_id = str(row.get("data_id", "")).strip()
            images = row.get("images") or []
            reasons = []
            if not source_id:
                reasons.append("missing_source_data_id")
            if len(images) != 1:
                reasons.append("missing_or_ambiguous_query_image")
            if reasons:
                for reason in reasons:
                    skipped_counts[reason] = (
                        skipped_counts.get(reason, 0) + 1
                    )
                rejected.append({
                    "source_dataset": self.source_name,
                    "source_data_id": source_id or "row-%d" % index,
                    "source_row_index": index,
                    "rejection_reasons": reasons,
                })
                continue
            reward = row.get("reward_model") or {}
            canonical = str(reward.get("ground_truth", "")).strip()
            aliases = unique_aliases([
                canonical, *_candidate_answers(reward.get("candidate_answers"))
            ])
            raw_image, extension = image_bytes(images[0])
            category = str(row.get("category", "")).casefold().replace("-", "_")
            suggested = (
                "search_free" if category == "search_free"
                else "visual_search_required"
                if category == "search_required" else None
            )
            entry = cache.get(source_id) if cache is not None else None
            titles = tuple(title for _, title in entry.usable_titles) if entry else ()
            candidate = SourceCandidate(
                source_dataset=self.source_name,
                source_data_id=source_id,
                question=_question(row.get("prompt")),
                image_bytes=raw_image,
                image_extension=extension,
                answer_aliases=aliases,
                suggested_task_type=suggested,
                image_search_records=titles,
                text_corpus_records=titles,
                source_metadata={
                    "source_row_index": index,
                    "data_source": row.get("data_source"),
                    "task_type_requires_manual_review": suggested is None,
                    "query_image_path": str(
                        images[0].get("path") or "embedded_bytes"
                    ),
                    "query_image_role_verified": True,
                    "query_image_role_source": "dataset_annotation",
                    "retrieval_result_images_excluded_from_input": True,
                    "online_access": False,
                },
            )
            candidate.validate()
            candidates.append(candidate)
        return SourceScan(
            self.source_name, True, len(candidates),
            "FVQA Official Test scanned; missing task labels require review",
            tuple(candidates),
            input_file_hashes({
                "parquet": self.parquet_path,
                "image_search_cache": self.cache_path,
            }),
            skipped_counts,
            raw_record_count=len(rows),
            hard_rejected_rows=tuple(rejected),
            inventory={
                "configured": True,
                "root_exists": True,
                "annotation_files_found": 1,
                "image_files_found": len(candidates),
                "evidence_files_found": int(cache is not None),
                "status": "partial" if rejected else "available",
            },
        )
