from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from .generic_heldout import GenericHeldoutSourceAdapter
from .base import (
    SourceCandidate,
    SourceScan,
    image_bytes,
    input_file_hashes,
    read_tabular,
    unique_aliases,
)


def _website_records(row: Mapping[str, Any]) -> tuple[str, ...]:
    records = []
    for index in range(8):
        value = row.get("website%d_info" % index)
        if not isinstance(value, Mapping):
            continue
        text = " | ".join(
            str(value.get(key, "")).strip()
            for key in ("title", "snippet") if str(value.get(key, "")).strip()
        )
        if text:
            records.append(text)
    return tuple(records)


class MMSearchAdapter:
    source_name = "mmsearch"

    def __init__(
        self,
        end2end_path: Path | None,
        rerank_path: Path | None = None,
        summarization_path: Path | None = None,
    ):
        self.end2end_path = Path(end2end_path) if end2end_path else None
        self.rerank_path = Path(rerank_path) if rerank_path else None
        self.summarization_path = (
            Path(summarization_path) if summarization_path else None
        )

    def scan(self) -> SourceScan:
        if self.end2end_path is None or not self.end2end_path.is_file():
            return SourceScan(self.source_name, False, 0, "MMSearch unavailable")
        end_rows = read_tabular(self.end2end_path, [
            "sample_id", "query", "query_image", "timestamp", "area",
            "subfield", "gt_requery", "gt_answer", "alternative_gt_answers",
        ])
        rerank = {
            str(row["sample_id"]): row
            for row in read_tabular(
                self.rerank_path,
                ["sample_id"] + [
                    "website%d_info" % index for index in range(8)
                ],
            )
        } if self.rerank_path and self.rerank_path.is_file() else {}
        summaries: dict[str, list[str]] = {}
        if self.summarization_path and self.summarization_path.is_file():
            for row in read_tabular(self.summarization_path, [
                "sample_id", "website_retrieved_content",
                "website_original_content", "website_snippet",
            ]):
                source_id = str(row["sample_id"])
                text = str(
                    row.get("website_retrieved_content")
                    or row.get("website_original_content")
                    or row.get("website_snippet")
                    or ""
                ).strip()
                if text:
                    summaries.setdefault(source_id, []).append(text)
        candidates = []
        rejected = []
        skipped_missing_image = 0
        skipped_unusable_image = 0
        for index, row in enumerate(end_rows):
            source_id = str(row.get("sample_id", "")).strip()
            if not source_id:
                rejected.append({
                    "source_dataset": self.source_name,
                    "source_data_id": "row-%d" % index,
                    "source_row_index": index,
                    "rejection_reasons": ["missing_source_data_id"],
                })
                continue
            query_image = row.get("query_image")
            if query_image is None:
                skipped_missing_image += 1
                rejected.append({
                    "source_dataset": self.source_name,
                    "source_data_id": source_id,
                    "source_row_index": index,
                    "rejection_reasons": ["missing_query_image"],
                    "retrieval_result_images_excluded_from_input": True,
                })
                continue
            try:
                raw_image, extension = image_bytes(query_image)
            except (FileNotFoundError, TypeError, ValueError):
                skipped_unusable_image += 1
                rejected.append({
                    "source_dataset": self.source_name,
                    "source_data_id": source_id,
                    "source_row_index": index,
                    "rejection_reasons": ["unusable_query_image"],
                    "retrieval_result_images_excluded_from_input": True,
                })
                continue
            image_records = _website_records(rerank.get(source_id, {}))
            text_records = tuple(summaries.get(source_id, ())) or image_records
            candidate = SourceCandidate(
                source_dataset=self.source_name,
                source_data_id=source_id,
                question=str(row.get("query", "")).strip(),
                image_bytes=raw_image,
                image_extension=extension,
                answer_aliases=unique_aliases([
                    row.get("gt_answer", ""),
                    *(row.get("alternative_gt_answers") or []),
                ]),
                # MMSearch end-to-end requires visual retrieval followed by
                # evidence reading; human review remains mandatory.
                suggested_task_type="mixed_search_required",
                image_search_records=image_records,
                text_corpus_records=text_records,
                source_metadata={
                    "source_row_index": index,
                    "timestamp": row.get("timestamp"),
                    "area": row.get("area"),
                    "subfield": row.get("subfield"),
                    "task_type_requires_manual_review": True,
                    "query_image_path": str(
                        query_image.get("path") or "embedded_bytes"
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
            "MMSearch held-out data scanned; task type and reachability require review",
            tuple(candidates),
            input_file_hashes({
                "end2end": self.end2end_path,
                "rerank": self.rerank_path,
                "summarization": self.summarization_path,
            }),
            {
                "missing_source_data_id": sum(
                    "missing_source_data_id" in row["rejection_reasons"]
                    for row in rejected
                ),
                "missing_query_image": skipped_missing_image,
                "unusable_query_image": skipped_unusable_image,
            },
            raw_record_count=len(end_rows),
            hard_rejected_rows=tuple(rejected),
            inventory={
                "configured": True,
                "root_exists": True,
                "annotation_files_found": 1,
                "image_files_found": len(candidates),
                "evidence_files_found": int(bool(rerank))
                + int(bool(summaries)),
                "status": "partial" if rejected else "available",
            },
        )


class MMSearchHeldOutAdapter(GenericHeldoutSourceAdapter):
    source_name = "mmsearch_heldout"

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        project_root: Path,
    ):
        normalized = dict(config)
        field_map = {
            "source_id": ("sample_id", "source_data_id", "id"),
            "question": ("query", "question"),
            "answers": (
                "alternative_gt_answers",
                "answer_aliases",
                "gt_answer",
            ),
            "query_image": "query_image",
            "image_evidence": "image_search_records",
            "text_evidence": (
                "offline_evidence_records",
                "text_corpus_records",
                "evidence",
            ),
            "task_type": "task_type",
            "eligible_task_types": "eligible_task_types",
            **dict(normalized.get("field_map", {})),
        }
        normalized["field_map"] = field_map
        super().__init__(
            normalized,
            project_root=project_root,
            source_name=self.source_name,
        )

    def scan(self) -> SourceScan:
        scan = super().scan()
        candidates = []
        for candidate in scan.candidates:
            metadata = dict(candidate.source_metadata)
            metadata.update({
                "query_image_role_verified": True,
                "query_image_role_source": "dataset_annotation",
                "retrieval_result_images_excluded_from_input": True,
            })
            candidates.append(replace(
                candidate, source_metadata=metadata
            ))
        return replace(scan, candidates=tuple(candidates))
