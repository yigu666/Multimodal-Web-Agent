from __future__ import annotations

import csv
import hashlib
from itertools import chain
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..answer_metrics import normalize_answer
from ..leakage import LeakageReference, normalized_question
from .sources.visual_infoseek import (
    ALLOWED_CANDIDATE_SPLITS,
    FORBIDDEN_CANDIDATE_SPLITS,
    normalize_infoseek_annotation,
)


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "%s has invalid JSONL at line %d"
                    % (path, line_number)
                ) from exc
            if not isinstance(row, Mapping):
                raise ValueError(
                    "%s line %d is not an object"
                    % (path, line_number)
                )
            yield dict(row)


def load_infoseek_candidates(
    annotations: Sequence[Path],
    *,
    allow_val_as_eval_dev_only: bool = False,
    supplemental_annotations: Sequence[Path] = (),
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    supplemental_by_id: dict[str, dict[str, Any]] = {}
    for path in supplemental_annotations:
        for row in read_jsonl(path):
            source_id = str(
                row.get("data_id") or row.get("source_data_id") or ""
            )
            if source_id:
                supplemental_by_id[source_id] = row
    candidates = []
    rejected = []
    for path in annotations:
        for row in read_jsonl(path):
            source_id = str(
                row.get("data_id") or row.get("source_data_id") or ""
            )
            supplemental = supplemental_by_id.get(source_id)
            if supplemental:
                # Public GCS Test annotations omit answer labels. Fill only
                # their missing fields from the matching official gated OVEN
                # evaluation metadata while retaining GCS as the primary
                # annotation source.
                row = {
                    **supplemental,
                    **{
                        key: value
                        for key, value in row.items()
                        if value not in (None, "", [])
                    },
                }
            normalized = normalize_infoseek_annotation(
                row, annotation_path=path
            )
            reasons = []
            split = normalized["source_split"]
            if split in FORBIDDEN_CANDIDATE_SPLITS:
                reasons.append("forbidden_candidate_split")
            elif split == "val":
                if not allow_val_as_eval_dev_only:
                    reasons.append("val_not_enabled")
                normalized["eval_dev_only"] = True
            elif split not in ALLOWED_CANDIDATE_SPLITS:
                reasons.append("unsupported_candidate_split")
            if not normalized["source_data_id"]:
                reasons.append("missing_source_data_id")
            if not normalized["question"]:
                reasons.append("empty_question")
            if not normalized["answer_aliases"]:
                reasons.append("empty_answer_aliases")
            if not normalized["query_image_id"]:
                reasons.append("missing_query_image_id")
            normalized["metadata_stage_rejection_reasons"] = reasons
            (rejected if reasons else candidates).append(normalized)
    return candidates, rejected


def join_qtype_metadata(
    candidates: Sequence[dict[str, Any]],
    qtype_paths: Sequence[Path],
) -> None:
    by_id = {
        row["source_data_id"]: row for row in candidates
    }
    for path in qtype_paths:
        for row in read_jsonl(path):
            source_id = str(
                row.get("data_id") or row.get("sample_id") or ""
            )
            target = by_id.get(source_id)
            if target is None:
                continue
            target["question_type"] = (
                row.get("question_type")
                or row.get("qtype")
                or row.get("type")
            )


def deterministic_stratified_order(
    candidates: Sequence[dict[str, Any]],
    *,
    seed: int,
    maximum_raw_candidates: int,
) -> list[dict[str, Any]]:
    strata: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in candidates:
        key = (
            str(row.get("source_split") or "unknown"),
            str(row.get("question_type") or "unknown"),
            str(row.get("entity_type") or "unknown"),
        )
        strata.setdefault(key, []).append(row)
    for key, rows in strata.items():
        rows.sort(key=lambda row: hashlib.sha256(
            ("%d:%s" % (seed, row["source_data_id"])).encode("utf-8")
        ).hexdigest())
    result = []
    keys = sorted(strata)
    while keys and len(result) < maximum_raw_candidates:
        remaining = []
        for key in keys:
            rows = strata[key]
            if rows and len(result) < maximum_raw_candidates:
                result.append(rows.pop(0))
            if rows:
                remaining.append(key)
        keys = remaining
    return result


def metadata_prefilter(
    candidates: Sequence[dict[str, Any]],
    references: Sequence[LeakageReference],
    *,
    baseline_image_ids: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source_ids = {
        row.source_data_id for row in references if row.source_data_id
    }
    questions = {
        normalized_question(row.question)
        for row in references if row.question
    }
    qa_pairs = {
        (normalized_question(row.question), normalize_answer(alias))
        for row in references
        for alias in row.answer_aliases
        if row.question and alias
    }
    image_ids = set(baseline_image_ids or ())
    seen_source_ids: set[str] = set()
    seen_questions: set[str] = set()
    seen_qa: set[tuple[str, str]] = set()
    survivors = []
    rejected = []
    for row in candidates:
        reasons = []
        source_id = row["source_data_id"]
        question = normalized_question(row["question"])
        pairs = {
            (question, normalize_answer(alias))
            for alias in row["answer_aliases"]
        }
        image_id = row["query_image_id"]
        if source_id in source_ids:
            reasons.append("historical_source_data_id_overlap")
        if question in questions:
            reasons.append("historical_exact_question_overlap")
        if pairs & qa_pairs:
            reasons.append("historical_question_answer_overlap")
        if image_id in image_ids:
            reasons.append("baseline_image_id_overlap")
        if source_id in seen_source_ids:
            reasons.append("internal_source_data_id_duplicate")
        if question in seen_questions:
            reasons.append("internal_exact_question_duplicate")
        if pairs & seen_qa:
            reasons.append("internal_question_answer_duplicate")
        seen_source_ids.add(source_id)
        seen_questions.add(question)
        seen_qa.update(pairs)
        if reasons:
            rejected.append({
                **row,
                "metadata_stage_rejection_reasons": sorted(set(reasons)),
            })
        else:
            survivors.append(row)
    return survivors, rejected


def load_official_entity_metadata(
    paths: Sequence[Path],
    required_image_ids: set[str],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in paths:
        for row in read_jsonl(path):
            image_id = str(
                row.get("image_id")
                or row.get("oven_image_id")
                or row.get("id")
                or ""
            )
            if image_id not in required_image_ids:
                continue
            target = result.setdefault(image_id, {})
            for key in (
                "entity_id", "wikidata_id", "wikipedia_id",
                "wikipedia_title", "entity_title", "title",
                "entity_type", "source_dataset",
            ):
                if row.get(key) not in (None, ""):
                    target.setdefault(key, row[key])
    return result


def load_oven_image_mapping(
    path: Path,
    required_image_ids: set[str],
) -> dict[str, dict[str, str]]:
    """Load official OVEN paths without requiring a user-prepared CSV.

    The released ``ovenid2impath.csv`` is headerless, while small development
    fixtures sometimes use a header. Both representations are supported. The
    official OVEN merge layout renames files under
    ``oven_images/<id-prefix>/<oven_id><extension>``. The Hugging Face
    snapshot TARs use a separate physical shard layout, so a headerless CSV
    does not by itself identify which TAR contains an image.
    """
    result = {}
    with Path(path).open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        reader = csv.reader(handle)
        first = next(reader, None)
        if not first:
            raise ValueError("ovenid2impath.csv is empty")
        header = {
            value.strip().casefold(): index
            for index, value in enumerate(first)
        }
        has_header = bool(
            {"image_id", "oven_id", "ovenid", "id"} & set(header)
        )

        def field(
            row: list[str],
            names: Sequence[str],
            fallback: int | None,
        ) -> str:
            for name in names:
                index = header.get(name)
                if index is not None and index < len(row):
                    return str(row[index]).strip()
            if not has_header and fallback is not None and fallback < len(row):
                return str(row[fallback]).strip()
            return ""

        rows = reader if has_header else chain((first,), reader)
        for row in rows:
            image_id = field(
                row, ("image_id", "oven_id", "ovenid", "id"), 0
            )
            if image_id not in required_image_ids:
                continue
            original_path = field(
                row, ("image_path", "path", "impath"), 1
            )
            if not original_path:
                continue
            suffix = Path(original_path).suffix.casefold()
            match = re.fullmatch(r"oven_(\d{2})\d+", image_id.casefold())
            if match:
                directory = match.group(1)
                snapshot_path = (
                    f"oven_images/{directory}/{image_id}{suffix}"
                )
            else:
                snapshot_path = original_path.replace("\\", "/")
            explicit_shard = field(
                row, ("shard", "tar_file", "archive"), None
            )
            result[image_id] = {
                "image_id": image_id,
                "image_path": snapshot_path,
                "source_image_path": original_path.replace("\\", "/"),
                # Only a real column may assert physical membership. The
                # official headerless mapping has no shard column; inferring
                # it from the OVEN ID prefix is incorrect for the HF snapshot.
                "shard": explicit_shard,
            }
    return result
