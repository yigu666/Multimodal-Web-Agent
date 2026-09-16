from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ..answer_aliases import extract_answer_aliases
from .local_archive import LocalArchiveAcquisitionPlugin


ALLOWED_CANDIDATE_SPLITS = {"test", "human"}
FORBIDDEN_CANDIDATE_SPLITS = {"train"}


def infoseek_split(row: Mapping[str, Any], path: Path | None = None) -> str:
    value = str(row.get("data_split") or row.get("split") or "").casefold()
    if value:
        return "human" if "human" in value else value
    name = (path.name if path else "").casefold()
    if "human" in name:
        return "human"
    if "test" in name:
        return "test"
    if "val" in name:
        return "val"
    if "train" in name:
        return "train"
    return "unknown"


def infoseek_aliases(row: Mapping[str, Any]) -> tuple[str, ...]:
    if row.get("answer_eval") not in (None, "", []):
        return extract_answer_aliases(
            {"answer_eval": row["answer_eval"]}
        )
    return extract_answer_aliases({"answer": row.get("answer")})


def normalize_infoseek_annotation(
    row: Mapping[str, Any],
    *,
    annotation_path: Path | None = None,
) -> dict[str, Any]:
    split = infoseek_split(row, annotation_path)
    return {
        "source_dataset": "visual_infoseek_2023",
        "dataset_family": "visual_infoseek_2023",
        "source_split": split,
        "source_data_id": str(
            row.get("data_id") or row.get("sample_id") or ""
        ).strip(),
        "question": str(row.get("question") or "").strip(),
        "answer_aliases": list(infoseek_aliases(row)),
        "query_image_id": str(
            row.get("image_id") or ""
        ).strip(),
        "official_annotation": dict(row),
    }


class VisualInfoSeekAcquisitionPlugin(LocalArchiveAcquisitionPlugin):
    id_fields = ("data_id", "sample_id", "id")
    question_fields = ("question",)

    def split(
        self, row: Mapping[str, Any], annotation: Path
    ) -> str | None:
        return infoseek_split(row, annotation)
