from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

from ..answer_metrics import answer_reachable


EVIDENCE_FIELDS = (
    "offline_evidence_records",
    "evidence",
    "documents",
    "supporting_documents",
    "text_search_results",
    "image_search_results",
    "retrieval_results",
    "page",
    "wiki",
)


def _text(value: Any) -> str:
    if isinstance(value, Mapping):
        return " | ".join(
            str(value.get(key, "")).strip()
            for key in (
                "title", "snippet", "text", "content", "answer",
                "page_title", "url",
            )
            if str(value.get(key, "")).strip()
        )
    return str(value).strip()


def build_official_evidence(
    candidate_id: str,
    row: Mapping[str, Any],
    answer_aliases: Sequence[str],
) -> dict[str, Any]:
    records = []
    for field in EVIDENCE_FIELDS:
        value = row.get(field)
        if value is None:
            continue
        values = value if isinstance(value, (list, tuple)) else [value]
        for index, item in enumerate(values):
            text = _text(item)
            if not text:
                continue
            records.append({
                "record_id": "%s:%s:%d" % (
                    candidate_id, field, index
                ),
                "source_type": "official_dataset_evidence",
                "source_reference": (
                    item.get("url") or item.get("page_id") or field
                    if isinstance(item, Mapping) else field
                ),
                "text": text,
                "sha256": hashlib.sha256(
                    text.encode("utf-8")
                ).hexdigest(),
            })
    reachable = answer_reachable(
        answer_aliases, [item["text"] for item in records]
    )
    return {
        "candidate_id": candidate_id,
        "evidence_records": records,
        "answer_reachable": reachable,
        "evidence_generated_from_answer": False,
    }
