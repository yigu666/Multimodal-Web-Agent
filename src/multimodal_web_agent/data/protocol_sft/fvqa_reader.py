from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Sequence

from .answer_normalizer import accepted_answer_list


@dataclass(frozen=True)
class FVQARecord:
    data_id: str
    question: str
    canonical_answer: str
    accepted_answers: List[str]
    category: str
    row_index: int
    image_ref: Dict[str, Any]
    data_source: str


def _question_from_row(row: Mapping[str, Any]) -> str:
    prompt = row.get("prompt")
    if prompt is None:
        prompt = row.get("question", row.get("content"))
    if isinstance(prompt, str):
        return prompt.strip()
    if isinstance(prompt, Sequence):
        user_contents = []
        all_contents = []
        for message in prompt:
            if isinstance(message, Mapping):
                content = str(message.get("content", "")).strip()
                if content:
                    all_contents.append(content)
                    if str(message.get("role", "")).casefold() == "user":
                        user_contents.append(content)
        if user_contents:
            return user_contents[-1]
        if all_contents:
            return all_contents[-1]
    return ""


def _record_from_row(row: Mapping[str, Any], row_index: int) -> FVQARecord:
    data_id = str(row.get("data_id", "")).strip()
    question = _question_from_row(row)
    reward = row.get("reward_model")
    if not isinstance(reward, Mapping):
        reward = row
    canonical = str(reward.get("ground_truth", "")).strip()
    candidates = reward.get("candidate_answers")
    accepted = accepted_answer_list(canonical, candidates)
    category = str(row.get("category", "")).strip().casefold().replace("-", "_")
    if not data_id:
        raise ValueError("FVQA row %d has no data_id" % row_index)
    if not question:
        raise ValueError("FVQA row %d has no question" % row_index)
    if not canonical or not accepted:
        raise ValueError("FVQA row %d has no accepted answer" % row_index)
    if category not in {"search_free", "search_required"}:
        raise ValueError("FVQA row %d has unsupported category %r" % (row_index, category))
    image_ref = row.get("image_ref")
    if not isinstance(image_ref, Mapping):
        image_ref = {
            "kind": "fvqa_parquet_row",
            "data_id": data_id,
            "row_index": row_index,
            "image_index": 0,
        }
    return FVQARecord(
        data_id=data_id,
        question=question,
        canonical_answer=canonical,
        accepted_answers=accepted,
        category=category,
        row_index=row_index,
        image_ref=dict(image_ref),
        data_source=str(row.get("data_source", "mmsearch_r1/fvqa_train")),
    )


def _read_jsonl(path: Path) -> Iterator[FVQARecord]:
    with path.open("r", encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise TypeError("fixture JSONL rows must be objects")
            yield _record_from_row(value, row_index)


def _read_parquet(path: Path, batch_size: int) -> Iterator[FVQARecord]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "Reading FVQA parquet requires pyarrow; install it only in the intended data-build environment."
        ) from exc

    parquet = pq.ParquetFile(path)
    required = {"prompt", "images", "reward_model", "data_id", "category"}
    available = set(parquet.schema_arrow.names)
    missing = required - available
    if missing:
        raise ValueError("FVQA parquet is missing columns: %s" % sorted(missing))
    columns = ["prompt", "reward_model", "data_source", "data_id", "category"]
    columns = [name for name in columns if name in available]
    row_index = 0
    for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
        for row in batch.to_pylist():
            yield _record_from_row(row, row_index)
            row_index += 1


def read_fvqa_records(path: Path, batch_size: int = 256) -> List[FVQARecord]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.casefold() in {".jsonl", ".json"}:
        records = list(_read_jsonl(path))
    elif path.suffix.casefold() in {".parquet", ".pq"}:
        records = list(_read_parquet(path, batch_size))
    else:
        raise ValueError("unsupported FVQA source format: %s" % path.suffix)
    seen = set()
    duplicates = []
    for record in records:
        if record.data_id in seen:
            duplicates.append(record.data_id)
        seen.add(record.data_id)
    if duplicates:
        raise ValueError("duplicate FVQA data_id values: %s" % sorted(set(duplicates)))
    return records
