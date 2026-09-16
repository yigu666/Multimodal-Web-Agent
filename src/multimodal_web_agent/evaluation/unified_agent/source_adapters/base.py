from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol

from ..schema import TASK_TYPES


@dataclass(frozen=True)
class SourceCandidate:
    source_dataset: str
    source_data_id: str
    question: str
    image_bytes: bytes
    image_extension: str
    answer_aliases: tuple[str, ...]
    suggested_task_type: str | None
    image_search_records: tuple[str, ...] = ()
    text_corpus_records: tuple[str, ...] = ()
    source_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def image_sha256(self) -> str:
        return hashlib.sha256(self.image_bytes).hexdigest()

    @property
    def candidate_key(self) -> str:
        return "%s:%s" % (self.source_dataset, self.source_data_id)

    @property
    def candidate_sha256(self) -> str:
        value = {
            "source_dataset": self.source_dataset,
            "source_data_id": self.source_data_id,
            "question": self.question,
            "image_sha256": self.image_sha256,
            "answer_aliases": self.answer_aliases,
            "suggested_task_type": self.suggested_task_type,
            "image_search_records": self.image_search_records,
            "text_corpus_records": self.text_corpus_records,
        }
        return hashlib.sha256(json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")).hexdigest()

    def validate(self) -> None:
        if not self.source_dataset or not self.source_data_id:
            raise ValueError("source identifiers cannot be empty")
        if not self.question.strip() or not self.image_bytes:
            raise ValueError("candidate question/image cannot be empty")
        if not self.answer_aliases:
            raise ValueError("candidate answer aliases cannot be empty")
        if (
            self.suggested_task_type is not None
            and self.suggested_task_type not in TASK_TYPES
        ):
            raise ValueError("unsupported suggested task type")


@dataclass(frozen=True)
class SourceScan:
    source_name: str
    available: bool
    candidate_count: int
    message: str
    candidates: tuple[SourceCandidate, ...] = ()
    input_files_sha256: dict[str, str] = field(default_factory=dict)
    skipped_candidate_counts: dict[str, int] = field(default_factory=dict)
    raw_record_count: int | None = None
    hard_rejected_rows: tuple[dict[str, Any], ...] = ()
    inventory: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "source_name": self.source_name,
            "available": self.available,
            "candidate_count": self.candidate_count,
            "message": self.message,
            "input_files_sha256": dict(self.input_files_sha256),
            "skipped_candidate_counts": dict(self.skipped_candidate_counts),
            "raw_record_count": (
                self.raw_record_count
                if self.raw_record_count is not None
                else self.candidate_count + sum(
                    self.skipped_candidate_counts.values()
                )
            ),
            "hard_rejected_count": len(self.hard_rejected_rows),
            "inventory": dict(self.inventory),
        }


class SourceAdapter(Protocol):
    source_name: str

    def scan(self) -> SourceScan:
        ...


def image_bytes(value: Any) -> tuple[bytes, str]:
    if not isinstance(value, Mapping):
        raise ValueError("image value must be a mapping")
    raw = value.get("bytes")
    if isinstance(raw, (bytes, bytearray)) and raw:
        return bytes(raw), ".jpg"
    path = value.get("path")
    if path:
        candidate = Path(str(path))
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        return candidate.read_bytes(), candidate.suffix or ".jpg"
    raise ValueError("image has neither bytes nor a readable path")


def unique_aliases(values: Iterable[Any]) -> tuple[str, ...]:
    result = []
    seen = set()
    for value in values:
        text = str(value).strip()
        folded = text.casefold()
        if text and folded not in seen:
            seen.add(folded)
            result.append(text)
    if not result:
        raise ValueError("source row contains no answer aliases")
    return tuple(result)


def input_file_hashes(paths: Mapping[str, Path | None]) -> dict[str, str]:
    result = {}
    for label, value in paths.items():
        if value is None:
            continue
        path = Path(value)
        if not path.is_file():
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        result[label] = digest.hexdigest()
    return result


def read_tabular(path: Path, columns: list[str] | None = None) -> list[dict]:
    path = Path(path)
    if path.suffix.casefold() == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, list):
            return [dict(row) for row in value]
        if isinstance(value, Mapping):
            for key in ("data", "records", "items", "examples"):
                if isinstance(value.get(key), list):
                    return [dict(row) for row in value[key]]
            return [dict(value)]
        raise ValueError("JSON annotation root must be an object or array")
    if path.suffix.casefold() == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
        return rows
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Parquet source adapters require pyarrow") from exc
    available = set(pq.ParquetFile(path).schema_arrow.names)
    selected = (
        [name for name in columns if name in available]
        if columns is not None else None
    )
    return pq.read_table(path, columns=selected).to_pylist()
