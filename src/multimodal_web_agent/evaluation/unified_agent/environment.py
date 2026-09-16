from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from multimodal_web_agent.data.protocol_sft.information_formatter import (
    format_frozen_information,
)
from multimodal_web_agent.data.protocol_sft.text_retriever import (
    BootstrapTextRetriever,
    TextDocument,
)

from .answer_metrics import answer_reachable
from .source_adapters.base import SourceCandidate


ENVIRONMENT_SCHEMA = "unified-agent-frozen-environment-v1"
TOKENIZER_VERSION = "protocol-query-tokens-v1"
RETRIEVER_VERSION = "deterministic-bm25-v1"
FORMATTER_VERSION = "unified-information-xml-v1"
IMAGE_RECORD_MAX_CHARS = 512
TEXT_RECORD_MAX_CHARS = 1000


def truncate_record(value: str, maximum_chars: int) -> str:
    text = " ".join(str(value).split())
    return text if len(text) <= maximum_chars else text[:maximum_chars].rstrip()


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(
                dict(row), ensure_ascii=False, sort_keys=True
            ) + "\n")


def build_frozen_environment(
    root: Path,
    candidates: Sequence[SourceCandidate],
    *,
    image_search_top_k: int = 5,
    text_search_top_k: int = 5,
) -> dict[str, Any]:
    root = Path(root)
    image_rows = []
    document_rows = []
    seen_documents = set()
    reachability = {}
    for candidate in sorted(candidates, key=lambda item: item.candidate_key):
        image_records = []
        for value in candidate.image_search_records[:image_search_top_k]:
            text = truncate_record(value, IMAGE_RECORD_MAX_CHARS)
            if text:
                image_records.append(text)
        image_rows.append({
            "image_sha256": candidate.image_sha256,
            "source_dataset": candidate.source_dataset,
            "source_data_id": candidate.source_data_id,
            "results": image_records,
        })
        for index, raw_text in enumerate(candidate.text_corpus_records):
            text = truncate_record(raw_text, TEXT_RECORD_MAX_CHARS)
            if not text:
                continue
            document_id = "%s:%s:text:%d" % (
                candidate.source_dataset,
                candidate.source_data_id,
                index,
            )
            if document_id in seen_documents:
                raise ValueError("duplicate frozen text document ID")
            seen_documents.add(document_id)
            document_rows.append({
                "document_id": document_id,
                "text": str(text),
                "source_dataset": candidate.source_dataset,
                "source_data_id": candidate.source_data_id,
            })
        records = image_records + list(candidate.text_corpus_records)
        reachability[candidate.candidate_key] = answer_reachable(
            candidate.answer_aliases, records
        )
    _write_jsonl(root / "image_search/results.jsonl", image_rows)
    image_results_path = root / "image_search/results.jsonl"
    _write_jsonl(root / "text_corpus/documents.jsonl", document_rows)
    corpus_path = root / "text_corpus/documents.jsonl"
    index_manifest = {
        "schema_version": "unified-agent-text-index-v1",
        "retriever": RETRIEVER_VERSION,
        "tokenizer": TOKENIZER_VERSION,
        "document_count": len(document_rows),
        "corpus_sha256": hashlib.sha256(
            corpus_path.read_bytes()
        ).hexdigest(),
        "top_k": text_search_top_k,
        "tie_breaking": "score_desc_document_id_asc",
    }
    (root / "text_index").mkdir(parents=True, exist_ok=True)
    (root / "text_index/index_manifest.json").write_text(
        json.dumps(index_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": ENVIRONMENT_SCHEMA,
        "online_access": False,
        "image_search": {
            "top_k": image_search_top_k,
            "result_order": "source_frozen_order",
            "formatter": FORMATTER_VERSION,
            "record_max_chars": IMAGE_RECORD_MAX_CHARS,
            "result_count": len(image_rows),
            "results_sha256": hashlib.sha256(
                image_results_path.read_bytes()
            ).hexdigest(),
        },
        "text_search": {
            **index_manifest,
            "formatter": FORMATTER_VERSION,
            "record_max_chars": TEXT_RECORD_MAX_CHARS,
        },
        "reachability": {
            "checked_count": len(reachability),
            "reachable_count": sum(reachability.values()),
            "by_candidate": reachability,
        },
    }
    (root / "environment_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return manifest


class FrozenToolEnvironment:
    def __init__(
        self,
        root: Path,
        *,
        image_search_top_k: int = 5,
        text_search_top_k: int = 5,
    ):
        self.root = Path(root)
        manifest = json.loads(
            (self.root / "environment_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        if manifest.get("schema_version") != ENVIRONMENT_SCHEMA:
            raise ValueError("Frozen environment schema mismatch")
        if manifest.get("online_access") is not False:
            raise PermissionError("Frozen environment permits online access")
        if image_search_top_k != manifest["image_search"]["top_k"]:
            raise ValueError("image search top-k differs from frozen environment")
        if text_search_top_k != manifest["text_search"]["top_k"]:
            raise ValueError("text search top-k differs from frozen environment")
        image_path = self.root / "image_search/results.jsonl"
        corpus_path = self.root / "text_corpus/documents.jsonl"
        if hashlib.sha256(image_path.read_bytes()).hexdigest() != (
            manifest["image_search"]["results_sha256"]
        ):
            raise RuntimeError("frozen image-search results Hash mismatch")
        if hashlib.sha256(corpus_path.read_bytes()).hexdigest() != (
            manifest["text_search"]["corpus_sha256"]
        ):
            raise RuntimeError("frozen text corpus Hash mismatch")
        self.image_search_top_k = image_search_top_k
        self.text_search_top_k = text_search_top_k
        self._images = {}
        with image_path.open(
            "r", encoding="utf-8"
        ) as handle:
            for line in handle:
                row = json.loads(line)
                self._images[row["image_sha256"]] = tuple(row["results"])
        self._documents = []
        with corpus_path.open(
            "r", encoding="utf-8"
        ) as handle:
            for line in handle:
                row = json.loads(line)
                self._documents.append(TextDocument(
                    document_id=row["document_id"],
                    text=row["text"],
                    source_data_id=row["source_data_id"],
                    cache_result_index=0,
                    raw_result_hash=hashlib.sha256(
                        row["text"].encode("utf-8")
                    ).hexdigest(),
                ))
        self._documents.sort(key=lambda item: item.document_id)
        self._retriever = BootstrapTextRetriever(self._documents)

    def image_search(self, image_sha256: str) -> str:
        records = self._images.get(str(image_sha256), ())
        try:
            return format_frozen_information(
                "Image Search", records[:self.image_search_top_k]
            )
        except ValueError as exc:
            raise LookupError(str(exc)) from exc

    def text_search(self, query: str) -> str:
        hits = self._retriever.retrieve(
            query, top_k=self.text_search_top_k
        )
        try:
            return format_frozen_information(
                "Text Search", [hit.document.text for hit in hits]
            )
        except ValueError as exc:
            raise LookupError(str(exc)) from exc

    def deterministic_query_digest(self, query: str) -> str:
        result = self.text_search(query)
        return hashlib.sha256(result.encode("utf-8")).hexdigest()
