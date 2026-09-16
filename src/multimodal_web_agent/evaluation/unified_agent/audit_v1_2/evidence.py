from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import unicodedata
from typing import Any, Iterable, Mapping, Sequence

from multimodal_web_agent.data.protocol_sft.information_formatter import (
    format_frozen_information,
)
from multimodal_web_agent.data.protocol_sft.text_retriever import (
    BootstrapTextRetriever,
    TextDocument,
)
from multimodal_web_agent.evaluation.unified_agent.answer_metrics import (
    normalize_answer,
)
from multimodal_web_agent.evaluation.unified_agent.environment import (
    ENVIRONMENT_SCHEMA,
    truncate_record,
)
from multimodal_web_agent.evaluation.unified_agent.fingerprints import (
    sha256_file,
)

from .answer_equivalence import (
    deterministic_alias_equivalence,
    quantity_equivalence,
    question_requires_numeric_answer,
)


EXPECTED_ENVIRONMENT_MANIFEST_SHA256 = (
    "427761c3b704deef478398b8023eca1a108dcb9046a1f58e8dce6f37d6e9fc74"
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def exact_string_normalize(value: str) -> str:
    return " ".join(
        unicodedata.normalize("NFKC", str(value)).casefold().strip().split()
    )


def _token_subsequence(needle: str, haystack: str) -> bool:
    left = normalize_answer(needle).split()
    right = normalize_answer(haystack).split()
    if not left or len(left) > len(right):
        return False
    width = len(left)
    return any(right[index:index + width] == left for index in range(
        len(right) - width + 1
    ))


def _alias_variant_hit(alias: str, text: str) -> bool:
    normalized_text = normalize_answer(text)
    candidates = set()
    tokens = normalize_answer(alias).split()
    if tokens:
        last = tokens[-1]
        if last.endswith("s") and len(last) > 3:
            candidates.add(" ".join([*tokens[:-1], last[:-1]]))
        else:
            candidates.add(" ".join([*tokens[:-1], last + "s"]))
    if any(_token_subsequence(candidate, normalized_text) for candidate in candidates):
        return True
    # A title/core match is only accepted against a bounded capitalized span.
    spans = re.findall(
        r"\b(?:[A-Z][\w'-]*|of|the)(?:\s+(?:[A-Z][\w'-]*|of|the)){1,7}\b",
        str(text),
    )
    return any(deterministic_alias_equivalence(alias, span) for span in spans)


def evidence_match_layers(
    text: str,
    accepted_answers: Iterable[str],
    *,
    question: str,
) -> dict[str, bool]:
    aliases = tuple(str(value) for value in accepted_answers if str(value).strip())
    exact_text = exact_string_normalize(text)
    exact = any(
        exact_string_normalize(alias) in exact_text
        for alias in aliases if exact_string_normalize(alias)
    )
    normalized = any(_token_subsequence(alias, text) for alias in aliases)
    numeric = False
    unit = False
    if question_requires_numeric_answer(question):
        for alias in aliases:
            numeric_match, unit_match, _ = quantity_equivalence(
                alias, text, question=question
            )
            numeric = numeric or numeric_match
            unit = unit or unit_match
    alias_hit = any(_alias_variant_hit(alias, text) for alias in aliases)
    return {
        "exact_answer_string_hit": exact,
        "normalized_answer_hit": normalized,
        "numeric_answer_hit": numeric,
        "unit_equivalent_hit": unit,
        "alias_answer_hit": alias_hit,
        "evidence_hit_any": exact or normalized or numeric or unit or alias_hit,
    }


def aggregate_evidence_matches(
    retrieved: Sequence[Mapping[str, Any]],
    accepted_answers: Iterable[str],
    *,
    question: str,
) -> dict[str, Any]:
    layer_names = (
        "exact_answer_string_hit",
        "normalized_answer_hit",
        "numeric_answer_hit",
        "unit_equivalent_hit",
        "alias_answer_hit",
    )
    layers = {name: False for name in layer_names}
    first = None
    for call in retrieved:
        if call.get("status") != "success":
            continue
        for result in call.get("results", []):
            match = evidence_match_layers(
                str(result["text"]), accepted_answers, question=question
            )
            for name in layer_names:
                layers[name] = layers[name] or bool(match[name])
            if match["evidence_hit_any"] and first is None:
                first = {
                    "evidence_hit_rank": int(result["rank"]),
                    "evidence_hit_tool": str(call["tool"]),
                    "evidence_hit_turn": int(call["turn"]),
                }
    return {
        **layers,
        "evidence_hit_any": any(layers.values()),
        "evidence_hit_rank": first["evidence_hit_rank"] if first else None,
        "evidence_hit_tool": first["evidence_hit_tool"] if first else None,
        "evidence_hit_turn": first["evidence_hit_turn"] if first else None,
    }


class FrozenEvidenceStore:
    """Read-only deterministic view of the frozen v1.1 tool environment."""

    def __init__(self, project_root: Path):
        self.project_root = Path(project_root).resolve()
        self.dataset_root = (
            self.project_root / "data/processed/unified_agent_eval_v1_1"
        )
        self.root = self.dataset_root / "environment"
        self.manifest_path = self.root / "environment_manifest.json"
        manifest_hash = sha256_file(self.manifest_path)
        if manifest_hash != EXPECTED_ENVIRONMENT_MANIFEST_SHA256:
            raise RuntimeError(
                "Frozen environment manifest Hash mismatch: %s" % manifest_hash
            )
        self.manifest_sha256 = manifest_hash
        self.manifest = json.loads(
            self.manifest_path.read_text(encoding="utf-8")
        )
        if self.manifest.get("schema_version") != ENVIRONMENT_SCHEMA:
            raise ValueError("Frozen environment schema mismatch")
        if self.manifest.get("online_access") is not False:
            raise PermissionError("Frozen environment permits online access")
        if self.manifest["image_search"]["top_k"] != 5:
            raise ValueError("frozen Image Search top-k is not five")
        if self.manifest["text_search"]["top_k"] != 5:
            raise ValueError("frozen Text Search top-k is not five")
        image_path = self.root / "image_search/results.jsonl"
        corpus_path = self.root / "text_corpus/documents.jsonl"
        if sha256_file(image_path) != self.manifest["image_search"][
            "results_sha256"
        ]:
            raise RuntimeError("frozen Image Search results Hash mismatch")
        if sha256_file(corpus_path) != self.manifest["text_search"][
            "corpus_sha256"
        ]:
            raise RuntimeError("frozen Text Search corpus Hash mismatch")
        self.image_results_sha256 = sha256_file(image_path)
        self.text_corpus_sha256 = sha256_file(corpus_path)
        self._images = {
            row["image_sha256"]: row for row in read_jsonl(image_path)
        }
        documents = []
        for row in read_jsonl(corpus_path):
            documents.append(TextDocument(
                document_id=row["document_id"],
                text=row["text"],
                source_data_id=row["source_data_id"],
                cache_result_index=0,
                raw_result_hash=hashlib.sha256(
                    row["text"].encode("utf-8")
                ).hexdigest(),
            ))
        documents.sort(key=lambda item: item.document_id)
        self._retriever = BootstrapTextRetriever(documents)
        self._evidence_cache: dict[str, dict[str, Any]] = {}

    def _evidence(self, example: Mapping[str, Any]) -> dict[str, Any]:
        relative = str(example["evidence_path"])
        if relative in self._evidence_cache:
            return self._evidence_cache[relative]
        path = self.dataset_root / relative
        value = json.loads(path.read_text(encoding="utf-8"))
        content_hash = value.get("content_sha256")
        payload = {key: item for key, item in value.items() if key != "content_sha256"}
        actual = hashlib.sha256(json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")).hexdigest()
        if content_hash != actual:
            raise RuntimeError("frozen candidate evidence Hash mismatch")
        expected_candidate = "%s:%s" % (
            example["source_dataset"], example["source_data_id"]
        )
        if value.get("candidate_id") != expected_candidate:
            raise RuntimeError("candidate evidence identity mismatch")
        self._evidence_cache[relative] = value
        return value

    def image_search(
        self, example: Mapping[str, Any]
    ) -> dict[str, Any]:
        image_hash = str(example["image_sha256"])
        row = self._images.get(image_hash)
        if row is None or not row.get("results"):
            raise LookupError("Image Search has no frozen result")
        returned = list(row["results"][:5])
        full = list(self._evidence(example).get("image_search_records") or [])[:5]
        results = []
        for index, text in enumerate(returned):
            full_text = full[index] if index < len(full) else None
            expected = (
                truncate_record(full_text, 512) if full_text is not None else None
            )
            if expected is not None and expected != text:
                raise RuntimeError("Image Search reconstruction differs from source")
            results.append({
                "rank": index + 1,
                "text": text,
                "document_id": "%s:image:%d" % (
                    example["source_data_id"], index
                ),
                "returned_chars": len(text),
                "full_source_text": full_text,
                "full_source_chars": len(full_text) if full_text is not None else None,
                "truncated": bool(full_text is not None and text != " ".join(
                    str(full_text).split()
                )),
            })
        return {
            "tool": "image_search",
            "query": {
                "kind": "query_image",
                "image_sha256": image_hash,
                "source_data_id": example["source_data_id"],
            },
            "information": format_frozen_information("Image Search", returned),
            "results": results,
        }

    def text_search(self, query: str) -> dict[str, Any]:
        hits = self._retriever.retrieve(str(query), top_k=5)
        if not hits:
            raise LookupError("Text Search has no frozen result")
        records = [hit.document.text for hit in hits]
        return {
            "tool": "text_search",
            "query": {"kind": "text", "text": str(query)},
            "information": format_frozen_information("Text Search", records),
            "results": [
                {
                    "rank": index,
                    "text": hit.document.text,
                    "document_id": hit.document.document_id,
                    "returned_chars": len(hit.document.text),
                    "full_source_text": None,
                    "full_source_chars": None,
                    "truncated": None,
                    "truncation_audit_status": (
                        "unknown_full_source_unavailable_without_opening_"
                        "non_dev_candidate_evidence"
                    ),
                    "bm25_score": float(hit.score),
                }
                for index, hit in enumerate(hits, 1)
            ],
        }

    def source_hashes(self) -> dict[str, Any]:
        return {
            "environment_manifest_sha256": self.manifest_sha256,
            "expected_environment_manifest_sha256": (
                EXPECTED_ENVIRONMENT_MANIFEST_SHA256
            ),
            "environment_manifest_match": True,
            "image_results_sha256": self.image_results_sha256,
            "image_results_manifest_match": True,
            "text_corpus_sha256": self.text_corpus_sha256,
            "text_corpus_manifest_match": True,
        }
