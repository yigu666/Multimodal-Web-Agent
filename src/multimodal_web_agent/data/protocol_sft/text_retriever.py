from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Set

from .cache_reader import CACHE_SOURCE, CACHE_VERSION, ImageSearchCache
from .query_builder import query_tokens


@dataclass(frozen=True)
class TextDocument:
    document_id: str
    text: str
    source_data_id: str
    cache_result_index: int
    raw_result_hash: str


@dataclass(frozen=True)
class TextSearchHit:
    document: TextDocument
    score: float


class BootstrapTextRetriever:
    """Small deterministic BM25 backend over real cached webpage titles only."""

    def __init__(self, documents: Sequence[TextDocument], k1: float = 1.2, b: float = 0.75):
        self.documents = sorted(documents, key=lambda item: item.document_id)
        self.k1 = k1
        self.b = b
        self._term_frequencies: List[Counter[str]] = []
        self._document_frequency: Dict[str, int] = defaultdict(int)
        self._posting_lists: Dict[str, List[int]] = defaultdict(list)
        self._lengths: List[int] = []
        for document in self.documents:
            tokens = [token.casefold() for token in query_tokens(document.text)]
            frequencies = Counter(tokens)
            self._term_frequencies.append(frequencies)
            self._lengths.append(len(tokens))
            for token in frequencies:
                self._document_frequency[token] += 1
                self._posting_lists[token].append(len(self._term_frequencies) - 1)
        self._average_length = (
            sum(self._lengths) / len(self._lengths) if self._lengths else 0.0
        )

    @classmethod
    def from_image_cache(cls, cache: ImageSearchCache) -> "BootstrapTextRetriever":
        documents = []
        for entry in cache.entries():
            for result_index, title in entry.usable_titles:
                documents.append(
                    TextDocument(
                        document_id="%s:title:%d" % (entry.data_id, result_index),
                        text=title,
                        source_data_id=entry.data_id,
                        cache_result_index=result_index,
                        raw_result_hash=entry.raw_result_hash,
                    )
                )
        return cls(documents)

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        exclude_source_data_id: str = "",
        exclude_document_ids: Iterable[str] = (),
    ) -> List[TextSearchHit]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        query_terms = sorted(set(token.casefold() for token in query_tokens(query)))
        if not query_terms or not self.documents:
            return []
        total_documents = len(self.documents)
        candidate_indices = sorted(
            {
                index
                for term in query_terms
                for index in self._posting_lists.get(term, [])
            }
        )
        excluded_documents: Set[str] = set(exclude_document_ids)
        scored = []
        for index in candidate_indices:
            document = self.documents[index]
            if exclude_source_data_id and document.source_data_id == exclude_source_data_id:
                continue
            if document.document_id in excluded_documents:
                continue
            frequencies = self._term_frequencies[index]
            length = self._lengths[index]
            score = 0.0
            for term in query_terms:
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                document_frequency = self._document_frequency[term]
                inverse_document_frequency = math.log(
                    1.0 + (total_documents - document_frequency + 0.5) / (document_frequency + 0.5)
                )
                normalization = frequency + self.k1 * (
                    1.0 - self.b
                    + self.b * length / (self._average_length or 1.0)
                )
                score += inverse_document_frequency * frequency * (self.k1 + 1.0) / normalization
            if score > 0.0:
                scored.append(TextSearchHit(document=document, score=score))
        scored.sort(key=lambda hit: (-hit.score, hit.document.document_id))
        return scored[:top_k]


def text_backend_provenance(cache: ImageSearchCache, hits: Iterable[TextSearchHit]) -> Dict[str, object]:
    selected = list(hits)
    return {
        "backend": "deterministic_bm25_over_fvqa_cache_titles",
        "cache_source": CACHE_SOURCE,
        "cache_version": CACHE_VERSION,
        "cache_label": cache.label,
        "cache_file_sha256": cache.file_sha256,
        "document_ids": [hit.document.document_id for hit in selected],
        "document_raw_result_hashes": [hit.document.raw_result_hash for hit in selected],
        "scores": [round(hit.score, 8) for hit in selected],
        "online_access": False,
        "ground_truth_used_to_build_corpus": False,
    }
