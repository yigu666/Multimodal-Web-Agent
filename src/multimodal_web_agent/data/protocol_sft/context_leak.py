from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple


_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "by",
    "do", "does", "for", "from", "had", "has", "have", "he", "her", "here",
    "his", "how", "i", "if", "in", "into", "is", "it", "its", "of", "on",
    "or", "our", "she", "that", "the", "their", "them", "there", "these",
    "they", "this", "those", "to", "was", "we", "were", "what", "when",
    "where", "which", "who", "why", "with", "you", "your",
}


@dataclass(frozen=True)
class ContextLeakResult:
    leaked: bool
    matched_source: Optional[str]
    matched_text: Optional[str]
    overlap_ratio: float
    longest_matching_ngram: int
    reason: Optional[str]


def normalize_context_tokens(text: str) -> List[str]:
    normalized = unicodedata.normalize("NFKC", str(text)).casefold()
    return [
        token
        for token in _TOKEN_RE.findall(normalized)
        if token not in _STOPWORDS
    ]


def _contains_ngram(tokens: Sequence[str], ngram: Tuple[str, ...]) -> bool:
    size = len(ngram)
    return any(tuple(tokens[index:index + size]) == ngram for index in range(len(tokens) - size + 1))


def _longest_unavailable_ngram(
    query_tokens: Sequence[str],
    unavailable_tokens: Sequence[str],
    visible_sequences: Sequence[Sequence[str]],
) -> int:
    maximum = min(len(query_tokens), len(unavailable_tokens))
    unavailable_ngrams = {
        size: {
            tuple(unavailable_tokens[index:index + size])
            for index in range(len(unavailable_tokens) - size + 1)
        }
        for size in range(1, maximum + 1)
    }
    for size in range(maximum, 0, -1):
        for index in range(len(query_tokens) - size + 1):
            ngram = tuple(query_tokens[index:index + size])
            if ngram not in unavailable_ngrams[size]:
                continue
            if any(_contains_ngram(visible, ngram) for visible in visible_sequences):
                continue
            return size
    return 0


def detect_unavailable_context_leak(
    query: str,
    visible_texts: Sequence[str],
    unavailable_texts: Sequence[Tuple[str, str]],
    *,
    min_ngram: int = 4,
    overlap_threshold: float = 0.5,
) -> ContextLeakResult:
    if min_ngram < 1:
        raise ValueError("min_ngram must be positive")
    if not 0.0 <= overlap_threshold <= 1.0:
        raise ValueError("overlap_threshold must be between zero and one")

    query_tokens = normalize_context_tokens(query)
    visible_sequences = [normalize_context_tokens(text) for text in visible_texts]
    visible_token_set = {token for sequence in visible_sequences for token in sequence}
    if not query_tokens:
        return ContextLeakResult(False, None, None, 0.0, 0, None)

    best = ContextLeakResult(False, None, None, 0.0, 0, None)
    for source, unavailable_text in unavailable_texts:
        unavailable_tokens = normalize_context_tokens(unavailable_text)
        if not unavailable_tokens:
            continue
        longest = _longest_unavailable_ngram(
            query_tokens, unavailable_tokens, visible_sequences
        )
        novel_query = [token for token in query_tokens if token not in visible_token_set]
        overlap_count = sum(
            (Counter(novel_query) & Counter(unavailable_tokens)).values()
        )
        overlap_ratio = overlap_count / len(query_tokens)
        leaked = longest >= min_ngram or overlap_ratio > overlap_threshold
        reason = None
        if longest >= min_ngram:
            reason = "unavailable_contiguous_ngram"
        elif overlap_ratio > overlap_threshold:
            reason = "unavailable_token_overlap"
        candidate = ContextLeakResult(
            leaked,
            source if leaked else None,
            unavailable_text if leaked else None,
            round(overlap_ratio, 6),
            longest,
            reason,
        )
        if leaked:
            return candidate
        if (candidate.longest_matching_ngram, candidate.overlap_ratio) > (
            best.longest_matching_ngram,
            best.overlap_ratio,
        ):
            best = candidate
    return best
