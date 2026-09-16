from __future__ import annotations

import string
from html import unescape
from typing import Iterable, Sequence

from .answer_normalizer import normalize_answer


def information_supports_answer(information_text: str, accepted_answers: Sequence[str]) -> bool:
    def evidence_normalize(value: str) -> str:
        # Information blocks are HTML-escaped by the formatter.  Decode them
        # before punctuation normalization so an entity such as "A & B" does
        # not become the three-token string "A amp B" during evidence checks.
        decoded = unescape(value)
        spaced = "".join(
            " " if character in string.punctuation else character
            for character in decoded
        )
        return normalize_answer(spaced)

    information_normalized = evidence_normalize(information_text)
    normalized_information = " %s " % information_normalized
    compact_information = information_normalized.replace(" ", "")
    for answer in accepted_answers:
        normalized = evidence_normalize(answer)
        if not normalized:
            continue
        if " %s " % normalized in normalized_information:
            return True
        if any(ord(character) > 127 for character in normalized) and normalized in compact_information:
            return True
    return False


def identified_entity_candidates(titles: Iterable[str], max_tokens: int = 14) -> Iterable[str]:
    """Yield deterministic title-derived entity hints; no answer is accepted here."""

    seen = set()
    for title in titles:
        cleaned = " ".join(str(title).replace("|", " ").split())
        tokens = cleaned.split()
        for width in (max_tokens, 10, 6):
            candidate = " ".join(tokens[:width]).strip(" -:;,.")
            key = candidate.casefold()
            if candidate and key not in seen:
                seen.add(key)
                yield candidate
