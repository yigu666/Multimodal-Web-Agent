from __future__ import annotations

import re
import unicodedata
from html import unescape
from typing import Iterable


TOKEN_RE = re.compile(r"[^\W_]+(?:['’-][^\W_]+)*", re.UNICODE)

QUESTION_WORDS = {
    "what", "which", "who", "whom", "whose", "where", "when", "why",
    "how", "is", "are", "was", "were", "do", "does", "did", "can",
    "could", "would", "should", "has", "have", "had",
}

STOP_WORDS = QUESTION_WORDS | {
    "a", "an", "the", "of", "to", "in", "on", "at", "for", "from",
    "with", "and", "or", "by", "as", "it", "its", "be", "been", "being",
    "after",
}

VISUAL_REFERENCE_TERMS = {
    "image", "photo", "picture", "shown", "depicted", "visible", "this",
    "that", "these", "those", "object", "person", "character", "building",
    "logo", "emblem", "event", "name", "type", "located", "place", "thing",
    "model", "product", "structure", "street", "bridge", "tower",
    "government",
}

VISUAL_CATEGORY_TERMS = {
    "animal", "bird", "breed", "car", "company", "food", "fruit", "goat",
    "item", "man", "monument", "plant", "root", "symbol", "vegetable",
    "vehicle", "woman",
}

GENERIC_QUERY_TERMS = STOP_WORDS | VISUAL_REFERENCE_TERMS | {
    "answer", "identify", "identification", "find", "tell", "represented",
    "represent", "wearing", "performing", "happening", "officially",
    "named",
} | VISUAL_CATEGORY_TERMS


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", unescape(str(value))).casefold()
    return " ".join(TOKEN_RE.findall(value))


def text_tokens(value: str) -> tuple[str, ...]:
    return tuple(TOKEN_RE.findall(normalize_text(value)))


def content_tokens(
    value: str,
    *,
    excluded: Iterable[str] = (),
) -> tuple[str, ...]:
    blocked = GENERIC_QUERY_TERMS | {token.casefold() for token in excluded}
    return tuple(token for token in text_tokens(value) if token not in blocked)
