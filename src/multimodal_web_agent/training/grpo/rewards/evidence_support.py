from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import html
import re
import unicodedata
from typing import Iterable

from multimodal_web_agent.evaluation.unified_agent.answer_metrics import normalize_answer
from multimodal_web_agent.evaluation.unified_agent.audit_v1_2.answer_equivalence import (
    deterministic_alias_equivalence,
    evaluate_answer_v2,
    extract_quantities,
)


MATCHER_VERSION = "grounded-evidence-support-v2"
_WORD_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)?", re.UNICODE)
_INFORMATION_RE = re.compile(r"<information>(.*?)</information>", re.I | re.S)
_ARTICLES = {"a", "an", "the"}
_NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16",
    "seventeen": "17", "eighteen": "18", "nineteen": "19", "twenty": "20",
}
_QUANTITY_SPAN_RE = re.compile(
    r"(?:[$€£]\s*)?(?:[-+]?\d[\d,]*(?:\.\d+)?|"
    r"(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|"
    r"twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|"
    r"nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|"
    r"hundred|thousand|million|billion)(?:[\s-]+(?:and[\s-]+)?"
    r"(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|hundred|"
    r"thousand|million|billion)){0,4})"
    r"(?:\s*(?:-|–|—|to|:|/)?\s*[-+]?\d+(?:\.\d+)?)?"
    r"(?:\s*(?:thousand|million|billion|k))?"
    r"(?:\s*(?:kilometers?|kilometres?|kilograms?|kilowatts?|"
    r"centur(?:y|ies)|percentage|fahrenheit|celsius|millimeters?|"
    r"millimetres?|centimeters?|centimetres?|miles?|minutes?|months?|"
    r"seconds?|inches?|pounds?|dollars?|percent|meters?|metres?|years?|"
    r"weeks?|hours?|grams?|watts?|euros?|feet|foot|days?|kelvin|"
    r"mm|cm|km|kg|kw|ft|mi|lb|lbs|usd|eur|gbp|°c|°f|m|g|w|%))?"
    r"(?!\w)",
    re.IGNORECASE,
)
_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "as", "at", "be", "because", "been", "before",
    "being", "below", "between", "both", "but", "by", "can", "could", "did",
    "do", "does", "doing", "down", "during", "each", "few", "for", "from",
    "further", "had", "has", "have", "having", "he", "her", "here", "hers",
    "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is",
    "it", "its", "itself", "just", "me", "more", "most", "my", "myself", "no",
    "nor", "not", "now", "of", "off", "on", "once", "only", "or", "other",
    "our", "ours", "ourselves", "out", "over", "own", "same", "she", "should",
    "so", "some", "such", "than", "that", "the", "their", "theirs", "them",
    "themselves", "then", "there", "these", "they", "this", "those", "through",
    "to", "too", "under", "until", "up", "very", "was", "we", "were", "what",
    "when", "where", "which", "while", "who", "whom", "why", "will", "with",
    "would", "you", "your", "yours", "yourself", "yourselves",
}


@dataclass(frozen=True)
class SupportMatch:
    accepted_answer: str | None
    match_type: str
    matched_span: str | None
    support_score: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class _Token:
    normalized: str
    raw: str
    start: int
    end: int


def information_payload(text: str) -> str:
    matches = _INFORMATION_RE.findall(str(text or ""))
    return "\n".join(html.unescape(value) for value in matches) if matches else str(text or "")


def _tokens(text: str) -> list[_Token]:
    result = []
    value = unicodedata.normalize("NFKC", information_payload(text))
    for match in _WORD_RE.finditer(value):
        normalized = unicodedata.normalize("NFKC", match.group(0)).casefold()
        if normalized in _ARTICLES:
            continue
        normalized = _NUMBER_WORDS.get(normalized, normalized)
        result.append(_Token(normalized, match.group(0), match.start(), match.end()))
    return result


def normalized_evidence_tokens(text: str) -> tuple[str, ...]:
    return tuple(token.normalized for token in _tokens(text))


def deterministic_quantity_signatures(text: str) -> tuple[tuple[str, str], ...]:
    signatures = set()
    for match in _QUANTITY_SPAN_RE.finditer(information_payload(text)):
        for quantity in extract_quantities(match.group(0)):
            dimension = str(quantity.dimension or "number")
            value = quantity.canonical_value if quantity.dimension else quantity.value
            signatures.add((dimension, format(float(value), ".9g")))
    return tuple(sorted(signatures))


def _span(text: str, tokens: list[_Token], start: int, end: int) -> str:
    payload = information_payload(text)
    return payload[tokens[start].start:tokens[end - 1].end]


def _singular(token: str) -> str:
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("s") and len(token) > 3 and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def _full_match(alias: str, information_text: str, *, question: str) -> SupportMatch | None:
    info_tokens = _tokens(information_text)
    alias_tokens = normalize_answer(alias).split()
    if not info_tokens or not alias_tokens:
        return None
    normalized_info = [token.normalized for token in info_tokens]
    width = len(alias_tokens)
    for start in range(0, len(info_tokens) - width + 1):
        if normalized_info[start:start + width] == alias_tokens:
            return SupportMatch(alias, "normalized_exact", _span(information_text, info_tokens, start, start + width), 1.0)
    singular_alias = [_singular(token) for token in alias_tokens]
    singular_info = [_singular(token) for token in normalized_info]
    for start in range(0, len(info_tokens) - width + 1):
        if singular_info[start:start + width] == singular_alias:
            return SupportMatch(alias, "alias_equivalent", _span(information_text, info_tokens, start, start + width), 1.0)
    # Title/person aliases may change token width; only those aliases need the
    # more expensive bounded deterministic-title scan.
    alias_normalized = normalize_answer(alias)
    title_like = alias_normalized.startswith((
        "grand duke ", "grand duchess ", "duke ", "duchess ", "king ",
        "queen ", "prince ", "princess ", "emperor ", "empress ",
        "president ", "saint ", "sir ", "dame ", "dr ", "doctor ",
        "professor ",
    )) or " of " in alias_normalized
    if title_like:
        minimum = max(1, width - 4)
        maximum = min(len(info_tokens), width + 4, 16)
        for window in range(minimum, maximum + 1):
            for start in range(0, len(info_tokens) - window + 1):
                candidate = _span(information_text, info_tokens, start, start + window)
                if deterministic_alias_equivalence(candidate, alias):
                    return SupportMatch(alias, "alias_equivalent", candidate, 1.0)
    # Numeric and unit equivalence only needs to inspect quantity-like spans.
    # The same tested evaluator remains the authority for accepting a match.
    payload = information_payload(information_text)
    for quantity in _QUANTITY_SPAN_RE.finditer(payload):
        candidate = quantity.group(0)
        result = evaluate_answer_v2(candidate, (alias,), question=question,
                                    count_semantic_as_strict=False)
        reasons = set(result.equivalence_reasons)
        if "unit_equivalence" in reasons:
            return SupportMatch(alias, "unit_equivalent", candidate, 1.0)
        if "numeric_equivalence" in reasons:
            return SupportMatch(alias, "numeric_equivalent", candidate, 1.0)
    return None


def support_match(
    accepted_answers: Iterable[str],
    information_text: str,
    *,
    question: str = "",
) -> SupportMatch:
    aliases = tuple(str(value) for value in accepted_answers if str(value).strip())
    for alias in aliases:
        full = _full_match(alias, information_text, question=question)
        if full is not None:
            return full
    info_tokens = _tokens(information_text)
    info_counter = Counter(
        token.normalized for token in info_tokens if token.normalized not in _STOPWORDS
    )
    best = SupportMatch(None, "none", None, 0.0)
    for alias in aliases:
        all_alias_tokens = normalize_answer(alias).split()
        core = [token for token in all_alias_tokens if token not in _STOPWORDS]
        # Single-token answers and answers reduced to one core token never receive
        # fuzzy support; this blocks common-word false positives.
        if len(all_alias_tokens) <= 1 or len(core) <= 1:
            continue
        overlap = Counter(core) & info_counter
        score = sum(overlap.values()) / len(core)
        if score <= best.support_score:
            continue
        matched = {token for token, count in overlap.items() if count}
        selected = [token for token in info_tokens if token.normalized in matched]
        matched_span = None
        if selected:
            payload = information_payload(information_text)
            matched_span = payload[selected[0].start:selected[-1].end]
        best = SupportMatch(alias, "core_token_recall", matched_span, float(score))
    return best


def support(
    accepted_answers: Iterable[str], information_text: str, *, question: str = ""
) -> float:
    return support_match(accepted_answers, information_text, question=question).support_score


def support_prediction(prediction: str, information_text: str) -> SupportMatch:
    if not str(prediction or "").strip():
        return SupportMatch(None, "none", None, 0.0)
    return support_match((prediction,), information_text)
