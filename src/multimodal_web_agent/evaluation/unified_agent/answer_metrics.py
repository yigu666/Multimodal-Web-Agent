from __future__ import annotations

from collections import Counter
from datetime import datetime
import re
import string
import unicodedata
from typing import Iterable


_ARTICLES = {"a", "an", "the"}
_NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16",
    "seventeen": "17", "eighteen": "18", "nineteen": "19",
    "twenty": "20",
}
_DATE_FORMATS = (
    "%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%d/%m/%Y",
    "%B %d %Y", "%b %d %Y", "%d %B %Y", "%d %b %Y",
)


def _normalize_date(text: str) -> str:
    candidate = " ".join(text.replace(",", " ").split())
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(candidate, fmt).strftime("%Y %m %d")
        except ValueError:
            continue
    return text


def normalize_answer(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value)).casefold().strip()
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    text = _normalize_date(text)
    punctuation = string.punctuation + "“”‘’–—…"
    text = "".join(" " if char in punctuation else char for char in text)
    tokens = []
    for token in text.split():
        if token in _ARTICLES:
            continue
        tokens.append(_NUMBER_WORDS.get(token, token))
    return " ".join(tokens)


def normalized_exact_match(
    prediction: str | None,
    answer_aliases: Iterable[str],
) -> int:
    if prediction is None:
        return 0
    normalized = normalize_answer(prediction)
    if not normalized:
        return 0
    return int(any(
        normalized == normalize_answer(alias) for alias in answer_aliases
    ))


def _token_f1(prediction: str, target: str) -> float:
    predicted = normalize_answer(prediction).split()
    gold = normalize_answer(target).split()
    if not predicted or not gold:
        return float(predicted == gold and bool(predicted))
    common = Counter(predicted) & Counter(gold)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(gold)
    return 2 * precision * recall / (precision + recall)


def maximum_alias_token_f1(
    prediction: str | None,
    answer_aliases: Iterable[str],
) -> float:
    if prediction is None:
        return 0.0
    aliases = tuple(answer_aliases)
    return max((_token_f1(prediction, alias) for alias in aliases), default=0.0)


def answer_reachable(
    answer_aliases: tuple[str, ...],
    frozen_environment_records: list[str],
) -> bool:
    records = [normalize_answer(record) for record in frozen_environment_records]
    for alias in answer_aliases:
        normalized = normalize_answer(alias)
        if normalized and any(normalized in record for record in records):
            return True
    return False
