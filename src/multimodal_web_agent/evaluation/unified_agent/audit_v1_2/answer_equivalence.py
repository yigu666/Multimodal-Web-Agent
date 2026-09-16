from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import math
import re
import unicodedata
from typing import Iterable, Sequence

from multimodal_web_agent.evaluation.unified_agent.answer_metrics import (
    maximum_alias_token_f1,
    normalize_answer,
    normalized_exact_match,
)

from .schema import ANSWER_EVALUATOR_SCHEMA


_NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
_MULTIPLIERS = {"hundred": 100, "thousand": 1_000, "million": 1_000_000}
_NUMERIC_QUESTION = re.compile(
    r"\b(when|year|date|how many|how much|number|percent|percentage|"
    r"length|distance|height|wide|width|long|weight|mass|power|"
    r"temperature|duration|centur(?:y|ies)|age|old|cost|price)\b",
    re.IGNORECASE,
)
_CLOCK_QUESTION = re.compile(
    r"\b(when|what time|close|closes|closed|open|opens|opened)\b",
    re.IGNORECASE,
)
_CLOCK_RE = re.compile(
    r"\b(?P<hour>\d{1,2})(?::(?P<minute>\d{2}))?\s*"
    r"(?P<ampm>a\.?m\.?|p\.?m\.?)?(?!\w)",
    re.IGNORECASE,
)
_RANGE_RE = re.compile(
    r"(?P<left>[-+]?\d+(?:\.\d+)?)\s*(?:-|–|—|to)\s*"
    r"(?P<right>[-+]?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
_CURRENCY_RE = re.compile(
    r"(?:(?P<prefix>usd|eur|gbp|\$|€|£)\s*)?"
    r"(?P<number>[-+]?\d+(?:\.\d+)?)\s*"
    r"(?P<scale>thousand|million|billion|k)?\s*"
    r"(?P<suffix>usd|dollars?|eur|euros?|gbp|pounds?)?",
    re.IGNORECASE,
)
_QUANTITY_RE = re.compile(
    r"(?P<number>[-+]?\d+(?:\.\d+)?)\s*"
    r"(?P<scale>thousand|million|billion|k)?\s*"
    r"(?P<unit>millimet(?:er|re)s?|mm|centimet(?:er|re)s?|cm|"
    r"kilomet(?:er|re)s?|km|met(?:er|re)s?|m|inches|inch|in|"
    r"feet|foot|ft|miles?|mi|grams?|g|kilograms?|kg|pounds?|lb|lbs|"
    r"watts?|w|kilowatts?|kw|seconds?|secs?|minutes?|mins?|hours?|hrs?|"
    r"days?|weeks?|months?|years?|centur(?:y|ies)|%|percent|percentage|"
    r"celsius|fahrenheit|kelvin|°c|°f|usd|dollars?|eur|euros?|gbp|pounds?)?"
    r"(?!\w)",
    re.IGNORECASE,
)
_TITLE_PREFIXES = (
    "grand duke", "grand duchess", "duke", "duchess", "king", "queen",
    "prince", "princess", "emperor", "empress", "president", "saint",
    "sir", "dame", "dr", "doctor", "professor",
)


@dataclass(frozen=True)
class Quantity:
    value: float
    dimension: str | None
    canonical_value: float
    unit: str | None


@dataclass(frozen=True)
class AnswerEvaluationV2:
    schema_version: str
    em_v1: int
    token_f1_v1: float
    em_v2_strict: int
    token_f1_v2: float
    semantic_equivalence_v2: bool
    numeric_equivalence_v2: bool
    unit_equivalence_v2: bool
    alias_equivalence_v2: bool
    strict_semantic_enabled: bool
    unit_relative_error: float | None
    matched_alias: str | None
    equivalence_reasons: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


def _plain(value: str | None) -> str:
    return " ".join(
        unicodedata.normalize("NFKC", str(value or "")).casefold().split()
    )


def _words_to_numbers(value: str) -> str:
    tokens = re.findall(r"[a-z]+|\d+(?:\.\d+)?|[^\w\s]", _plain(value))
    output: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token not in _NUMBER_WORDS and token not in _MULTIPLIERS:
            output.append(token)
            index += 1
            continue
        total = 0
        current = 0
        consumed = False
        while index < len(tokens):
            word = tokens[index]
            if word in _NUMBER_WORDS:
                current += _NUMBER_WORDS[word]
            elif word == "hundred":
                current = max(current, 1) * 100
            elif word in {"thousand", "million"}:
                total += max(current, 1) * _MULTIPLIERS[word]
                current = 0
            elif word == "and":
                index += 1
                continue
            else:
                break
            consumed = True
            index += 1
        if consumed:
            output.append(str(total + current))
        else:
            output.append(token)
            index += 1
    return " ".join(output)


def _unit(value: str | None) -> tuple[str | None, float, str | None]:
    raw = _plain(value or "")
    aliases = {
        "mm": ("length", 0.001, "mm"),
        "millimeter": ("length", 0.001, "mm"),
        "millimeters": ("length", 0.001, "mm"),
        "millimetre": ("length", 0.001, "mm"),
        "millimetres": ("length", 0.001, "mm"),
        "cm": ("length", 0.01, "cm"),
        "centimeter": ("length", 0.01, "cm"),
        "centimeters": ("length", 0.01, "cm"),
        "centimetre": ("length", 0.01, "cm"),
        "centimetres": ("length", 0.01, "cm"),
        "m": ("length", 1.0, "m"),
        "meter": ("length", 1.0, "m"),
        "meters": ("length", 1.0, "m"),
        "metre": ("length", 1.0, "m"),
        "metres": ("length", 1.0, "m"),
        "km": ("length", 1000.0, "km"),
        "kilometer": ("length", 1000.0, "km"),
        "kilometers": ("length", 1000.0, "km"),
        "kilometre": ("length", 1000.0, "km"),
        "kilometres": ("length", 1000.0, "km"),
        "in": ("length", 0.0254, "in"),
        "inch": ("length", 0.0254, "in"),
        "inches": ("length", 0.0254, "in"),
        "ft": ("length", 0.3048, "ft"),
        "foot": ("length", 0.3048, "ft"),
        "feet": ("length", 0.3048, "ft"),
        "mi": ("length", 1609.344, "mi"),
        "mile": ("length", 1609.344, "mi"),
        "miles": ("length", 1609.344, "mi"),
        "g": ("mass", 0.001, "g"),
        "gram": ("mass", 0.001, "g"),
        "grams": ("mass", 0.001, "g"),
        "kg": ("mass", 1.0, "kg"),
        "kilogram": ("mass", 1.0, "kg"),
        "kilograms": ("mass", 1.0, "kg"),
        "lb": ("mass", 0.45359237, "lb"),
        "lbs": ("mass", 0.45359237, "lb"),
        "pound": ("mass", 0.45359237, "lb"),
        "pounds": ("mass", 0.45359237, "lb"),
        "w": ("power", 1.0, "w"),
        "watt": ("power", 1.0, "w"),
        "watts": ("power", 1.0, "w"),
        "kw": ("power", 1000.0, "kw"),
        "kilowatt": ("power", 1000.0, "kw"),
        "kilowatts": ("power", 1000.0, "kw"),
        "second": ("duration", 1.0, "second"),
        "seconds": ("duration", 1.0, "second"),
        "sec": ("duration", 1.0, "second"),
        "secs": ("duration", 1.0, "second"),
        "minute": ("duration", 60.0, "minute"),
        "minutes": ("duration", 60.0, "minute"),
        "min": ("duration", 60.0, "minute"),
        "mins": ("duration", 60.0, "minute"),
        "hour": ("duration", 3600.0, "hour"),
        "hours": ("duration", 3600.0, "hour"),
        "hr": ("duration", 3600.0, "hour"),
        "hrs": ("duration", 3600.0, "hour"),
        "day": ("duration", 86400.0, "day"),
        "days": ("duration", 86400.0, "day"),
        "week": ("duration", 604800.0, "week"),
        "weeks": ("duration", 604800.0, "week"),
        "month": ("duration", 2629800.0, "month"),
        "months": ("duration", 2629800.0, "month"),
        "year": ("duration", 31557600.0, "year"),
        "years": ("duration", 31557600.0, "year"),
        "century": ("duration", 3155760000.0, "century"),
        "centuries": ("duration", 3155760000.0, "century"),
        "%": ("percentage", 0.01, "%"),
        "percent": ("percentage", 0.01, "%"),
        "percentage": ("percentage", 0.01, "%"),
        "usd": ("currency:usd", 1.0, "usd"),
        "dollar": ("currency:usd", 1.0, "usd"),
        "dollars": ("currency:usd", 1.0, "usd"),
        "eur": ("currency:eur", 1.0, "eur"),
        "euro": ("currency:eur", 1.0, "eur"),
        "euros": ("currency:eur", 1.0, "eur"),
        "gbp": ("currency:gbp", 1.0, "gbp"),
    }
    if raw in {"celsius", "°c"}:
        return "temperature:c", 1.0, "celsius"
    if raw in {"fahrenheit", "°f"}:
        return "temperature:f", 1.0, "fahrenheit"
    if raw == "kelvin":
        return "temperature:k", 1.0, "kelvin"
    return aliases.get(raw, (None, 1.0, None))


def extract_quantities(value: str | None) -> tuple[Quantity, ...]:
    text = _words_to_numbers(_plain(value or "").replace(",", ""))
    result = []
    for match in _QUANTITY_RE.finditer(text):
        number = float(match.group("number"))
        scale = _plain(match.group("scale") or "")
        if scale in {"thousand", "k"}:
            number *= 1_000
        elif scale == "million":
            number *= 1_000_000
        elif scale == "billion":
            number *= 1_000_000_000
        dimension, factor, unit = _unit(match.group("unit"))
        canonical = number * factor
        if dimension == "temperature:f":
            canonical = (number - 32.0) * 5.0 / 9.0
            dimension = "temperature"
        elif dimension == "temperature:k":
            canonical = number - 273.15
            dimension = "temperature"
        elif dimension == "temperature:c":
            canonical = number
            dimension = "temperature"
        result.append(Quantity(number, dimension, canonical, unit))
    return tuple(result)


def question_requires_numeric_answer(question: str) -> bool:
    return bool(_NUMERIC_QUESTION.search(question or ""))


def _clock_minutes(value: str | None) -> int | None:
    match = _CLOCK_RE.search(_plain(value or ""))
    if not match:
        return None
    hour = int(match.group("hour"))
    minute = int(match.group("minute") or 0)
    ampm = (match.group("ampm") or "").replace(".", "").casefold()
    if minute >= 60 or hour >= 24:
        return None
    if ampm:
        if not 1 <= hour <= 12:
            return None
        hour %= 12
        if ampm == "pm":
            hour += 12
    return hour * 60 + minute


def _numeric_range(value: str | None) -> tuple[float, float] | None:
    text = _words_to_numbers(_plain(value or "").replace(",", ""))
    match = _RANGE_RE.search(text)
    if not match:
        return None
    return float(match.group("left")), float(match.group("right"))


def _currency(value: str | None) -> tuple[str, float] | None:
    text = _plain(value or "").replace(",", "")
    aliases = {
        "$": "usd", "usd": "usd", "dollar": "usd", "dollars": "usd",
        "€": "eur", "eur": "eur", "euro": "eur", "euros": "eur",
        "£": "gbp", "gbp": "gbp", "pound": "gbp", "pounds": "gbp",
    }
    for match in _CURRENCY_RE.finditer(text):
        token = _plain(match.group("prefix") or match.group("suffix") or "")
        if token not in aliases:
            continue
        number = float(match.group("number"))
        scale = _plain(match.group("scale") or "")
        if scale in {"thousand", "k"}:
            number *= 1_000
        elif scale == "million":
            number *= 1_000_000
        elif scale == "billion":
            number *= 1_000_000_000
        return aliases[token], number
    return None


def _close(left: float, right: float, tolerance: float = 0.01) -> tuple[bool, float]:
    denominator = max(abs(left), abs(right), 1e-12)
    error = abs(left - right) / denominator
    if math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-9):
        return True, 0.0
    return error <= tolerance, error


def quantity_equivalence(
    left: str | None,
    right: str | None,
    *,
    question: str = "",
) -> tuple[bool, bool, float | None]:
    if _CLOCK_QUESTION.search(question or ""):
        left_clock = _clock_minutes(left)
        right_clock = _clock_minutes(right)
        if left_clock is not None and right_clock is not None:
            equal = left_clock == right_clock
            return equal, False, 0.0 if equal else None
    left_range = _numeric_range(left)
    right_range = _numeric_range(right)
    if left_range is not None and right_range is not None:
        equal = all(
            _close(left_value, right_value, 1e-9)[0]
            for left_value, right_value in zip(left_range, right_range)
        )
        return equal, False, 0.0 if equal else None
    left_currency = _currency(left)
    right_currency = _currency(right)
    if left_currency is not None and right_currency is not None:
        same_nominal = left_currency[0] == right_currency[0]
        equal, error = _close(left_currency[1], right_currency[1])
        matched = same_nominal and equal
        return False, matched, error if matched else None
    left_values = extract_quantities(left)
    right_values = extract_quantities(right)
    if not left_values or not right_values:
        return False, False, None
    numeric_allowed = question_requires_numeric_answer(question) or bool(
        any(item.dimension for item in left_values + right_values)
    )
    if not numeric_allowed:
        return False, False, None
    numeric = False
    unit = False
    best_error = None
    for l_value in left_values:
        for r_value in right_values:
            if l_value.dimension is None or r_value.dimension is None:
                equal, error = _close(l_value.value, r_value.value, 1e-9)
                numeric = numeric or equal
                if equal:
                    best_error = min(best_error, error) if best_error is not None else error
                continue
            if l_value.dimension != r_value.dimension:
                continue
            equal, error = _close(l_value.canonical_value, r_value.canonical_value)
            unit = unit or equal
            if equal:
                best_error = min(best_error, error) if best_error is not None else error
    return numeric, unit, best_error


def _singular_tokens(value: str) -> tuple[str, ...]:
    output = []
    for token in normalize_answer(value).split():
        if token.endswith("ies") and len(token) > 4:
            token = token[:-3] + "y"
        elif token.endswith("s") and len(token) > 3 and not token.endswith(
            ("ss", "us", "is")
        ):
            token = token[:-1]
        output.append(token)
    return tuple(output)


def _person_core(value: str) -> tuple[str, ...]:
    text = normalize_answer(value)
    for prefix in _TITLE_PREFIXES:
        if text.startswith(prefix + " "):
            text = text[len(prefix):].strip()
            break
    text = re.sub(r"\s+of\s+(?:the\s+)?[a-z][a-z\s-]+$", "", text)
    return tuple(text.split())


def deterministic_alias_equivalence(left: str, right: str) -> bool:
    if not left or not right:
        return False
    if _singular_tokens(left) == _singular_tokens(right):
        return normalize_answer(left) != normalize_answer(right)
    left_core = _person_core(left)
    right_core = _person_core(right)
    return (
        left_core == right_core
        and len(left_core) >= 2
        and normalize_answer(left) != normalize_answer(right)
    )


def deterministic_semantic_equivalence(left: str, right: str) -> bool:
    groups = (
        {"0", "free", "no cost", "zero cost", "without charge"},
    )
    normalized = {normalize_answer(left), normalize_answer(right)}
    return any(normalized <= group for group in groups)


def _token_f1_v2(prediction: str, alias: str) -> float:
    predicted = list(_singular_tokens(_words_to_numbers(prediction)))
    target = list(_singular_tokens(_words_to_numbers(alias)))
    if not predicted or not target:
        return float(predicted == target and bool(predicted))
    overlap = sum((Counter(predicted) & Counter(target)).values())
    if not overlap:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(target)
    return 2 * precision * recall / (precision + recall)


def evaluate_answer_v2(
    prediction: str | None,
    answer_aliases: Iterable[str],
    *,
    question: str = "",
    count_semantic_as_strict: bool = False,
) -> AnswerEvaluationV2:
    aliases = tuple(str(value) for value in answer_aliases)
    value = str(prediction or "")
    em_v1 = normalized_exact_match(prediction, aliases)
    f1_v1 = maximum_alias_token_f1(prediction, aliases)
    numeric = unit = alias_equivalent = semantic = False
    unit_error = None
    matched_alias = None
    reasons = []
    for alias in aliases:
        numeric_match, unit_match, error = quantity_equivalence(
            value, alias, question=question
        )
        alias_match = deterministic_alias_equivalence(value, alias)
        semantic_match = deterministic_semantic_equivalence(value, alias)
        if numeric_match or unit_match or alias_match or semantic_match:
            matched_alias = matched_alias or alias
        numeric = numeric or numeric_match
        unit = unit or unit_match
        alias_equivalent = alias_equivalent or alias_match
        semantic = semantic or semantic_match
        if error is not None:
            unit_error = min(unit_error, error) if unit_error is not None else error
    if em_v1:
        reasons.append("v1_normalized_exact")
    if numeric:
        reasons.append("numeric_equivalence")
    if unit:
        reasons.append("unit_equivalence")
    if alias_equivalent:
        reasons.append("deterministic_alias_equivalence")
    if semantic:
        reasons.append("closed_semantic_equivalence")
    strict = bool(
        em_v1 or numeric or unit or alias_equivalent
        or (count_semantic_as_strict and semantic)
    )
    f1_v2 = max(
        [f1_v1, *(_token_f1_v2(value, alias) for alias in aliases)],
        default=0.0,
    )
    if strict or semantic:
        f1_v2 = 1.0
    return AnswerEvaluationV2(
        schema_version=ANSWER_EVALUATOR_SCHEMA,
        em_v1=em_v1,
        token_f1_v1=f1_v1,
        em_v2_strict=int(strict),
        token_f1_v2=f1_v2,
        semantic_equivalence_v2=semantic,
        numeric_equivalence_v2=numeric,
        unit_equivalence_v2=unit,
        alias_equivalence_v2=alias_equivalent,
        strict_semantic_enabled=count_semantic_as_strict,
        unit_relative_error=unit_error,
        matched_alias=matched_alias,
        equivalence_reasons=tuple(reasons),
    )
