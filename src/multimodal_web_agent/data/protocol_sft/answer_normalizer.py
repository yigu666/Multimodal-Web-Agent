from __future__ import annotations

import json
import re
import string
from typing import Any, Iterable, List


def normalize_answer(value: str) -> str:
    text = str(value).casefold()
    text = "".join(character for character in text if character not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def parse_candidate_answers(value: Any) -> List[str]:
    if value is None:
        return []
    parsed = value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
        except (TypeError, ValueError):
            parsed = [stripped]
    if not isinstance(parsed, (list, tuple, set)):
        parsed = [parsed]
    return [str(item).strip() for item in parsed if str(item).strip()]


def accepted_answer_list(canonical: Any, candidates: Any) -> List[str]:
    values: Iterable[Any] = [canonical] + parse_candidate_answers(candidates)
    result: List[str] = []
    seen = set()
    for value in values:
        text = str(value).strip()
        normalized = normalize_answer(text)
        if text and normalized and normalized not in seen:
            seen.add(normalized)
            result.append(text)
    return result
