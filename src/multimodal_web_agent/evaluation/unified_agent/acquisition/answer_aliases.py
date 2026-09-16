from __future__ import annotations

import json
import re
import unicodedata
from typing import Any, Mapping, Sequence


ANSWER_FIELDS = (
    "answer_aliases",
    "alternative_gt_answers",
    "answers",
    "gt_answer",
    "answer",
    "answer_eval",
    "original_answer",
    "entity_aliases",
)


def normalize_alias(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value))
    return re.sub(r"\s+", " ", text).strip()


def extract_answer_aliases(
    row: Mapping[str, Any],
    fields: Sequence[str] = ANSWER_FIELDS,
) -> tuple[str, ...]:
    values = []
    for field in fields:
        value = row.get(field)
        if value is None or value == "":
            continue
        if isinstance(value, str):
            try:
                decoded = json.loads(value)
            except json.JSONDecodeError:
                decoded = value
            value = decoded
        if isinstance(value, Mapping):
            value = (
                value.get("aliases")
                or value.get("answers")
                or value.get("text")
                or ()
            )
        if isinstance(value, (list, tuple, set)):
            values.extend(value)
        else:
            values.append(value)
    result = []
    seen = set()
    for value in values:
        alias = normalize_alias(value)
        key = alias.casefold()
        if alias and key not in seen:
            seen.add(key)
            result.append(alias)
    return tuple(result)
