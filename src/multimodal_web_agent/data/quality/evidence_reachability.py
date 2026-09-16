from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from .generic_terms import normalize_text


def alias_in_text(aliases: Sequence[str], text: str) -> bool:
    normalized = " %s " % normalize_text(text)
    return any(
        alias.strip()
        and (" %s " % normalize_text(alias)) in normalized
        for alias in aliases
    )


def result_text(results: Iterable[Any] | None) -> str:
    values = []
    for result in results or ():
        if isinstance(result, Mapping):
            values.extend(
                str(result.get(key, ""))
                for key in ("title", "text", "content")
            )
        else:
            values.append(str(result))
    return "\n".join(values)


def evidence_document_ids(results: Iterable[Any] | None) -> tuple[str, ...]:
    ids = []
    for result in results or ():
        if isinstance(result, Mapping):
            value = result.get("document_id", result.get("id", ""))
            if value:
                ids.append(str(value))
    return tuple(ids)


def answer_evidence_reachable(
    aliases: Sequence[str],
    results: Iterable[Any] | None,
) -> bool:
    return alias_in_text(aliases, result_text(results))
