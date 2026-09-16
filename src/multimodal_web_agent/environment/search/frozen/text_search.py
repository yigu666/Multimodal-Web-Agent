from __future__ import annotations

import hashlib

from ..base import EpisodeContext, TextSearchBackend
from ..schemas import SearchRecord, SearchResult


def _records_from_frozen_information(information: str) -> tuple[str, ...]:
    lines = information.splitlines()
    if len(lines) < 4 or lines[0] != "<information>" or lines[-1] != "</information>":
        raise ValueError("unexpected frozen information format")
    records = []
    for index, line in enumerate(lines[2:-1], 1):
        prefix = "%d. " % index
        if not line.startswith(prefix):
            raise ValueError("unexpected frozen record ordering")
        records.append(line[len(prefix):])
    return tuple(records)


class FrozenTextSearchBackend(TextSearchBackend):
    """Thin adapter: the official FrozenToolEnvironment remains authoritative."""

    def __init__(self, legacy_environment):
        self.legacy_environment = legacy_environment

    def search(self, query: str, episode_context: EpisodeContext) -> SearchResult:
        information = self.legacy_environment.text_search(query)
        records = _records_from_frozen_information(information)
        return SearchResult(
            tool_type="text_search",
            backend="frozen_bm25",
            request={"query": str(query)},
            timestamp="",
            records=tuple(
                SearchRecord(
                    rank=index,
                    content=text,
                    source="frozen",
                    content_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                )
                for index, text in enumerate(records, 1)
            ),
            information_text=information,
            metadata={"online_access": False, "legacy_implementation_reused": True},
        )

