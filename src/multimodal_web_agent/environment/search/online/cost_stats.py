from __future__ import annotations

from collections import Counter
from threading import Lock
from typing import Mapping


COUNTERS = (
    "text_tool_calls",
    "visual_tool_calls",
    "serper_remote_requests",
    "searxng_remote_requests",
    "vision_remote_requests",
    "serpapi_image_upload_requests",
    "serpapi_lens_search_requests",
    "serpapi_visual_transactions",
    "jina_remote_requests",
    "alibaba_websearch_requests",
    "text_cache_hits",
    "visual_cache_hits",
    "page_cache_hits",
    "replay_hits",
    "local_page_reads",
    "jina_fallback_reads",
)


class CostStatistics:
    """Run-local counters that distinguish agent actions from real remote calls."""

    def __init__(self) -> None:
        self._values: Counter[str] = Counter()
        self._lock = Lock()

    def increment(self, name: str, amount: int = 1) -> None:
        if name not in COUNTERS:
            raise KeyError("unknown cost counter: %s" % name)
        if int(amount) < 0:
            raise ValueError("cost counters cannot decrease")
        with self._lock:
            self._values[name] += int(amount)

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {name: int(self._values[name]) for name in COUNTERS}

    @staticmethod
    def delta(before: Mapping[str, int], after: Mapping[str, int]) -> dict[str, int]:
        return {
            name: int(after.get(name, 0)) - int(before.get(name, 0))
            for name in COUNTERS
        }

    @staticmethod
    def remote_total(values: Mapping[str, int]) -> int:
        return sum(
            int(values.get(name, 0))
            for name in (
                "serper_remote_requests",
                "searxng_remote_requests",
                "vision_remote_requests",
                "serpapi_image_upload_requests",
                "serpapi_lens_search_requests",
                "jina_remote_requests",
                "alibaba_websearch_requests",
            )
        )
