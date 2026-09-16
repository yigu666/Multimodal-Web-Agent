from __future__ import annotations

from typing import Any, Mapping

from ..base import EpisodeContext, TextSearchBackend, VisualSearchBackend
from ..online.cache import JsonCache, text_cache_key, visual_cache_key
from ..online.cost_stats import CostStatistics
from ..schemas import SearchBackendError, SearchResult


class ReplayTextSearchBackend(TextSearchBackend):
    def __init__(
        self,
        *,
        cache: JsonCache,
        provider: str,
        parameters: Mapping[str, Any],
        backend_version: str,
        statistics: CostStatistics | None = None,
    ):
        self.cache = cache
        self.provider = str(provider)
        self.parameters = dict(parameters)
        self.backend_version = str(backend_version)
        self.statistics = statistics

    def search(self, query: str, episode_context: EpisodeContext) -> SearchResult:
        key = text_cache_key(
            self.provider, query, self.parameters, self.backend_version
        )
        result = self.cache.get_search_result("text", key)
        if result is None:
            raise SearchBackendError(
                "REPLAY_CACHE_MISS", "no cached result for the exact text request"
            )
        if self.statistics is not None:
            self.statistics.increment("replay_hits")
        return result


class ReplayVisualSearchBackend(VisualSearchBackend):
    def __init__(
        self,
        *,
        cache: JsonCache,
        provider: str,
        parameters: Mapping[str, Any],
        backend_version: str,
        statistics: CostStatistics | None = None,
        namespace: str = "visual",
    ):
        self.cache = cache
        self.provider = str(provider)
        self.parameters = dict(parameters)
        self.backend_version = str(backend_version)
        self.statistics = statistics
        self.namespace = str(namespace)

    def search(self, image: Any, episode_context: EpisodeContext) -> SearchResult:
        image_sha256 = episode_context.image_sha256 or str(image)
        key = visual_cache_key(
            self.provider,
            image_sha256,
            self.parameters,
            self.backend_version,
        )
        result = self.cache.get_search_result(self.namespace, key)
        if result is None:
            raise SearchBackendError(
                "REPLAY_CACHE_MISS", "no cached result for the exact visual request"
            )
        if self.statistics is not None:
            self.statistics.increment("replay_hits")
        return result
