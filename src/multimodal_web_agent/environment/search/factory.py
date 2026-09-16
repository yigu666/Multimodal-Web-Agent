from __future__ import annotations

from pathlib import Path
import hashlib
import time
from typing import Any, Mapping

from multimodal_web_agent.evaluation.unified_agent.environment import (
    FrozenToolEnvironment,
)

from .base import EpisodeContext
from .frozen import FrozenTextSearchBackend, FrozenVisualSearchBackend
from .online.cache import JsonCache
from .online.budget import ApiBudgetGuard
from .online.cost_stats import CostStatistics
from .online.page_reader import JinaPageReader
from .online.page_reader_local import FallbackPageReader, LocalPageReader
from .online.provenance import ProvenanceWriter
from .online.text_search_serper import (
    SERPER_BACKEND_VERSION,
    SerperClient,
    SerperTextSearchBackend,
)
from .online.text_search_searxng import (
    SEARXNG_BACKEND_VERSION,
    SearXNGClient,
    SearXNGTextSearchBackend,
)
from .online.visual_search_google import (
    GOOGLE_VISUAL_BACKEND_VERSION,
    GoogleVisionClient,
    GoogleVisualSearchBackend,
)
from .online.visual_search_serpapi_lens import (
    SERPAPI_LENS_BACKEND_VERSION,
    SERPAPI_VISUAL_CACHE_NAMESPACE,
    SerpApiGoogleLensClient,
    SerpApiGoogleLensVisualSearchBackend,
)
from .replay import ReplayTextSearchBackend, ReplayVisualSearchBackend
from .schemas import SearchBackendError, SearchResult


class SearchToolEnvironment:
    """Runner-compatible environment with unified Frozen/Live/Replay results."""

    def __init__(
        self,
        *,
        mode: str,
        text_backend,
        visual_backend,
        provenance: ProvenanceWriter | None = None,
        statistics: CostStatistics | None = None,
        budget: ApiBudgetGuard | None = None,
    ):
        if mode not in {"frozen", "live", "replay"}:
            raise ValueError("backend_mode must be frozen, live, or replay")
        self.mode = mode
        self.text_backend = text_backend
        self.visual_backend = visual_backend
        self.provenance = provenance
        self.statistics = statistics or CostStatistics()
        self.budget = budget
        self.context = EpisodeContext(episode_id="unbound")
        self.events: list[dict[str, Any]] = []
        self.last_result: SearchResult | None = None

    def begin_episode(self, example, image: Any) -> None:
        self.context = EpisodeContext(
            episode_id=str(example.eval_id),
            image_sha256=str(example.image_sha256),
            image=image,
            metadata={
                "task_type": str(example.task_type),
                "source_dataset": str(example.source_dataset),
            },
        )
        self.events = []
        self.last_result = None

    def _execute(self, tool: str, request: Mapping[str, Any], callback) -> str:
        started = time.perf_counter()
        before_cost = self.statistics.snapshot()
        self.statistics.increment("text_tool_calls" if tool == "text_search" else "visual_tool_calls")
        event: dict[str, Any] = {
            "tool": tool,
            "backend_mode": self.mode,
            "request": dict(request),
            "status": "success",
            "cache_hit": self.mode == "replay",
        }
        try:
            result = callback()
            self.last_result = result
            backend_object = self.text_backend if tool == "text_search" else self.visual_backend
            event["cache_hit"] = self.mode == "replay" or bool(
                getattr(backend_object, "last_cache_hit", False)
            )
            event.update({
                "backend": result.backend,
                "record_count": len(result.records),
                "provider_status": "success",
                "urls": [record.url for record in result.records if record.url],
                "provider_latency_seconds": float(result.metadata.get("latency_seconds", 0.0)),
                "information_sha256": hashlib.sha256(
                    result.information_text.encode("utf-8")
                ).hexdigest(),
                "page_fetch_success_count": int(result.metadata.get("page_fetch_success_count", 0)),
                "page_fetch_failure_count": int(result.metadata.get("page_fetch_failure_count", 0)),
                "retry_count": int(result.metadata.get("retry_count", 0)),
                "raw_mcp_response_sha256": result.metadata.get("raw_mcp_response_sha256"),
                "raw_mcp_response_path": result.metadata.get("raw_mcp_response_path"),
                "request_hash": result.metadata.get("request_hash"),
                "response_hash": result.metadata.get("response_hash"),
            })
            if self.provenance is not None:
                path = self.provenance.write(self.context.episode_id, result, event)
                event["provenance_path"] = str(path) if path is not None else None
            return result.information_text
        except SearchBackendError as exc:
            event.update({
                "status": exc.code,
                "provider_status": exc.code,
                "error_metadata": exc.metadata,
            })
            raise
        finally:
            after_cost = self.statistics.snapshot()
            cost_delta = CostStatistics.delta(before_cost, after_cost)
            event["cost_delta"] = cost_delta
            event["remote_provider_calls"] = CostStatistics.remote_total(cost_delta)
            event["latency_seconds"] = time.perf_counter() - started
            self.events.append(event)

    def image_search(self, image_sha256: str) -> str:
        if self.context.episode_id == "unbound":
            self.context = EpisodeContext(
                episode_id="direct", image_sha256=str(image_sha256), image=image_sha256
            )
        return self._execute(
            "visual_search",
            {"image_sha256": str(image_sha256)},
            lambda: self.visual_backend.search(self.context.image, self.context),
        )

    def text_search(self, query: str) -> str:
        return self._execute(
            "text_search",
            {"query": str(query)},
            lambda: self.text_backend.search(str(query), self.context),
        )

    def episode_log(self) -> list[dict[str, Any]]:
        return [dict(event) for event in self.events]

    def cost_statistics(self) -> dict[str, int]:
        return self.statistics.snapshot()

    def budget_status(self) -> dict[str, Any] | None:
        return self.budget.snapshot() if self.budget is not None else None


def _search_parameters(config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    text = config["text_search"]
    visual = config["visual_search"]
    text_parameters = {
        "search_top_k": int(text["provider_top_k"]),
        "result_top_k": int(text["top_k"]),
        "record_max_chars": int(text["record_max_chars"]),
        "chunk_words": int(text["chunk_words"]),
        "overlap_words": int(text["chunk_overlap_words"]),
        "locale": str(text.get("locale", "en-US")),
    }
    effective_visual_provider = str(visual.get("cache_provider", visual.get("provider", "")))
    if effective_visual_provider == "serpapi_google_lens":
        visual_parameters = {
            "auto_crop": bool(visual.get("auto_crop", False)),
            "locale": str(visual.get("locale", "en-US")),
            "country": str(visual.get("country", "us")),
            "safe": str(visual.get("safe", "active")),
            "visual_match_top_k": int(visual.get("visual_match_top_k", visual["top_k"])),
            "related_content_top_k": int(visual.get("related_content_top_k", 2)),
            "result_top_k": int(visual["top_k"]),
            "record_max_chars": int(visual["record_max_chars"]),
            "read_matching_pages": int(visual["read_matching_pages"]),
            "max_upload_bytes": int(visual.get("max_upload_bytes", 500000)),
            "image_preparation_version": "serpapi-image-prep-v1",
            "region_aware": False,
            "retrieved_images_injected": False,
        }
    else:
        visual_parameters = {
            "entity_top_k": int(visual["entity_top_k"]),
            "matching_page_top_k": int(visual["matching_page_top_k"]),
            "result_top_k": int(visual["top_k"]),
            "record_max_chars": int(visual["record_max_chars"]),
            "read_matching_pages": int(visual["read_matching_pages"]),
            "region_aware": False,
            "retrieved_images_injected": False,
        }
    return text_parameters, visual_parameters


def build_search_environment(
    *,
    project_root: Path,
    config: Mapping[str, Any],
    serper_client=None,
    vision_client=None,
    serpapi_client=None,
    page_reader=None,
) -> SearchToolEnvironment:
    root = Path(project_root)
    mode = str(config["backend_mode"])
    if mode == "frozen":
        legacy = FrozenToolEnvironment(
            root / str(config["environment_dir"]),
            image_search_top_k=int(config["visual_search"]["top_k"]),
            text_search_top_k=int(config["text_search"]["top_k"]),
        )
        return SearchToolEnvironment(
            mode=mode,
            text_backend=FrozenTextSearchBackend(legacy),
            visual_backend=FrozenVisualSearchBackend(legacy),
        )

    cache = JsonCache(
        root / str(config["cache"]["root"]),
        enabled=bool(config["cache"].get("enabled", True)),
    )
    provenance = ProvenanceWriter(
        root / str(config["provenance"]["root"]),
        enabled=bool(config["provenance"].get("enabled", True)),
    )
    statistics = CostStatistics()
    budget = ApiBudgetGuard.from_config(root, config)
    text_parameters, visual_parameters = _search_parameters(config)
    if mode == "replay":
        text_provider = str(config["text_search"].get("provider", "serper"))
        replay_visual_provider = str(
            config["visual_search"].get(
                "cache_provider",
                config["visual_search"].get("provider", "serpapi_google_lens"),
            )
        )
        replay_visual_version = (
            SERPAPI_LENS_BACKEND_VERSION
            if replay_visual_provider == "serpapi_google_lens"
            else GOOGLE_VISUAL_BACKEND_VERSION
        )
        replay_namespace = (
            SERPAPI_VISUAL_CACHE_NAMESPACE
            if replay_visual_provider == "serpapi_google_lens"
            else "visual"
        )
        text_version = (
            SERPER_BACKEND_VERSION if text_provider == "serper" else SEARXNG_BACKEND_VERSION
        )
        return SearchToolEnvironment(
            mode=mode,
            text_backend=ReplayTextSearchBackend(
                cache=cache,
                provider=text_provider,
                parameters=text_parameters,
                backend_version=text_version,
                statistics=statistics,
            ),
            visual_backend=ReplayVisualSearchBackend(
                cache=cache,
                provider=replay_visual_provider,
                parameters=visual_parameters,
                backend_version=replay_visual_version,
                statistics=statistics,
                namespace=replay_namespace,
            ),
            provenance=provenance,
            statistics=statistics,
            budget=budget,
        )

    page_config = config["page_reader"]
    if page_reader is not None:
        reader = page_reader
    elif str(page_config.get("primary", page_config.get("provider", "jina"))) == "local":
        local = LocalPageReader(
            cache=cache,
            statistics=statistics,
            timeout_seconds=float(page_config["timeout_seconds"]),
            max_response_bytes=int(page_config["max_response_bytes"]),
            max_redirects=int(page_config["max_redirects"]),
            minimum_usable_chars=int(page_config.get("minimum_usable_chars", 200)),
        )
        fallback_name = str(page_config.get("fallback", "none"))
        fallback = None
        if fallback_name == "jina":
            fallback = JinaPageReader(
                cache=cache,
                timeout_seconds=float(page_config["timeout_seconds"]),
                max_response_bytes=int(page_config["max_response_bytes"]),
                max_redirects=int(page_config["max_redirects"]),
                budget=budget,
                statistics=statistics,
            )
        elif fallback_name != "none":
            raise ValueError("unsupported page reader fallback: %s" % fallback_name)
        reader = FallbackPageReader(
            primary=local, fallback=fallback, cache=cache, statistics=statistics
        )
    else:
        reader = JinaPageReader(
            cache=cache,
            timeout_seconds=float(page_config["timeout_seconds"]),
            max_response_bytes=int(page_config["max_response_bytes"]),
            max_redirects=int(page_config["max_redirects"]),
            budget=budget,
            statistics=statistics,
        )
    text_provider = str(config["text_search"].get("provider", "serper"))
    common_text = dict(
        page_reader=reader,
        cache=cache,
        search_top_k=text_parameters["search_top_k"],
        result_top_k=text_parameters["result_top_k"],
        record_max_chars=text_parameters["record_max_chars"],
        chunk_words=text_parameters["chunk_words"],
        overlap_words=text_parameters["overlap_words"],
        locale=text_parameters["locale"],
        budget=budget,
        statistics=statistics,
    )
    if text_provider == "serper":
        text_backend = SerperTextSearchBackend(
            client=serper_client or SerperClient(
                timeout_seconds=float(config["text_search"]["timeout_seconds"]),
                locale=text_parameters["locale"],
            ),
            **common_text,
        )
    elif text_provider == "searxng":
        text_backend = SearXNGTextSearchBackend(
            client=serper_client or SearXNGClient(
                timeout_seconds=float(config["text_search"]["timeout_seconds"]),
                locale=text_parameters["locale"],
            ),
            **common_text,
        )
    else:
        raise ValueError("unsupported live text provider: %s" % text_provider)
    visual_provider = str(config["visual_search"].get("provider"))
    if visual_provider == "cache_only_or_replay":
        cache_provider = str(config["visual_search"].get("cache_provider", "serpapi_google_lens"))
        visual_backend = ReplayVisualSearchBackend(
            cache=cache,
            provider=cache_provider,
            parameters=visual_parameters,
            backend_version=SERPAPI_LENS_BACKEND_VERSION if cache_provider == "serpapi_google_lens" else GOOGLE_VISUAL_BACKEND_VERSION,
            statistics=statistics,
            namespace=SERPAPI_VISUAL_CACHE_NAMESPACE if cache_provider == "serpapi_google_lens" else "visual",
        )
    elif visual_provider == "serpapi_google_lens":
        visual_backend = SerpApiGoogleLensVisualSearchBackend(
            client=serpapi_client or SerpApiGoogleLensClient(
                timeout_seconds=float(config["visual_search"].get("timeout_seconds", 30))
            ),
            page_reader=reader,
            cache=cache,
            budget=budget,
            statistics=statistics,
            auto_crop=visual_parameters["auto_crop"],
            locale=visual_parameters["locale"],
            country=visual_parameters["country"],
            safe=visual_parameters["safe"],
            visual_match_top_k=visual_parameters["visual_match_top_k"],
            related_content_top_k=visual_parameters["related_content_top_k"],
            result_top_k=visual_parameters["result_top_k"],
            record_max_chars=visual_parameters["record_max_chars"],
            read_matching_pages=visual_parameters["read_matching_pages"],
            max_upload_bytes=visual_parameters["max_upload_bytes"],
        )
    elif visual_provider == "google_vision_web_detection":
        visual_backend = GoogleVisualSearchBackend(
            client=vision_client or GoogleVisionClient(),
            page_reader=reader,
            cache=cache,
            entity_top_k=visual_parameters["entity_top_k"],
            matching_page_top_k=visual_parameters["matching_page_top_k"],
            result_top_k=visual_parameters["result_top_k"],
            record_max_chars=visual_parameters["record_max_chars"],
            read_matching_pages=visual_parameters["read_matching_pages"],
            budget=budget,
            statistics=statistics,
        )
    else:
        raise ValueError("unsupported live visual provider: %s" % visual_provider)
    return SearchToolEnvironment(
        mode=mode,
        text_backend=text_backend,
        visual_backend=visual_backend,
        provenance=provenance,
        statistics=statistics,
        budget=budget,
    )
