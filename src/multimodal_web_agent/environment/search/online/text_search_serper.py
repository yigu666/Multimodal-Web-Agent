from __future__ import annotations

import hashlib
import json
import os
import socket
import time
from typing import Any, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from multimodal_web_agent.data.protocol_sft.information_formatter import (
    format_frozen_information,
)
from multimodal_web_agent.evaluation.unified_agent.environment import truncate_record

from ..base import EpisodeContext, TextSearchBackend
from ..schemas import SearchBackendError, SearchRecord, SearchResult
from .cache import JsonCache, text_cache_key
from .budget import ApiBudgetGuard
from .chunk_selector import PageChunk, chunk_page, clean_text, select_chunks
from .cost_stats import CostStatistics
from .provenance import utc_now


SERPER_BACKEND_VERSION = "serper-local-jina-bm25-v2"


class SerperClient:
    def __init__(self, *, api_key: str | None = None, timeout_seconds: float = 12.0, locale: str = "en-US"):
        self.api_key = api_key if api_key is not None else os.environ.get("SERPER_API_KEY", "")
        self.timeout_seconds = float(timeout_seconds)
        self.locale = str(locale)

    def validate_ready(self) -> None:
        if not self.api_key:
            raise SearchBackendError("ONLINE_CREDENTIAL_MISSING", "SERPER_API_KEY is not configured")

    def search(self, query: str, *, top_k: int) -> list[dict[str, Any]]:
        self.validate_ready()
        language, _, country = self.locale.partition("-")
        payload = {"q": query, "num": int(top_k), "hl": language or "en"}
        if country:
            payload["gl"] = country.casefold()
        body = json.dumps(payload).encode("utf-8")
        request = Request(
            "https://google.serper.dev/search",
            data=body,
            method="POST",
            headers={"X-API-KEY": self.api_key, "Content-Type": "application/json"},
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, socket.timeout, json.JSONDecodeError) as exc:
            code = "TIMEOUT" if isinstance(exc, (TimeoutError, socket.timeout)) else "SEARCH_API_ERROR"
            raise SearchBackendError(code, "Serper request failed") from exc
        rows = []
        for rank, item in enumerate(raw.get("organic", [])[:top_k], 1):
            rows.append({
                "rank": rank,
                "title": clean_text(str(item.get("title", ""))),
                "url": str(item.get("link", "")),
                "snippet": clean_text(str(item.get("snippet", ""))),
            })
        return rows


class SerperTextSearchBackend(TextSearchBackend):
    def __init__(
        self,
        *,
        client: Any,
        page_reader: Any,
        cache: JsonCache,
        search_top_k: int = 5,
        result_top_k: int = 5,
        record_max_chars: int = 1000,
        chunk_words: int = 320,
        overlap_words: int = 48,
        locale: str = "en-US",
        provider_name: str = "serper",
        backend_version: str = SERPER_BACKEND_VERSION,
        budget: ApiBudgetGuard | None = None,
        statistics: CostStatistics | None = None,
    ):
        self.client = client
        self.page_reader = page_reader
        self.cache = cache
        self.search_top_k = int(search_top_k)
        self.result_top_k = int(result_top_k)
        self.record_max_chars = int(record_max_chars)
        self.chunk_words = int(chunk_words)
        self.overlap_words = int(overlap_words)
        self.locale = str(locale)
        self.provider_name = str(provider_name)
        self.backend_version = str(backend_version)
        self.budget = budget
        self.statistics = statistics
        self.parameters = {
            "search_top_k": self.search_top_k,
            "result_top_k": self.result_top_k,
            "record_max_chars": self.record_max_chars,
            "chunk_words": self.chunk_words,
            "overlap_words": self.overlap_words,
            "locale": self.locale,
        }

    def cache_key(self, query: str) -> str:
        return text_cache_key(self.provider_name, query, self.parameters, self.backend_version)

    def search(self, query: str, episode_context: EpisodeContext) -> SearchResult:
        self.last_cache_hit = False
        query = " ".join(str(query).split())
        if not query:
            raise SearchBackendError("EMPTY_RETRIEVAL", "text query is empty")
        key = self.cache_key(query)
        cached = self.cache.get_search_result("text", key)
        if cached is not None:
            self.last_cache_hit = True
            if self.statistics is not None:
                self.statistics.increment("text_cache_hits")
            return cached
        started = time.perf_counter()
        validator = getattr(self.client, "validate_ready", None)
        if validator is not None:
            validator()
        if self.provider_name == "serper" and self.budget is not None:
            self.budget.reserve("serper")
        if self.statistics is not None:
            self.statistics.increment(
                "serper_remote_requests" if self.provider_name == "serper" else "searxng_remote_requests"
            )
        provider_rows = self.client.search(query, top_k=self.search_top_k)
        if not provider_rows:
            raise SearchBackendError("EMPTY_RETRIEVAL", "Serper returned no organic results")
        chunks: list[PageChunk] = []
        fetch_rows: dict[int, dict[str, Any]] = {}
        for row in provider_rows:
            rank = int(row["rank"])
            try:
                page = self.page_reader.read(str(row.get("url", "")))
                fetch_rows[rank] = {
                    "status": page.fetch_status,
                    "latency_seconds": page.latency_seconds,
                    "content_sha256": page.content_sha256,
                    "fallback": False,
                    "reader_provider": str(page.metadata.get("reader_provider", page.metadata.get("provider", "unknown"))),
                    "reader_fallback": bool(page.metadata.get("reader_fallback", False)),
                    "reader_fallback_reason": page.metadata.get("reader_fallback_reason"),
                }
                chunks.extend(chunk_page(
                    page.clean_text,
                    url=page.url,
                    title=page.title or str(row.get("title", "")),
                    page_rank=rank,
                    chunk_words=self.chunk_words,
                    overlap_words=self.overlap_words,
                ))
            except SearchBackendError as exc:
                snippet = clean_text(str(row.get("snippet", "")))
                fetch_rows[rank] = {
                    "status": exc.code,
                    "latency_seconds": 0.0,
                    "content_sha256": hashlib.sha256(snippet.encode("utf-8")).hexdigest(),
                    "fallback": True,
                    "reader_provider": "snippet",
                    "reader_fallback": False,
                    "reader_fallback_reason": exc.code,
                }
                if snippet:
                    chunks.extend(chunk_page(
                        snippet,
                        url=str(row.get("url", "")),
                        title=str(row.get("title", "")),
                        page_rank=rank,
                        chunk_words=self.chunk_words,
                        overlap_words=self.overlap_words,
                    ))
        selected = select_chunks(query, chunks, top_k=self.result_top_k)
        if not selected:
            raise SearchBackendError("EMPTY_RETRIEVAL", "no page text or snippets were usable")
        provider_by_rank = {int(row["rank"]): row for row in provider_rows}
        records = []
        formatted = []
        for rank, chunk in enumerate(selected, 1):
            provider = provider_by_rank[chunk.page_rank]
            text = truncate_record(
                "title: %s | url: %s | content: %s"
                % (chunk.title or provider.get("title", ""), chunk.url, chunk.text),
                self.record_max_chars,
            )
            formatted.append(text)
            records.append(SearchRecord(
                rank=rank,
                title=chunk.title or str(provider.get("title", "")),
                url=chunk.url,
                snippet=str(provider.get("snippet", "")),
                content=truncate_record(chunk.text, self.record_max_chars),
                source="%s+%s" % (self.provider_name, fetch_rows[chunk.page_rank]["reader_provider"]),
                content_sha256=hashlib.sha256(chunk.text.encode("utf-8")).hexdigest(),
                metadata={
                    "provider_rank": chunk.page_rank,
                    "chunk_index": chunk.chunk_index,
                    "page_fetch_status": fetch_rows[chunk.page_rank]["status"],
                    "fetch_fallback": fetch_rows[chunk.page_rank]["fallback"],
                    "reader_provider": fetch_rows[chunk.page_rank]["reader_provider"],
                    "reader_fallback": fetch_rows[chunk.page_rank]["reader_fallback"],
                    "reader_fallback_reason": fetch_rows[chunk.page_rank]["reader_fallback_reason"],
                },
            ))
        result = SearchResult(
            tool_type="text_search",
            backend=self.provider_name,
            request={"query": query},
            timestamp=utc_now(),
            records=tuple(records),
            information_text=format_frozen_information("Text Search", formatted),
            metadata={
                "provider_parameters": self.parameters,
                "backend_version": self.backend_version,
                "latency_seconds": time.perf_counter() - started,
                "search_api_success": True,
                "page_fetch_success_count": sum(row["status"] == "success" for row in fetch_rows.values()),
                "page_fetch_failure_count": sum(row["status"] != "success" for row in fetch_rows.values()),
                "local_page_read_count": sum(row["reader_provider"] == "local" for row in fetch_rows.values()),
                "jina_fallback_read_count": sum(row["reader_provider"] == "jina" and row["reader_fallback"] for row in fetch_rows.values()),
                "cache_key": key,
            },
        )
        self.cache.put_search_result("text", key, result)
        return result
