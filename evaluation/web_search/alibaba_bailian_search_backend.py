"""Alibaba Bailian WebSearch MCP adapter for the public E-VQA evaluator.

The adapter keeps one Streamable HTTP MCP session per worker, normalizes the
provider's ``pages`` records into the project's frozen ``SearchResult``
representation, and writes a separate redacted/raw-response provenance file.
No provider other than Bailian is referenced by this module.
"""
from __future__ import annotations

import asyncio
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
import time
from typing import Any, Mapping
from urllib.parse import urlparse

import anyio
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from multimodal_web_agent.data.protocol_sft.information_formatter import format_frozen_information
from multimodal_web_agent.environment.search.base import EpisodeContext, TextSearchBackend
from multimodal_web_agent.environment.search.online.cache import JsonCache, text_cache_key
from multimodal_web_agent.environment.search.online.cost_stats import CostStatistics
from multimodal_web_agent.environment.search.online.provenance import utc_now
from multimodal_web_agent.environment.search.schemas import SearchBackendError, SearchRecord, SearchResult


ALIBABA_BACKEND = "ALIBABA_BAILIAN_WEBSEARCH_MCP"
ALIBABA_TOOL = "bailian_web_search"
ALIBABA_ENDPOINT = "https://dashscope.aliyuncs.com/api/v1/mcps/WebSearch/mcp"
ALIBABA_BACKEND_VERSION = "alibaba-bailian-websearch-mcp-v1"


def _dump_model(value: Any) -> Any:
    """Convert an MCP/Pydantic object into JSON-compatible data."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(k): _dump_model(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_dump_model(v) for v in value]
    method = getattr(value, "model_dump", None)
    if method is not None:
        try:
            return method(mode="json")
        except TypeError:
            return method()
    method = getattr(value, "dict", None)
    if method is not None:
        return method()
    return str(value)


def _find_pages(value: Any) -> list[dict[str, Any]]:
    """Find the provider's pages list in structured or text MCP content."""
    if isinstance(value, Mapping):
        pages = value.get("pages")
        if isinstance(pages, list):
            return [dict(item) for item in pages if isinstance(item, Mapping)]
        if isinstance(value.get("text"), str):
            found = _find_pages(value["text"])
            if found:
                return found
        for key in ("structuredContent", "structured_content", "content", "data", "result"):
            if key in value:
                found = _find_pages(value[key])
                if found:
                    return found
        return []
    if isinstance(value, list):
        for item in value:
            found = _find_pages(item)
            if found:
                return found
        return []
    if isinstance(value, str):
        text = value.strip()
        candidates = [text]
        if text.startswith("```"):
            candidates.append(text.strip("`").replace("json\n", "", 1).strip())
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            candidates.append(text[start : end + 1])
        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
            except Exception:
                continue
            found = _find_pages(parsed)
            if found:
                return found
    return []


def _response_payload(result: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    dumped = _dump_model(result)
    pages = _find_pages(dumped)
    if not pages:
        raise SearchBackendError("ALIBABA_RESPONSE_INVALID", "MCP response did not contain pages[]")
    return (dumped if isinstance(dumped, dict) else {"value": dumped}, pages)


class _McpSessionWorker:
    """Persistent async MCP session hosted by a small private event-loop thread."""

    def __init__(self, endpoint: str, api_key: str, timeout_seconds: float = 45.0):
        self.endpoint = endpoint
        self.api_key = api_key
        self.timeout_seconds = float(timeout_seconds)
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._thread_main, name="bailian-mcp-session", daemon=True)
        self.ready = threading.Event()
        self.stop_event: asyncio.Event | None = None
        self.session: ClientSession | None = None
        self.server_name: str | None = None
        self.server_version: str | None = None
        self.protocol_version: str | None = None
        self.tools: list[dict[str, Any]] = []
        self.start_exception: Exception | None = None
        self._serve_task: asyncio.Task[Any] | None = None
        self.thread.start()
        if not self.ready.wait(max(60.0, self.timeout_seconds + 15.0)):
            raise RuntimeError("Alibaba MCP session initialization timed out")
        if self.start_exception is not None:
            raise self.start_exception

    def _thread_main(self) -> None:
        asyncio.set_event_loop(self.loop)
        self._serve_task = self.loop.create_task(self._serve())
        self.loop.run_forever()
        pending = asyncio.all_tasks(self.loop)
        for task in pending:
            task.cancel()
        if pending:
            self.loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        self.loop.close()

    async def _serve(self) -> None:
        self.stop_event = asyncio.Event()
        headers = {
            "Authorization": "Bearer " + self.api_key,
            "Content-Type": "application/json",
        }
        try:
            async with streamablehttp_client(
                self.endpoint,
                headers=headers,
                timeout=self.timeout_seconds,
                sse_read_timeout=self.timeout_seconds + 10.0,
                terminate_on_close=True,
            ) as (read_stream, write_stream, _get_session_id):
                self.session = ClientSession(
                    read_stream,
                    write_stream,
                    read_timeout_seconds=timedelta(seconds=self.timeout_seconds + 10.0),
                )
                initialized = await self.session.initialize()
                server_info = getattr(initialized, "serverInfo", None)
                self.server_name = str(getattr(server_info, "name", "")) or None
                self.server_version = str(getattr(server_info, "version", "")) or None
                self.protocol_version = str(getattr(initialized, "protocolVersion", "")) or None
                listed = await self.session.list_tools()
                self.tools = []
                for tool in getattr(listed, "tools", []) or []:
                    self.tools.append({
                        "name": str(getattr(tool, "name", "")),
                        "description": str(getattr(tool, "description", "") or ""),
                        "inputSchema": _dump_model(getattr(tool, "inputSchema", {})),
                    })
                self.ready.set()
                await self.stop_event.wait()
        except Exception as exc:
            self.start_exception = exc
            self.ready.set()
        finally:
            self.session = None

    async def _call(self, arguments: dict[str, Any]) -> Any:
        if self.session is None:
            raise RuntimeError("Alibaba MCP session is not ready")
        return await self.session.call_tool(ALIBABA_TOOL, arguments)

    def call(self, arguments: dict[str, Any]) -> Any:
        if self.start_exception is not None:
            raise self.start_exception
        future = asyncio.run_coroutine_threadsafe(self._call(arguments), self.loop)
        return future.result(timeout=self.timeout_seconds + 20.0)

    async def _stop(self) -> None:
        if self.stop_event is not None:
            self.stop_event.set()
        if self._serve_task is not None:
            try:
                await self._serve_task
            except Exception:
                pass

    def close(self) -> None:
        if not self.thread.is_alive():
            return
        try:
            future = asyncio.run_coroutine_threadsafe(self._stop(), self.loop)
            future.result(timeout=30.0)
        except Exception:
            pass
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=30.0)


class _McpOneShotWorker:
    """Safe fallback for the SDK's sync evaluator boundary.

    The official MCP 1.29.0 context manager works reliably when entered and
    exited inside one ``anyio.run`` call. Keeping that context alive from a
    background asyncio thread caused the server's initialize response to time
    out, so this worker reconnects only after that transport-specific failure.
    The reason is persisted in the run metadata; no parallel connections are
    opened.
    """

    def __init__(self, endpoint: str, api_key: str, timeout_seconds: float = 45.0):
        self.endpoint = endpoint
        self.api_key = api_key
        self.timeout_seconds = float(timeout_seconds)
        self.server_name: str | None = None
        self.server_version: str | None = None
        self.protocol_version: str | None = None
        self.tools: list[dict[str, Any]] = []
        self._probe()

    async def _probe_async(self) -> None:
        async with streamablehttp_client(
            self.endpoint,
            headers={"Authorization": "Bearer " + self.api_key},
            timeout=self.timeout_seconds,
            sse_read_timeout=self.timeout_seconds + 10.0,
            terminate_on_close=True,
        ) as (read_stream, write_stream, _get_session_id):
            async with ClientSession(
                read_stream,
                write_stream,
                read_timeout_seconds=timedelta(seconds=self.timeout_seconds + 10.0),
            ) as session:
                initialized = await session.initialize()
                server_info = getattr(initialized, "serverInfo", None)
                self.server_name = str(getattr(server_info, "name", "")) or None
                self.server_version = str(getattr(server_info, "version", "")) or None
                self.protocol_version = str(getattr(initialized, "protocolVersion", "")) or None
                listed = await session.list_tools()
                self.tools = [{
                    "name": str(getattr(tool, "name", "")),
                    "description": str(getattr(tool, "description", "") or ""),
                    "inputSchema": _dump_model(getattr(tool, "inputSchema", {})),
                } for tool in (getattr(listed, "tools", []) or [])]

    async def _call_async(self, arguments: dict[str, Any]) -> Any:
        async with streamablehttp_client(
            self.endpoint,
            headers={"Authorization": "Bearer " + self.api_key},
            timeout=self.timeout_seconds,
            sse_read_timeout=self.timeout_seconds + 10.0,
            terminate_on_close=True,
        ) as (read_stream, write_stream, _get_session_id):
            async with ClientSession(
                read_stream,
                write_stream,
                read_timeout_seconds=timedelta(seconds=self.timeout_seconds + 10.0),
            ) as session:
                initialized = await session.initialize()
                server_info = getattr(initialized, "serverInfo", None)
                self.server_name = str(getattr(server_info, "name", "")) or None
                self.server_version = str(getattr(server_info, "version", "")) or None
                self.protocol_version = str(getattr(initialized, "protocolVersion", "")) or None
                return await session.call_tool(ALIBABA_TOOL, arguments)

    def _probe(self) -> None:
        anyio.run(self._probe_async)

    def call(self, arguments: dict[str, Any]) -> Any:
        return anyio.run(self._call_async, arguments)

    def close(self) -> None:
        return None


class AlibabaBailianWebSearchBackend(TextSearchBackend):
    """Synchronous project backend backed by a reusable Bailian MCP session."""

    def __init__(
        self,
        *,
        cache: JsonCache,
        raw_response_root: Path,
        statistics: CostStatistics | None = None,
        api_key: str | None = None,
        endpoint: str = ALIBABA_ENDPOINT,
        tool_name: str = ALIBABA_TOOL,
        search_count: int = 5,
        max_remote_calls: int = 640,
        timeout_seconds: float = 45.0,
    ) -> None:
        self.api_key = api_key if api_key is not None else os.environ.get("DASHSCOPE_API_KEY", "")
        if not self.api_key:
            raise SearchBackendError("ONLINE_CREDENTIAL_MISSING", "DASHSCOPE_API_KEY is not configured")
        if str(tool_name) != ALIBABA_TOOL:
            raise ValueError("R2 requires bailian_web_search")
        self.endpoint = str(endpoint)
        self.tool_name = str(tool_name)
        self.search_count = int(search_count)
        self.max_remote_calls = int(max_remote_calls)
        self.cache = cache
        self.raw_response_root = Path(raw_response_root)
        self.statistics = statistics
        self.backend_version = ALIBABA_BACKEND_VERSION
        self.parameters = {"tool": ALIBABA_TOOL, "count": self.search_count}
        self.last_cache_hit = False
        self.last_retry_count = 0
        self.last_request_hash: str | None = None
        self.last_response_hash: str | None = None
        self.remote_call_count = 0
        self._call_lock = threading.Lock()
        self._worker = _McpOneShotWorker(self.endpoint, self.api_key, timeout_seconds=timeout_seconds)
        if ALIBABA_TOOL not in {tool.get("name") for tool in self._worker.tools}:
            self.close()
            raise SearchBackendError("ALIBABA_TOOL_MISSING", "bailian_web_search was not advertised by MCP")

    @property
    def server_metadata(self) -> dict[str, Any]:
        return {
            "server_name": self._worker.server_name,
            "server_version": self._worker.server_version,
            "negotiated_protocol": self._worker.protocol_version,
            "tool_name": self.tool_name,
            "endpoint": self.endpoint,
            "tool_schema": next((tool.get("inputSchema") for tool in self._worker.tools if tool.get("name") == ALIBABA_TOOL), {}),
        }

    def close(self) -> None:
        self._worker.close()

    def _write_raw(self, request_hash: str, payload: Mapping[str, Any]) -> tuple[str, str]:
        data = json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        response_hash = hashlib.sha256(data.encode("utf-8")).hexdigest()
        directory = self.raw_response_root / "raw_mcp"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / (request_hash + ".json")
        descriptor, temporary_name = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".tmp", dir=str(directory))
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, path)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)
        return response_hash, str(path)

    @staticmethod
    def _retryable(exc: Exception) -> bool:
        text = str(exc).upper()
        return any(token in text for token in ("TIMEOUT", "TIMED OUT", "503", "502", "504", "DISCONNECT", "TRANSPORT"))

    def _remote_search(self, query: str) -> tuple[dict[str, Any], list[dict[str, Any]], int, float]:
        request_hash = hashlib.sha256(json.dumps({"query": query, "count": self.search_count}, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        last_exc: Exception | None = None
        attempts = 0
        started = time.perf_counter()
        while attempts < 2:
            attempts += 1
            with self._call_lock:
                if self.remote_call_count >= self.max_remote_calls:
                    raise SearchBackendError("ALIBABA_BUDGET_EXHAUSTED", "Alibaba formal search budget exhausted")
                self.remote_call_count += 1
            if self.statistics is not None:
                self.statistics.increment("alibaba_websearch_requests")
            try:
                result = self._worker.call({"query": query, "count": self.search_count})
                dumped, pages = _response_payload(result)
                if bool(getattr(result, "isError", False)):
                    raise SearchBackendError("ALIBABA_TOOL_ERROR", "bailian_web_search returned isError=true")
                dumped["_mcp_is_error"] = False
                response_hash, raw_path = self._write_raw(request_hash, dumped)
                dumped["_raw_response_path"] = raw_path
                self.last_request_hash = request_hash
                self.last_response_hash = response_hash
                return dumped, pages, attempts - 1, time.perf_counter() - started
            except SearchBackendError:
                raise
            except Exception as exc:
                last_exc = exc
                if attempts >= 2 or not self._retryable(exc):
                    break
        raise SearchBackendError("ALIBABA_SEARCH_ERROR", "Alibaba MCP search failed", metadata={"attempts": attempts, "error_type": type(last_exc).__name__ if last_exc else "unknown"}) from last_exc

    def search(self, query: str, episode_context: EpisodeContext) -> SearchResult:
        del episode_context
        self.last_cache_hit = False
        self.last_retry_count = 0
        normalized = " ".join(str(query).split())
        if not normalized:
            raise SearchBackendError("EMPTY_RETRIEVAL", "text query is empty")
        key = text_cache_key(ALIBABA_BACKEND, normalized, self.parameters, self.backend_version)
        cached = self.cache.get_search_result("text", key)
        if cached is not None:
            self.last_cache_hit = True
            self.last_request_hash = hashlib.sha256(json.dumps({"query": normalized, "count": self.search_count}, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
            self.last_response_hash = hashlib.sha256(cached.information_text.encode("utf-8")).hexdigest()
            if self.statistics is not None:
                self.statistics.increment("text_cache_hits")
            return cached
        dumped, pages, retry_count, latency = self._remote_search(normalized)
        self.last_retry_count = retry_count
        records: list[SearchRecord] = []
        formatted: list[str] = []
        for rank, page in enumerate(pages[: self.search_count], 1):
            title = str(page.get("title", page.get("name", "")) or "")
            url = str(page.get("url", page.get("link", "")) or "")
            snippet = str(page.get("snippet", page.get("description", page.get("content", ""))) or "")
            hostname = str(page.get("hostname", "") or "")
            if not hostname:
                hostname = str(urlparse(url).hostname or "")
            content = snippet.strip()
            records.append(SearchRecord(
                rank=rank,
                title=title,
                url=url,
                snippet=snippet,
                content=content,
                source=ALIBABA_BACKEND,
                content_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
                metadata={"hostname": hostname, "provider": ALIBABA_BACKEND, "tool_name": ALIBABA_TOOL},
            ))
            formatted.append("title: %s | url: %s | hostname: %s | snippet: %s" % (title, url, hostname, snippet))
        if not records:
            raise SearchBackendError("EMPTY_RETRIEVAL", "Alibaba returned no usable pages")
        info = format_frozen_information("Text Search", formatted)
        raw_path = dumped.get("_raw_response_path")
        result = SearchResult(
            tool_type="text_search",
            backend=ALIBABA_BACKEND,
            request={"query": normalized, "count": self.search_count, "tool": ALIBABA_TOOL},
            timestamp=utc_now(),
            records=tuple(records),
            information_text=info,
            metadata={
                "provider_parameters": self.parameters,
                "backend_version": self.backend_version,
                "tool_name": ALIBABA_TOOL,
                "latency_seconds": latency,
                "search_api_success": True,
                "page_fetch_success_count": 0,
                "page_fetch_failure_count": 0,
                "raw_mcp_response_sha256": self.last_response_hash,
                "raw_mcp_response_path": raw_path,
                "request_hash": self.last_request_hash,
                "response_hash": self.last_response_hash,
                "retry_count": retry_count,
                **self.server_metadata,
            },
        )
        self.cache.put_search_result("text", key, result)
        return result
