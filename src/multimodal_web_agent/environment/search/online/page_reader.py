from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
import socket
import time
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, build_opener, HTTPRedirectHandler

from ..schemas import SearchBackendError
from .cache import JsonCache, page_cache_key
from .budget import ApiBudgetGuard
from .chunk_selector import clean_text
from .cost_stats import CostStatistics
from .provenance import utc_now
from .security import validate_public_url


JINA_READER_VERSION = "jina-reader-v1"


@dataclass(frozen=True)
class PageReadResult:
    url: str
    title: str
    clean_text: str
    fetch_status: str
    timestamp: str
    content_sha256: str
    latency_seconds: float
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PageReadResult":
        return cls(
            url=str(value["url"]),
            title=str(value.get("title", "")),
            clean_text=str(value.get("clean_text", "")),
            fetch_status=str(value.get("fetch_status", "")),
            timestamp=str(value.get("timestamp", "")),
            content_sha256=str(value.get("content_sha256", "")),
            latency_seconds=float(value.get("latency_seconds", 0.0)),
            metadata=dict(value.get("metadata", {})),
        )


class _LimitedRedirectHandler(HTTPRedirectHandler):
    def __init__(self, maximum: int):
        super().__init__()
        self.maximum = int(maximum)

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        count = int(req.headers.get("X-Online-Redirect-Count", "0")) + 1
        if count > self.maximum:
            raise HTTPError(req.full_url, code, "redirect limit exceeded", headers, fp)
        request = super().redirect_request(req, fp, code, msg, headers, newurl)
        if request is not None:
            request.add_header("X-Online-Redirect-Count", str(count))
        return request


class JinaPageReader:
    def __init__(
        self,
        *,
        cache: JsonCache | None = None,
        api_key: str | None = None,
        timeout_seconds: float = 15.0,
        max_response_bytes: int = 2_000_000,
        max_redirects: int = 3,
        opener=None,
        url_validator=validate_public_url,
        budget: ApiBudgetGuard | None = None,
        statistics: CostStatistics | None = None,
    ):
        self.cache = cache
        self.api_key = api_key if api_key is not None else os.environ.get("JINA_API_KEY", "")
        self.timeout_seconds = float(timeout_seconds)
        self.max_response_bytes = int(max_response_bytes)
        self.url_validator = url_validator
        self.opener = opener or build_opener(_LimitedRedirectHandler(max_redirects))
        self.budget = budget
        self.statistics = statistics

    def read(self, url: str) -> PageReadResult:
        canonical = self.url_validator(url)
        key = page_cache_key(canonical, JINA_READER_VERSION)
        if self.cache is not None:
            cached = self.cache.get_json("pages", key)
            if cached is not None:
                if self.statistics is not None:
                    self.statistics.increment("page_cache_hits")
                return PageReadResult.from_dict(cached)
        # The source URL is embedded in the Jina endpoint path. Provider URLs
        # can legally contain decoded spaces (for example Lens result URLs),
        # while urllib rejects control characters in Request paths.
        encoded_source_url = quote(
            canonical,
            safe=":/?&=%+,$;@-._~!()*'[]",
        )
        reader_url = "https://r.jina.ai/%s" % encoded_source_url
        headers = {
            "Accept": "application/json",
            "User-Agent": "multimodal-web-agent-v1/1.0",
        }
        if self.api_key:
            headers["Authorization"] = "Bearer %s" % self.api_key
        started = time.perf_counter()
        if self.budget is not None:
            self.budget.reserve("jina")
        if self.statistics is not None:
            self.statistics.increment("jina_remote_requests")
        try:
            response = self.opener.open(
                Request(reader_url, headers=headers, method="GET"),
                timeout=self.timeout_seconds,
            )
            payload = response.read(self.max_response_bytes + 1)
            if len(payload) > self.max_response_bytes:
                raise SearchBackendError("PAGE_FETCH_ERROR", "reader response is too large")
            decoded = payload.decode("utf-8", errors="replace")
        except (HTTPError, URLError, TimeoutError, socket.timeout) as exc:
            code = "TIMEOUT" if isinstance(exc, (TimeoutError, socket.timeout)) else "PAGE_FETCH_ERROR"
            raise SearchBackendError(code, "Jina Reader request failed") from exc
        title = ""
        content = decoded
        try:
            raw = json.loads(decoded)
            data = raw.get("data", raw) if isinstance(raw, dict) else {}
            if isinstance(data, dict):
                title = str(data.get("title", ""))
                content = str(data.get("content") or data.get("text") or "")
        except json.JSONDecodeError:
            pass
        content = clean_text(content)
        if not content:
            raise SearchBackendError("PAGE_FETCH_ERROR", "Jina Reader returned empty text")
        result = PageReadResult(
            url=canonical,
            title=clean_text(title),
            clean_text=content,
            fetch_status="success",
            timestamp=utc_now(),
            content_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
            latency_seconds=time.perf_counter() - started,
            metadata={
                "provider": "jina",
                "reader_provider": "jina",
                "reader_fallback": False,
                "reader_fallback_reason": None,
                "reader_version": JINA_READER_VERSION,
                "http_status": int(getattr(response, "status", 200)),
                "canonical_url": canonical,
            },
        )
        if self.cache is not None:
            self.cache.put_json("pages", key, result.to_dict())
        return result
