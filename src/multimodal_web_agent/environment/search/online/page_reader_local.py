from __future__ import annotations

from html.parser import HTMLParser
import hashlib
import time
from typing import Any
from urllib.parse import urljoin

import requests

from ..schemas import SearchBackendError
from .cache import JsonCache, page_cache_key
from .chunk_selector import clean_text
from .cost_stats import CostStatistics
from .page_reader import PageReadResult
from .provenance import utc_now
from .security import validate_public_url


LOCAL_READER_VERSION = "local-html-reader-v1"
PAGE_READER_PIPELINE_VERSION = "local-primary-jina-fallback-v1"


class _DeterministicHTMLExtractor(HTMLParser):
    SKIP = {"script", "style", "noscript", "svg", "canvas", "template"}
    BLOCK = {
        "article", "aside", "blockquote", "br", "div", "footer", "h1", "h2",
        "h3", "h4", "h5", "h6", "header", "li", "main", "nav", "p", "pre",
        "section", "table", "td", "th", "tr",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.skip_depth = 0
        self.in_title = False
        self.parts: list[str] = []
        self.title_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        tag = tag.casefold()
        if tag in self.SKIP:
            self.skip_depth += 1
        if tag == "title" and self.skip_depth == 0:
            self.in_title = True
        if tag in self.BLOCK and self.skip_depth == 0:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.casefold()
        if tag in self.SKIP and self.skip_depth:
            self.skip_depth -= 1
        if tag == "title":
            self.in_title = False
        if tag in self.BLOCK and self.skip_depth == 0:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self.skip_depth:
            return
        self.parts.append(data)
        if self.in_title:
            self.title_parts.append(data)

    def result(self) -> tuple[str, str]:
        return clean_text(" ".join(self.title_parts)), clean_text(" ".join(self.parts))


def extract_html_text(value: str) -> tuple[str, str]:
    parser = _DeterministicHTMLExtractor()
    try:
        parser.feed(str(value))
        parser.close()
    except (ValueError, AssertionError) as exc:
        raise SearchBackendError("HTML_EXTRACTION_FAILURE", "local HTML extraction failed") from exc
    return parser.result()


class LocalPageReader:
    def __init__(
        self,
        *,
        cache: JsonCache | None = None,
        statistics: CostStatistics | None = None,
        timeout_seconds: float = 15.0,
        max_response_bytes: int = 2_000_000,
        max_redirects: int = 3,
        minimum_usable_chars: int = 200,
        session=None,
        url_validator=validate_public_url,
    ) -> None:
        self.cache = cache
        self.statistics = statistics
        self.timeout_seconds = float(timeout_seconds)
        self.max_response_bytes = int(max_response_bytes)
        self.max_redirects = int(max_redirects)
        self.minimum_usable_chars = int(minimum_usable_chars)
        self.session = session or requests.Session()
        self.url_validator = url_validator

    def _fetch(self, url: str):
        current = self.url_validator(url)
        response = None
        for redirect_count in range(self.max_redirects + 1):
            try:
                response = self.session.get(
                    current,
                    headers={"User-Agent": "multimodal-web-agent-v1-cost-revision/1.0"},
                    timeout=self.timeout_seconds,
                    allow_redirects=False,
                    stream=True,
                )
            except requests.Timeout as exc:
                raise SearchBackendError("TIMEOUT", "local page fetch timed out") from exc
            except requests.RequestException as exc:
                raise SearchBackendError("PAGE_FETCH_ERROR", "local page fetch failed") from exc
            if int(response.status_code) in {301, 302, 303, 307, 308}:
                location = response.headers.get("Location", "")
                response.close()
                if not location:
                    raise SearchBackendError("PAGE_FETCH_ERROR", "redirect is missing Location")
                if redirect_count >= self.max_redirects:
                    raise SearchBackendError("PAGE_FETCH_ERROR", "redirect limit exceeded")
                current = self.url_validator(urljoin(current, location))
                continue
            break
        if response is None:
            raise SearchBackendError("PAGE_FETCH_ERROR", "local page fetch produced no response")
        final_url = self.url_validator(str(getattr(response, "url", current) or current))
        return final_url, response

    def read(self, url: str) -> PageReadResult:
        canonical = self.url_validator(url)
        key = page_cache_key(canonical, LOCAL_READER_VERSION)
        if self.cache is not None:
            cached = self.cache.get_json("pages", key)
            if cached is not None:
                if self.statistics is not None:
                    self.statistics.increment("page_cache_hits")
                return PageReadResult.from_dict(cached)
        if self.statistics is not None:
            self.statistics.increment("local_page_reads")
        started = time.perf_counter()
        final_url, response = self._fetch(canonical)
        status = int(response.status_code)
        try:
            if status < 200 or status >= 300:
                raise SearchBackendError(
                    "PAGE_FETCH_ERROR", "local page returned HTTP %d" % status,
                    metadata={"http_status": status},
                )
            content_type = str(response.headers.get("Content-Type", "")).casefold()
            if "html" not in content_type and "text/" not in content_type:
                raise SearchBackendError(
                    "UNSUPPORTED_PAGE", "local reader only accepts HTML/text responses",
                    metadata={"content_type": content_type, "http_status": status},
                )
            parts: list[bytes] = []
            total = 0
            for block in response.iter_content(chunk_size=65536):
                if not block:
                    continue
                total += len(block)
                if total > self.max_response_bytes:
                    raise SearchBackendError("PAGE_FETCH_ERROR", "page response is too large")
                parts.append(bytes(block))
            encoding = response.encoding or "utf-8"
            decoded = b"".join(parts).decode(encoding, errors="replace")
        finally:
            response.close()
        if "html" in content_type:
            title, content = extract_html_text(decoded)
        else:
            title, content = "", clean_text(decoded)
        if len(content) < self.minimum_usable_chars:
            raise SearchBackendError(
                "USABLE_TEXT_TOO_SHORT",
                "local page text is shorter than the configured minimum",
                metadata={"usable_chars": len(content), "minimum_usable_chars": self.minimum_usable_chars},
            )
        result = PageReadResult(
            url=final_url,
            title=title,
            clean_text=content,
            fetch_status="success",
            timestamp=utc_now(),
            content_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
            latency_seconds=time.perf_counter() - started,
            metadata={
                "provider": "local",
                "reader_provider": "local",
                "reader_fallback": False,
                "reader_fallback_reason": None,
                "reader_version": LOCAL_READER_VERSION,
                "http_status": status,
                "canonical_url": canonical,
            },
        )
        if self.cache is not None:
            self.cache.put_json("pages", key, result.to_dict())
        return result


class FallbackPageReader:
    ALLOWED_FALLBACK_CODES = {
        "PAGE_FETCH_ERROR", "TIMEOUT", "USABLE_TEXT_TOO_SHORT",
        "HTML_EXTRACTION_FAILURE", "UNSUPPORTED_PAGE",
    }

    def __init__(
        self,
        *,
        primary: Any,
        fallback: Any | None,
        cache: JsonCache | None = None,
        statistics: CostStatistics | None = None,
        url_validator=validate_public_url,
    ) -> None:
        self.primary = primary
        self.fallback = fallback
        self.cache = cache
        self.statistics = statistics
        self.url_validator = url_validator

    def read(self, url: str) -> PageReadResult:
        canonical = self.url_validator(url)
        key = page_cache_key(canonical, PAGE_READER_PIPELINE_VERSION)
        if self.cache is not None:
            cached = self.cache.get_json("pages", key)
            if cached is not None:
                if self.statistics is not None:
                    self.statistics.increment("page_cache_hits")
                return PageReadResult.from_dict(cached)
        try:
            result = self.primary.read(canonical)
        except SearchBackendError as exc:
            if exc.code not in self.ALLOWED_FALLBACK_CODES or self.fallback is None:
                raise
            fallback = self.fallback.read(canonical)
            if self.statistics is not None:
                self.statistics.increment("jina_fallback_reads")
            result = PageReadResult(
                url=fallback.url,
                title=fallback.title,
                clean_text=fallback.clean_text,
                fetch_status=fallback.fetch_status,
                timestamp=fallback.timestamp,
                content_sha256=fallback.content_sha256,
                latency_seconds=fallback.latency_seconds,
                metadata={
                    **fallback.metadata,
                    "reader_provider": "jina",
                    "reader_fallback": True,
                    "reader_fallback_reason": exc.code,
                    "local_failure_metadata": exc.metadata,
                    "reader_pipeline_version": PAGE_READER_PIPELINE_VERSION,
                },
            )
        else:
            result = PageReadResult(
                url=result.url,
                title=result.title,
                clean_text=result.clean_text,
                fetch_status=result.fetch_status,
                timestamp=result.timestamp,
                content_sha256=result.content_sha256,
                latency_seconds=result.latency_seconds,
                metadata={
                    **result.metadata,
                    "reader_provider": "local",
                    "reader_fallback": False,
                    "reader_fallback_reason": None,
                    "reader_pipeline_version": PAGE_READER_PIPELINE_VERSION,
                },
            )
        if self.cache is not None:
            self.cache.put_json("pages", key, result.to_dict())
        return result
