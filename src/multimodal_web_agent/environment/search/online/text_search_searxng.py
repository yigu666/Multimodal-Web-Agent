from __future__ import annotations

import os
from typing import Any
from urllib.parse import urljoin

import requests

from ..schemas import SearchBackendError
from .chunk_selector import clean_text
from .text_search_serper import SerperTextSearchBackend


SEARXNG_BACKEND_VERSION = "searxng-local-bm25-v1"


class SearXNGClient:
    def __init__(
        self,
        *,
        base_url: str | None = None,
        timeout_seconds: float = 12.0,
        locale: str = "en-US",
        session=None,
    ) -> None:
        self.base_url = (base_url if base_url is not None else os.environ.get("SEARXNG_BASE_URL", "")).rstrip("/")
        self.timeout_seconds = float(timeout_seconds)
        self.locale = str(locale)
        self.session = session or requests.Session()

    def validate_ready(self) -> None:
        if not self.base_url:
            raise SearchBackendError("SEARXNG_UNAVAILABLE", "SEARXNG_BASE_URL is not configured")

    def search(self, query: str, *, top_k: int) -> list[dict[str, Any]]:
        self.validate_ready()
        try:
            response = self.session.get(
                urljoin(self.base_url + "/", "search"),
                params={"q": query, "format": "json", "language": self.locale},
                headers={"Accept": "application/json", "User-Agent": "multimodal-web-agent-v1-cost-revision/1.0"},
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            raw = response.json()
        except requests.Timeout as exc:
            raise SearchBackendError("TIMEOUT", "SearXNG request timed out") from exc
        except (requests.RequestException, ValueError) as exc:
            raise SearchBackendError("SEARCH_API_ERROR", "SearXNG request failed") from exc
        rows = []
        for rank, item in enumerate(raw.get("results", [])[: int(top_k)], 1):
            rows.append({
                "rank": rank,
                "title": clean_text(str(item.get("title", ""))),
                "url": str(item.get("url", "")),
                "snippet": clean_text(str(item.get("content", ""))),
            })
        return rows


class SearXNGTextSearchBackend(SerperTextSearchBackend):
    def __init__(self, **kwargs) -> None:
        super().__init__(
            provider_name="searxng",
            backend_version=SEARXNG_BACKEND_VERSION,
            **kwargs,
        )
