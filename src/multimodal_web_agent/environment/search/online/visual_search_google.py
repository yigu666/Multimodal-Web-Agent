from __future__ import annotations

from io import BytesIO
import hashlib
import os
import time
from typing import Any, Mapping

from multimodal_web_agent.data.protocol_sft.information_formatter import (
    format_frozen_information,
)
from multimodal_web_agent.evaluation.unified_agent.environment import truncate_record

from ..base import EpisodeContext, VisualSearchBackend
from ..schemas import SearchBackendError, SearchRecord, SearchResult
from .cache import JsonCache, visual_cache_key
from .budget import ApiBudgetGuard
from .chunk_selector import clean_text
from .cost_stats import CostStatistics
from .provenance import utc_now


GOOGLE_VISUAL_BACKEND_VERSION = "google-vision-web-detection-local-reader-v2"


def _image_bytes(image: Any) -> bytes:
    if isinstance(image, bytes):
        return image
    if isinstance(image, bytearray):
        return bytes(image)
    if hasattr(image, "save"):
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()
    raise SearchBackendError("MODEL_ERROR", "visual search requires current image pixels")


class GoogleVisionClient:
    def __init__(self, client=None):
        self.client = client

    def validate_ready(self) -> None:
        credential = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
        if self.client is None and not credential:
            raise SearchBackendError(
                "ONLINE_CREDENTIAL_MISSING",
                "GOOGLE_APPLICATION_CREDENTIALS is not configured",
            )
        if self.client is None:
            try:
                import google.cloud.vision  # noqa: F401
            except ImportError as exc:
                raise SearchBackendError(
                    "ONLINE_DEPENDENCY_MISSING", "google-cloud-vision is not installed"
                ) from exc

    def detect(self, image: Any) -> dict[str, Any]:
        self.validate_ready()
        credential = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
        if self.client is None and not credential:
            raise SearchBackendError(
                "ONLINE_CREDENTIAL_MISSING",
                "GOOGLE_APPLICATION_CREDENTIALS is not configured",
            )
        try:
            from google.cloud import vision
        except ImportError as exc:
            if self.client is None:
                raise SearchBackendError(
                    "ONLINE_DEPENDENCY_MISSING", "google-cloud-vision is not installed"
                ) from exc
            vision = None
        client = self.client or vision.ImageAnnotatorClient()
        content = _image_bytes(image)
        request_image = vision.Image(content=content) if vision is not None else content
        try:
            response = client.web_detection(image=request_image)
        except Exception as exc:
            raise SearchBackendError("SEARCH_API_ERROR", "Google Vision request failed") from exc
        error = getattr(response, "error", None)
        if error is not None and getattr(error, "message", ""):
            raise SearchBackendError("SEARCH_API_ERROR", str(error.message))
        annotation = getattr(response, "web_detection", response)
        return {
            "web_entities": [
                {
                    "description": str(getattr(item, "description", "")),
                    "score": float(getattr(item, "score", 0.0)),
                    "entity_id": str(getattr(item, "entity_id", "")),
                }
                for item in (getattr(annotation, "web_entities", None) or [])
            ],
            "matching_pages": [
                {
                    "url": str(getattr(item, "url", "")),
                    "title": str(getattr(item, "page_title", "")),
                    "score": float(getattr(item, "score", 0.0)),
                }
                for item in (getattr(annotation, "pages_with_matching_images", None) or [])
            ],
            "full_matching_images": [
                str(getattr(item, "url", ""))
                for item in (getattr(annotation, "full_matching_images", None) or [])
            ],
            "partial_matching_images": [
                str(getattr(item, "url", ""))
                for item in (getattr(annotation, "partial_matching_images", None) or [])
            ],
            "visually_similar_images": [
                str(getattr(item, "url", ""))
                for item in (getattr(annotation, "visually_similar_images", None) or [])
            ],
        }


class GoogleVisualSearchBackend(VisualSearchBackend):
    def __init__(
        self,
        *,
        client: Any,
        page_reader: Any,
        cache: JsonCache,
        entity_top_k: int = 3,
        matching_page_top_k: int = 5,
        result_top_k: int = 5,
        record_max_chars: int = 512,
        read_matching_pages: int = 2,
        budget: ApiBudgetGuard | None = None,
        statistics: CostStatistics | None = None,
    ):
        self.client = client
        self.page_reader = page_reader
        self.cache = cache
        self.entity_top_k = int(entity_top_k)
        self.matching_page_top_k = int(matching_page_top_k)
        self.result_top_k = int(result_top_k)
        self.record_max_chars = int(record_max_chars)
        self.read_matching_pages = int(read_matching_pages)
        self.budget = budget
        self.statistics = statistics
        self.parameters = {
            "entity_top_k": self.entity_top_k,
            "matching_page_top_k": self.matching_page_top_k,
            "result_top_k": self.result_top_k,
            "record_max_chars": self.record_max_chars,
            "read_matching_pages": self.read_matching_pages,
            "region_aware": False,
            "retrieved_images_injected": False,
        }

    def cache_key(self, image_sha256: str) -> str:
        return visual_cache_key(
            "google_vision_web_detection",
            image_sha256,
            self.parameters,
            GOOGLE_VISUAL_BACKEND_VERSION,
        )

    def search(self, image: Any, episode_context: EpisodeContext) -> SearchResult:
        self.last_cache_hit = False
        image_sha256 = episode_context.image_sha256
        if not image_sha256:
            image_sha256 = hashlib.sha256(_image_bytes(image)).hexdigest()
        key = self.cache_key(image_sha256)
        cached = self.cache.get_search_result("visual", key)
        if cached is not None:
            self.last_cache_hit = True
            if self.statistics is not None:
                self.statistics.increment("visual_cache_hits")
            return cached
        started = time.perf_counter()
        validator = getattr(self.client, "validate_ready", None)
        if validator is not None:
            validator()
        if self.budget is not None:
            self.budget.reserve("google_vision")
        if self.statistics is not None:
            self.statistics.increment("vision_remote_requests")
        raw = self.client.detect(image)
        entities = [
            item for item in raw.get("web_entities", [])
            if clean_text(str(item.get("description", "")))
        ][:self.entity_top_k]
        pages = [
            item for item in raw.get("matching_pages", [])
            if str(item.get("url", "")).strip()
        ][:self.matching_page_top_k]
        page_content: dict[str, tuple[str, str, bool, str, bool, str | None]] = {}
        for item in pages[:self.read_matching_pages]:
            url = str(item.get("url", ""))
            try:
                page = self.page_reader.read(url)
                page_content[url] = (
                    truncate_record(page.clean_text, self.record_max_chars),
                    page.fetch_status,
                    False,
                    str(page.metadata.get("reader_provider", page.metadata.get("provider", "unknown"))),
                    bool(page.metadata.get("reader_fallback", False)),
                    page.metadata.get("reader_fallback_reason"),
                )
            except SearchBackendError as exc:
                page_content[url] = ("", exc.code, True, "none", False, exc.code)
        candidates: list[SearchRecord] = []
        for item in entities:
            description = clean_text(str(item.get("description", "")))
            candidates.append(SearchRecord(
                rank=len(candidates) + 1,
                title=description,
                content=description,
                source="google_vision_web_entity",
                content_sha256=hashlib.sha256(description.encode("utf-8")).hexdigest(),
                metadata={"score": float(item.get("score", 0.0)), "entity_id": str(item.get("entity_id", ""))},
            ))
        for item in pages:
            url = str(item.get("url", ""))
            content, status, fallback, reader_provider, reader_fallback, fallback_reason = page_content.get(
                url, ("", "not_fetched", False, "none", False, None)
            )
            title = clean_text(str(item.get("title", "")))
            candidates.append(SearchRecord(
                rank=len(candidates) + 1,
                title=title,
                url=url,
                content=content,
                source="google_vision_matching_page",
                content_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
                metadata={
                    "score": float(item.get("score", 0.0)),
                    "page_fetch_status": status,
                    "fetch_fallback": fallback,
                    "reader_provider": reader_provider,
                    "reader_fallback": reader_fallback,
                    "reader_fallback_reason": fallback_reason,
                },
            ))
        candidates = candidates[:self.result_top_k]
        if not candidates:
            raise SearchBackendError("EMPTY_RETRIEVAL", "Vision Web Detection returned no usable entities/pages")
        records = tuple(
            SearchRecord(**{**record.to_dict(), "rank": rank})
            for rank, record in enumerate(candidates, 1)
        )
        formatted = [
            truncate_record(
                "entity/title: %s%s%s"
                % (
                    record.title,
                    " | url: %s" % record.url if record.url else "",
                    " | content: %s" % record.content if record.content else "",
                ),
                self.record_max_chars,
            )
            for record in records
        ]
        result = SearchResult(
            tool_type="visual_search",
            backend="google_vision_web_detection",
            request={"image_sha256": image_sha256},
            timestamp=utc_now(),
            records=records,
            information_text=format_frozen_information("Image Search", formatted),
            metadata={
                "provider_parameters": self.parameters,
                "backend_version": GOOGLE_VISUAL_BACKEND_VERSION,
                "latency_seconds": time.perf_counter() - started,
                "search_api_success": True,
                "web_entity_count": len(raw.get("web_entities", [])),
                "matching_page_count": len(raw.get("matching_pages", [])),
                "page_fetch_success_count": sum(
                    status == "success" for _, status, _, _, _, _ in page_content.values()
                ),
                "page_fetch_failure_count": sum(
                    status != "success" for _, status, _, _, _, _ in page_content.values()
                ),
                "local_page_read_count": sum(
                    provider == "local" for _, _, _, provider, _, _ in page_content.values()
                ),
                "jina_fallback_read_count": sum(
                    provider == "jina" and fallback
                    for _, _, _, provider, fallback, _ in page_content.values()
                ),
                "full_matching_image_count": len(raw.get("full_matching_images", [])),
                "partial_matching_image_count": len(raw.get("partial_matching_images", [])),
                "visually_similar_image_count": len(raw.get("visually_similar_images", [])),
                "cache_key": key,
            },
        )
        self.cache.put_search_result("visual", key, result)
        return result
