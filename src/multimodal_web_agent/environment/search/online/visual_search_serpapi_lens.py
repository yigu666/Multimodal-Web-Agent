from __future__ import annotations

from dataclasses import asdict, dataclass
from io import BytesIO
import hashlib
import os
import time
from typing import Any

from PIL import Image, ImageOps
import requests

from multimodal_web_agent.data.protocol_sft.information_formatter import format_frozen_information
from multimodal_web_agent.evaluation.unified_agent.environment import truncate_record

from ..base import EpisodeContext, VisualSearchBackend
from ..schemas import SearchBackendError, SearchRecord, SearchResult
from .budget import ApiBudgetGuard
from .cache import JsonCache, visual_cache_key
from .chunk_selector import clean_text
from .cost_stats import CostStatistics
from .provenance import utc_now
from .security import canonical_url


SERPAPI_LENS_BACKEND_VERSION = "serpapi-google-lens-local-reader-v1"
SERPAPI_IMAGE_PREPARATION_VERSION = "serpapi-image-prep-v1"
SERPAPI_VISUAL_CACHE_NAMESPACE = "visual/serpapi_google_lens"
SUPPORTED_FORMATS = {"JPEG": ("jpg", "image/jpeg"), "PNG": ("png", "image/png"), "WEBP": ("webp", "image/webp")}


@dataclass(frozen=True)
class PreparedSerpApiImage:
    data: bytes
    filename: str
    mime_type: str
    original_image_sha256: str
    uploaded_image_sha256: str
    original_size: int
    uploaded_size: int
    original_dimensions: tuple[int, int]
    uploaded_dimensions: tuple[int, int]
    compression_applied: bool
    preparation_version: str = SERPAPI_IMAGE_PREPARATION_VERSION

    def provenance(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("data")
        value["original_dimensions"] = list(self.original_dimensions)
        value["uploaded_dimensions"] = list(self.uploaded_dimensions)
        return value


def _source_image(image: Any) -> tuple[bytes, Image.Image, str | None]:
    if isinstance(image, (bytes, bytearray)):
        source = bytes(image)
        opened = Image.open(BytesIO(source))
        original_format = str(opened.format or "").upper() or None
        return source, ImageOps.exif_transpose(opened).copy(), original_format
    if hasattr(image, "save"):
        copied = ImageOps.exif_transpose(image).copy()
        original_format = str(getattr(image, "format", "") or "").upper() or None
        buffer = BytesIO()
        save_format = original_format if original_format in SUPPORTED_FORMATS else "PNG"
        copied.save(buffer, format=save_format)
        return buffer.getvalue(), copied, save_format
    raise SearchBackendError("MODEL_ERROR", "SerpApi Lens requires image pixels")


def prepare_serpapi_image(
    image: Any,
    *,
    original_image_sha256: str | None = None,
    max_bytes: int = 500_000,
) -> PreparedSerpApiImage:
    source, pil_image, source_format = _source_image(image)
    original_dimensions = tuple(int(value) for value in pil_image.size)
    original_digest = str(original_image_sha256 or hashlib.sha256(source).hexdigest())
    if source_format in SUPPORTED_FORMATS and len(source) <= int(max_bytes):
        extension, mime = SUPPORTED_FORMATS[source_format]
        uploaded = source
        dimensions = original_dimensions
        compressed = False
    else:
        base = pil_image.convert("RGBA" if "A" in pil_image.getbands() else "RGB")
        resampling = getattr(Image, "Resampling", Image).LANCZOS
        uploaded = b""
        dimensions = original_dimensions
        for scale in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1):
            width = max(1, round(original_dimensions[0] * scale))
            height = max(1, round(original_dimensions[1] * scale))
            candidate = base if (width, height) == original_dimensions else base.resize((width, height), resampling)
            for quality in (92, 86, 80, 74, 68, 60):
                buffer = BytesIO()
                candidate.save(buffer, format="WEBP", quality=quality, method=6)
                value = buffer.getvalue()
                if len(value) <= int(max_bytes):
                    uploaded = value
                    dimensions = (width, height)
                    break
            if uploaded:
                break
        if not uploaded:
            raise SearchBackendError("IMAGE_PREPARATION_ERROR", "image cannot be reduced below SerpApi upload limit")
        extension, mime, compressed = "webp", "image/webp", True
    return PreparedSerpApiImage(
        data=uploaded,
        filename="visual-search.%s" % extension,
        mime_type=mime,
        original_image_sha256=original_digest,
        uploaded_image_sha256=hashlib.sha256(uploaded).hexdigest(),
        original_size=len(source),
        uploaded_size=len(uploaded),
        original_dimensions=original_dimensions,
        uploaded_dimensions=dimensions,
        compression_applied=compressed,
    )


class SerpApiGoogleLensClient:
    def __init__(self, *, api_key: str | None = None, timeout_seconds: float = 30.0, session=None) -> None:
        self.api_key = api_key if api_key is not None else os.environ.get("SERPAPI_API_KEY", "")
        self.timeout_seconds = float(timeout_seconds)
        self.session = session or requests.Session()

    def validate_ready(self) -> None:
        if not self.api_key:
            raise SearchBackendError("ONLINE_CREDENTIAL_MISSING", "SERPAPI_API_KEY is not configured")

    def upload_image(self, prepared: PreparedSerpApiImage) -> str:
        self.validate_ready()
        try:
            response = self.session.post(
                "https://serpapi.com/image",
                data={"api_key": self.api_key},
                files={"image": (prepared.filename, prepared.data, prepared.mime_type)},
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            value = response.json()
        except requests.Timeout as exc:
            raise SearchBackendError("TIMEOUT", "SerpApi image upload timed out") from exc
        except (requests.RequestException, ValueError) as exc:
            raise SearchBackendError("SEARCH_API_ERROR", "SerpApi image upload failed") from exc
        if value.get("error") or not value.get("image_id"):
            raise SearchBackendError("SEARCH_API_ERROR", "SerpApi image upload returned an error")
        return str(value["image_id"])

    def search_lens(self, image_id: str, *, auto_crop: bool, locale: str, country: str, safe: str) -> dict[str, Any]:
        self.validate_ready()
        language = str(locale).partition("-")[0] or "en"
        params = {
            "engine": "google_lens",
            "image_id": image_id,
            "auto_crop": "true" if auto_crop else "false",
            "hl": language,
            "country": str(country),
            "safe": str(safe),
            "no_cache": "false",
            "output": "json",
            "api_key": self.api_key,
        }
        try:
            response = self.session.get("https://serpapi.com/search.json", params=params, timeout=self.timeout_seconds)
            response.raise_for_status()
            value = response.json()
        except requests.Timeout as exc:
            raise SearchBackendError("TIMEOUT", "SerpApi Google Lens timed out") from exc
        except (requests.RequestException, ValueError) as exc:
            raise SearchBackendError("SEARCH_API_ERROR", "SerpApi Google Lens request failed") from exc
        if value.get("error"):
            raise SearchBackendError("SEARCH_API_ERROR", "SerpApi Google Lens returned an error")
        return dict(value)


class SerpApiGoogleLensVisualSearchBackend(VisualSearchBackend):
    def __init__(
        self,
        *,
        client: Any,
        page_reader: Any,
        cache: JsonCache,
        budget: ApiBudgetGuard | None = None,
        statistics: CostStatistics | None = None,
        auto_crop: bool = False,
        locale: str = "en-US",
        country: str = "us",
        safe: str = "active",
        visual_match_top_k: int = 5,
        related_content_top_k: int = 2,
        result_top_k: int = 5,
        record_max_chars: int = 512,
        read_matching_pages: int = 2,
        max_upload_bytes: int = 500_000,
    ) -> None:
        if auto_crop:
            raise ValueError("V1 whole-image semantics require auto_crop=false")
        self.client = client
        self.page_reader = page_reader
        self.cache = cache
        self.budget = budget
        self.statistics = statistics
        self.auto_crop = False
        self.locale = str(locale)
        self.country = str(country)
        self.safe = str(safe)
        self.visual_match_top_k = int(visual_match_top_k)
        self.related_content_top_k = int(related_content_top_k)
        self.result_top_k = int(result_top_k)
        self.record_max_chars = int(record_max_chars)
        self.read_matching_pages = int(read_matching_pages)
        self.max_upload_bytes = int(max_upload_bytes)
        self.parameters = {
            "auto_crop": False,
            "locale": self.locale,
            "country": self.country,
            "safe": self.safe,
            "visual_match_top_k": self.visual_match_top_k,
            "related_content_top_k": self.related_content_top_k,
            "result_top_k": self.result_top_k,
            "record_max_chars": self.record_max_chars,
            "read_matching_pages": self.read_matching_pages,
            "max_upload_bytes": self.max_upload_bytes,
            "image_preparation_version": SERPAPI_IMAGE_PREPARATION_VERSION,
            "region_aware": False,
            "retrieved_images_injected": False,
        }

    def cache_key(self, image_sha256: str) -> str:
        return visual_cache_key("serpapi_google_lens", image_sha256, self.parameters, SERPAPI_LENS_BACKEND_VERSION)

    def search(self, image: Any, episode_context: EpisodeContext) -> SearchResult:
        self.last_cache_hit = False
        image_sha256 = episode_context.image_sha256
        if not image_sha256:
            image_sha256 = hashlib.sha256(_source_image(image)[0]).hexdigest()
        key = self.cache_key(image_sha256)
        cached = self.cache.get_search_result(SERPAPI_VISUAL_CACHE_NAMESPACE, key)
        if cached is not None:
            self.last_cache_hit = True
            if self.statistics is not None:
                self.statistics.increment("visual_cache_hits")
            return cached
        prepared = prepare_serpapi_image(image, original_image_sha256=image_sha256, max_bytes=self.max_upload_bytes)
        validator = getattr(self.client, "validate_ready", None)
        if validator is not None:
            validator()
        if self.budget is not None:
            self.budget.reserve("serpapi_lens")
        if self.statistics is not None:
            self.statistics.increment("serpapi_visual_transactions")
            self.statistics.increment("serpapi_image_upload_requests")
        started = time.perf_counter()
        image_id = self.client.upload_image(prepared)
        if self.statistics is not None:
            self.statistics.increment("serpapi_lens_search_requests")
        raw = self.client.search_lens(
            image_id, auto_crop=False, locale=self.locale, country=self.country, safe=self.safe
        )
        matches: list[dict[str, Any]] = []
        seen_urls: set[str] = set()
        for item in raw.get("visual_matches", []):
            title = clean_text(str(item.get("title", "")))
            link = str(item.get("link", "")).strip()
            if not title or not link:
                continue
            try:
                link = canonical_url(link)
            except SearchBackendError:
                continue
            if link in seen_urls:
                continue
            seen_urls.add(link)
            matches.append({
                "position": int(item.get("position", len(matches) + 1)),
                "title": title,
                "link": link,
                "source": clean_text(str(item.get("source", ""))),
                "thumbnail": str(item.get("thumbnail", "")),
                "thumbnail_width": int(item.get("thumbnail_width", 0) or 0),
                "thumbnail_height": int(item.get("thumbnail_height", 0) or 0),
                "exact_matches": bool(item.get("exact_matches", False)),
            })
            if len(matches) >= self.visual_match_top_k:
                break
        related = []
        for item in raw.get("related_content", [])[: self.related_content_top_k]:
            query = clean_text(str(item.get("query", "")))
            if query:
                related.append({"query": query, "link": str(item.get("link", "")), "thumbnail": str(item.get("thumbnail", ""))})
        page_content: dict[str, tuple[str, str, str, bool, str | None]] = {}
        for item in matches[: self.read_matching_pages]:
            url = item["link"]
            try:
                page = self.page_reader.read(url)
                page_content[url] = (
                    truncate_record(page.clean_text, self.record_max_chars),
                    page.fetch_status,
                    str(page.metadata.get("reader_provider", page.metadata.get("provider", "unknown"))),
                    bool(page.metadata.get("reader_fallback", False)),
                    page.metadata.get("reader_fallback_reason"),
                )
            except SearchBackendError as exc:
                page_content[url] = ("", exc.code, "none", False, exc.code)
        records: list[SearchRecord] = []
        for item in matches:
            content, status, reader_provider, reader_fallback, fallback_reason = page_content.get(
                item["link"], ("", "not_fetched", "none", False, None)
            )
            records.append(SearchRecord(
                rank=len(records) + 1,
                title=item["title"],
                url=item["link"],
                snippet=item["source"],
                content=content,
                source="serpapi_google_lens_visual_match",
                content_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
                metadata={
                    "provider_position": item["position"],
                    "thumbnail": item["thumbnail"],
                    "thumbnail_width": item["thumbnail_width"],
                    "thumbnail_height": item["thumbnail_height"],
                    "exact_matches": item["exact_matches"],
                    "page_fetch_status": status,
                    "reader_provider": reader_provider,
                    "reader_fallback": reader_fallback,
                    "reader_fallback_reason": fallback_reason,
                },
            ))
        for item in related:
            if len(records) >= self.result_top_k:
                break
            records.append(SearchRecord(
                rank=len(records) + 1,
                title=item["query"],
                url=item["link"],
                snippet="related visual query",
                source="serpapi_google_lens_related_content",
                metadata={"thumbnail": item["thumbnail"]},
            ))
        records = records[: self.result_top_k]
        if not records:
            raise SearchBackendError("EMPTY_RETRIEVAL", "SerpApi Google Lens returned no usable results")
        formatted = [
            truncate_record(
                "entity/title: %s%s%s" % (
                    record.title,
                    " | url: %s" % record.url if record.url else "",
                    " | content: %s" % record.content if record.content else "",
                ),
                self.record_max_chars,
            )
            for record in records
        ]
        preparation = prepared.provenance()
        result = SearchResult(
            tool_type="visual_search",
            backend="serpapi_google_lens",
            request={"image_sha256": image_sha256},
            timestamp=utc_now(),
            records=tuple(records),
            information_text=format_frozen_information("Image Search", formatted),
            metadata={
                "provider_parameters": self.parameters,
                "provider_request": {"auto_crop": False, "locale": self.locale, "country": self.country, "safe": self.safe},
                "backend_version": SERPAPI_LENS_BACKEND_VERSION,
                "latency_seconds": time.perf_counter() - started,
                "search_api_success": True,
                **preparation,
                "visual_matches": matches,
                "related_content": related,
                "selected_pages": [item["link"] for item in matches[: self.read_matching_pages]],
                "page_fetch_success_count": sum(value[1] == "success" for value in page_content.values()),
                "page_fetch_failure_count": sum(value[1] != "success" for value in page_content.values()),
                "local_page_read_count": sum(value[2] == "local" for value in page_content.values()),
                "jina_fallback_read_count": sum(value[2] == "jina" and value[3] for value in page_content.values()),
                "image_id_retained": False,
                "cache_key": key,
                "cache_namespace": SERPAPI_VISUAL_CACHE_NAMESPACE,
            },
        )
        self.cache.put_search_result(SERPAPI_VISUAL_CACHE_NAMESPACE, key, result)
        return result
