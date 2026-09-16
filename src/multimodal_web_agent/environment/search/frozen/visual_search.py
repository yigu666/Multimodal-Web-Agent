from __future__ import annotations

import hashlib

from ..base import EpisodeContext, VisualSearchBackend
from ..schemas import SearchRecord, SearchResult
from .text_search import _records_from_frozen_information


class FrozenVisualSearchBackend(VisualSearchBackend):
    """Thin adapter: calls the official frozen image-search method verbatim."""

    def __init__(self, legacy_environment):
        self.legacy_environment = legacy_environment

    def search(self, image, episode_context: EpisodeContext) -> SearchResult:
        image_sha256 = episode_context.image_sha256 or str(image)
        information = self.legacy_environment.image_search(image_sha256)
        records = _records_from_frozen_information(information)
        return SearchResult(
            tool_type="visual_search",
            backend="frozen_image_search",
            request={"image_sha256": image_sha256},
            timestamp="",
            records=tuple(
                SearchRecord(
                    rank=index,
                    title=text,
                    source="frozen",
                    content_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                )
                for index, text in enumerate(records, 1)
            ),
            information_text=information,
            metadata={"online_access": False, "legacy_implementation_reused": True},
        )
