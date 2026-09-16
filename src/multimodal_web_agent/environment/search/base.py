from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from .schemas import SearchResult


@dataclass
class EpisodeContext:
    episode_id: str
    image_sha256: str = ""
    image: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


class TextSearchBackend(ABC):
    @abstractmethod
    def search(self, query: str, episode_context: EpisodeContext) -> SearchResult:
        raise NotImplementedError


class VisualSearchBackend(ABC):
    @abstractmethod
    def search(self, image: Any, episode_context: EpisodeContext) -> SearchResult:
        raise NotImplementedError


class RegionAwareVisualSearchBackend(VisualSearchBackend):
    """V2 extension point. Online Web Agent V1 never enables regions."""

    enabled = False

