from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


SEARCH_RESULT_SCHEMA = "online-web-search-result-v1"


@dataclass(frozen=True)
class SearchRecord:
    rank: int
    title: str = ""
    url: str = ""
    snippet: str = ""
    content: str = ""
    source: str = ""
    content_sha256: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SearchRecord":
        return cls(
            rank=int(value["rank"]),
            title=str(value.get("title", "")),
            url=str(value.get("url", "")),
            snippet=str(value.get("snippet", "")),
            content=str(value.get("content", "")),
            source=str(value.get("source", "")),
            content_sha256=str(value.get("content_sha256", "")),
            metadata=dict(value.get("metadata", {})),
        )


@dataclass(frozen=True)
class SearchResult:
    tool_type: str
    backend: str
    request: dict[str, Any]
    timestamp: str
    records: tuple[SearchRecord, ...]
    information_text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SEARCH_RESULT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != SEARCH_RESULT_SCHEMA:
            raise ValueError("SearchResult schema mismatch")
        if self.tool_type not in {"text_search", "visual_search"}:
            raise ValueError("unsupported search tool type")
        if not self.backend:
            raise ValueError("search backend is required")
        if not self.information_text.startswith("<information>\n"):
            raise ValueError("SearchResult information must use the frozen XML wrapper")
        if not self.information_text.endswith("\n</information>"):
            raise ValueError("SearchResult information XML is incomplete")
        ranks = [record.rank for record in self.records]
        if ranks != list(range(1, len(ranks) + 1)):
            raise ValueError("SearchResult ranks must be consecutive")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["records"] = [record.to_dict() for record in self.records]
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SearchResult":
        return cls(
            tool_type=str(value["tool_type"]),
            backend=str(value["backend"]),
            request=dict(value.get("request", {})),
            timestamp=str(value.get("timestamp", "")),
            records=tuple(
                SearchRecord.from_dict(record)
                for record in value.get("records", [])
            ),
            information_text=str(value["information_text"]),
            metadata=dict(value.get("metadata", {})),
            schema_version=str(value.get("schema_version", SEARCH_RESULT_SCHEMA)),
        )


class SearchBackendError(LookupError):
    """A categorized provider error that the existing runner handles safely."""

    def __init__(self, code: str, message: str, *, metadata: Mapping[str, Any] | None = None):
        super().__init__("%s: %s" % (code, message))
        self.code = str(code)
        self.metadata = dict(metadata or {})

