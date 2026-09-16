from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import uuid
from typing import Any, Mapping

from ..schemas import SearchResult


SECRET_NAME = re.compile(r"(api[_-]?key|authorization|credential|token|secret)", re.I)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def redact_secrets(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): ("[REDACTED]" if SECRET_NAME.search(str(key)) else redact_secrets(item))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_secrets(item) for item in value]
    if isinstance(value, tuple):
        return [redact_secrets(item) for item in value]
    return value


class ProvenanceWriter:
    def __init__(self, root: Path, *, enabled: bool = True):
        self.root = Path(root)
        self.enabled = bool(enabled)

    def write(self, episode_id: str, result: SearchResult, event: Mapping[str, Any]) -> Path | None:
        if not self.enabled:
            return None
        safe_episode = re.sub(r"[^A-Za-z0-9_.-]+", "_", episode_id)[:120] or "unknown"
        directory = self.root / safe_episode
        directory.mkdir(parents=True, exist_ok=True)
        timestamp = utc_now()
        payload = {
            "schema_version": "online-web-provenance-v2",
            "episode_id": episode_id,
            "tool": result.tool_type,
            "backend": result.backend,
            "timestamp": result.timestamp or timestamp,
            "request": result.request,
            "provider_parameters": result.metadata.get("provider_parameters", {}),
            "results": [record.to_dict() for record in result.records],
            "information_sha256": hashlib.sha256(
                result.information_text.encode("utf-8")
            ).hexdigest(),
            "event": dict(event),
        }
        if result.tool_type == "visual_search" and result.backend == "serpapi_google_lens":
            payload.update({
                "original_image_sha256": result.metadata.get("original_image_sha256"),
                "uploaded_image_sha256": result.metadata.get("uploaded_image_sha256"),
                "original_size": result.metadata.get("original_size"),
                "uploaded_size": result.metadata.get("uploaded_size"),
                "original_dimensions": result.metadata.get("original_dimensions"),
                "uploaded_dimensions": result.metadata.get("uploaded_dimensions"),
                "compression_applied": result.metadata.get("compression_applied"),
                "provider_request": result.metadata.get("provider_request", {}),
                "visual_matches": result.metadata.get("visual_matches", []),
                "related_content": result.metadata.get("related_content", []),
                "selected_pages": result.metadata.get("selected_pages", []),
                "cache_hit": bool(event.get("cache_hit", False)),
            })
        payload = redact_secrets(payload)
        path = directory / ("%s_%s.json" % (result.tool_type, uuid.uuid4().hex))
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return path
