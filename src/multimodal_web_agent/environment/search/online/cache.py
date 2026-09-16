from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

from ..schemas import SearchResult
from .security import canonical_url


CACHE_SCHEMA = "online-web-cache-v1"


def stable_digest(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def normalized_query(query: str) -> str:
    return " ".join(str(query).split()).casefold()


def text_cache_key(provider: str, query: str, parameters: Mapping[str, Any], version: str) -> str:
    return stable_digest({
        "provider": provider,
        "normalized_query": normalized_query(query),
        "provider_parameters": dict(parameters),
        "backend_version": version,
    })


def visual_cache_key(provider: str, image_sha256: str, parameters: Mapping[str, Any], version: str) -> str:
    return stable_digest({
        "provider": provider,
        "input_image_sha256": image_sha256,
        "provider_parameters": dict(parameters),
        "backend_version": version,
    })


def page_cache_key(url: str, reader_version: str) -> str:
    return stable_digest({"canonical_url": canonical_url(url), "reader_version": reader_version})


class JsonCache:
    def __init__(self, root: Path, *, enabled: bool = True):
        self.root = Path(root)
        self.enabled = bool(enabled)

    def _path(self, namespace: str, key: str) -> Path:
        return self.root / namespace / (str(key) + ".json")

    def get_json(self, namespace: str, key: str) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        path = self._path(namespace, key)
        if not path.is_file():
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("schema_version") != CACHE_SCHEMA or value.get("key") != key:
            raise RuntimeError("online cache envelope mismatch")
        payload = value["payload"]
        expected = value["payload_sha256"]
        if stable_digest(payload) != expected:
            raise RuntimeError("online cache payload hash mismatch")
        return dict(payload)

    def put_json(self, namespace: str, key: str, payload: Mapping[str, Any]) -> Path | None:
        if not self.enabled:
            return None
        path = self._path(namespace, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        envelope = {
            "schema_version": CACHE_SCHEMA,
            "key": key,
            "payload": dict(payload),
            "payload_sha256": stable_digest(payload),
        }
        data = json.dumps(envelope, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".%s." % path.name, suffix=".tmp", dir=str(path.parent)
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, path)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)
        return path

    def get_search_result(self, namespace: str, key: str) -> SearchResult | None:
        value = self.get_json(namespace, key)
        return SearchResult.from_dict(value) if value is not None else None

    def put_search_result(self, namespace: str, key: str, result: SearchResult) -> Path | None:
        return self.put_json(namespace, key, result.to_dict())

