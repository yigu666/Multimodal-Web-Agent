from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple
from urllib.parse import urlparse


CACHE_VERSION = "lmms-lab/FVQA@bb4a4ff4c9c3fd0382d11f5d7fccd66d0b8428b5"
CACHE_SOURCE = "lmms-lab/FVQA official image search cache"
CLEANER_VERSION = "protocol-sft-v0-sanitize-titles"
TITLE_FIELD = "tool_returned_web_title_list"
IMAGE_FIELD = "tool_returned_images_urls"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _raw_hash(value: Any) -> str:
    serialized = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    return hashlib.sha256(serialized).hexdigest()


def _sequence(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return []


def _thumbnail_descriptor(value: Any) -> Optional[Dict[str, Any]]:
    if isinstance(value, str):
        stripped = value.strip()
        parsed = urlparse(stripped)
        if stripped and parsed.scheme in {"http", "https"} and parsed.netloc:
            return {"kind": "remote_url", "url": stripped}
        return None
    if isinstance(value, (bytes, bytearray)):
        return {"kind": "embedded_cache_bytes"}
    module = type(value).__module__
    if module.startswith("PIL."):
        return {"kind": "embedded_pil_image"}
    return None


@dataclass(frozen=True)
class CacheEntry:
    data_id: str
    titles: List[str]
    thumbnail_descriptors: List[Optional[Dict[str, Any]]]
    raw_result_hash: str
    cache_file_sha256: str
    cache_label: str

    @property
    def usable_titles(self) -> List[Tuple[int, str]]:
        return [(index, title) for index, title in enumerate(self.titles) if title.strip()]

    @property
    def usable_image_results(self) -> List[Tuple[int, str, Dict[str, Any]]]:
        results = []
        for index, title in self.usable_titles:
            descriptor = (
                self.thumbnail_descriptors[index]
                if index < len(self.thumbnail_descriptors)
                else None
            )
            if descriptor is not None:
                results.append((index, title, descriptor))
        return results

    def provenance(self, selected_indices: Iterable[int]) -> Dict[str, Any]:
        return {
            "backend": "fvqa_image_search_cache",
            "cache_source": CACHE_SOURCE,
            "cache_version": CACHE_VERSION,
            "cache_label": self.cache_label,
            "cache_file_sha256": self.cache_file_sha256,
            "raw_result_hash": self.raw_result_hash,
            "cleaner_version": CLEANER_VERSION,
            "source_data_id": self.data_id,
            "selected_result_indices": list(selected_indices),
            "online_access": False,
        }


class CacheMissError(KeyError):
    pass


class ImageSearchCache:
    def __init__(self, entries: Mapping[str, CacheEntry], file_sha256: str, label: str):
        self._entries = dict(entries)
        self.file_sha256 = file_sha256
        self.label = label

    @classmethod
    def load(cls, path: Path, label: Optional[str] = None) -> "ImageSearchCache":
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open("rb") as handle:
            raw_cache = pickle.load(handle)
        if not isinstance(raw_cache, Mapping):
            raise TypeError("image-search cache top level must be a mapping")
        file_hash = sha256_file(path)
        cache_label = label or path.name
        entries: Dict[str, CacheEntry] = {}
        for raw_key, raw_record in raw_cache.items():
            data_id = str(raw_key)
            if not isinstance(raw_record, Mapping):
                continue
            raw_titles = _sequence(raw_record.get(TITLE_FIELD))
            raw_thumbnails = _sequence(raw_record.get(IMAGE_FIELD))
            titles = [str(value).strip() if isinstance(value, str) else "" for value in raw_titles]
            thumbnails = [_thumbnail_descriptor(value) for value in raw_thumbnails]
            entries[data_id] = CacheEntry(
                data_id=data_id,
                titles=titles,
                thumbnail_descriptors=thumbnails,
                raw_result_hash=_raw_hash(raw_record),
                cache_file_sha256=file_hash,
                cache_label=cache_label,
            )
        return cls(entries, file_hash, cache_label)

    def get(self, data_id: str) -> Optional[CacheEntry]:
        return self._entries.get(str(data_id))

    def require(self, data_id: str) -> CacheEntry:
        entry = self.get(data_id)
        if entry is None:
            raise CacheMissError("cache miss for source_data_id=%s" % data_id)
        if not entry.usable_image_results:
            raise CacheMissError(
                "cache entry has no usable title/thumbnail pairs for source_data_id=%s" % data_id
            )
        return entry

    def entries(self) -> List[CacheEntry]:
        return [self._entries[key] for key in sorted(self._entries)]
