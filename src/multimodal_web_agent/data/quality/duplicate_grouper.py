from __future__ import annotations

import hashlib
import io
import re
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from PIL import Image

from .generic_terms import normalize_text


TITLE_SPLIT_RE = re.compile(r"\s+(?:[-–—|:])\s+")


def stable_group_id(prefix: str, value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]
    return "%s:%s" % (prefix, digest)


def canonical_entity_from_titles(
    titles: Sequence[str],
    *,
    question: str = "",
) -> str:
    for title in titles:
        head = TITLE_SPLIT_RE.split(str(title).strip(), maxsplit=1)[0]
        normalized = normalize_text(head)
        if len(normalized) >= 3:
            return normalized
    normalized_question = normalize_text(question)
    return normalized_question or "unknown"


def entity_group_id(
    *,
    titles: Sequence[str] = (),
    question: str = "",
    explicit_entity: str = "",
) -> str:
    entity = (
        normalize_text(explicit_entity)
        or canonical_entity_from_titles(titles, question=question)
    )
    return stable_group_id("entity", entity)


def image_sha256(image_bytes: bytes) -> str:
    return hashlib.sha256(image_bytes).hexdigest()


def image_dhash(image_bytes: bytes, hash_size: int = 8) -> str:
    with Image.open(io.BytesIO(image_bytes)) as image:
        grayscale = image.convert("L").resize(
            (hash_size + 1, hash_size), Image.Resampling.LANCZOS
        )
        pixels = list(grayscale.getdata())
    bits = []
    for row in range(hash_size):
        offset = row * (hash_size + 1)
        bits.extend(
            pixels[offset + column] > pixels[offset + column + 1]
            for column in range(hash_size)
        )
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return ("%0*x" % ((hash_size * hash_size + 3) // 4, value))


def near_duplicate_group_id(
    *,
    source_hash: str = "",
    dhash: str = "",
    source_data_id: str = "",
) -> str:
    key = source_hash or ("dhash:%s" % dhash if dhash else source_data_id)
    return stable_group_id("near_duplicate", key)


def build_image_fingerprint_index(source_path: Path) -> Dict[str, Dict[str, Any]]:
    """Read only image/data_id columns; JSON fixtures may embed image bytes."""
    source_path = Path(source_path)
    result: Dict[str, Dict[str, Any]] = {}
    if source_path.suffix.casefold() in {".parquet", ".pq"}:
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("pyarrow is required for parquet fingerprints") from exc
        parquet = pq.ParquetFile(source_path)
        row_index = 0
        for batch in parquet.iter_batches(
            batch_size=128, columns=["data_id", "images"]
        ):
            for row in batch.to_pylist():
                data_id = str(row.get("data_id", ""))
                images = row.get("images") or ()
                raw = (
                    images[0].get("bytes")
                    if images and isinstance(images[0], Mapping)
                    else None
                )
                if raw:
                    with Image.open(io.BytesIO(raw)) as image:
                        dimensions = [int(image.width), int(image.height)]
                    source_hash = image_sha256(raw)
                    dhash = image_dhash(raw)
                    result[data_id] = {
                        "source_hash": source_hash,
                        "image_dhash": dhash,
                        "image_dimensions": dimensions,
                        "near_duplicate_group_id": near_duplicate_group_id(
                            source_hash=source_hash,
                            dhash=dhash,
                            source_data_id=data_id,
                        ),
                    }
                row_index += 1
        return result
    return result


def split_group_key(
    *,
    entity_group: str,
    near_duplicate_group: str,
) -> str:
    return "%s|%s" % (entity_group, near_duplicate_group)
