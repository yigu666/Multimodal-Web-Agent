from __future__ import annotations

from io import BytesIO
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image

from .provenance import sha256_file


IMAGE_FIELDS = (
    "query_image",
    "input_image",
    "image_path",
    "image",
)
FORBIDDEN_ROLE_TOKENS = (
    "retrieval", "result_image", "thumbnail", "webpage", "supporting"
)


def _validate_image(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError("query image has invalid dimensions")
    return width, height


def resolve_query_image(
    row: Mapping[str, Any],
    *,
    source_root: Path,
    image_files: Sequence[Path],
) -> dict[str, Any]:
    value = None
    role_source = None
    for field in IMAGE_FIELDS:
        if row.get(field) not in (None, ""):
            value = row[field]
            role_source = "dataset_annotation:%s" % field
            break
    if value is None and row.get("image_id") not in (None, ""):
        image_id = str(row["image_id"])
        matches = [
            path for path in image_files if path.stem == image_id
        ]
        if len(matches) == 1:
            value = str(matches[0])
            role_source = "official_image_id"
    if value is None:
        raise ValueError("query_image_role_unverified")
    if isinstance(value, Mapping):
        path_value = value.get("path")
        raw = value.get("bytes")
        if raw:
            content = bytes(raw)
            with Image.open(BytesIO(content)) as image:
                image.verify()
            with Image.open(BytesIO(content)) as image:
                width, height = image.size
                image_format = (image.format or "jpeg").casefold()
            if width <= 0 or height <= 0:
                raise ValueError("query image has invalid dimensions")
            extension = Path(str(path_value or "")).suffix.casefold()
            if not extension:
                extension = "." + {
                    "jpeg": "jpg",
                    "tiff": "tif",
                }.get(image_format, image_format)
            return {
                "path": None,
                "bytes": content,
                "extension": extension,
                "query_image_role_verified": True,
                "query_image_role_source": role_source,
                "query_image_sha256": hashlib.sha256(content).hexdigest(),
                "width": width,
                "height": height,
                "retrieval_result_images_excluded_from_input": True,
            }
        value = path_value
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError("query_image_role_unverified")
    value_text = str(value)
    if any(
        token in value_text.casefold() for token in FORBIDDEN_ROLE_TOKENS
    ):
        raise ValueError("query_image_role_unverified")
    raw_path = Path(value_text)
    candidates = (
        [raw_path] if raw_path.is_absolute()
        else [
            source_root / raw_path,
            *[
                path for path in image_files
                if path.name == raw_path.name
            ],
        ]
    )
    existing = []
    seen = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved.is_file() and resolved not in seen:
            existing.append(resolved)
            seen.add(resolved)
    if len(existing) != 1:
        raise ValueError("query_image_role_unverified")
    path = existing[0]
    width, height = _validate_image(path)
    return {
        "path": path,
        "bytes": None,
        "extension": path.suffix.casefold() or ".jpg",
        "query_image_role_verified": True,
        "query_image_role_source": role_source,
        "query_image_sha256": sha256_file(path),
        "width": width,
        "height": height,
        "retrieval_result_images_excluded_from_input": True,
    }
