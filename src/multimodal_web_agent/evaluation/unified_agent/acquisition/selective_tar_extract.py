from __future__ import annotations

from io import BytesIO
from pathlib import Path, PurePosixPath
import shutil
import tarfile
from typing import Any, Mapping

from PIL import Image

from .errors import SourceInvalidError
from .provenance import sha256_bytes


ALLOWED_IMAGE_SUFFIXES = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"
}


def _safe_name(value: str) -> str:
    pure = PurePosixPath(value.replace("\\", "/"))
    if pure.is_absolute() or ".." in pure.parts:
        raise SourceInvalidError(
            "OVEN shard path traversal detected"
        )
    return pure.as_posix().lstrip("./")


def _verify_image(raw: bytes) -> tuple[int, int, str]:
    try:
        with Image.open(BytesIO(raw)) as image:
            image.verify()
        with Image.open(BytesIO(raw)) as image:
            width, height = image.size
            image_format = (image.format or "jpeg").casefold()
    except OSError as exc:
        raise SourceInvalidError("OVEN image is not decodable") from exc
    if width <= 0 or height <= 0:
        raise SourceInvalidError("OVEN image has invalid dimensions")
    return width, height, image_format


def extract_selected_images(
    shard_path: Path,
    *,
    shard_name: str,
    image_ids: list[str],
    member_by_image_id: Mapping[str, str],
    output_root: Path,
    oven_revision: str,
) -> dict[str, dict[str, Any]]:
    expected = {
        _safe_name(member_by_image_id[image_id]): image_id
        for image_id in image_ids
        if image_id in member_by_image_id
    }
    expected_casefold = {
        member.casefold(): image_id
        for member, image_id in expected.items()
    }
    by_basename: dict[str, list[tuple[str, str]]] = {}
    by_stem: dict[str, list[tuple[str, str]]] = {}
    for member, image_id in expected.items():
        by_basename.setdefault(
            PurePosixPath(member).name.casefold(), []
        ).append((member, image_id))
        by_stem.setdefault(
            PurePosixPath(member).stem.casefold(), []
        ).append((member, image_id))
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    extracted: dict[str, dict[str, Any]] = {}
    with tarfile.open(shard_path, "r|*") as archive:
        for member in archive:
            safe = _safe_name(member.name)
            if member.issym() or member.islnk():
                raise SourceInvalidError(
                    "OVEN shard links are forbidden"
                )
            image_id = expected.get(safe)
            if image_id is None:
                image_id = expected_casefold.get(safe.casefold())
            if image_id is None:
                matches = by_basename.get(
                    PurePosixPath(safe).name.casefold(), ()
                )
                if len(matches) == 1:
                    image_id = matches[0][1]
            if image_id is None:
                # HF snapshot members use paths such as
                # ``06/oven_00790342.JPEG`` rather than the merge layout
                # recorded by ovenid2impath.csv. The OVEN ID stem is stable.
                matches = by_stem.get(
                    PurePosixPath(safe).stem.casefold(), ()
                )
                if len(matches) == 1:
                    image_id = matches[0][1]
            if image_id is None or not member.isfile():
                continue
            stream = archive.extractfile(member)
            if stream is None:
                raise SourceInvalidError(
                    "OVEN target image cannot be read"
                )
            raw = stream.read()
            try:
                width, height, image_format = _verify_image(raw)
            except SourceInvalidError:
                extracted[image_id] = {
                    "query_image_id": image_id,
                    "query_image_role_verified": False,
                    "image_validation_error": (
                        "oven_query_image_corrupt_or_undecodable"
                    ),
                    "oven_repository_revision": oven_revision,
                    "oven_shard": shard_name,
                    "oven_member_path": safe,
                }
                continue
            suffix = PurePosixPath(safe).suffix.casefold()
            if suffix not in ALLOWED_IMAGE_SUFFIXES:
                suffix = "." + {
                    "jpeg": "jpg", "tiff": "tif"
                }.get(image_format, image_format)
            digest = sha256_bytes(raw)
            target = output_root / (digest + suffix)
            if target.is_file():
                if target.read_bytes() != raw:
                    raise SourceInvalidError(
                        "SOURCE_CACHE_CONFLICT: %s" % target
                    )
                status = "SKIPPED_ALREADY_VERIFIED"
            else:
                temporary = target.with_name(target.name + ".partial")
                temporary.write_bytes(raw)
                temporary.replace(target)
                status = "EXTRACTED_AND_VERIFIED"
            extracted[image_id] = {
                "query_image_id": image_id,
                "query_image_path": str(target),
                "query_image_sha256": digest,
                "query_image_role_verified": True,
                "query_image_role_source": (
                    "infoseek_image_id_to_official_oven_snapshot"
                ),
                "oven_repository_revision": oven_revision,
                "oven_shard": shard_name,
                "oven_member_path": safe,
                "width": width,
                "height": height,
                "status": status,
            }
    return extracted
