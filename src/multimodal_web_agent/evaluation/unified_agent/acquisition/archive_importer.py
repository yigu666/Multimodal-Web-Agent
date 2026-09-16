from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import zipfile

from .errors import SourceInvalidError
from .provenance import sha256_file


ARCHIVE_SUFFIXES = (
    ".zip", ".tar", ".tar.gz", ".tgz", ".tar.xz",
)
DIRECT_SUFFIXES = (
    ".json", ".jsonl", ".parquet", ".arrow", ".csv",
)


def _safe_destination(root: Path, name: str) -> Path:
    pure = PurePosixPath(name.replace("\\", "/"))
    if pure.is_absolute() or ".." in pure.parts:
        raise SourceInvalidError("archive path traversal detected")
    destination = (root / Path(*pure.parts)).resolve()
    try:
        destination.relative_to(root.resolve())
    except ValueError as exc:
        raise SourceInvalidError(
            "archive member escaped extraction root"
        ) from exc
    return destination


def _extract_zip(path: Path, root: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            destination = _safe_destination(root, info.filename)
            if info.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            mode = (info.external_attr >> 16) & 0o170000
            if mode == 0o120000:
                raise SourceInvalidError("archive symlink is forbidden")
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, destination.open("xb") as out:
                shutil.copyfileobj(source, out)


def _extract_tar(path: Path, root: Path) -> None:
    with tarfile.open(path, "r:*") as archive:
        for member in archive.getmembers():
            destination = _safe_destination(root, member.name)
            if member.issym() or member.islnk():
                raise SourceInvalidError("archive links are forbidden")
            if member.isdir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise SourceInvalidError("archive member cannot be read")
            with source, destination.open("xb") as out:
                shutil.copyfileobj(source, out)


def _is_archive(path: Path) -> bool:
    name = path.name.casefold()
    return any(name.endswith(suffix) for suffix in ARCHIVE_SUFFIXES)


def import_archive(
    path: Path,
    *,
    source_name: str,
    extracted_root: Path,
    depth: int = 0,
) -> Path:
    path = Path(path)
    if depth > 3:
        raise SourceInvalidError("nested archive depth exceeds 3")
    digest = sha256_file(path)
    target = Path(extracted_root) / source_name / digest
    if target.is_dir():
        return target
    temporary = target.with_name(".%s.tmp-%d" % (digest, os.getpid()))
    if temporary.exists():
        raise SourceInvalidError("extraction temporary path exists")
    temporary.mkdir(parents=True)
    try:
        if zipfile.is_zipfile(path):
            _extract_zip(path, temporary)
        elif tarfile.is_tarfile(path):
            _extract_tar(path, temporary)
        elif path.suffix.casefold() in DIRECT_SUFFIXES:
            shutil.copy2(path, temporary / path.name)
        else:
            raise SourceInvalidError(
                "unsupported official archive/file: %s" % path.name
            )
        nested = sorted(
            item for item in temporary.rglob("*")
            if item.is_file() and _is_archive(item)
        )
        for item in nested:
            nested_target = temporary / (
                "_nested_%s" % sha256_file(item)[:16]
            )
            nested_target.mkdir()
            if zipfile.is_zipfile(item):
                _extract_zip(item, nested_target)
            elif tarfile.is_tarfile(item):
                _extract_tar(item, nested_target)
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, target)
    except Exception:
        if temporary.is_dir():
            shutil.rmtree(temporary)
        raise
    return target
