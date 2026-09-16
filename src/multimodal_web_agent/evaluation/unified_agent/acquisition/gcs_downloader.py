from __future__ import annotations

import json
import os
from pathlib import Path
import time
from typing import Any, Callable, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .errors import SourceInvalidError, UpstreamUnavailableError
from .provenance import sha256_file


def _metadata_path(target: Path) -> Path:
    return target.with_name(target.name + ".download.json")


def _set_read_timeout(response: Any, seconds: int) -> None:
    """Best-effort switch from the connect timeout to the read timeout."""
    candidates = (
        getattr(response, "fp", None),
        getattr(getattr(response, "fp", None), "raw", None),
    )
    for candidate in candidates:
        sock = getattr(candidate, "_sock", None)
        if sock is not None and hasattr(sock, "settimeout"):
            sock.settimeout(seconds)
            return


def _verified_existing(
    target: Path,
    *,
    url: str,
    expected_sha256: str | None,
) -> dict[str, Any] | None:
    if not target.is_file():
        return None
    digest = sha256_file(target)
    if expected_sha256 and digest != expected_sha256:
        raise SourceInvalidError(
            "SOURCE_CACHE_CONFLICT: %s" % target
        )
    metadata_path = _metadata_path(target)
    if metadata_path.is_file():
        metadata = json.loads(
            metadata_path.read_text(encoding="utf-8")
        )
        if metadata.get("official_resource_url") != url:
            raise SourceInvalidError(
                "SOURCE_CACHE_CONFLICT: URL changed for %s" % target
            )
        if metadata.get("downloaded_sha256") != digest:
            raise SourceInvalidError(
                "SOURCE_CACHE_CONFLICT: hash changed for %s" % target
            )
    else:
        metadata = {
            "official_resource_url": url,
            "resource_filename": target.name,
            "downloaded_sha256": digest,
            "downloaded_size": target.stat().st_size,
            "downloaded_at": None,
            "cache_metadata_reconstructed": True,
        }
        metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return {**metadata, "status": "SKIPPED_ALREADY_VERIFIED"}


def download_gcs_resource(
    resource: Mapping[str, Any],
    target: Path,
    *,
    retries: int = 5,
    connect_timeout_seconds: int = 30,
    read_timeout_seconds: int = 300,
    resume: bool = True,
    opener: Callable[..., Any] = urlopen,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Download one registry-pinned official resource atomically."""
    url = str(resource["official_resource_url"])
    expected_sha256 = resource.get("expected_sha256")
    expected_type = str(resource.get("expected_content_type") or "")
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    reused = _verified_existing(
        target, url=url, expected_sha256=expected_sha256
    )
    if reused:
        return reused
    partial = target.with_name(target.name + ".partial")
    last_error: Exception | None = None
    for attempt in range(max(1, retries)):
        offset = partial.stat().st_size if resume and partial.is_file() else 0
        headers = {
            "User-Agent": "multimodal-web-agent-infoseek/1",
        }
        if offset:
            headers["Range"] = "bytes=%d-" % offset
        try:
            response = opener(
                Request(url, headers=headers),
                timeout=connect_timeout_seconds,
            )
            with response:
                _set_read_timeout(response, read_timeout_seconds)
                status = int(getattr(response, "status", 200) or 200)
                append = offset > 0 and status == 206
                mode = "ab" if append else "wb"
                if not append:
                    offset = 0
                content_type = str(
                    response.headers.get("Content-Type") or ""
                ).split(";", 1)[0].strip()
                if expected_type and content_type and (
                    content_type != expected_type
                ):
                    raise SourceInvalidError(
                        "unexpected content type for %s: %s"
                        % (target.name, content_type)
                    )
                expected_remaining = response.headers.get("Content-Length")
                written = 0
                with partial.open(mode) as handle:
                    while True:
                        chunk = response.read(4 * 1024 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk)
                        written += len(chunk)
                    handle.flush()
                    os.fsync(handle.fileno())
                if (
                    expected_remaining is not None
                    and written != int(expected_remaining)
                ):
                    raise UpstreamUnavailableError(
                        "incomplete download for %s" % target.name
                    )
                metadata = {
                    "official_resource_url": url,
                    "resource_filename": target.name,
                    "expected_content_type": expected_type or None,
                    "actual_content_type": content_type or None,
                    "http_status": status,
                    "content_length": (
                        int(expected_remaining)
                        if expected_remaining is not None else None
                    ),
                    "etag": response.headers.get("ETag"),
                    "last_modified": response.headers.get("Last-Modified"),
                    "resumed_from_bytes": offset,
                }
            digest = sha256_file(partial)
            if expected_sha256 and digest != expected_sha256:
                raise SourceInvalidError(
                    "download hash mismatch for %s" % target.name
                )
            os.replace(partial, target)
            metadata.update({
                "downloaded_sha256": digest,
                "downloaded_size": target.stat().st_size,
                "downloaded_at": int(time.time()),
                "status": "DOWNLOADED_AND_VERIFIED",
            })
            _metadata_path(target).write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            return metadata
        except SourceInvalidError:
            raise
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            last_error = exc
            if attempt + 1 < max(1, retries):
                sleep(min(60.0, float(2 ** attempt)))
    raise UpstreamUnavailableError(
        "official GCS resource unavailable after retries: %s (%r)"
        % (url, last_error)
    )


def probe_gcs_resource(
    resource: Mapping[str, Any],
    *,
    timeout_seconds: int = 30,
    opener: Callable[..., Any] = urlopen,
) -> dict[str, Any]:
    url = str(resource["official_resource_url"])
    try:
        response = opener(
            Request(url, method="HEAD"), timeout=timeout_seconds
        )
        with response:
            return {
                "official_resource_url": url,
                "available": True,
                "http_status": int(
                    getattr(response, "status", 200) or 200
                ),
                "content_length": (
                    int(response.headers["Content-Length"])
                    if response.headers.get("Content-Length") else None
                ),
                "content_type": response.headers.get("Content-Type"),
                "etag": response.headers.get("ETag"),
                "last_modified": response.headers.get("Last-Modified"),
            }
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        return {
            "official_resource_url": url,
            "available": False,
            "error_type": type(exc).__name__,
        }
