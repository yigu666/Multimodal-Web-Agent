from __future__ import annotations

import json
import os
from pathlib import Path
import time
from typing import Any
from urllib.request import Request, urlopen

from .errors import ArchiveRequiredError, SourceInvalidError
from .provenance import sha256_file
from .schema import AcquisitionSource


def _download_url(
    url: str,
    target: Path,
    *,
    expected_sha256: str | None,
) -> dict[str, Any]:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file():
        digest = sha256_file(target)
        if expected_sha256 and digest != expected_sha256:
            raise SourceInvalidError(
                "existing download hash mismatch: %s" % target
            )
        return {
            "path": str(target),
            "sha256": digest,
            "size": target.stat().st_size,
            "reused": True,
        }
    partial = target.with_name(target.name + ".partial")
    offset = partial.stat().st_size if partial.is_file() else 0
    headers = {"User-Agent": "multimodal-web-agent-acquisition/1"}
    if offset:
        headers["Range"] = "bytes=%d-" % offset
    request = Request(url, headers=headers)
    with urlopen(request, timeout=60) as response:
        status = int(getattr(response, "status", 200) or 200)
        mode = "ab" if offset and status == 206 else "wb"
        if mode == "wb":
            offset = 0
        with partial.open(mode) as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
        metadata = {
            "url": url,
            "http_status": status,
            "etag": response.headers.get("ETag"),
            "last_modified": response.headers.get("Last-Modified"),
            "content_type": response.headers.get("Content-Type"),
            "downloaded_at_unix": int(time.time()),
            "resumed_from_bytes": offset,
        }
    digest = sha256_file(partial)
    if expected_sha256 and digest != expected_sha256:
        raise SourceInvalidError(
            "download hash mismatch for %s" % url
        )
    os.replace(partial, target)
    metadata.update({
        "path": str(target),
        "sha256": digest,
        "size": target.stat().st_size,
        "reused": False,
    })
    return metadata


def download_public_source(
    source: AcquisitionSource,
    downloaded_root: Path,
    log_root: Path,
) -> list[Path]:
    if not source.enabled:
        return []
    output = Path(downloaded_root) / source.name
    logs = []
    paths = []
    if source.download_urls:
        for index, item in enumerate(source.download_urls):
            url = str(item["url"])
            filename = str(
                item.get("filename")
                or Path(url.split("?", 1)[0]).name
                or "download-%d.bin" % index
            )
            record = _download_url(
                url,
                output / filename,
                expected_sha256=item.get("sha256"),
            )
            logs.append(record)
            paths.append(Path(record["path"]))
    elif source.official_dataset_id:
        if source.requires_user_license_acceptance is True:
            raise ArchiveRequiredError(source.name)
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise ArchiveRequiredError(source.name) from exc
        target = output / "huggingface_snapshot"
        endpoint = os.environ.get(
            "HF_ENDPOINT", "https://huggingface.co"
        ).rstrip("/")
        if not endpoint.startswith("https://"):
            raise SourceInvalidError(
                "HF_ENDPOINT must use HTTPS"
            )
        if not target.exists():
            temporary = output / ".huggingface_snapshot.tmp"
            temporary.parent.mkdir(parents=True, exist_ok=True)
            snapshot_download(
                repo_id=source.official_dataset_id,
                repo_type="dataset",
                local_dir=str(temporary),
                endpoint=endpoint,
            )
            os.replace(temporary, target)
        paths.append(target)
        logs.append({
            "dataset_id": source.official_dataset_id,
            "path": str(target),
            "channel": "huggingface_dataset_snapshot",
            "endpoint": endpoint,
        })
    else:
        raise ArchiveRequiredError(source.name)
    log_path = Path(log_root) / ("%s_download.json" % source.name)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        json.dumps(logs, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return paths
