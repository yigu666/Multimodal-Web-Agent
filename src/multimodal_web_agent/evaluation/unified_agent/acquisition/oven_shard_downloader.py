from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import time
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urlparse

from .errors import (
    InsufficientDiskError,
    SourceInvalidError,
    UpstreamUnavailableError,
)
from .oven_access import OvenAccess
from .provenance import sha256_file


def _is_repository_access_error(
    exc: Exception,
    endpoints: Sequence[str],
) -> bool:
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    if status not in {401, 403}:
        return False
    response_url = str(getattr(response, "url", "") or "")
    if not response_url:
        return True
    response_host = (urlparse(response_url).hostname or "").casefold()
    endpoint_hosts = {
        (urlparse(endpoint).hostname or "").casefold()
        for endpoint in endpoints
    }
    # A 401/403 from the repository API is an authorization failure. A
    # 401/403 from a signed CAS/Xet/CDN URL is normally an expired transfer
    # URL after a long interrupted download; a fresh hf_hub_download call
    # obtains a new signed URL and resumes the same .incomplete file.
    return response_host in endpoint_hosts


def check_disk_budget(
    target_root: Path,
    files: Sequence[Mapping[str, Any]],
    *,
    maximum_download_bytes: int,
    minimum_free_disk_bytes_after_download: int,
    disk_usage: Callable[[str | os.PathLike[str]], Any] = shutil.disk_usage,
) -> int:
    missing_sizes = [
        str(row.get("path") or "<unknown>")
        for row in files
        if int(row.get("size") or 0) <= 0
    ]
    if missing_sizes:
        raise SourceInvalidError(
            "OVEN shard size metadata is missing: %s"
            % ", ".join(missing_sizes)
        )
    required = sum(int(row["size"]) for row in files)
    if required > int(maximum_download_bytes):
        raise InsufficientDiskError(
            "INSUFFICIENT_DISK_FOR_INFOSEEK: planned OVEN shards exceed "
            "maximum_download_bytes"
        )
    Path(target_root).mkdir(parents=True, exist_ok=True)
    free = int(disk_usage(target_root).free)
    if free - required < int(minimum_free_disk_bytes_after_download):
        raise InsufficientDiskError(
            "INSUFFICIENT_DISK_FOR_INFOSEEK: free disk safety floor would "
            "be violated"
        )
    return required


def download_oven_files(
    access: OvenAccess,
    filenames: Sequence[str],
    target_root: Path,
    *,
    token: str,
    retries: int = 1,
    retry_backoff_seconds: float = 1.0,
    download: Callable[..., str] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> list[dict[str, Any]]:
    if download is None:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise UpstreamUnavailableError(
                "huggingface_hub is required for OVEN downloads"
            ) from exc
        download = hf_hub_download
    inventory = {row["path"]: row for row in access.files}
    missing = sorted(set(filenames) - set(inventory))
    if missing:
        raise SourceInvalidError(
            "OVEN required files missing at revision %s: %s"
            % (access.revision, ", ".join(missing))
        )
    target_root = Path(target_root)
    target_root.mkdir(parents=True, exist_ok=True)
    reports = []
    for filename in filenames:
        target = target_root / filename
        metadata_path = target.with_name(target.name + ".hf.json")
        expected = inventory[filename]
        expected_sha256 = expected.get("lfs_sha256")
        expected_size = int(expected.get("size") or 0)
        if target.is_file() and metadata_path.is_file():
            metadata = json.loads(
                metadata_path.read_text(encoding="utf-8")
            )
            digest = sha256_file(target)
            if (
                metadata.get("sha256") != digest
                or metadata.get("revision") != access.revision
            ):
                raise SourceInvalidError(
                    "SOURCE_CACHE_CONFLICT: %s" % target
                )
            reports.append({
                **metadata,
                "status": "SKIPPED_ALREADY_VERIFIED",
            })
            continue
        if target.is_file():
            digest = sha256_file(target)
            if (
                not expected_sha256
                or digest != expected_sha256
                or (
                    expected_size > 0
                    and target.stat().st_size != expected_size
                )
            ):
                raise SourceInvalidError(
                    "SOURCE_CACHE_CONFLICT: unverified existing OVEN "
                    "file %s" % target
                )
            metadata = {
                "dataset_id": access.dataset_id,
                "endpoint": access.endpoint,
                "revision": access.revision,
                "filename": filename,
                "size": target.stat().st_size,
                "sha256": digest,
                "downloaded_at": None,
                "cache_metadata_reconstructed": True,
            }
            metadata_path.write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            reports.append({
                **metadata,
                "status": "SKIPPED_ALREADY_VERIFIED",
            })
            continue
        downloaded = None
        access_errors = []
        upstream_errors = []
        used_endpoint = None
        attempt_count = 0
        for attempt in range(max(1, int(retries))):
            round_access_errors = 0
            for endpoint in access.endpoints:
                attempt_count += 1
                try:
                    downloaded = Path(download(
                        repo_id=access.dataset_id,
                        repo_type="dataset",
                        filename=filename,
                        revision=access.revision,
                        endpoint=endpoint,
                        token=token,
                        local_dir=str(target_root),
                        force_download=False,
                    ))
                    used_endpoint = endpoint
                    break
                except Exception as exc:
                    if _is_repository_access_error(
                        exc, access.endpoints
                    ):
                        access_errors.append(exc)
                        round_access_errors += 1
                    else:
                        upstream_errors.append(exc)
            if downloaded is not None:
                break
            if (
                round_access_errors == len(access.endpoints)
                and not upstream_errors
            ):
                break
            if attempt + 1 < max(1, int(retries)):
                sleep(min(
                    60.0,
                    float(retry_backoff_seconds) * float(2 ** attempt),
                ))
        if downloaded is None:
            if access_errors and not upstream_errors:
                from .errors import OvenAccessRequiredError
                raise OvenAccessRequiredError(
                    "OVEN_ACCESS_REQUIRED during file download"
                ) from access_errors[-1]
            cause = (
                upstream_errors[-1]
                if upstream_errors else access_errors[-1]
            )
            raise UpstreamUnavailableError(
                "OVEN file download failed at all configured endpoints: %s"
                % filename
            ) from cause
        if downloaded.resolve() != target.resolve():
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                raise SourceInvalidError(
                    "SOURCE_CACHE_CONFLICT: %s" % target
                )
            shutil.copy2(downloaded, target)
        digest = sha256_file(target)
        if expected_size > 0 and target.stat().st_size != expected_size:
            raise SourceInvalidError(
                "OVEN downloaded file size mismatch: %s" % filename
            )
        if expected_sha256 and digest != expected_sha256:
            raise SourceInvalidError(
                "OVEN downloaded file hash mismatch: %s" % filename
            )
        metadata = {
            "dataset_id": access.dataset_id,
            "endpoint": used_endpoint,
            "revision": access.revision,
            "filename": filename,
            "size": target.stat().st_size,
            "sha256": digest,
            "downloaded_at": int(time.time()),
            "download_attempts": attempt_count,
            "status": "DOWNLOADED_AND_VERIFIED",
        }
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        reports.append(metadata)
    return reports
