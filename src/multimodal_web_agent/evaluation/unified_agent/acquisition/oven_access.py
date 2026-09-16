from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Callable, Sequence

from .errors import OvenAccessRequiredError, UpstreamUnavailableError


@dataclass(frozen=True)
class OvenAccess:
    dataset_id: str
    endpoint: str
    endpoints: tuple[str, ...]
    token_source: str
    revision: str
    files: tuple[dict[str, Any], ...]


def discover_hf_token(
    cached_token_getter: Callable[[], str | None] | None = None,
) -> tuple[str, str]:
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = os.environ.get(name)
        if value:
            return value, name
    if cached_token_getter is None:
        try:
            from huggingface_hub import get_token
        except ImportError:
            get_token = None
        cached_token_getter = get_token
    if cached_token_getter is not None:
        value = cached_token_getter()
        if value:
            return value, "huggingface_cached_token"
    raise OvenAccessRequiredError(
        "OVEN_ACCESS_REQUIRED: authorize ychenNLP/oven and configure a "
        "Hugging Face token; no manual data download is required"
    )


def _status_code(exc: Exception) -> int | None:
    response = getattr(exc, "response", None)
    return getattr(response, "status_code", None)


def probe_oven_access(
    *,
    dataset_id: str,
    endpoints: Sequence[str],
    token: str | None = None,
    token_source: str | None = None,
    api_factory: Callable[..., Any] | None = None,
) -> OvenAccess:
    if token is None:
        token, detected_source = discover_hf_token()
        token_source = token_source or detected_source
    if api_factory is None:
        try:
            from huggingface_hub import HfApi
        except ImportError as exc:
            raise UpstreamUnavailableError(
                "huggingface_hub is required for OVEN acquisition"
            ) from exc
        api_factory = HfApi
    configured_endpoints = tuple(dict.fromkeys(
        str(value).rstrip("/") for value in endpoints
    ))
    strict_mode = str(os.environ.get(
        "HF_ENDPOINT_STRICT", ""
    )).casefold() in {"1", "true", "yes", "on"}
    probe_endpoints = configured_endpoints
    if strict_mode:
        strict_endpoint = str(
            os.environ.get("HF_ENDPOINT")
            or configured_endpoints[0]
        ).rstrip("/")
        probe_endpoints = (strict_endpoint,)
    access_errors = []
    upstream_errors = []
    for endpoint in probe_endpoints:
        try:
            api = api_factory(endpoint=endpoint, token=token)
            info = api.dataset_info(
                dataset_id, files_metadata=True, token=token
            )
            siblings = []
            for item in getattr(info, "siblings", ()) or ():
                name = getattr(item, "rfilename", None)
                if not name:
                    continue
                lfs = getattr(item, "lfs", None)
                lfs_sha256 = (
                    getattr(lfs, "sha256", None)
                    if lfs is not None else None
                )
                if isinstance(lfs, dict):
                    lfs_sha256 = lfs.get("sha256")
                siblings.append({
                    "path": str(name),
                    "size": getattr(item, "size", None),
                    "blob_id": getattr(item, "blob_id", None),
                    "lfs_sha256": lfs_sha256,
                })
            download_endpoints = probe_endpoints
            return OvenAccess(
                dataset_id=dataset_id,
                endpoint=endpoint,
                # Preserve configured download priority. In particular, a
                # metadata probe that succeeds only on huggingface.co must
                # never promote that endpoint ahead of hf-mirror.com.
                endpoints=download_endpoints,
                token_source=str(token_source or "provided"),
                revision=str(getattr(info, "sha", "") or "unknown"),
                files=tuple(sorted(
                    siblings, key=lambda row: row["path"]
                )),
            )
        except Exception as exc:
            if _status_code(exc) in {401, 403}:
                access_errors.append(exc)
            else:
                upstream_errors.append(exc)
    if access_errors:
        raise OvenAccessRequiredError(
            "OVEN_ACCESS_REQUIRED: official access is not authorized for "
            "the configured Hugging Face token"
        )
    raise UpstreamUnavailableError(
        "OVEN upstream endpoints are unavailable: %s"
        % ", ".join(probe_endpoints)
    )
