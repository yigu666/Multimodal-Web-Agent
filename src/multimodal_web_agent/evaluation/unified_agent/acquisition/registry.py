from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

from .errors import SourceInvalidError
from .schema import AcquisitionSource, REGISTRY_SCHEMA


def load_registry(path: Path) -> tuple[AcquisitionSource, ...]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or raw.get(
        "schema_version"
    ) != REGISTRY_SCHEMA:
        raise SourceInvalidError("source registry schema mismatch")
    result = []
    for name, value in raw.get("sources", {}).items():
        if not isinstance(value, Mapping):
            raise SourceInvalidError(
                "registry source %s must be a mapping" % name
            )
        urls = value.get("download_urls") or ()
        for item in urls:
            if not isinstance(item, Mapping) or not item.get("url"):
                raise SourceInvalidError(
                    "%s has an invalid trusted download URL" % name
                )
        result.append(AcquisitionSource(
            name=str(name),
            enabled=value.get("enabled") is True,
            acquisition_mode=str(
                value.get("acquisition_mode") or "archive_only"
            ),
            source_plugin=str(
                value.get("source_plugin") or "local_archive"
            ),
            official_dataset_name=(
                str(value["official_dataset_name"])
                if value.get("official_dataset_name") else None
            ),
            official_dataset_id=(
                str(value["official_dataset_id"])
                if value.get("official_dataset_id") else None
            ),
            official_repository=(
                str(value["official_repository"])
                if value.get("official_repository") else None
            ),
            preferred_splits=tuple(
                map(str, value.get("preferred_splits") or ())
            ),
            incoming_archive_globs=tuple(
                map(str, value.get("incoming_archive_globs") or ())
            ),
            allow_network_download=(
                value.get("allow_network_download") is True
            ),
            requires_user_license_acceptance=value.get(
                "requires_user_license_acceptance", "auto_detect"
            ),
            download_urls=tuple(dict(item) for item in urls),
            managed_by=(
                str(value["managed_by"])
                if value.get("managed_by") else None
            ),
            dataset_family=(
                str(value["dataset_family"])
                if value.get("dataset_family") else None
            ),
        ))
    if len({item.name for item in result}) != len(result):
        raise SourceInvalidError("duplicate source registry name")
    return tuple(result)
