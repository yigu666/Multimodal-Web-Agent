from __future__ import annotations

from dataclasses import dataclass
from typing import Any


REGISTRY_SCHEMA = "unified-agent-eval-source-registry-v1"
ACQUISITION_SCHEMA = "unified-agent-eval-acquisition-v1"
NORMALIZED_MANIFEST_SCHEMA = "unified-eval-heldout-source-v2"


@dataclass(frozen=True)
class AcquisitionSource:
    name: str
    enabled: bool
    acquisition_mode: str
    source_plugin: str
    official_dataset_name: str | None
    official_dataset_id: str | None
    official_repository: str | None
    preferred_splits: tuple[str, ...]
    incoming_archive_globs: tuple[str, ...]
    allow_network_download: bool
    requires_user_license_acceptance: bool | str
    download_urls: tuple[dict[str, Any], ...]
    managed_by: str | None = None
    dataset_family: str | None = None

    @property
    def official_source_reference(self) -> str | None:
        return self.official_dataset_id or self.official_repository
