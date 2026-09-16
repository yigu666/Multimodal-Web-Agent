from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from threading import Lock
from typing import Any, Mapping

from ..schemas import SearchBackendError


BUDGET_SCHEMA = "online-web-api-budget-v1"
PROVIDER_KEYS = {
    "serper": "max_serper_requests",
    "google_vision": "max_google_vision_requests",
    "serpapi_lens": "max_serpapi_lens_requests",
    "jina": "max_jina_requests",
}


class ApiBudgetGuard:
    """Persistent, fail-closed budget ledger for billable remote providers."""

    def __init__(
        self,
        *,
        enabled: bool,
        limits: Mapping[str, int],
        allow_paid_overage: bool,
        ledger_path: Path | None = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.limits = {name: int(value) for name, value in limits.items()}
        self.allow_paid_overage = bool(allow_paid_overage)
        self.ledger_path = Path(ledger_path) if ledger_path is not None else None
        self._lock = Lock()
        self._counts = {provider: 0 for provider in PROVIDER_KEYS}
        self._load()

    @classmethod
    def from_config(cls, project_root: Path, config: Mapping[str, Any]) -> "ApiBudgetGuard":
        value = dict(config.get("online_budget") or config.get("budget") or {})
        ledger = value.get("ledger_path")
        return cls(
            enabled=bool(value.get("enabled", False)),
            limits={
                provider: int(value.get(key, 0))
                for provider, key in PROVIDER_KEYS.items()
            },
            allow_paid_overage=bool(value.get("allow_paid_overage", False)),
            ledger_path=(Path(project_root) / str(ledger)) if ledger else None,
        )

    def _load(self) -> None:
        if self.ledger_path is None or not self.ledger_path.is_file():
            return
        value = json.loads(self.ledger_path.read_text(encoding="utf-8"))
        if value.get("schema_version") != BUDGET_SCHEMA:
            raise RuntimeError("online budget ledger schema mismatch")
        for provider in PROVIDER_KEYS:
            self._counts[provider] = int(value.get("remote_request_counts", {}).get(provider, 0))

    def _write(self) -> None:
        if self.ledger_path is None:
            return
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": BUDGET_SCHEMA,
            "remote_request_counts": dict(self._counts),
        }
        data = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".%s." % self.ledger_path.name,
            suffix=".tmp",
            dir=str(self.ledger_path.parent),
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, self.ledger_path)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)

    def reserve(self, provider: str) -> None:
        provider = str(provider)
        if provider not in PROVIDER_KEYS:
            raise KeyError("unknown budget provider: %s" % provider)
        with self._lock:
            current = int(self._counts[provider])
            limit = int(self.limits.get(provider, 0))
            if self.enabled and not self.allow_paid_overage and current >= limit:
                raise SearchBackendError(
                    "ONLINE_API_BUDGET_EXHAUSTED",
                    "%s remote request budget is exhausted" % provider,
                    metadata={"provider": provider, "used": current, "limit": limit},
                )
            self._counts[provider] = current + 1
            self._write()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            counts = dict(self._counts)
        return {
            "enabled": self.enabled,
            "allow_paid_overage": self.allow_paid_overage,
            "limits": dict(self.limits),
            "remote_request_counts": counts,
            "remaining": {
                provider: max(0, int(self.limits.get(provider, 0)) - count)
                for provider, count in counts.items()
            },
        }
