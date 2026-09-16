from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any

from .fingerprints import registry_hashes, registered_models
from .schema import EMBARGO_SCHEMA_VERSION, V1_1_EMBARGO_SCHEMA_VERSION


REQUIRED_FINAL_MODELS = ("raw", "sft", "reward_v21", "stage2")


class FinalTestModelsIncomplete(RuntimeError):
    pass


class FinalTestAlreadyOpened(RuntimeError):
    pass


def initialize_test_embargo(path: Path) -> dict[str, Any]:
    path = Path(path)
    if path.exists():
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("schema_version") not in {
            EMBARGO_SCHEMA_VERSION,
            V1_1_EMBARGO_SCHEMA_VERSION,
        }:
            raise ValueError("Unified Eval Test embargo schema mismatch")
        return value
    schema_version = (
        V1_1_EMBARGO_SCHEMA_VERSION
        if "v1_1" in path.name else EMBARGO_SCHEMA_VERSION
    )
    value = {
        "schema_version": schema_version,
        "opened": False,
        "evaluation_count": 0,
        "allowed_model_count": len(REQUIRED_FINAL_MODELS),
        "registered_models": [],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return value


def assert_dev_authorized(path: Path) -> None:
    state = initialize_test_embargo(path)
    if state.get("opened") is not False or int(
        state.get("evaluation_count", 0)
    ) != 0:
        raise PermissionError("Frozen Eval Test embargo is not pristine")


def sync_registered_models(
    embargo_path: Path,
    registry_path: Path,
) -> dict[str, Any]:
    state = initialize_test_embargo(embargo_path)
    if state.get("opened") or int(state.get("evaluation_count", 0)) != 0:
        raise PermissionError("cannot update registrations after Test opening")
    models = registered_models(registry_path)
    updated = {
        **state,
        "registered_models": sorted(models),
        "registered_model_hashes": registry_hashes(registry_path),
    }
    path = Path(embargo_path)
    temporary = path.with_name(path.name + ".tmp-%d" % os.getpid())
    temporary.write_text(
        json.dumps(updated, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return updated


def open_final_test_once(
    embargo_path: Path,
    registry_path: Path,
    *,
    opened_at: str | None = None,
) -> dict[str, Any]:
    state = initialize_test_embargo(embargo_path)
    if state.get("opened") or int(state.get("evaluation_count", 0)) != 0:
        raise FinalTestAlreadyOpened("Frozen Eval Test was already opened")
    models = registered_models(registry_path)
    missing = [name for name in REQUIRED_FINAL_MODELS if name not in models]
    if missing:
        raise FinalTestModelsIncomplete(
            "FINAL_TEST_MODELS_INCOMPLETE: %s" % ", ".join(missing)
        )
    updated = {
        **state,
        "opened": True,
        "evaluation_count": 1,
        "opened_at": opened_at or datetime.now(timezone.utc).isoformat(),
        "registered_models": list(REQUIRED_FINAL_MODELS),
        "registered_model_hashes": registry_hashes(registry_path),
    }
    path = Path(embargo_path)
    temporary = path.with_name(path.name + ".opening-%d" % os.getpid())
    descriptor = os.open(
        temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(updated, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        latest = initialize_test_embargo(path)
        if latest != state:
            raise RuntimeError("Frozen Eval Test embargo changed concurrently")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return updated
