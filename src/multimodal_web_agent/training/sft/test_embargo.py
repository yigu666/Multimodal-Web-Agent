from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping


class TestEmbargoAlreadyOpenedError(RuntimeError):
    __test__ = False

    pass


def sha256_directory(path: Path) -> str:
    root = Path(path)
    if not root.is_dir():
        raise FileNotFoundError(root)
    digest = hashlib.sha256()
    files = sorted(item for item in root.rglob("*") if item.is_file())
    if not files:
        raise ValueError("adapter directory is empty: %s" % root)
    for file_path in files:
        relative = file_path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with file_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def read_embargo_state(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Test embargo state must be an object")
    return value


def initialize_embargo(path: Path) -> dict[str, Any]:
    path = Path(path)
    if path.exists():
        state = read_embargo_state(path)
        if "opened" not in state or "evaluation_count" not in state:
            raise ValueError("existing Test embargo state is invalid")
        return state
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {"opened": False, "evaluation_count": 0}
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return state


def open_embargo_once(
    path: Path,
    *,
    opened_at: str,
    selected_adapter_hash: str,
    checkpoint_selection_hash: str,
    allow_repeat_diagnostic: bool = False,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    path = Path(path)
    state = initialize_embargo(path)
    count = int(state.get("evaluation_count", 0))
    if count >= 1 and not allow_repeat_diagnostic:
        raise TestEmbargoAlreadyOpenedError(
            "Protocol-SFT v0.4 Test embargo has already been opened"
        )
    updated = {
        **state,
        "opened": True,
        "evaluation_count": count + 1,
        "opened_at": opened_at,
        "selected_adapter_hash": selected_adapter_hash,
        "checkpoint_selection_hash": checkpoint_selection_hash,
        "independent_final_evaluation": count == 0,
        "repeat_diagnostic": count >= 1,
        **dict(metadata or {}),
    }
    temporary = path.with_name(path.name + ".opening")
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
    except FileExistsError as exc:
        raise RuntimeError("another Test evaluation is opening the embargo") from exc
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(updated, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        latest = read_embargo_state(path)
        if int(latest.get("evaluation_count", 0)) != count:
            raise RuntimeError("Test embargo state changed concurrently")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return updated


def assert_eval_authorized(
    state: Mapping[str, Any],
    *,
    selected_adapter_hash: str | None = None,
    checkpoint_selection_hash: str | None = None,
) -> None:
    if state.get("opened") is not True:
        raise ValueError("Test embargo has not been opened")
    if int(state.get("evaluation_count", 0)) < 1:
        raise ValueError("Test evaluation_count is invalid")
    if (
        selected_adapter_hash is not None
        and state.get("selected_adapter_hash") != selected_adapter_hash
    ):
        raise ValueError("selected Adapter hash differs from embargo state")
    if (
        checkpoint_selection_hash is not None
        and state.get("checkpoint_selection_hash")
        != checkpoint_selection_hash
    ):
        raise ValueError("Checkpoint Selection hash differs from embargo state")
