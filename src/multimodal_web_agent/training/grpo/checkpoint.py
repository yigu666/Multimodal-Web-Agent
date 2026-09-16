from __future__ import annotations

from pathlib import Path
from typing import Any


def save_checkpoint(model: Any, path: Path, *, metadata: dict[str, Any] | None = None) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(path)
    if metadata is not None:
        import json
        (path / "grpo_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_checkpoint(model: Any, path: Path) -> Any:
    if hasattr(model, "load_adapter"):
        model.load_adapter(Path(path), is_trainable=True)
    return model
