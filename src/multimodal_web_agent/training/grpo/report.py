from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable, Mapping


def tree_sha256(root: Path, *, suffixes: tuple[str, ...] = (".py", ".sh", ".yaml")) -> str:
    digest = hashlib.sha256()
    paths = [path for path in Path(root).rglob("*") if path.is_file() and path.suffix in suffixes]
    for path in sorted(paths):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def write_baseline_report(path: Path, sections: Mapping[str, Any]) -> None:
    lines = ["# Reward v0 Baseline Report", ""]
    for title, value in sections.items():
        lines.extend([f"## {title}", "", str(value), ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
