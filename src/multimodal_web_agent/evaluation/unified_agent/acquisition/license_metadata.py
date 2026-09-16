from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping

from .provenance import sha256_file


def detect_license(
    root: Path,
    discovery: Mapping[str, Any],
) -> dict[str, Any]:
    candidates = [
        root / value for value in discovery.get("license_files", ())
    ]
    for path in candidates:
        if path.is_file() and path.stat().st_size > 0:
            text = path.read_text(encoding="utf-8", errors="replace")
            first = next(
                (line.strip() for line in text.splitlines() if line.strip()),
                path.name,
            )
            return {
                "license_name": first[:200],
                "license_source": path.relative_to(root).as_posix(),
                "license_file": path,
                "license_file_sha256": sha256_file(path),
                "license_verified": True,
            }
    for value in discovery.get("readme_files", ()):
        path = root / value
        text = path.read_text(encoding="utf-8", errors="replace")
        match = re.search(
            r"(?im)^(?:license|licence)\s*:\s*(.+?)\s*$", text
        )
        if match:
            return {
                "license_name": match.group(1).strip(),
                "license_source": path.relative_to(root).as_posix(),
                "license_file": path,
                "license_file_sha256": sha256_file(path),
                "license_verified": True,
            }
    return {
        "license_name": None,
        "license_source": None,
        "license_file": None,
        "license_file_sha256": None,
        "license_verified": False,
    }
