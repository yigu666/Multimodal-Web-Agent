from __future__ import annotations

from pathlib import Path

from .provenance import sha256_file


def build_checksums(root: Path, output: Path | None = None) -> Path:
    root = Path(root)
    output = Path(output or root / "files.sha256")
    files = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.resolve() != output.resolve()
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "\n".join(
            "%s  %s" % (
                sha256_file(path),
                path.relative_to(root).as_posix(),
            )
            for path in files
        ) + ("\n" if files else ""),
        encoding="utf-8",
    )
    return output
