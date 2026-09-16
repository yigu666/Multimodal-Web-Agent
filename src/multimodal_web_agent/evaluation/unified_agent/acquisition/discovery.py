from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..source_adapters.base import read_tabular


ANNOTATION_SUFFIXES = {".json", ".jsonl", ".parquet", ".arrow", ".csv"}
IMAGE_SUFFIXES = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"
}


def discover_source(root: Path) -> dict[str, Any]:
    root = Path(root)
    files = sorted(path for path in root.rglob("*") if path.is_file())
    annotations = [
        path for path in files
        if path.suffix.casefold() in ANNOTATION_SUFFIXES
    ]
    images = [
        path for path in files if path.suffix.casefold() in IMAGE_SUFFIXES
    ]
    licenses = [
        path for path in files
        if path.name.casefold().startswith(
            ("license", "copying", "notice")
        )
    ]
    readmes = [
        path for path in files
        if path.name.casefold().startswith(("readme", "dataset_card"))
    ]
    evidence = [
        path for path in files
        if any(
            token in part.casefold()
            for part in path.relative_to(root).parts
            for token in ("evidence", "document", "corpus", "wiki")
        )
    ]
    schema_candidates = []
    for annotation in annotations:
        try:
            rows = read_tabular(annotation)
            fields = sorted({
                str(key) for row in rows[:20] for key in row
            })
            schema_candidates.append({
                "path": annotation.relative_to(root).as_posix(),
                "record_count": len(rows),
                "fields": fields,
            })
        except Exception as exc:
            schema_candidates.append({
                "path": annotation.relative_to(root).as_posix(),
                "read_error": type(exc).__name__,
            })
    splits = sorted({
        split for path in annotations
        for split in ("test", "heldout", "validation", "val", "dev", "train")
        if split in path.name.casefold()
    })
    image_roots = sorted({
        path.parent.relative_to(root).as_posix() for path in images
    })
    result = {
        "schema_version": "unified-eval-source-discovery-v1",
        "root": str(root),
        "all_files": [
            path.relative_to(root).as_posix() for path in files
        ],
        "candidate_annotation_files": [
            path.relative_to(root).as_posix() for path in annotations
        ],
        "candidate_image_roots": image_roots,
        "candidate_evidence_files": [
            path.relative_to(root).as_posix() for path in evidence
        ],
        "license_files": [
            path.relative_to(root).as_posix() for path in licenses
        ],
        "readme_files": [
            path.relative_to(root).as_posix() for path in readmes
        ],
        "detected_formats": sorted({
            path.suffix.casefold() for path in files if path.suffix
        }),
        "detected_splits": splits,
        "schema_candidates": schema_candidates,
    }
    (root / "source_discovery.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return result
