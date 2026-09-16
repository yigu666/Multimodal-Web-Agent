from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def write_reserved_eval_manifests(
    *,
    accepted: Sequence[Mapping[str, Any]],
    quarantined: Sequence[Mapping[str, Any]],
    ids_path: Path,
    images_path: Path,
) -> dict[str, int]:
    rows = list(accepted) + list(quarantined)
    ids = sorted({
        str(row["source_data_id"])
        for row in rows if row.get("source_data_id")
    })
    image_rows = sorted({
        (
            str(row.get("query_image_sha256") or ""),
            str(row.get("query_image_id") or ""),
        )
        for row in rows if row.get("query_image_sha256")
    })
    ids_path = Path(ids_path)
    images_path = Path(images_path)
    ids_path.parent.mkdir(parents=True, exist_ok=True)
    ids_path.write_text(
        json.dumps({
            "schema_version": "unified-eval-infoseek-reserved-ids-v1",
            "dataset_family": "visual_infoseek_2023",
            "reserved_source_data_ids": ids,
            "future_training_use_forbidden": True,
        }, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    images_path.write_text(
        "".join(
            "%s  %s\n" % (digest, image_id)
            for digest, image_id in image_rows
        ),
        encoding="utf-8",
    )
    return {
        "reserved_id_count": len(ids),
        "reserved_image_count": len(image_rows),
    }
