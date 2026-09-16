from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable, Mapping

from PIL import Image

from multimodal_web_agent.evaluation.unified_agent.schema import UnifiedEvalExample


def _contact_sheet(images: list[Image.Image]) -> Image.Image:
    if not images:
        raise ValueError("MMSearch-Plus sample has no query images")
    converted = [image.convert("RGB") for image in images]
    width = max(image.width for image in converted)
    height = sum(image.height for image in converted)
    sheet = Image.new("RGB", (width, height), "white")
    offset = 0
    for image in converted:
        sheet.paste(image, (0, offset))
        offset += image.height
    return sheet


class MMSearchPlusExternalAdapter:
    """Adapter for the officially decrypted dataset; V1 keeps whole-image actions."""

    def materialize(self, rows: Iterable[Mapping[str, Any]], output_root: Path):
        output_root = Path(output_root)
        images_root = output_root / "images"
        images_root.mkdir(parents=True, exist_ok=False)
        examples = []
        for index, row in enumerate(rows):
            images = [row.get("img_%d" % number) for number in range(1, 6)]
            images = [image for image in images if isinstance(image, Image.Image)]
            sheet = _contact_sheet(images)
            path = images_root / ("%04d.png" % index)
            sheet.save(path, format="PNG")
            answers = tuple(str(value) for value in row.get("answer", []) if str(value).strip())
            example = UnifiedEvalExample(
                eval_id="mmsearch_plus:%04d" % index,
                source_dataset="mmsearch_plus_official",
                source_data_id=str(row.get("id", index)),
                question=str(row["question"]),
                image_path=path.relative_to(output_root).as_posix(),
                image_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                answer_aliases=answers,
                task_type="mixed_search_required",
                search_required=True,
                source_metadata={
                    "category": row.get("category"),
                    "difficulty": row.get("difficulty"),
                    "query_image_count": len(images),
                    "whole_image_contact_sheet": True,
                    "crop_action_enabled": False,
                    "zoom_action_enabled": False,
                    "labels_modified": False,
                },
            )
            example.validate()
            examples.append(example)
        if len(examples) != 311:
            raise RuntimeError("official MMSearch-Plus dataset must contain 311 samples")
        return examples
