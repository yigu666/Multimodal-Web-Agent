from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from PIL import Image

from multimodal_web_agent.evaluation.unified_agent.schema import UnifiedEvalExample


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))[:160]


def _aliases(row: Mapping[str, Any]) -> tuple[str, ...]:
    values = [row.get("gt_answer", "")]
    alternatives = row.get("alternative_gt_answers")
    if alternatives is not None:
        values.extend(list(alternatives))
    result = []
    seen = set()
    for value in values:
        text = " ".join(str(value).split())
        if text and text.casefold() not in seen:
            seen.add(text.casefold())
            result.append(text)
    return tuple(result)


def _query_image(value: Any) -> Image.Image | None:
    if value is None:
        return None
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, Mapping):
        if value.get("bytes"):
            return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
        if value.get("path"):
            return Image.open(value["path"]).convert("RGB")
    if isinstance(value, (str, Path)) and Path(value).is_file():
        return Image.open(value).convert("RGB")
    raise ValueError("unsupported MMSearch query image")


class MMSearchExternalAdapter:
    """Materialize the official 300-row end-to-end split without changing labels."""

    def __init__(self, parquet_path: Path, *, official_code_root: Path):
        self.parquet_path = Path(parquet_path)
        self.official_code_root = Path(official_code_root)

    def materialize(self, output_root: Path) -> list[UnifiedEvalExample]:
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError("pandas/pyarrow are required for MMSearch") from exc
        if not self.parquet_path.is_file():
            raise FileNotFoundError(self.parquet_path)
        if not (self.official_code_root / "score/f1_score.py").is_file():
            raise FileNotFoundError("official MMSearch scoring code is missing")
        rows = pd.read_parquet(self.parquet_path).to_dict(orient="records")
        if len(rows) != 300:
            raise RuntimeError("official MMSearch end2end split must contain 300 samples")
        output_root = Path(output_root)
        images = output_root / "images"
        images.mkdir(parents=True, exist_ok=False)
        examples = []
        text_only_count = 0
        for index, row in enumerate(rows):
            sample_id = str(row["sample_id"])
            image = _query_image(row.get("query_image"))
            text_only = image is None
            if text_only:
                text_only_count += 1
                # The current Agent renderer requires exactly one image. A neutral
                # pixel preserves the official text-only condition without adding clues.
                image = Image.new("RGB", (1, 1), (127, 127, 127))
            image_path = images / ("%04d_%s.png" % (index, _safe_name(sample_id)))
            image.save(image_path, format="PNG")
            digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
            example = UnifiedEvalExample(
                eval_id="mmsearch:%s" % sample_id,
                source_dataset="mmsearch_official",
                source_data_id=sample_id,
                question=str(row["query"]),
                image_path=image_path.relative_to(output_root).as_posix(),
                image_sha256=digest,
                answer_aliases=_aliases(row),
                task_type="mixed_search_required",
                search_required=True,
                source_metadata={
                    "official_row_index": index,
                    "area": str(row.get("area", "")),
                    "subfield": str(row.get("subfield", "")),
                    "timestamp": str(row.get("timestamp", "")),
                    "gt_requery": str(row.get("gt_requery", "")),
                    "official_text_only_query": text_only,
                    "neutral_image_adapter_used": text_only,
                    "labels_modified": False,
                },
            )
            example.validate()
            examples.append(example)
        manifest = {
            "schema_version": "online-external-mmsearch-adapter-v1",
            "source": self.parquet_path.as_posix(),
            "source_sha256": hashlib.sha256(self.parquet_path.read_bytes()).hexdigest(),
            "official_code_commit": "7b6ce517a7ba86c51bdb51ad1cddd9adb1c66e3d",
            "sample_count": len(examples),
            "text_only_sample_count": text_only_count,
            "labels_modified": False,
            "training_use": False,
            "frozen_test_accessed": False,
        }
        (output_root / "adapter_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (output_root / "examples.jsonl").write_text(
            "".join(json.dumps(item.to_dict(), ensure_ascii=False, sort_keys=True) + "\n" for item in examples),
            encoding="utf-8",
        )
        return examples


def official_mmsearch_f1(
    prediction: str | None,
    answers: Sequence[str],
    *,
    official_code_root: Path,
) -> float:
    path = Path(official_code_root) / "score/f1_score.py"
    spec = importlib.util.spec_from_file_location("mmsearch_official_f1", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load official MMSearch F1")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    predicted = prediction or ""
    return max(float(module.get_f1_score(predicted, answer)) for answer in answers)

