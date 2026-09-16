from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from multimodal_web_agent.data.protocol_sft.schema import StateActionExample, TransitionType

FORMAT_TRANSITIONS = (
    "initial_to_direct_answer",
    "initial_to_image_search",
    "image_information_to_answer",
    "image_information_to_text_search",
    "text_information_to_answer",
)


@dataclass(frozen=True)
class DatasetItem:
    example: StateActionExample
    image: Any


def load_split(path: Path, expected_count: Optional[int] = None) -> List[StateActionExample]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    examples: List[StateActionExample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                example = StateActionExample.from_dict(value)
                example.validate()
            except Exception as exc:
                raise ValueError("invalid Protocol-SFT example at %s:%d: %s" % (path, line_number, exc)) from exc
            examples.append(example)
    if expected_count is not None and len(examples) != expected_count:
        raise ValueError("%s contains %d examples; expected %d" % (path, len(examples), expected_count))
    if len({item.example_id for item in examples}) != len(examples):
        raise ValueError("duplicate example_id in %s" % path)
    return examples


def select_smoke_subset(
    examples: Sequence[StateActionExample],
    examples_per_transition: int = 2,
) -> List[StateActionExample]:
    if examples_per_transition != 2:
        raise ValueError("Protocol-SFT Smoke requires exactly two examples per transition")
    transitions = tuple(item.value for item in TransitionType)
    grouped: Dict[str, List[StateActionExample]] = {transition: [] for transition in transitions}
    for example in examples:
        grouped.setdefault(example.transition, []).append(example)
    selected: List[StateActionExample] = []
    for transition in transitions:
        values = sorted(grouped.get(transition, []), key=lambda item: item.example_id)
        if len(values) < examples_per_transition:
            raise ValueError("Smoke transition %s has only %d examples" % (transition, len(values)))
        selected.extend(values[:examples_per_transition])
    if len(selected) != 12 or len({item.example_id for item in selected}) != 12:
        raise AssertionError("Smoke subset must contain twelve unique examples")
    return selected


def select_format_smoke_subset(
    examples: Sequence[StateActionExample],
    examples_per_transition: int = 2,
) -> List[StateActionExample]:
    """Select two deterministic examples for each actual Format transition."""
    if examples_per_transition != 2:
        raise ValueError("Format Smoke requires two examples per transition")
    grouped: Dict[str, List[StateActionExample]] = {
        transition: [] for transition in FORMAT_TRANSITIONS
    }
    for example in examples:
        if example.transition in grouped:
            grouped[example.transition].append(example)
    selected: List[StateActionExample] = []
    for transition in FORMAT_TRANSITIONS:
        values = sorted(
            grouped[transition],
            key=lambda item: (
                getattr(item, "state_type", item.transition),
                item.example_id,
            ),
        )
        if len(values) < examples_per_transition:
            raise ValueError(
                "Format Smoke transition %s has only %d examples"
                % (transition, len(values))
            )
        selected.extend(values[:examples_per_transition])
    if len(selected) != 10 or len({item.example_id for item in selected}) != 10:
        raise AssertionError("Format Smoke subset must contain ten unique examples")
    return selected


class ImageStore:
    """Lazy loader for the single original FVQA image in each example.

    Cached search results are represented as text in Protocol-SFT and are never
    loaded here.  The store is intentionally injectable so local fixture tests
    do not need pyarrow or the full FVQA parquet.
    """

    def __init__(self, source_parquet: Optional[Path] = None):
        self.source_parquet = Path(source_parquet) if source_parquet else None
        self._rows: Dict[int, Mapping[str, Any]] = {}
        self._images: Dict[int, tuple[str, Any]] = {}
        self._parquet = None
        self._row_group_offsets: Optional[List[int]] = None

    def _ensure_parquet(self) -> Any:
        if self._parquet is not None:
            return self._parquet
        if self.source_parquet is None:
            raise FileNotFoundError("no source parquet configured for FVQA image loading")
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("FVQA image loading requires pyarrow on the server") from exc
        self._parquet = pq.ParquetFile(self.source_parquet)
        available = set(self._parquet.schema_arrow.names)
        missing = {"data_id", "images"} - available
        if missing:
            raise ValueError("FVQA source parquet is missing columns: %s" % sorted(missing))
        return self._parquet

    def _offsets(self) -> List[int]:
        if self._row_group_offsets is not None:
            return self._row_group_offsets
        parquet = self._ensure_parquet()
        offsets = [0]
        for group_index in range(parquet.metadata.num_row_groups):
            offsets.append(
                offsets[-1] + int(parquet.metadata.row_group(group_index).num_rows)
            )
        self._row_group_offsets = offsets
        return offsets

    def _load_row(self, row_index: int) -> Mapping[str, Any]:
        if row_index < 0:
            raise IndexError("FVQA source row index cannot be negative: %d" % row_index)
        if row_index in self._rows:
            return self._rows[row_index]
        parquet = self._ensure_parquet()
        offsets = self._offsets()
        if row_index >= offsets[-1]:
            raise IndexError("FVQA source row index not found: %d" % row_index)
        group_index = next(
            index
            for index in range(len(offsets) - 1)
            if offsets[index] <= row_index < offsets[index + 1]
        )
        table = parquet.read_row_group(
            group_index,
            columns=["data_id", "images"],
        )
        row = table.slice(row_index - offsets[group_index], 1).to_pylist()[0]
        self._rows[row_index] = row
        return row

    def load(self, example: StateActionExample) -> Any:
        for reference in example.image_refs:
            if reference.get("kind") == "fixture_image" and reference.get("image") is not None:
                return reference["image"]
            if reference.get("path"):
                return _open_image(Path(str(reference["path"])))
        row_index = example.source.get("source_row_index")
        if row_index is None:
            for reference in example.image_refs:
                if reference.get("row_index") is not None:
                    row_index = reference["row_index"]
                    break
        if row_index is None:
            raise ValueError("example has no original FVQA image row reference: %s" % example.example_id)
        row_index = int(row_index)
        cached = self._images.get(row_index)
        if cached is not None:
            data_id, image = cached
            if data_id != example.source_data_id:
                raise ValueError(
                    "FVQA cached row/data_id mismatch for %s" % example.example_id
                )
            return image
        row = self._load_row(row_index)
        if str(row.get("data_id")) != example.source_data_id:
            raise ValueError(
                "FVQA source row index/data_id mismatch for %s: expected %s, got %s"
                % (example.example_id, example.source_data_id, row.get("data_id"))
            )
        images = row.get("images")
        if not isinstance(images, list) or len(images) != 1 or not isinstance(images[0], Mapping):
            raise ValueError("example must contain exactly one original image: %s" % example.example_id)
        image = images[0]
        if image.get("bytes"):
            try:
                from PIL import Image
                import io
                with Image.open(io.BytesIO(image["bytes"])) as opened:
                    loaded = opened.convert("RGB")
            except ImportError as exc:
                raise RuntimeError("Pillow is required to load FVQA images") from exc
        elif image.get("path"):
            loaded = _open_image(Path(str(image["path"])))
        else:
            raise ValueError("FVQA image has neither bytes nor path: %s" % example.example_id)
        self._images[row_index] = (example.source_data_id, loaded)
        self._rows.pop(row_index, None)
        return loaded


def _open_image(path: Path) -> Any:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required to load images") from exc
    if not path.is_file():
        raise FileNotFoundError(path)
    with Image.open(path) as opened:
        return opened.convert("RGB")


class ProtocolSFTDataset:
    def __init__(
        self,
        examples: Sequence[StateActionExample],
        *,
        image_loader: Optional[Callable[[StateActionExample], Any]] = None,
        source_parquet: Optional[Path] = None,
    ):
        self.examples = list(examples)
        self.image_store = ImageStore(source_parquet) if image_loader is None else None
        self.image_loader = image_loader

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> DatasetItem:
        example = self.examples[index]
        image = self.image_loader(example) if self.image_loader else self.image_store.load(example)
        return DatasetItem(example=example, image=image)


def load_protocol_splits(
    data: Any,
    *,
    include_test: Optional[bool] = None,
) -> Dict[str, List[StateActionExample]]:
    result = {
        "train": load_split(data.train_file, data.expected_train_count),
        "dev": load_split(data.dev_file, data.expected_dev_count),
    }
    should_include_test = (
        bool(getattr(data, "allow_test_access", True))
        if include_test is None
        else bool(include_test)
    )
    if should_include_test and not bool(
        getattr(data, "allow_test_access", True)
    ):
        raise PermissionError("Test split access is embargoed by configuration")
    if should_include_test and data.test_file is not None:
        result["test"] = load_split(data.test_file, data.expected_test_count)
    return result
