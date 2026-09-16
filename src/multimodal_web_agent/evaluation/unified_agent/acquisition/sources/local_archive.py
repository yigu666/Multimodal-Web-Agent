from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from ...source_adapters.base import read_tabular


class LocalArchiveAcquisitionPlugin:
    id_fields = ("source_data_id", "sample_id", "data_id", "id")
    question_fields = ("question", "query")

    def __init__(self, preferred_splits: Sequence[str] = ()):
        self.preferred_splits = tuple(
            value.casefold() for value in preferred_splits
        )

    def _rank(self, path: Path) -> tuple[int, str]:
        name = path.name.casefold()
        if "train" in name:
            return (10_000, name)
        for index, split in enumerate(self.preferred_splits):
            if split in name:
                return (index, name)
        return (len(self.preferred_splits) + 1, name)

    def annotation_paths(
        self,
        root: Path,
        discovery: Mapping[str, Any],
    ) -> list[Path]:
        paths = [
            root / value
            for value in discovery["candidate_annotation_files"]
            if "train" not in Path(value).name.casefold()
        ]
        return sorted(paths, key=self._rank)

    def load_rows(
        self,
        root: Path,
        discovery: Mapping[str, Any],
    ) -> list[tuple[Path, int, dict[str, Any]]]:
        raw_rows = []
        for path in self.annotation_paths(root, discovery):
            try:
                rows = read_tabular(path)
            except Exception:
                continue
            for index, row in enumerate(rows):
                identifier = self.source_id(row)
                if identifier:
                    raw_rows.append((path, index, dict(row)))

        supplemental: dict[str, dict[str, Any]] = {}
        for _, _, row in raw_rows:
            merged = supplemental.setdefault(self.source_id(row), {})
            for key, value in row.items():
                if value not in (None, "", [], {}):
                    merged.setdefault(key, value)

        result = []
        seen = set()
        for path, index, row in raw_rows:
            identifier = self.source_id(row)
            if not self.question(row) or identifier in seen:
                continue
            seen.add(identifier)
            value = dict(supplemental[identifier])
            value.update(row)
            result.append((path, index, self.prepare_row(value)))
        return result

    def prepare_row(self, row: dict[str, Any]) -> dict[str, Any]:
        return row

    def source_id(self, row: Mapping[str, Any]) -> str:
        return next(
            (
                str(row[field]).strip() for field in self.id_fields
                if row.get(field) not in (None, "")
            ),
            "",
        )

    def question(self, row: Mapping[str, Any]) -> str:
        return next(
            (
                str(row[field]).strip() for field in self.question_fields
                if row.get(field) not in (None, "")
            ),
            "",
        )

    def split(
        self, row: Mapping[str, Any], annotation: Path
    ) -> str | None:
        value = row.get("split") or row.get("data_split")
        if value:
            return str(value)
        name = annotation.name.casefold()
        return next(
            (split for split in self.preferred_splits if split in name),
            None,
        )
