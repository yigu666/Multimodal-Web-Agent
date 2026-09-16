from __future__ import annotations

from pathlib import Path

from .base import (
    SourceCandidate,
    SourceScan,
    image_bytes,
    input_file_hashes,
    read_tabular,
    unique_aliases,
)


class GenericHeldOutAdapter:
    source_name = "generic"

    def __init__(self, path: Path | None):
        self.path = Path(path) if path else None

    def scan(self) -> SourceScan:
        if self.path is None or not self.path.is_file():
            return SourceScan(
                self.source_name, False, 0,
                "%s held-out source unavailable" % self.source_name,
            )
        candidates = []
        for index, row in enumerate(read_tabular(self.path)):
            images = row.get("images")
            image_value = row.get("image")
            if image_value is None and isinstance(images, (list, tuple)) and images:
                image_value = images[0]
            raw_image, extension = image_bytes(
                image_value
            )
            candidate = SourceCandidate(
                source_dataset=self.source_name,
                source_data_id=str(
                    row.get("source_data_id")
                    or row.get("data_id")
                    or row.get("id")
                ),
                question=str(row.get("question") or row.get("query") or ""),
                image_bytes=raw_image,
                image_extension=extension,
                answer_aliases=unique_aliases(
                    row.get("answer_aliases")
                    or [row.get("answer") or row.get("ground_truth")]
                ),
                suggested_task_type=row.get("task_type"),
                image_search_records=tuple(row.get("image_search_records", ())),
                text_corpus_records=tuple(row.get("text_corpus_records", ())),
                source_metadata={
                    "source_row_index": index,
                    "task_type_requires_manual_review": True,
                    "online_access": False,
                },
            )
            candidate.validate()
            candidates.append(candidate)
        return SourceScan(
            self.source_name, True, len(candidates),
            "%s held-out source scanned" % self.source_name,
            tuple(candidates),
            input_file_hashes({"source": self.path}),
        )
