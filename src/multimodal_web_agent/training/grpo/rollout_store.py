from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .schema import RolloutRecord


class RolloutStore:
    def __init__(self, path: Path):
        self.path = Path(path)

    def append(self, record: RolloutRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")

    def write(self, records: Iterable[RolloutRecord]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")

    def read(self) -> list[RolloutRecord]:
        if not self.path.exists():
            return []
        return [RolloutRecord.from_dict(json.loads(line)) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]
