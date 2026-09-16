from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping, Sequence

from .schema import RolloutRecord


def group_rollouts(records: Iterable[RolloutRecord | Mapping], *, group_size: int = 4) -> list[list[RolloutRecord | Mapping]]:
    groups: dict[str, list] = defaultdict(list)
    for record in records:
        uid = record.prompt_uid if isinstance(record, RolloutRecord) else str(record["prompt_uid"])
        groups[uid].append(record)
    result = []
    for uid in sorted(groups):
        group = sorted(groups[uid], key=lambda item: int(item.rollout_index if isinstance(item, RolloutRecord) else item.get("rollout_index", 0)))
        if len(group) != group_size:
            raise ValueError(f"prompt {uid} has {len(group)} rollouts; expected {group_size}")
        result.append(group)
    return result
