from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Mapping


def summarize_reward_components(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    n = len(rows) or 1
    rewards = [float(row.get("reward_total", 0.0)) for row in rows]
    return {
        "reward_mean": sum(rewards) / n,
        "reward_min": min(rewards) if rewards else 0.0,
        "reward_max": max(rewards) if rewards else 0.0,
        "protocol_valid_rate": sum(bool(row.get("protocol_valid")) for row in rows) / n,
        "answer_em": sum(float(row.get("answer_em", 0.0)) for row in rows) / n,
        "answer_f1": sum(float(row.get("answer_f1", 0.0)) for row in rows) / n,
        "first_action_distribution": dict(Counter(str(row.get("first_action", "")) for row in rows)),
        "image_search_rate": sum(bool(row.get("used_image_search")) for row in rows) / n,
        "text_search_rate": sum(bool(row.get("used_text_search")) for row in rows) / n,
    }
