from __future__ import annotations

from typing import Any, Iterable, Mapping

from .reward_v0_mmsearch_like import score_reward_v0


def evaluate_records(records: Iterable[Any], prompts: Mapping[str, Mapping[str, Any]]) -> dict[str, float]:
    rows = [score_reward_v0(record, prompts.get(record.prompt_uid, {})) for record in records]
    n = len(rows) or 1
    return {
        "overall_em": sum(row.answer_em for row in rows) / n,
        "overall_f1": sum(row.answer_f1 for row in rows) / n,
        "protocol_valid_rate": sum(row.protocol_valid for row in rows) / n,
        "finish_rate": sum(row.terminal_reason in {"answer", "cache_miss_then_answer"} for row in rows) / n,
    }
