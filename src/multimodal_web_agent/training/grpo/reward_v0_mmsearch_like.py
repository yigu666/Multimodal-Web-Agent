from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Iterable, Mapping

from multimodal_web_agent.agent.schema import ActionType
from multimodal_web_agent.data.protocol_sft.answer_normalizer import normalize_answer
from multimodal_web_agent.evaluation.unified_agent.answer_metrics import maximum_alias_token_f1, normalized_exact_match

from .reward_registry import register_reward


REWARD_NAME = "mmsearch_like_reward_v0"


def _value(rollout: Any, key: str, default: Any = None) -> Any:
    return getattr(rollout, key, default) if not isinstance(rollout, Mapping) else rollout.get(key, default)


@dataclass(frozen=True)
class RewardV0Components:
    answer_em: int
    answer_f1: float
    format_score: int
    search_count: int
    search_penalty_applied: bool
    answer_score_before_penalty: float
    answer_score_after_penalty: float
    reward_total: float
    protocol_valid: bool
    protocol_error: str | None
    terminal_reason: str
    category: str
    unnecessary_search: bool
    search_required_without_search: bool
    first_action: str
    used_image_search: bool
    used_text_search: bool
    duplicate_query: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def score_reward_v0(rollout: Any, prompt: Mapping[str, Any] | None = None) -> RewardV0Components:
    answer = str(_value(rollout, "answer_text", "") or "")
    aliases = tuple(str(x) for x in ((prompt or {}).get("candidate_answers") or (prompt or {}).get("accepted_answers") or []))
    if not aliases and prompt:
        aliases = (str(prompt.get("ground_truth", "")),)
    answer_em = normalized_exact_match(answer, aliases)
    answer_f1 = maximum_alias_token_f1(answer, aliases)
    search_count = int(_value(rollout, "search_count", 0) or 0)
    protocol_valid = bool(_value(rollout, "protocol_valid", False))
    # A valid protocol only gets the format point when the trajectory really ended in answer.
    terminal_reason = str(_value(rollout, "terminal_reason", "") or "")
    format_score = int(
        protocol_valid
        and bool(answer.strip())
        and terminal_reason not in {"protocol_error", "max_turns", "max_new_tokens", "environment_error"}
    )
    penalty = bool(answer_em and search_count > 0)
    before = float(answer_em)
    after = before * (0.9 if penalty else 1.0)
    total = 0.90 * after + 0.10 * format_score
    actions = _value(rollout, "actions", []) or []
    first = ""
    if actions:
        action = actions[0]
        action = action if isinstance(action, Mapping) else {"action_type": getattr(action, "action_type", "")}
        first = str(action.get("action_type", ""))
    category = str(_value(rollout, "category", "") or (prompt or {}).get("category", "") or "")
    return RewardV0Components(
        answer_em=int(answer_em), answer_f1=float(answer_f1), format_score=format_score,
        search_count=search_count, search_penalty_applied=penalty,
        answer_score_before_penalty=before, answer_score_after_penalty=after,
        reward_total=float(total), protocol_valid=protocol_valid,
        protocol_error=_value(rollout, "protocol_error", None), terminal_reason=terminal_reason,
        category=category, unnecessary_search=bool(category == "search_free" and search_count > 0),
        search_required_without_search=bool(category == "search_required" and search_count == 0),
        first_action=first,
        used_image_search=bool(_value(rollout, "used_image_search", False)),
        used_text_search=bool(_value(rollout, "used_text_search", False)),
        duplicate_query=bool(_value(rollout, "duplicate_query", False)),
    )


@register_reward(REWARD_NAME)
class MMSearchLikeRewardV0:
    name = REWARD_NAME

    def __call__(self, rollout: Any, prompt: Mapping[str, Any] | None = None) -> RewardV0Components:
        return score_reward_v0(rollout, prompt)

    def score(self, rollout: Any, prompt: Mapping[str, Any] | None = None) -> RewardV0Components:
        return self(rollout, prompt)
