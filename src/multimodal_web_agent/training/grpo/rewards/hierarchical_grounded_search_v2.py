from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, MutableMapping, Sequence

from multimodal_web_agent.agent.schema import ActionType
from multimodal_web_agent.training.grpo.reward_registry import register_reward

from .answer_score import score_answer
from .answer_reward_variants import (
    LEGACY_V2,
    answer_quality as official_answer_quality,
    score_variant_terminal,
)
from .config import RewardV2Config
from .coverage_cache import accepted_answers, prompt_id
from .evidence_support import support_match, support_prediction
from .image_retrieval_utility import image_marginal_evidence_gain
from .local_credit_assignment import (
    NOT_VERIFIABLE,
    assign_group_local_advantages,
    build_token_advantages,
    terminal_advantages,
)
from .reward_breakdown import group_summary
from .text_query_utility import extract_ranked_result_texts, score_text_query_utility


REWARD_VERSION = "hierarchical_grounded_search_v2"


def _value(value: Any, key: str, default: Any = None) -> Any:
    return value.get(key, default) if isinstance(value, Mapping) else getattr(value, key, default)


def _clip(value: float, lower: float, upper: float) -> float:
    return min(max(float(value), float(lower)), float(upper))


def _protocol_legal(rollout: Any) -> bool:
    actions = list(_value(rollout, "actions", []) or [])
    protocol_error = _value(rollout, "protocol_error", None)
    if protocol_error:
        return False
    if any(action.get("valid") is False for action in actions):
        return False
    # A valid sequence that exhausts its turn budget is legal but incomplete.
    return bool(_value(rollout, "protocol_valid", False) or actions)


def _finished_with_answer(rollout: Any) -> bool:
    answer = str(_value(rollout, "answer_text", "") or "").strip()
    if not answer:
        return False
    actions = list(_value(rollout, "actions", []) or [])
    return any(action.get("action_type") == ActionType.ANSWER.value and action.get("valid", True) for action in actions) or bool(_value(rollout, "finished_with_answer", False))


def _results_by_turn(rollout: Any) -> dict[int, Mapping[str, Any]]:
    result = {}
    for row in _value(rollout, "tool_results", []) or []:
        turn = int(row.get("turn", len(result)))
        if turn in result:
            raise ValueError(f"duplicate tool result at turn={turn}")
        result[turn] = row
    return result


@dataclass(frozen=True)
class GroupRewardResult:
    breakdowns: tuple[dict[str, Any], ...]
    token_advantages: tuple[Any | None, ...]
    group_metrics: dict[str, Any]


@register_reward(REWARD_VERSION)
class HierarchicalGroundedSearchRewardV2:
    """Global terminal reward plus action-local retrieval credit."""

    name = REWARD_VERSION

    def __init__(
        self,
        config: RewardV2Config,
        *,
        coverage_cache: Mapping[str, Mapping[str, Any]],
        question_baseline_cache: Mapping[str, Mapping[str, Any]],
    ):
        self.config = config
        self.coverage_cache = {str(key): dict(value) for key, value in coverage_cache.items()}
        self.question_baseline_cache = {str(key): dict(value) for key, value in question_baseline_cache.items()}

    def score_trajectory(
        self,
        rollout: Any,
        prompt: Mapping[str, Any],
    ) -> dict[str, Any]:
        aliases = accepted_answers(prompt)
        identifier = prompt_id(prompt) or str(_value(rollout, "prompt_uid", ""))
        coverage = self.coverage_cache.get(identifier)
        baseline = self.question_baseline_cache.get(identifier)
        if coverage is None or baseline is None:
            raise KeyError(f"Reward v2 caches have no prompt_id={identifier}")
        question = str(prompt.get("question", ""))
        prediction = str(_value(rollout, "answer_text", "") or "")
        answer = score_answer(prediction, aliases, question=question, weights=self.config.answer)
        actions = list(_value(rollout, "actions", []) or [])
        results = _results_by_turn(rollout)
        accumulated = ""
        search_actions = []
        for action_index, action in enumerate(actions):
            tool = str(action.get("action_type", ""))
            if tool not in {ActionType.TEXT_SEARCH.value, ActionType.IMAGE_SEARCH.value}:
                continue
            turn = int(action.get("turn", action_index))
            result = results.get(turn)
            executed = result is not None and bool(action.get("valid", True))
            information = str((result or {}).get("text", ""))
            before = accumulated
            if executed:
                accumulated = (accumulated + "\n" + information).strip()
            row: dict[str, Any] = {
                "turn": turn,
                "tool": tool,
                "query": str(action.get("content", "")) if tool == ActionType.TEXT_SEARCH.value else None,
                "executed": executed,
                "tool_execution_failure": bool(
                    not executed
                    or (result or {}).get("cache_miss", False)
                    or str((result or {}).get("status", "success")) not in {"", "success"}
                ),
            }
            if tool == ActionType.TEXT_SEARCH.value:
                actual_results = (result or {}).get("results")
                if actual_results and isinstance(actual_results[0], Mapping):
                    actual_results = [str(item.get("text", "")) for item in actual_results]
                utility = score_text_query_utility(
                    accepted_answers=aliases,
                    actual_results=actual_results,
                    actual_information=information,
                    coverage_mask=float(coverage["coverage_mask"]),
                    question_baseline_rank_utility=float(baseline["question_baseline_rank_utility"]),
                    tool_execution_failure=bool(row["tool_execution_failure"]),
                    question=question,
                    weights=self.config.text_query,
                    advantage_weights=self.config.text_query_advantage,
                )
                row.update(utility.to_dict())
                row["local_utility"] = utility.text_query_advantage
            else:
                utility = image_marginal_evidence_gain(
                    aliases, before, accumulated, question=question
                ) if executed else 0.0
                row.update({
                    "support_before": support_match(aliases, before, question=question).support_score,
                    "support_after": support_match(aliases, accumulated, question=question).support_score,
                    "image_retrieval_utility": utility,
                    "local_utility": utility,
                })
            search_actions.append(row)
        gold = support_match(aliases, accumulated, question=question)
        predicted = support_prediction(prediction, accumulated)
        searched = any(row["executed"] for row in search_actions)
        quality = official_answer_quality(
            em_v1=answer.em_v1, token_f1_v1=answer.token_f1_v1
        )
        diagnostic_answer_score = (
            answer.answer_score if self.config.mode == LEGACY_V2 else quality
        )
        evidence_use = (
            gold.support_score * predicted.support_score * diagnostic_answer_score
            if searched else 0.0
        )
        missed = gold.support_score * (1.0 - diagnostic_answer_score)
        wrong_span = bool(
            predicted.support_score >= self.config.evidence.wrong_span_prediction_support_threshold
            and diagnostic_answer_score <= self.config.evidence.wrong_span_answer_score_threshold
            and gold.support_score >= self.config.evidence.wrong_span_gold_support_threshold
        )
        wrong_weight = (
            self.config.evidence.wrong_span_copy_weight
            if self.config.mode == LEGACY_V2
            else float(self.config.negative_shaping.get(
                "wrong_span_copy_weight", 0.0
            ))
        )
        wrong_penalty = wrong_weight if wrong_span else 0.0
        protocol_valid = _protocol_legal(rollout)
        finished = _finished_with_answer(rollout)
        variant = score_variant_terminal(
            mode=self.config.mode,
            em_v1=answer.em_v1,
            token_f1_v1=answer.token_f1_v1,
            gold_support=gold.support_score,
            prediction_support=predicted.support_score,
            legacy_answer_score=answer.answer_score,
            legacy_evidence_use=evidence_use,
            legacy_missed_evidence=missed,
            legacy_wrong_span_penalty=wrong_penalty,
        )
        if not protocol_valid:
            terminal = self.config.terminal.invalid_protocol_reward
        elif not finished:
            terminal = self.config.terminal.missing_answer_reward
        else:
            terminal = variant.terminal_reward
        terminal = _clip(terminal, self.config.terminal.min_reward,
                         self.config.terminal.max_reward)
        breakdown = {
            "reward_version": (
                REWARD_VERSION if self.config.mode == LEGACY_V2
                else self.config.version
            ),
            "reward_mode": self.config.mode,
            "prompt_id": identifier,
            "rollout_uid": str(_value(rollout, "rollout_uid", "")),
            **answer.to_dict(),
            "answer_quality": variant.answer_quality,
            "answer_dominance_score": variant.answer_dominance_score,
            "grounded_quality": variant.grounded_quality,
            "positive_grounding_bonus": variant.positive_grounding_bonus,
            "negative_shaping_total": variant.negative_shaping_total,
            "gold_support_final": gold.support_score,
            "gold_support_match": gold.to_dict(),
            "prediction_support": predicted.support_score,
            "prediction_support_match": predicted.to_dict(),
            "evidence_use_score": float(evidence_use),
            "missed_evidence": float(missed),
            "wrong_span_copy": wrong_span,
            "wrong_span_copy_penalty": float(wrong_penalty),
            "wrong_span_evidence": {
                "prediction": prediction,
                "prediction_matched_span": predicted.matched_span,
                "gold_matched_span": gold.matched_span,
                "prediction_support": predicted.support_score,
                "gold_support_final": gold.support_score,
                "answer_score": diagnostic_answer_score,
            },
            "protocol_valid": protocol_valid,
            "finished_with_answer": finished,
            "terminal_reward": float(terminal),
            "search_actions": search_actions,
            "terminal_advantage": None,
            "token_advantage_stats": {"status": NOT_VERIFIABLE},
        }
        numeric = [answer.answer_score, quality, gold.support_score,
                   predicted.support_score, evidence_use, missed,
                   wrong_penalty, variant.grounded_quality,
                   variant.positive_grounding_bonus, terminal]
        numeric.extend(float(row["local_utility"]) for row in search_actions)
        if not all(math.isfinite(float(value)) for value in numeric):
            raise ValueError("Reward v2 produced a non-finite value")
        return breakdown

    def score_group(
        self,
        records: Sequence[Any],
        prompts: Sequence[Mapping[str, Any]] | Mapping[str, Any],
        *,
        tokenizer: Any | None = None,
    ) -> GroupRewardResult:
        prompt_rows = [prompts] * len(records) if isinstance(prompts, Mapping) else list(prompts)
        if len(records) != len(prompt_rows):
            raise ValueError("records and prompts differ in count")
        breakdowns = [self.score_trajectory(record, prompt) for record, prompt in zip(records, prompt_rows)]
        group_ids = {row["prompt_id"] for row in breakdowns}
        if len(group_ids) != 1:
            raise ValueError("Reward v2 local normalization crossed prompt groups")
        local_stats = assign_group_local_advantages(breakdowns, weights=self.config.local_credit)
        terminal = terminal_advantages(
            [row["terminal_reward"] for row in breakdowns], next(iter(group_ids))
        )
        token_values = []
        for record, row, advantage in zip(records, breakdowns, terminal):
            row["terminal_advantage"] = advantage
            if tokenizer is None or _value(record, "full_input_ids", None) is None:
                token_values.append(None)
                row["token_advantage_stats"] = {"status": NOT_VERIFIABLE}
                continue
            token_values.append(build_token_advantages(
                record=record,
                breakdown=row,
                terminal_advantage=advantage,
                tokenizer=tokenizer,
                weights=self.config.local_credit,
            ))
        summary = group_summary(breakdowns)[0]
        summary["local_normalization"] = local_stats
        token_stds = [
            float(row["token_advantage_stats"]["std"])
            for row in breakdowns
            if isinstance(row.get("token_advantage_stats", {}).get("std"), (int, float))
        ]
        summary["token_advantage_std"] = sum(token_stds) / len(token_stds) if token_stds else None
        return GroupRewardResult(tuple(deepcopy(breakdowns)), tuple(token_values), summary)
