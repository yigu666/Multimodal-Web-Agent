from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, MutableMapping, Sequence

import torch

from multimodal_web_agent.agent.schema import ActionType
from multimodal_web_agent.training.grpo.advantages import compute_group_advantages
from multimodal_web_agent.training.sft.target_segments import target_segments

from .config import LocalCreditWeights
from .answer_reward_variants import (
    combined_text_advantage,
    effective_text_query_advantage,
)


NOT_VERIFIABLE = "not_verifiable_from_persisted_rollouts"


@dataclass(frozen=True)
class LocalNormalization:
    local_utility_count: int
    local_utility_mean: float
    local_utility_std: float
    local_advantage: float
    local_advantage_fallback_reason: str | None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ActionTokenSpan:
    response_start: int
    response_end: int
    full_sequence_start: int
    full_sequence_end: int
    alignment_method: str

    def to_dict(self) -> dict:
        return asdict(self)


def normalize_valid_local_utilities(
    utilities: Sequence[float],
    *,
    min_valid_actions: int = 2,
    epsilon: float = 1e-6,
    variance_epsilon: float = 1e-12,
) -> list[LocalNormalization]:
    values = [float(value) for value in utilities]
    if any(not math.isfinite(value) for value in values):
        raise ValueError("local utility is non-finite")
    count = len(values)
    mean = sum(values) / count if count else 0.0
    variance = sum((value - mean) ** 2 for value in values) / count if count else 0.0
    std = math.sqrt(variance)
    if count < min_valid_actions:
        reason = "fewer_than_min_valid_actions"
    elif std <= variance_epsilon:
        reason = "zero_variance"
    else:
        reason = None
    return [
        LocalNormalization(
            local_utility_count=count,
            local_utility_mean=mean,
            local_utility_std=std,
            local_advantage=0.0 if reason else (value - mean) / (std + epsilon),
            local_advantage_fallback_reason=reason,
        )
        for value in values
    ]


def assign_group_local_advantages(
    breakdowns: Sequence[MutableMapping[str, Any]],
    *,
    weights: LocalCreditWeights,
) -> dict[str, dict[str, float | int | bool]]:
    by_tool: dict[str, list[tuple[int, int, float]]] = defaultdict(list)
    for record_index, breakdown in enumerate(breakdowns):
        for action_index, action in enumerate(breakdown.get("search_actions", [])):
            if not action.get("executed", False):
                continue
            tool = str(action.get("tool", ""))
            if tool not in {ActionType.TEXT_SEARCH.value, ActionType.IMAGE_SEARCH.value}:
                continue
            by_tool[tool].append((record_index, action_index, float(action["local_utility"])))
    stats = {}
    for tool in (ActionType.TEXT_SEARCH.value, ActionType.IMAGE_SEARCH.value):
        items = by_tool.get(tool, [])
        if tool == ActionType.TEXT_SEARCH.value:
            normalized = [
                LocalNormalization(
                    local_utility_count=1,
                    local_utility_mean=value,
                    local_utility_std=0.0,
                    local_advantage=effective_text_query_advantage(
                        value,
                        enabled=weights.text_local_enabled,
                        positive_only=weights.text_local_positive_only,
                    ),
                    local_advantage_fallback_reason=(
                        "text_local_disabled"
                        if not weights.text_local_enabled
                        else "negative_query_clipped_to_zero"
                        if weights.text_local_positive_only and value < 0.0
                        else "counterfactual_absolute_no_group_normalization"
                    ),
                )
                for _, _, value in items
            ]
        else:
            normalized = normalize_valid_local_utilities(
                [item[2] for item in items],
                min_valid_actions=weights.min_valid_actions,
                epsilon=weights.advantage_epsilon,
                variance_epsilon=weights.variance_epsilon,
            )
        for (record_index, action_index, _), row in zip(items, normalized):
            action = breakdowns[record_index]["search_actions"][action_index]
            action["raw_local_advantage"] = float(
                action.get("local_utility", 0.0)
            )
            action["negative_query_contribution_removed"] = bool(
                tool == ActionType.TEXT_SEARCH.value
                and weights.text_local_positive_only
                and float(action["raw_local_advantage"]) < 0.0
            )
            action.update(row.to_dict())
        count = len(items)
        mean = sum(item[2] for item in items) / count if count else 0.0
        std = math.sqrt(sum((item[2] - mean) ** 2 for item in items) / count) if count else 0.0
        stats[tool] = {
            "count": count,
            "mean": mean,
            "std": std,
            "nonzero": any(abs(row.local_advantage) > 0 for row in normalized),
        }
    return stats


def terminal_advantages(terminal_rewards: Sequence[float], prompt_group_id: str) -> list[float]:
    tensor = torch.tensor([float(value) for value in terminal_rewards], dtype=torch.float32)
    values = compute_group_advantages(tensor, [prompt_group_id] * len(terminal_rewards))
    return [float(value) for value in values]


def _encoded(tokenizer: Any, text: str) -> list[int]:
    try:
        values = tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        values = tokenizer.encode(text)
    return [int(value) for value in values]


def _decoded(tokenizer: Any, token_ids: Sequence[int]) -> str:
    decoder = getattr(tokenizer, "decode", None)
    if decoder is None:
        batch_decoder = getattr(tokenizer, "batch_decode", None)
        if batch_decoder is None:
            raise TypeError("verified generated-token alignment requires decode")
        return str(batch_decoder(
            [list(token_ids)], skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0])
    try:
        return str(decoder(
            list(token_ids), skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ))
    except TypeError:
        return str(decoder(
            list(token_ids), skip_special_tokens=True
        ))


def _verified_character_token_span(
    *, tokenizer: Any, token_ids: Sequence[int], text: str,
    char_start: int, char_end: int,
) -> tuple[int, int]:
    ids = [int(value) for value in token_ids]
    expected = str(text)
    if _decoded(tokenizer, ids) != expected:
        raise ValueError("verified generated content does not decode to raw action")
    prefixes = [_decoded(tokenizer, ids[:index]) for index in range(len(ids) + 1)]
    start_prefix = expected[:char_start]
    end_prefix = expected[:char_end]
    target = expected[char_start:char_end]
    candidates = [
        (start, end)
        for start, start_text in enumerate(prefixes)
        if start_text == start_prefix
        for end, end_text in enumerate(prefixes)
        if end >= start
        and end_text == end_prefix
        and _decoded(tokenizer, ids[start:end]) == target
    ]
    shortest = min((end - start for start, end in candidates), default=None)
    candidates = [
        pair for pair in candidates if pair[1] - pair[0] == shortest
    ]
    if len(candidates) != 1:
        raise ValueError(
            "generated action character/token alignment is not unique: "
            f"matches={candidates}"
        )
    return candidates[0]


def _subsequence(haystack: Sequence[int], needle: Sequence[int]) -> int:
    if not needle:
        raise ValueError("cannot align an empty token sequence")
    matches = [
        start for start in range(0, len(haystack) - len(needle) + 1)
        if list(haystack[start:start + len(needle)]) == list(needle)
    ]
    if len(matches) != 1:
        raise ValueError(f"action token alignment is not unique: matches={matches}")
    return matches[0]


def locate_search_action_token_span(
    *,
    raw_action: str,
    action_type: str,
    full_input_ids: Sequence[int] | torch.Tensor,
    assistant_span: Sequence[int],
    tokenizer: Any,
    verified_content_span: Sequence[int] | None = None,
) -> ActionTokenSpan:
    if action_type not in {ActionType.TEXT_SEARCH.value, ActionType.IMAGE_SEARCH.value}:
        raise ValueError("only executed search actions have local token spans")
    original_raw = str(raw_action)
    raw = original_raw.strip()
    leading_trim = len(original_raw) - len(original_raw.lstrip())
    weights = {
        "reason": 1.0,
        "image_search_action": 1.0,
        "text_search_open_close_tags": 1.0,
        "text_query_payload": 1.0,
        "answer_open_close_tags": 1.0,
        "answer_payload": 1.0,
        "assistant_end": 1.0,
    }
    segments = target_segments(raw, weights)
    search_segments = [segment for segment in segments if segment.name != "reason"]
    if not search_segments:
        raise ValueError("parsed search action has no action segment")
    char_start = min(segment.start for segment in search_segments)
    char_end = max(segment.end for segment in search_segments)
    if verified_content_span is not None:
        full = [int(value) for value in torch.as_tensor(full_input_ids).tolist()]
        content_start, content_end = map(int, verified_content_span)
        assistant_start, assistant_end = map(int, assistant_span)
        if not (
            assistant_start <= content_start < content_end <= assistant_end
        ):
            raise ValueError("verified generated content left assistant span")
        token_start, token_end = _verified_character_token_span(
            tokenizer=tokenizer,
            token_ids=full[content_start:content_end],
            text=original_raw,
            char_start=leading_trim + char_start,
            char_end=leading_trim + char_end,
        )
        full_start = content_start + token_start
        full_end = content_start + token_end
        if full_start <= 0:
            raise ValueError("action begins before the first predicted token")
        return ActionTokenSpan(
            response_start=full_start - 1,
            response_end=full_end - 1,
            full_sequence_start=full_start,
            full_sequence_end=full_end,
            alignment_method="generated_token_decode_prefix_alignment",
        )
    if (
        hasattr(tokenizer, "decode")
        or hasattr(tokenizer, "batch_decode")
    ):
        full = [int(value) for value in torch.as_tensor(full_input_ids).tolist()]
        assistant_start, assistant_end = map(int, assistant_span)
        assistant_ids = full[assistant_start:assistant_end]
        assistant_text = _decoded(tokenizer, assistant_ids)
        raw_offsets = [
            index
            for index in range(len(assistant_text) - len(original_raw) + 1)
            if assistant_text[index:index + len(original_raw)] == original_raw
        ]
        if len(raw_offsets) == 1:
            raw_offset = raw_offsets[0]
            try:
                token_start, token_end = _verified_character_token_span(
                    tokenizer=tokenizer,
                    token_ids=assistant_ids,
                    text=assistant_text,
                    char_start=raw_offset + leading_trim + char_start,
                    char_end=raw_offset + leading_trim + char_end,
                )
            except ValueError:
                pass
            else:
                full_start = assistant_start + token_start
                full_end = assistant_start + token_end
                if full_start <= 0:
                    raise ValueError(
                        "action begins before the first predicted token"
                    )
                return ActionTokenSpan(
                    response_start=full_start - 1,
                    response_end=full_end - 1,
                    full_sequence_start=full_start,
                    full_sequence_end=full_end,
                    alignment_method=(
                        "assistant_decode_prefix_character_alignment"
                    ),
                )
    try:
        encoded = tokenizer(raw, add_special_tokens=False, return_offsets_mapping=True)
        ids = encoded["input_ids"]
        offsets = encoded["offset_mapping"]
        if ids and isinstance(ids[0], (list, tuple)):
            ids = ids[0]
        if offsets and isinstance(offsets[0][0], (list, tuple)):
            offsets = offsets[0]
        selected = [
            index for index, (start, end) in enumerate(offsets)
            if int(start) >= char_start and int(end) <= char_end and int(end) > int(start)
        ]
        if not selected or selected != list(range(selected[0], selected[-1] + 1)):
            raise ValueError("search action offset mapping is discontinuous")
        if any(
            int(start) < char_start < int(end) or int(start) < char_end < int(end)
            for start, end in offsets
        ):
            raise ValueError("token crosses search action character boundary")
        raw_ids = [int(value) for value in ids]
        action_token_start, action_token_end = selected[0], selected[-1] + 1
        method = "parser_plus_tokenizer_offset_mapping"
    except (TypeError, AttributeError, KeyError, NotImplementedError):
        raw_ids = _encoded(tokenizer, raw)
        prefix_ids = _encoded(tokenizer, raw[:char_start])
        action_ids = _encoded(tokenizer, raw[char_start:char_end])
        if prefix_ids + action_ids != raw_ids:
            raise ValueError("verified segment tokenization cannot reconstruct action")
        action_token_start = len(prefix_ids)
        action_token_end = len(raw_ids)
        method = "parser_plus_verified_segment_tokenization"
    full = [int(value) for value in torch.as_tensor(full_input_ids).tolist()]
    assistant_start, assistant_end = map(int, assistant_span)
    relative = _subsequence(full[assistant_start:assistant_end], raw_ids)
    full_start = assistant_start + relative + action_token_start
    full_end = assistant_start + relative + action_token_end
    if full_start <= 0:
        raise ValueError("action begins before the first predicted token")
    return ActionTokenSpan(
        response_start=full_start - 1,
        response_end=full_end - 1,
        full_sequence_start=full_start,
        full_sequence_end=full_end,
        alignment_method=method,
    )


def build_token_advantages(
    *,
    record: Any,
    breakdown: MutableMapping[str, Any],
    terminal_advantage: float,
    tokenizer: Any,
    weights: LocalCreditWeights,
) -> torch.Tensor:
    policy_mask = torch.as_tensor(record.policy_action_mask, dtype=torch.float32)
    information_mask = torch.as_tensor(record.information_mask, dtype=torch.float32)
    if policy_mask.shape != information_mask.shape:
        raise ValueError("policy and information masks differ in shape")
    if int((policy_mask * information_mask).sum().item()) != 0:
        raise ValueError("environment information token entered the policy mask")
    advantages = policy_mask * float(terminal_advantage)
    assistant_spans = list(record.assistant_turn_spans)
    actions = list(record.actions)
    if len(assistant_spans) != len(actions):
        raise ValueError("assistant spans and actions differ in count")
    search_by_turn = {int(row["turn"]): row for row in breakdown.get("search_actions", [])}
    verified_by_turn = {
        int(row["turn"]): (
            int(row["full_sequence_start"]),
            int(row["full_sequence_end"]),
        )
        for row in getattr(record, "exploration_metadata", {}).get(
            "generated_content_alignment", []
        )
    }
    for action_index, action in enumerate(actions):
        tool = str(action.get("action_type", ""))
        turn = int(action.get("turn", action_index))
        if turn not in search_by_turn:
            continue
        local_row = search_by_turn[turn]
        span = locate_search_action_token_span(
            raw_action=str(action.get("raw", "")),
            action_type=tool,
            full_input_ids=record.full_input_ids,
            assistant_span=assistant_spans[action_index],
            tokenizer=tokenizer,
            verified_content_span=verified_by_turn.get(turn),
        )
        local_advantage = float(local_row.get("local_advantage", 0.0))
        if tool == ActionType.TEXT_SEARCH.value:
            value = combined_text_advantage(
                terminal_advantage=terminal_advantage,
                query_advantage=local_advantage,
                terminal_weight=weights.text_terminal_weight,
                query_weight=weights.text_local_weight,
            )
            local_row["text_query_advantage"] = local_advantage
            local_row["text_query_advantage_positive_only"] = bool(
                weights.text_local_positive_only
            )
            local_row["text_query_local_enabled"] = bool(
                weights.text_local_enabled
            )
            local_row["combined_text_token_advantage"] = float(value)
        else:
            value = terminal_advantage
            local_row["image_local_head_used_in_loss"] = False
            local_row["combined_image_token_advantage"] = float(value)
        advantages[span.response_start:span.response_end] = value
        local_row["token_span"] = span.to_dict()
        local_row["token_advantage"] = float(value)
    advantages *= policy_mask
    if not torch.isfinite(advantages).all():
        raise ValueError("token advantage is non-finite")
    active = advantages[policy_mask.bool()]
    breakdown["token_advantage_stats"] = {
        "min": float(active.min()) if active.numel() else 0.0,
        "max": float(active.max()) if active.numel() else 0.0,
        "mean": float(active.mean()) if active.numel() else 0.0,
        "std": float(active.std(unbiased=False)) if active.numel() else 0.0,
        "nonzero_tokens": int((active != 0).sum().item()),
        "policy_token_count": int(active.numel()),
    }
    return advantages
