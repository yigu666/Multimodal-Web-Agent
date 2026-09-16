from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence

import torch

from multimodal_web_agent.agent import ActionType, parse_action


SEGMENT_CONTEXT = 0
SEGMENT_REASON = 1
SEGMENT_ACTION_TAG = 2
SEGMENT_QUERY_PAYLOAD = 3
SEGMENT_ANSWER_PAYLOAD = 4
SEGMENT_ASSISTANT_END = 5

SEGMENT_NAMES = {
    SEGMENT_CONTEXT: "context",
    SEGMENT_REASON: "reason",
    SEGMENT_ACTION_TAG: "action_tag",
    SEGMENT_QUERY_PAYLOAD: "query_payload",
    SEGMENT_ANSWER_PAYLOAD: "answer_payload",
    SEGMENT_ASSISTANT_END: "assistant_end",
}


class TargetWeightAlignmentError(ValueError):
    pass


@dataclass(frozen=True)
class TargetSegment:
    name: str
    start: int
    end: int
    segment_id: int
    weight: float


@dataclass(frozen=True)
class TargetWeightAlignment:
    token_ids: tuple[int, ...]
    token_weights: tuple[float, ...]
    segment_ids: tuple[int, ...]
    method: str


def _weight(weights: Any, name: str) -> float:
    value = (
        getattr(weights, name)
        if hasattr(weights, name)
        else weights[name]
    )
    value = float(value)
    if not value > 0:
        raise ValueError(f"loss weight must be positive: {name}")
    return value


def target_segments(
    target_text: str,
    weights: Any,
) -> list[TargetSegment]:
    if target_text != target_text.strip():
        raise TargetWeightAlignmentError(
            "target text must not contain outer whitespace"
        )
    parsed = parse_action(target_text)
    if not parsed.valid or parsed.action_type is None:
        raise TargetWeightAlignmentError("target is not Strict-Parser valid")
    reason_close = "</reason>"
    reason_end = target_text.index(reason_close) + len(reason_close)
    action_start = reason_end
    while (
        action_start < len(target_text)
        and target_text[action_start].isspace()
    ):
        action_start += 1
    segments = [
        TargetSegment(
            "reason",
            0,
            action_start,
            SEGMENT_REASON,
            _weight(weights, "reason"),
        )
    ]
    if parsed.action_type == ActionType.IMAGE_SEARCH:
        action = "<search><img></search>"
        if target_text[action_start:] != action:
            raise TargetWeightAlignmentError(
                "image-search target structure differs from frozen protocol"
            )
        segments.append(TargetSegment(
            "image_search_action",
            action_start,
            len(target_text),
            SEGMENT_ACTION_TAG,
            _weight(weights, "image_search_action"),
        ))
        return segments
    if parsed.action_type == ActionType.TEXT_SEARCH:
        opening, closing = "<text_search>", "</text_search>"
        payload_segment = SEGMENT_QUERY_PAYLOAD
        payload_name = "text_query_payload"
        tag_weight_name = "text_search_open_close_tags"
    else:
        opening, closing = "<answer>", "</answer>"
        payload_segment = SEGMENT_ANSWER_PAYLOAD
        payload_name = "answer_payload"
        tag_weight_name = "answer_open_close_tags"
    if not target_text.startswith(opening, action_start):
        raise TargetWeightAlignmentError("action opening tag is misaligned")
    close_start = target_text.rfind(closing)
    if close_start < action_start + len(opening):
        raise TargetWeightAlignmentError("action closing tag is misaligned")
    if close_start + len(closing) != len(target_text):
        raise TargetWeightAlignmentError(
            "unexpected text after action closing tag"
        )
    segments.extend([
        TargetSegment(
            "action_open_tag",
            action_start,
            action_start + len(opening),
            SEGMENT_ACTION_TAG,
            _weight(weights, tag_weight_name),
        ),
        TargetSegment(
            payload_name,
            action_start + len(opening),
            close_start,
            payload_segment,
            _weight(weights, payload_name),
        ),
        TargetSegment(
            "action_close_tag",
            close_start,
            len(target_text),
            SEGMENT_ACTION_TAG,
            _weight(weights, tag_weight_name),
        ),
    ])
    if any(segment.start >= segment.end for segment in segments):
        raise TargetWeightAlignmentError("target contains an empty segment")
    return segments


def _encoded_ids(tokenizer: Any, text: str) -> list[int]:
    values = tokenizer.encode(text, add_special_tokens=False)
    return [int(value) for value in values]


def _offset_alignment(
    tokenizer: Any,
    target_text: str,
    segments: Sequence[TargetSegment],
) -> TargetWeightAlignment | None:
    try:
        encoded = tokenizer(
            target_text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        ids = encoded["input_ids"]
        offsets = encoded["offset_mapping"]
    except (
        TypeError, AttributeError, KeyError, NotImplementedError, ValueError
    ):
        return None
    if ids and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    if offsets and isinstance(offsets[0][0], (list, tuple)):
        offsets = offsets[0]
    if len(ids) != len(offsets):
        raise TargetWeightAlignmentError(
            "Tokenizer ID/offset lengths differ"
        )
    token_weights = []
    segment_ids = []
    for start, end in offsets:
        start, end = int(start), int(end)
        if end <= start:
            raise TargetWeightAlignmentError(
                "Target tokenizer returned an empty offset"
            )
        matches = [
            segment for segment in segments
            if segment.start <= start and end <= segment.end
        ]
        if len(matches) != 1:
            raise TargetWeightAlignmentError(
                "Target token crosses a structured segment boundary"
            )
        token_weights.append(matches[0].weight)
        segment_ids.append(matches[0].segment_id)
    return TargetWeightAlignment(
        tuple(int(value) for value in ids),
        tuple(token_weights),
        tuple(segment_ids),
        "offset_mapping",
    )


def align_target_tokens(
    tokenizer: Any,
    target_text: str,
    weights: Any,
) -> TargetWeightAlignment:
    segments = target_segments(target_text, weights)
    offset_result = _offset_alignment(tokenizer, target_text, segments)
    if offset_result is not None:
        expected = _encoded_ids(tokenizer, target_text)
        if list(offset_result.token_ids) != expected:
            raise TargetWeightAlignmentError(
                "Offset-mapped Target IDs differ from tokenizer.encode"
            )
        return offset_result
    combined_ids = []
    token_weights = []
    segment_ids = []
    for segment in segments:
        ids = _encoded_ids(
            tokenizer, target_text[segment.start:segment.end]
        )
        combined_ids.extend(ids)
        token_weights.extend([segment.weight] * len(ids))
        segment_ids.extend([segment.segment_id] * len(ids))
    expected = _encoded_ids(tokenizer, target_text)
    if combined_ids != expected:
        raise TargetWeightAlignmentError(
            "Per-segment token IDs do not exactly reconstruct Target IDs"
        )
    return TargetWeightAlignment(
        tuple(combined_ids),
        tuple(token_weights),
        tuple(segment_ids),
        "verified_segment_tokenization",
    )


def build_sequence_token_weights(
    tokenizer: Any,
    target_text: str,
    full_input_ids: torch.Tensor,
    *,
    target_start: int,
    target_end: int,
    weights: Any,
) -> tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    if full_input_ids.ndim != 1:
        raise TargetWeightAlignmentError(
            "full_input_ids must be one-dimensional"
        )
    aligned = align_target_tokens(tokenizer, target_text, weights)
    active_ids = [
        int(value)
        for value in full_input_ids[target_start:target_end].tolist()
    ]
    target_ids = list(aligned.token_ids)
    if active_ids[:len(target_ids)] != target_ids:
        raise TargetWeightAlignmentError(
            "Target token IDs do not match the Full suffix"
        )
    assistant_end_count = len(active_ids) - len(target_ids)
    if assistant_end_count <= 0:
        raise TargetWeightAlignmentError(
            "Full suffix does not contain an Assistant End token"
        )
    sequence_weights = torch.zeros(
        full_input_ids.shape, dtype=torch.float32
    )
    sequence_segments = torch.full(
        full_input_ids.shape,
        SEGMENT_CONTEXT,
        dtype=torch.long,
    )
    target_token_end = target_start + len(target_ids)
    sequence_weights[target_start:target_token_end] = torch.tensor(
        aligned.token_weights, dtype=torch.float32
    )
    sequence_segments[target_start:target_token_end] = torch.tensor(
        aligned.segment_ids, dtype=torch.long
    )
    sequence_weights[target_token_end:target_end] = _weight(
        weights, "assistant_end"
    )
    sequence_segments[target_token_end:target_end] = SEGMENT_ASSISTANT_END
    return sequence_weights, sequence_segments, {
        "alignment_method": aligned.method,
        "target_text_token_count": len(target_ids),
        "assistant_end_token_count": assistant_end_count,
        "target_weight_alignment_failure_count": 0,
    }


def audit_weight_tensors(
    labels: torch.Tensor,
    token_weights: torch.Tensor,
    segment_ids: torch.Tensor,
) -> None:
    if labels.shape != token_weights.shape or labels.shape != segment_ids.shape:
        raise ValueError("labels, token_weights and segment_ids must align")
    active = labels.ne(-100)
    if int(active.sum().item()) <= 0:
        raise ValueError("weighted target contains no active token")
    if torch.any(token_weights[~active] != 0):
        raise ValueError("Context or padding token has a non-zero weight")
    active_weights = token_weights[active]
    if not torch.isfinite(active_weights).all():
        raise ValueError("Active Target weights must be finite")
    if torch.any(active_weights <= 0):
        raise ValueError("Active Target weights must be positive")
