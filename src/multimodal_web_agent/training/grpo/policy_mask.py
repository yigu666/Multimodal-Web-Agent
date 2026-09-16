from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence


def _span(value: Any) -> tuple[int, int]:
    if isinstance(value, Mapping):
        start = value.get("start", value.get("begin", value.get("start_index")))
        end = value.get("end", value.get("stop", value.get("end_index")))
        if start is None or end is None:
            raise ValueError(f"invalid span: {value!r}")
        return int(start), int(end)
    if len(value) != 2:
        raise ValueError(f"invalid span: {value!r}")
    return int(value[0]), int(value[1])


def _mask(length: int, spans: Iterable[Any]) -> list[int]:
    output = [0] * length
    for raw in spans:
        start, end = _span(raw)
        if start < 0 or end < start or end > length:
            raise ValueError(f"span outside sequence: {(start, end)}")
        for index in range(start, end):
            output[index] = 1
    return output


def build_policy_action_mask(
    sequence_length: int,
    assistant_turn_spans: Sequence[Any],
    information_spans: Sequence[Any] = (),
    attention_mask: Any = None,
) -> Any:
    """Mask all assistant generations across turns; never mask environment info."""
    values = _mask(sequence_length, assistant_turn_spans)
    for start, end in (_span(span) for span in information_spans):
        for index in range(start, end):
            values[index] = 0
    if attention_mask is not None:
        values = [int(value and bool(attention_mask[index])) for index, value in enumerate(values)]
    try:
        import torch
        return torch.tensor(values, dtype=torch.long, device=attention_mask.device if torch.is_tensor(attention_mask) else None)
    except ImportError:
        return values


def build_information_mask(sequence_length: int, information_spans: Sequence[Any], attention_mask: Any = None) -> Any:
    values = _mask(sequence_length, information_spans)
    if attention_mask is not None:
        values = [int(value and bool(attention_mask[index])) for index, value in enumerate(values)]
    try:
        import torch
        return torch.tensor(values, dtype=torch.long, device=attention_mask.device if torch.is_tensor(attention_mask) else None)
    except ImportError:
        return values


def build_grpo_masks(sequence_length, assistant_turn_spans, information_spans, attention_mask=None):
    return (
        build_policy_action_mask(sequence_length, assistant_turn_spans, information_spans, attention_mask),
        build_information_mask(sequence_length, information_spans, attention_mask),
    )
