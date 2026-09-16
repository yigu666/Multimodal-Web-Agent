from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Mapping, Optional

import torch

from multimodal_web_agent.data.protocol_sft.schema import StateActionExample

from .renderer import ProtocolRenderer


class PrefixMismatchError(ValueError):
    pass


class SequenceOverflowError(ValueError):
    pass


class EmptyTargetError(ValueError):
    pass


@dataclass(frozen=True)
class MaskMetadata:
    sample_id: str
    trajectory_id: str
    route: str
    state_type: str
    split: str
    input_length: int
    prefix_length: int
    target_length: int
    target_start: int
    target_end: int
    active_label_count: int
    target_truncated: bool
    history_assistant_turn_count: int
    information_turn_count: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TokenizedExample:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    pixel_values: Optional[torch.Tensor]
    image_grid_thw: Optional[torch.Tensor]
    metadata: MaskMetadata
    image: Any = None
    token_weights: Optional[torch.Tensor] = None
    segment_ids: Optional[torch.Tensor] = None
    weight_alignment: Optional[Mapping[str, Any]] = None

    def model_inputs(self) -> Dict[str, torch.Tensor]:
        values: Dict[str, torch.Tensor] = {
            "input_ids": self.input_ids,
            "attention_mask": self.attention_mask,
            "labels": self.labels,
        }
        if self.pixel_values is not None:
            values["pixel_values"] = self.pixel_values
        if self.image_grid_thw is not None:
            values["image_grid_thw"] = self.image_grid_thw
        return values


def _tensor(value: Any, name: str) -> torch.Tensor:
    if not torch.is_tensor(value):
        value = torch.as_tensor(value)
    if value.ndim == 1:
        value = value.unsqueeze(0)
    if value.ndim < 2:
        raise ValueError("processor output %s must have at least two dimensions" % name)
    return value


def _encode(processor: Any, rendered_text: str, image: Any) -> Mapping[str, Any]:
    return processor(
        text=[rendered_text],
        images=[image],
        padding=False,
        truncation=False,
        return_tensors="pt",
    )


def tokenize_current_turn(
    processor: Any,
    renderer: ProtocolRenderer,
    example: StateActionExample,
    image: Any,
    *,
    max_seq_len: int = 1536,
    loss_weights: Any = None,
) -> TokenizedExample:
    rendered = renderer.render(example)
    prefix_batch = _encode(processor, rendered.prefix_text, image)
    full_batch = _encode(processor, rendered.full_text, image)
    prefix_ids = _tensor(prefix_batch["input_ids"], "prefix input_ids")[0]
    full_ids = _tensor(full_batch["input_ids"], "full input_ids")[0]
    if not torch.equal(full_ids[: prefix_ids.shape[0]], prefix_ids):
        raise PrefixMismatchError(
            "chat-template prefix differs between generation-prompt and full rendering for %s"
            % example.example_id
        )
    input_length = int(full_ids.shape[0])
    prefix_length = int(prefix_ids.shape[0])
    target_length = input_length - prefix_length
    if target_length <= 0:
        raise EmptyTargetError("current target has no tokens: %s" % example.example_id)
    if input_length > max_seq_len:
        raise SequenceOverflowError(
            "full encoded sequence has %d tokens, exceeds max_seq_len=%d: %s"
            % (input_length, max_seq_len, example.example_id)
        )
    labels = torch.full_like(full_ids, -100)
    labels[prefix_length:] = full_ids[prefix_length:]
    if int((labels != -100).sum().item()) != target_length:
        raise AssertionError("active label count does not equal current target length")
    history_turns = sum(message.role == "assistant" for message in example.state)
    information_turns = sum(
        message.role == "tool" or message.content.lstrip().startswith("<information>")
        for message in example.state
    )
    metadata = MaskMetadata(
        sample_id=example.example_id,
        trajectory_id=example.trajectory_id,
        route=example.route,
        state_type=example.transition,
        split=example.split,
        input_length=input_length,
        prefix_length=prefix_length,
        target_length=target_length,
        target_start=prefix_length,
        target_end=input_length,
        active_label_count=target_length,
        target_truncated=False,
        history_assistant_turn_count=history_turns,
        information_turn_count=information_turns,
    )
    pixel_values = full_batch.get("pixel_values")
    image_grid_thw = full_batch.get("image_grid_thw")
    if pixel_values is not None and not torch.is_tensor(pixel_values):
        pixel_values = torch.as_tensor(pixel_values)
    if image_grid_thw is not None and not torch.is_tensor(image_grid_thw):
        image_grid_thw = torch.as_tensor(image_grid_thw)
    tokenized = TokenizedExample(
        input_ids=full_ids,
        attention_mask=_tensor(full_batch.get("attention_mask", torch.ones_like(full_ids)), "attention_mask")[0],
        labels=labels,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        metadata=metadata,
        image=image,
    )
    if loss_weights is not None:
        from .target_segments import build_sequence_token_weights
        tokenizer = getattr(processor, "tokenizer", processor)
        (
            tokenized.token_weights,
            tokenized.segment_ids,
            tokenized.weight_alignment,
        ) = build_sequence_token_weights(
            tokenizer,
            example.target,
            full_ids,
            target_start=prefix_length,
            target_end=input_length,
            weights=loss_weights,
        )
    audit_tokenized_example(tokenized)
    return tokenized


def audit_tokenized_example(example: TokenizedExample) -> None:
    metadata = example.metadata
    if metadata.active_label_count <= 0 or metadata.target_length <= 0:
        raise ValueError("empty active target: %s" % metadata.sample_id)
    if metadata.target_truncated:
        raise ValueError("target was truncated: %s" % metadata.sample_id)
    labels = example.labels
    ids = example.input_ids
    if labels.ndim != 1 or ids.ndim != 1:
        raise ValueError("tokenized example tensors must be one-dimensional")
    if labels.shape != ids.shape:
        raise ValueError("labels and input_ids shape mismatch")
    if torch.any(labels[: metadata.target_start] != -100):
        raise ValueError("context contains active labels: %s" % metadata.sample_id)
    if not torch.equal(labels[metadata.target_start: metadata.target_end], ids[metadata.target_start: metadata.target_end]):
        raise ValueError("active labels differ from target input_ids: %s" % metadata.sample_id)
    if torch.any(labels[metadata.target_end:] != -100):
        raise ValueError("tokens after target are active: %s" % metadata.sample_id)
    if example.token_weights is not None or example.segment_ids is not None:
        if example.token_weights is None or example.segment_ids is None:
            raise ValueError("partial Target weight alignment tensors")
        from .target_segments import audit_weight_tensors
        audit_weight_tensors(
            labels, example.token_weights, example.segment_ids
        )
