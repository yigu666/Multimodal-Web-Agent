from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Mapping, Sequence

import torch

from multimodal_web_agent.data.protocol_sft.schema import StateActionExample

from .renderer import ProtocolRenderer


@dataclass(frozen=True)
class GenerationRecord:
    sample_id: str
    trajectory_id: str
    route: str
    state_type: str
    target_rendered: str
    generated_text: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _model_device(model: Any) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def generate_for_example(
    model: Any,
    processor: Any,
    renderer: ProtocolRenderer,
    example: StateActionExample,
    image: Any,
    *,
    max_new_tokens: int = 96,
    generation_kwargs: Mapping[str, Any] | None = None,
) -> GenerationRecord:
    rendered = renderer.render(example)
    inputs = processor(
        text=[rendered.prefix_text],
        images=[image],
        padding=True,
        return_tensors="pt",
    )
    device = _model_device(model)
    model_inputs = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in inputs.items()
    }
    frozen_generation = {
        "do_sample": False,
        "num_beams": 1,
        "max_new_tokens": max_new_tokens,
        "repetition_penalty": 1.0,
    }
    frozen_generation.update(dict(generation_kwargs or {}))
    generated = model.generate(
        **model_inputs,
        **frozen_generation,
    )
    prompt_length = int(model_inputs["input_ids"].shape[-1])
    generated_only = generated[:, prompt_length:]
    if hasattr(processor, "batch_decode"):
        text = processor.batch_decode(
            generated_only,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
    else:
        text = processor.tokenizer.batch_decode(
            generated_only,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
    return GenerationRecord(
        sample_id=example.example_id,
        trajectory_id=example.trajectory_id,
        route=example.route,
        state_type=example.transition,
        target_rendered=example.target,
        generated_text=str(text).strip(),
    )


def generate_records(
    model: Any,
    processor: Any,
    renderer: ProtocolRenderer,
    examples: Sequence[StateActionExample],
    images: Sequence[Any],
    *,
    max_new_tokens: int = 96,
    generation_kwargs: Mapping[str, Any] | None = None,
) -> list[GenerationRecord]:
    if len(examples) != len(images):
        raise ValueError("examples and images must have equal lengths")
    return [
        generate_for_example(
            model, processor, renderer, example, image,
            max_new_tokens=max_new_tokens,
            generation_kwargs=generation_kwargs,
        )
        for example, image in zip(examples, images)
    ]
