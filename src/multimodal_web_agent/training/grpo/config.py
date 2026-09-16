from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass(frozen=True)
class GRPOConfig:
    name: str = "lightweight_multimodal_grpo_v1_reward_v0"
    seed: int = 20260730
    group_size: int = 4
    prompt_groups_per_update: int = 4
    max_turns: int = 3
    max_new_tokens_per_turn: int = 96
    max_seq_len: int = 1536
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 0
    clip_ratio: float = 0.2
    learning_rate: float = 5e-7
    fp16: bool = True
    bf16: bool = False
    quant_type: str = "nf4"
    double_quant: bool = True
    freeze_visual: bool = True
    reward_name: str = "mmsearch_like_reward_v0"

    def validate(self) -> None:
        if self.group_size < 1 or self.prompt_groups_per_update < 1:
            raise ValueError("group sizes must be positive")
        if self.max_turns != 3 or self.max_new_tokens_per_turn != 96:
            raise ValueError("GRPO v1 frozen context contract is max_turns=3 and max_new_tokens=96")
        if self.max_seq_len != 1536 or self.clip_ratio != 0.2:
            raise ValueError("GRPO v1 frozen sequence/clip contract mismatch")
        if self.quant_type.casefold() != "nf4" or not self.double_quant or not self.fp16 or self.bf16:
            raise ValueError("GRPO v1 requires NF4 double-quantized FP16 and forbids BF16")
        if not self.freeze_visual:
            raise ValueError("visual encoder/projector must remain frozen")
        if self.reward_name != "mmsearch_like_reward_v0":
            raise ValueError("only Reward v0 is registered in this task")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_grpo_config(path: Path) -> GRPOConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    run, rollout, context, optimizer, grpo, quantization, precision, vision = (raw.get(name, {}) for name in ("run", "rollout", "context", "optimizer", "grpo", "quantization", "precision", "vision"))
    config = GRPOConfig(
        name=str(run.get("name", GRPOConfig.name)), seed=int(run.get("seed", GRPOConfig.seed)),
        group_size=int(rollout.get("group_size", 4)), prompt_groups_per_update=int(rollout.get("prompt_groups_per_update", 4)),
        max_turns=int(context.get("max_turns", 3)), max_new_tokens_per_turn=int(context.get("max_new_tokens_per_turn", 96)), max_seq_len=int(context.get("max_seq_len", 1536)),
        temperature=float(rollout.get("temperature", 0.7)), top_p=float(rollout.get("top_p", 0.9)), top_k=int(rollout.get("top_k", 0)),
        clip_ratio=float(grpo.get("clip_ratio", 0.2)), learning_rate=float(optimizer.get("actor_learning_rate", 5e-7)),
        fp16=bool(precision.get("fp16", True)), bf16=bool(precision.get("bf16", False)), quant_type=str(quantization.get("quant_type", "nf4")), double_quant=bool(quantization.get("double_quant", True)), freeze_visual=bool(vision.get("freeze_encoder", True) and vision.get("freeze_projector", True)), reward_name=str(grpo.get("reward_name", "mmsearch_like_reward_v0")),
    )
    config.validate()
    return config
