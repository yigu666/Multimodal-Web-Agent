from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import torch

from .advantages import compute_group_advantages
from .policy_loss import trajectory_balanced_policy_loss
from .reward_v0_mmsearch_like import score_reward_v0


@dataclass
class UpdateResult:
    loss: float
    group_count: int
    trajectory_count: int


class GRPOTrainer:
    """Replay exact stored rollout tensors; no input reconstruction is performed."""

    def __init__(self, model: Any, optimizer: Any, *, clip_ratio: float = 0.2):
        self.model = model
        self.optimizer = optimizer
        self.clip_ratio = clip_ratio

    def update(self, records: Sequence[Any], group_ids: Sequence[str], forward_log_probs: Callable[[Any], torch.Tensor]) -> UpdateResult:
        def field(record, name):
            return getattr(record, name) if hasattr(record, name) else record[name]
        rewards = torch.stack([torch.as_tensor(field(record, "reward_total")) for record in records])
        advantages = compute_group_advantages(rewards, group_ids)
        losses = []
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        for record, advantage in zip(records, advantages):
            old = torch.as_tensor(field(record, "old_log_probs"))
            new = forward_log_probs(record)
            mask = torch.as_tensor(field(record, "policy_action_mask"))
            loss = trajectory_balanced_policy_loss(old, new, torch.ones_like(old) * advantage, mask, clip_ratio=self.clip_ratio)
            losses.append(loss)
        total = torch.stack(losses).mean() if losses else torch.zeros((), requires_grad=True)
        total.backward()
        self.optimizer.step()
        return UpdateResult(float(total.detach()), len(set(group_ids)), len(records))
