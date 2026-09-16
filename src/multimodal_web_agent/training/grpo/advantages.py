from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping, Sequence

import torch


def compute_group_advantages(
    rewards: Sequence[torch.Tensor] | torch.Tensor,
    group_ids: Sequence[str] | torch.Tensor,
    *,
    epsilon: float = 1e-6,
    variance_epsilon: float = 1e-12,
) -> torch.Tensor:
    """Return scalar group-relative advantages, preserving tensor device/dtype."""
    values = list(rewards) if not torch.is_tensor(rewards) else list(rewards.unbind(0))
    if len(values) != len(group_ids):
        raise ValueError("rewards and group_ids must have equal length")
    if not values:
        return torch.empty(0)
    result = torch.zeros(len(values), dtype=values[0].dtype, device=values[0].device)
    groups: dict[str, list[int]] = defaultdict(list)
    for index, group_id in enumerate(group_ids):
        key = str(group_id.item()) if torch.is_tensor(group_id) else str(group_id)
        groups[key].append(index)
    for indices in groups.values():
        if len(indices) == 1:
            continue
        group = torch.stack([values[index] for index in indices])
        mean = group.mean()
        std = group.std(unbiased=False)
        if float(std.detach().abs()) <= variance_epsilon:
            continue
        for index, value in zip(indices, group):
            result[index] = (value - mean) / (std + epsilon)
    return result


def group_advantage_statistics(rewards, group_ids, advantages, *, variance_epsilon=1e-12):
    values = list(rewards) if not torch.is_tensor(rewards) else list(rewards.unbind(0))
    groups: dict[str, list[int]] = defaultdict(list)
    for index, group_id in enumerate(group_ids):
        groups[str(group_id)] .append(index)
    rows = []
    for group_id, indices in sorted(groups.items()):
        group = torch.stack([values[i] for i in indices])
        adv = torch.stack([advantages[i] for i in indices])
        std = group.std(unbiased=False)
        rows.append({
            "prompt_uid": group_id,
            "group_reward_mean": float(group.mean()),
            "group_reward_std": float(std),
            "advantage_mean": float(adv.mean()),
            "advantage_std": float(adv.std(unbiased=False)),
            "positive_advantage_ratio": float((adv > 0).float().mean()),
            "negative_advantage_ratio": float((adv < 0).float().mean()),
            "zero_variance_group": bool(len(indices) == 1 or float(std) <= variance_epsilon),
        })
    return rows


compute_grpo_outcome_advantage = compute_group_advantages
