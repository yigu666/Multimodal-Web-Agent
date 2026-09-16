from __future__ import annotations

from typing import Sequence

import torch


def trajectory_balanced_policy_loss(
    old_log_probs: torch.Tensor,
    new_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    policy_action_mask: torch.Tensor,
    *,
    clip_ratio: float = 0.2,
) -> torch.Tensor:
    if not (old_log_probs.shape == new_log_probs.shape == advantages.shape == policy_action_mask.shape):
        raise ValueError("log probabilities, advantages and mask must have equal shape")
    if clip_ratio <= 0 or clip_ratio >= 1:
        raise ValueError("clip_ratio must be in (0, 1)")
    mask_bool = policy_action_mask.to(dtype=torch.bool)
    advantage_values = advantages.to(dtype=torch.float32)
    # QLoRA replay log-probabilities are stored in fp16. Computing exp in fp16
    # can overflow before PPO clipping, and 0 * inf then becomes NaN even for a
    # zero-advantage or masked token. Those tokens are mathematically inert, so
    # give them the neutral ratio 1 before exponentiation. Active ratios are
    # evaluated in fp32 without changing the PPO objective or its normalization.
    ratio_active = mask_bool & advantage_values.ne(0)
    log_ratio = new_log_probs.to(dtype=torch.float32) - old_log_probs.to(dtype=torch.float32)
    log_ratio = torch.where(ratio_active, log_ratio, torch.zeros_like(log_ratio))
    ratio = torch.exp(log_ratio)
    clipped_ratio = ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio)
    unclipped = -advantage_values * ratio
    clipped = -advantage_values * clipped_ratio
    token_loss = torch.maximum(unclipped, clipped)
    mask = policy_action_mask.to(dtype=token_loss.dtype)
    if token_loss.ndim == 1:
        return (token_loss * mask).sum() / mask.sum().clamp_min(1.0)
    per_trajectory = (token_loss * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1.0)
    return per_trajectory.mean()


def compute_policy_loss(*args, **kwargs):
    return trajectory_balanced_policy_loss(*args, **kwargs)


def policy_loss_metrics(old_log_probs, new_log_probs, policy_action_mask, clip_ratio=0.2):
    mask_bool = policy_action_mask.to(dtype=torch.bool)
    log_ratio = new_log_probs.to(dtype=torch.float32) - old_log_probs.to(dtype=torch.float32)
    log_ratio = torch.where(mask_bool, log_ratio, torch.zeros_like(log_ratio))
    ratio = torch.exp(log_ratio)
    mask = policy_action_mask.to(dtype=ratio.dtype)
    return {
        "policy_ratio_mean": float((ratio * mask).sum() / mask.sum().clamp_min(1.0)),
        "policy_ratio_max": float((ratio * mask).max()) if bool(mask.any()) else 0.0,
        "clip_fraction": float((((ratio < 1 - clip_ratio) | (ratio > 1 + clip_ratio)).float() * mask).sum() / mask.sum().clamp_min(1.0)),
    }
