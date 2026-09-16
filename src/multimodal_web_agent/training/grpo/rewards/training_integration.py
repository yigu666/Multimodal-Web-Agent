from __future__ import annotations

import math
import statistics
from typing import Any, Sequence

import torch

from multimodal_web_agent.training.grpo.input_identity import assert_input_identity
from multimodal_web_agent.training.grpo.logprob_alignment import assert_logprob_alignment
from multimodal_web_agent.training.grpo.policy_loss import (
    policy_loss_metrics,
    trajectory_balanced_policy_loss,
)


def _identity(record: Any) -> dict[str, Any]:
    return {
        "input_ids": record.full_input_ids,
        "response_ids": record.response_ids,
        "policy_action_mask": record.policy_action_mask,
        "information_mask": record.information_mask,
        "image_sha256": record.image_sha256,
    }


def update_records_v2(
    *,
    model: Any,
    trajectory_runner: Any,
    optimizer: Any,
    records: Sequence[Any],
    token_advantages: Sequence[torch.Tensor],
    clip_ratio: float = 0.2,
    max_grad_norm: float = 1.0,
    verify_alignment: bool = True,
) -> dict[str, Any]:
    """Apply the existing trajectory-balanced clipped loss to v2 token credit."""
    if len(records) != len(token_advantages) or not records:
        raise ValueError("records and Reward v2 token advantages must be non-empty and aligned")
    identity_failures = response_mismatches = information_leaks = 0
    processor_mismatches = target_truncations = 0
    maximum_alignment_error = 0.0
    maximum_behavior_alignment_error = 0.0
    policy_tokens = local_override_tokens = 0
    for record, advantages in zip(records, token_advantages):
        input_ids = torch.as_tensor(record.full_input_ids)
        response_ids = torch.as_tensor(record.response_ids)
        policy_mask = torch.as_tensor(record.policy_action_mask)
        information_mask = torch.as_tensor(record.information_mask)
        values = torch.as_tensor(advantages)
        if not torch.equal(input_ids[1:], response_ids):
            response_mismatches += 1
        if record.processor_hash != trajectory_runner.processor_hash:
            processor_mismatches += 1
        if int(input_ids.numel()) > trajectory_runner.max_seq_len:
            target_truncations += 1
        information_leaks += int((policy_mask * information_mask).sum().item())
        policy_tokens += int(policy_mask.sum().item())
        if values.shape != policy_mask.shape or not torch.isfinite(values).all():
            raise RuntimeError("Reward v2 token advantage identity/finite contract failed")
        if int((values[~policy_mask.bool()] != 0).sum().item()) != 0:
            raise RuntimeError("Reward v2 assigned advantage outside the policy mask")
        terminal = float(record.reward_components["terminal_advantage"])
        local_override_tokens += int(((values != terminal) & policy_mask.bool()).sum().item())
        try:
            assert_input_identity(_identity(record), {
                "input_ids": input_ids, "response_ids": input_ids[1:],
                "policy_action_mask": policy_mask, "information_mask": information_mask,
                "image_sha256": record.image_sha256,
            })
        except AssertionError:
            identity_failures += 1
    failures = {
        "input_identity_failure_count": identity_failures,
        "response_token_mismatch_count": response_mismatches,
        "processor_hash_mismatch_count": processor_mismatches,
        "target_truncation_count": target_truncations,
        "information_token_train_mask_sum": information_leaks,
    }
    if any(failures.values()) or policy_tokens <= 0:
        raise RuntimeError(f"Reward v2 rollout/update identity contract failed: {failures}")
    if verify_alignment:
        model.eval()
        if hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()
        if hasattr(model, "config"):
            model.config.use_cache = True
        for record in records:
            if (
                bool(getattr(record, "exploration_metadata", {}).get(
                    "exploration_selected", False
                ))
                and hasattr(trajectory_runner, "replay_behavior_log_probs")
            ):
                recomputed = trajectory_runner.replay_behavior_log_probs(
                    record
                ).detach().cpu()
            else:
                recomputed = trajectory_runner.replay_log_probs(
                    record, grad=False
                ).detach().cpu()
            error = assert_logprob_alignment(
                torch.as_tensor(record.old_log_probs), recomputed,
                mask=torch.as_tensor(record.policy_action_mask), tolerance=1e-3,
            )
            maximum_alignment_error = max(maximum_alignment_error, error)
            if bool(getattr(record, "exploration_metadata", {}).get(
                "exploration_selected", False
            )):
                maximum_behavior_alignment_error = max(
                    maximum_behavior_alignment_error, error
                )
    if hasattr(model, "config"):
        model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses, ratio_rows = [], []
    for record, advantages in zip(records, token_advantages):
        new = trajectory_runner.replay_log_probs(record, grad=True)
        old = torch.as_tensor(record.old_log_probs, device=new.device)
        mask = torch.as_tensor(record.policy_action_mask, device=new.device)
        token_values = torch.as_tensor(advantages, dtype=new.dtype, device=new.device)
        new_flat = new.detach().reshape(-1).cpu()
        old_flat = old.detach().reshape(-1).cpu()
        for trace in getattr(record, "exploration_trace", []):
            position = trace.get("action_boundary_index")
            if position is None:
                continue
            response_position = int(position) - 1
            if not 0 <= response_position < new_flat.numel():
                raise RuntimeError("exploration action boundary left the response")
            new_value = float(new_flat[response_position])
            behavior_value = float(old_flat[response_position])
            ratio = math.exp(new_value - behavior_value)
            if not all(math.isfinite(value) for value in (
                new_value, behavior_value, ratio
            )):
                raise RuntimeError("non-finite exploration PPO ratio")
            trace["new_policy_logprob"] = new_value
            trace["ppo_ratio_pi_new_over_mu_old"] = ratio
        loss = trajectory_balanced_policy_loss(old, new, token_values, mask, clip_ratio=clip_ratio)
        if not torch.isfinite(loss):
            active = mask.to(dtype=torch.bool)
            nonzero = active & token_values.ne(0)
            log_ratio = new.detach().float() - old.detach().float()
            finite_log_ratio = log_ratio[nonzero & torch.isfinite(log_ratio)]
            log_ratio_min = (
                float(finite_log_ratio.min().cpu()) if finite_log_ratio.numel() else None
            )
            log_ratio_max = (
                float(finite_log_ratio.max().cpu()) if finite_log_ratio.numel() else None
            )
            raise RuntimeError(
                "non-finite Reward v2 policy loss: "
                f"rollout_uid={getattr(record, 'rollout_uid', '')} "
                f"policy_tokens={int(active.sum().cpu())} "
                f"nonzero_advantage_tokens={int(nonzero.sum().cpu())} "
                f"nonfinite_old={int((active & ~torch.isfinite(old)).sum().cpu())} "
                f"nonfinite_new={int((active & ~torch.isfinite(new)).sum().cpu())} "
                f"nonfinite_advantage={int((active & ~torch.isfinite(token_values)).sum().cpu())} "
                f"active_log_ratio_min={log_ratio_min} "
                f"active_log_ratio_max={log_ratio_max}"
            )
        (loss / len(records)).backward()
        losses.append(float(loss.detach()))
        ratio_rows.append(policy_loss_metrics(old.detach(), new.detach(), mask, clip_ratio=clip_ratio))
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    gradient_finite = all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all()) for parameter in trainable)
    if not gradient_finite:
        raise RuntimeError("non-finite Reward v2 gradient")
    gradient_norm = float(torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm))
    if not math.isfinite(gradient_norm):
        raise RuntimeError("non-finite Reward v2 gradient norm")
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return {
        "loss": statistics.mean(losses),
        "gradient_norm": gradient_norm,
        "gradient_finite": gradient_finite,
        "clip_fraction": statistics.mean(row["clip_fraction"] for row in ratio_rows),
        "policy_ratio_mean": statistics.mean(row["policy_ratio_mean"] for row in ratio_rows),
        "policy_ratio_max": max(row["policy_ratio_max"] for row in ratio_rows),
        "policy_action_token_count": policy_tokens,
        "local_override_token_count": local_override_tokens,
        "max_logprob_alignment_error": maximum_alignment_error,
        "exploration_behavior_logprob_error_max": (
            maximum_behavior_alignment_error
        ),
        **failures,
    }
