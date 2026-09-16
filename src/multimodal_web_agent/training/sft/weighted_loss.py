from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn.functional as F

from .target_segments import (
    SEGMENT_ACTION_TAG,
    SEGMENT_ANSWER_PAYLOAD,
    SEGMENT_QUERY_PAYLOAD,
    SEGMENT_REASON,
    audit_weight_tensors,
)


def weighted_causal_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    token_weights: torch.Tensor,
    segment_ids: torch.Tensor,
) -> tuple[torch.Tensor, Dict[str, Any]]:
    if logits.ndim != 3 or labels.ndim != 2:
        raise ValueError("expected [batch, sequence, vocab] logits")
    if logits.shape[:2] != labels.shape:
        raise ValueError("logits and labels sequence dimensions differ")
    audit_weight_tensors(labels, token_weights, segment_ids)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_weights = token_weights[..., 1:].contiguous()
    shift_segments = segment_ids[..., 1:].contiguous()
    vocab_size = shift_logits.shape[-1]
    per_token_loss = F.cross_entropy(
        shift_logits.view(-1, vocab_size),
        shift_labels.view(-1),
        reduction="none",
        ignore_index=-100,
    ).view_as(shift_labels)
    active = shift_labels.ne(-100)
    active_count = int(active.sum().item())
    if active_count <= 0:
        raise ValueError("Shifted target contains no active token")
    active_weights = shift_weights[active]
    if not torch.isfinite(active_weights).all():
        raise ValueError("Shifted active weights must be finite")
    if torch.any(active_weights <= 0):
        raise ValueError("Shifted active weights must be positive")
    weight_sum = active_weights.sum()
    if not torch.isfinite(weight_sum) or float(weight_sum.item()) <= 0:
        raise ValueError("Shifted active weight sum must be positive")
    total = (
        per_token_loss[active] * active_weights
    ).sum() / weight_sum

    def segment_stats(segment_id: int) -> tuple[float, int]:
        selected = active & shift_segments.eq(segment_id)
        count = int(selected.sum().item())
        return (
            float(per_token_loss[selected].mean().detach().cpu())
            if count else 0.0,
            count,
        )

    reason_loss, reason_count = segment_stats(SEGMENT_REASON)
    action_loss, action_count = segment_stats(SEGMENT_ACTION_TAG)
    query_loss, query_count = segment_stats(SEGMENT_QUERY_PAYLOAD)
    answer_loss, answer_count = segment_stats(SEGMENT_ANSWER_PAYLOAD)
    stats: Dict[str, Any] = {
        "total_weighted_loss": float(total.detach().cpu()),
        "unweighted_target_loss": float(
            per_token_loss[active].mean().detach().cpu()
        ),
        "reason_loss": reason_loss,
        "action_tag_loss": action_loss,
        "query_payload_loss": query_loss,
        "answer_payload_loss": answer_loss,
        "reason_active_token_count": reason_count,
        "action_active_token_count": action_count,
        "query_active_token_count": query_count,
        "answer_active_token_count": answer_count,
        "active_token_count": active_count,
        "active_weight_sum": float(weight_sum.detach().cpu()),
    }
    return total, stats
