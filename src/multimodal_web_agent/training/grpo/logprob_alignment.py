from __future__ import annotations

from typing import Any

import torch


def selected_token_log_probs(logits: torch.Tensor, response_ids: torch.Tensor, *, temperature: float = 1.0) -> torch.Tensor:
    if logits.ndim != 3 or response_ids.ndim != 2:
        raise ValueError("logits must be [batch,time,vocab] and response_ids [batch,time]")
    if logits.shape[:2] != response_ids.shape:
        raise ValueError("logits and response_ids are not aligned")
    return torch.log_softmax(logits / temperature, dim=-1).gather(-1, response_ids.unsqueeze(-1)).squeeze(-1)


def max_logprob_alignment_error(stored: torch.Tensor, recomputed: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    if stored.shape != recomputed.shape:
        raise AssertionError("response log-prob shape mismatch")
    diff = (stored - recomputed).abs()
    if mask is not None:
        values = diff[mask.bool()]
        return float(values.max()) if values.numel() else 0.0
    return float(diff.max()) if diff.numel() else 0.0


def assert_logprob_alignment(stored, recomputed, *, mask=None, tolerance: float = 1e-3) -> float:
    error = max_logprob_alignment_error(stored, recomputed, mask)
    if error >= tolerance:
        raise AssertionError(f"max_abs_logprob_error={error} >= tolerance={tolerance}")
    return error
