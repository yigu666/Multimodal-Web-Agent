from __future__ import annotations

from typing import Any


def build_paged_adamw_8bit(parameters, *, learning_rate: float = 5e-7, weight_decay: float = 0.0):
    try:
        import bitsandbytes as bnb
        return bnb.optim.PagedAdamW8bit(parameters, lr=learning_rate, weight_decay=weight_decay)
    except ImportError:
        import torch
        return torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
