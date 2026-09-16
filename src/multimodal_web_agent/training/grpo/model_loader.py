from __future__ import annotations

from pathlib import Path
from typing import Any


def load_grpo_adapter(config: Any, *, adapter_path: Path | None = None):
    """Reuse the frozen SFT QLoRA loader; GRPO never creates a new adapter."""
    from multimodal_web_agent.training.sft.model_factory import load_qwen_lora
    return load_qwen_lora(config, adapter_path=adapter_path or getattr(config, "adapter_path", None))
