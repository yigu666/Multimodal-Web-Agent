from __future__ import annotations

import importlib.metadata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .config import SFTConfig


@dataclass(frozen=True)
class ModelAudit:
    matched_lora_module_names: tuple[str, ...]
    matched_lora_module_count: int
    trainable_parameter_count: int
    total_parameter_count: int
    trainable_parameter_ratio: float
    trainable_visual_parameter_count: int
    trainable_base_parameter_count: int
    gradient_checkpointing: bool
    quantization: str
    compute_dtype: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_quantization_config(config: SFTConfig) -> Any:
    if config.model.quant_type != "nf4" or config.model.compute_dtype != "float16":
        raise ValueError("Protocol-SFT requires NF4 with float16 compute")
    try:
        from transformers import BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError("transformers is required for NF4 QLoRA") from exc
    try:
        import torch
        dtype = torch.float16
    except ImportError as exc:
        raise RuntimeError("torch is required for NF4 QLoRA") from exc
    return BitsAndBytesConfig(
        load_in_4bit=config.model.load_in_4bit,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=config.model.double_quant,
        bnb_4bit_compute_dtype=dtype,
    )


def select_lora_target_modules(
    model: Any,
    target_suffixes: Sequence[str],
    excluded_name_fragments: Sequence[str],
) -> List[str]:
    """Return language-module paths only; suffix matching alone is insufficient."""
    suffixes = set(target_suffixes)
    excluded = tuple(fragment.casefold() for fragment in excluded_name_fragments)
    language_markers = (
        "language_model",
        "model.layers",
        "transformer.layers",
        "text_model",
    )
    names: List[str] = []
    for name, _module in model.named_modules():
        lowered = name.casefold()
        leaf = name.rsplit(".", 1)[-1]
        if leaf not in suffixes:
            continue
        if any(fragment in lowered for fragment in excluded):
            continue
        if not any(marker in lowered for marker in language_markers):
            continue
        names.append(name)
    return sorted(set(names))


def freeze_visual_modules(model: Any, excluded_fragments: Sequence[str]) -> None:
    excluded = tuple(fragment.casefold() for fragment in excluded_fragments)
    for name, parameter in model.named_parameters():
        if any(fragment in name.casefold() for fragment in excluded):
            parameter.requires_grad = False


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def audit_model(model: Any, target_names: Sequence[str], config: SFTConfig) -> ModelAudit:
    trainable = 0
    total = 0
    visual_trainable = 0
    base_trainable = 0
    excluded = tuple(fragment.casefold() for fragment in config.lora.excluded_name_fragments)
    for name, parameter in model.named_parameters():
        total += parameter.numel()
        if parameter.requires_grad:
            trainable += parameter.numel()
            if "lora_" not in name.casefold():
                base_trainable += parameter.numel()
            if any(fragment in name.casefold() for fragment in excluded):
                visual_trainable += parameter.numel()
    if not target_names:
        raise RuntimeError("no language-model LoRA target modules matched")
    if trainable <= 0:
        raise RuntimeError("LoRA has no trainable parameters")
    if visual_trainable != 0:
        raise RuntimeError("visual modules have trainable parameters")
    if base_trainable != 0:
        raise RuntimeError("base-model parameters remain trainable")
    checkpointing = bool(getattr(model, "is_gradient_checkpointing", False))
    return ModelAudit(
        matched_lora_module_names=tuple(target_names),
        matched_lora_module_count=len(target_names),
        trainable_parameter_count=trainable,
        total_parameter_count=total,
        trainable_parameter_ratio=trainable / total if total else 0.0,
        trainable_visual_parameter_count=visual_trainable,
        trainable_base_parameter_count=base_trainable,
        gradient_checkpointing=checkpointing,
        quantization="NF4",
        compute_dtype="float16",
    )


def load_processor(config: SFTConfig) -> Any:
    try:
        from transformers import AutoProcessor
    except ImportError as exc:
        raise RuntimeError("transformers is required for Protocol-SFT") from exc
    effective_pixels = int(config.data.pixel_budget)
    return AutoProcessor.from_pretrained(
        config.model.path,
        min_pixels=effective_pixels,
        max_pixels=effective_pixels,
        local_files_only=config.model.local_files_only,
        use_fast=False,
    )


def load_qwen_lora(
    config: SFTConfig,
    *,
    processor: Any = None,
    adapter_path: Optional[Path] = None,
) -> tuple[Any, Any, ModelAudit]:
    try:
        from peft import (
            LoraConfig,
            PeftModel,
            get_peft_model,
            prepare_model_for_kbit_training,
        )
        from transformers import Qwen2_5_VLForConditionalGeneration
    except ImportError as exc:
        raise RuntimeError("torch, transformers and peft are required for QLoRA training") from exc
    model, loaded_processor = load_qwen_base(
        config,
        Qwen2_5_VLForConditionalGeneration,
        processor=processor,
    )
    processor = loaded_processor
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    if config.model.gradient_checkpointing:
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": config.model.use_reentrant}
            )
        except TypeError:
            model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    freeze_visual_modules(model, config.lora.excluded_name_fragments)
    target_names = select_lora_target_modules(
        model, config.lora.target_suffixes, config.lora.excluded_name_fragments
    )
    if adapter_path is None:
        lora_config = LoraConfig(
            r=config.lora.r,
            lora_alpha=config.lora.alpha,
            lora_dropout=config.lora.dropout,
            bias=config.lora.bias,
            task_type=config.lora.task_type,
            target_modules=target_names,
        )
        model = get_peft_model(model, lora_config)
    else:
        model = PeftModel.from_pretrained(
            model,
            Path(adapter_path),
            is_trainable=True,
        )
    freeze_visual_modules(model, config.lora.excluded_name_fragments)
    audit = audit_model(model, target_names, config)
    return model, processor, audit


def load_qwen_base(
    config: SFTConfig,
    model_class: Any = None,
    *,
    processor: Any = None,
    attn_implementation: str = "eager",
) -> tuple[Any, Any]:
    try:
        import torch
        from transformers import Qwen2_5_VLForConditionalGeneration
    except ImportError as exc:
        raise RuntimeError("torch and transformers are required for Qwen2.5-VL") from exc
    if model_class is None:
        model_class = Qwen2_5_VLForConditionalGeneration
    quantization_config = build_quantization_config(config)
    model = model_class.from_pretrained(
        config.model.path,
        quantization_config=quantization_config,
        torch_dtype=torch.float16,
        device_map={"": 0},
        attn_implementation=attn_implementation,
        local_files_only=config.model.local_files_only,
        low_cpu_mem_usage=True,
    )
    if processor is None:
        processor = load_processor(config)
    model.config.use_cache = False
    return model, processor


def environment_versions() -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "python": __import__("sys").version,
        "torch": _package_version("torch"),
        "transformers": _package_version("transformers"),
        "accelerate": _package_version("accelerate"),
        "peft": _package_version("peft"),
        "bitsandbytes": _package_version("bitsandbytes"),
    }
    try:
        import torch
        result["cuda"] = {
            "available": bool(torch.cuda.is_available()),
            "runtime_version": torch.version.cuda,
            "device_count": int(torch.cuda.device_count()),
        }
        if torch.cuda.is_available():
            index = int(torch.cuda.current_device())
            properties = torch.cuda.get_device_properties(index)
            result["cuda"].update({
                "current_device": index,
                "device_name": torch.cuda.get_device_name(index),
                "compute_capability": list(torch.cuda.get_device_capability(index)),
                "total_memory": int(properties.total_memory),
            })
    except (ImportError, RuntimeError):
        result["cuda"] = {"available": False}
    return result
