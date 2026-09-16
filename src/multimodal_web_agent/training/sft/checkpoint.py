from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import yaml

from .config import SFTConfig, sha256_file


def snapshot_trainable_parameters(model: Any) -> Dict[str, str]:
    snapshots: Dict[str, str] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        value = parameter.detach().cpu().contiguous().numpy().tobytes()
        snapshots[name] = hashlib.sha256(value).hexdigest()
    return snapshots


def changed_parameter_count(before: Mapping[str, str], model: Any) -> int:
    after = snapshot_trainable_parameters(model)
    return sum(before.get(name) != digest for name, digest in after.items())


def save_adapter_checkpoint(
    model: Any,
    processor: Any,
    output_dir: Path,
    *,
    config: SFTConfig,
    data_manifest_hash: str,
    model_audit: Mapping[str, Any],
    training_manifest: Optional[Mapping[str, Any]] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    if hasattr(processor, "save_pretrained"):
        processor.save_pretrained(output_dir / "processor")
    metadata: Dict[str, Any] = {
        "base_model_path": str(config.model.path).replace("\\", "/"),
        "data_manifest_hash": data_manifest_hash,
        "config_hash": config.config_hash,
        "lora_config": config.to_dict(relative=False)["lora"],
        "model_audit": dict(model_audit),
    }
    metadata.update(dict(extra or {}))
    (output_dir / "checkpoint_contract.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(
            config.to_dict(relative=True),
            allow_unicode=True,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    if training_manifest is not None:
        (output_dir / "train_manifest.json").write_text(
            json.dumps(
                dict(training_manifest),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    return output_dir


def validate_resume_checkpoint(
    checkpoint_dir: Path,
    config: SFTConfig,
    data_manifest_hash: str,
) -> Dict[str, Any]:
    checkpoint_dir = Path(checkpoint_dir)
    contract_path = checkpoint_dir / "checkpoint_contract.json"
    if not contract_path.is_file():
        raise ValueError("checkpoint contract is missing: %s" % checkpoint_dir)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    expected_model = str(config.model.path).replace("\\", "/")
    if contract.get("base_model_path") != expected_model:
        raise ValueError("checkpoint base model path does not match current config")
    if contract.get("data_manifest_hash") != data_manifest_hash:
        raise ValueError("checkpoint data Manifest hash does not match current data")
    if contract.get("config_hash") != config.config_hash:
        raise ValueError("checkpoint training config hash does not match current config")
    expected_lora = config.to_dict(relative=False)["lora"]
    if contract.get("lora_config") != expected_lora:
        raise ValueError("checkpoint LoRA configuration does not match current config")
    return contract


def load_adapter_into_model(
    model: Any,
    checkpoint_dir: Path,
    *,
    is_trainable: bool = False,
) -> Any:
    try:
        from peft import PeftModel
    except ImportError as exc:
        raise RuntimeError("peft is required to load an adapter checkpoint") from exc
    return PeftModel.from_pretrained(
        model,
        Path(checkpoint_dir),
        is_trainable=is_trainable,
    )


def load_adapter_for_resume(model: Any, checkpoint_dir: Path) -> Any:
    """Load a named resume adapter into an already prepared PEFT model."""
    if not hasattr(model, "load_adapter"):
        raise RuntimeError("model does not expose the PEFT load_adapter API")
    model.load_adapter(str(Path(checkpoint_dir)), adapter_name="resume", is_trainable=True)
    if hasattr(model, "set_adapter"):
        model.set_adapter("resume")
    return model


def data_manifest_hash(manifest_path: Path) -> str:
    return sha256_file(Path(manifest_path))
