from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from .schema import (
    EvaluatedModel,
    REGISTRY_SCHEMA_VERSION,
    V1_1_REGISTRY_SCHEMA_VERSION,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(path: Path) -> str:
    root = Path(path)
    if not root.is_dir():
        raise FileNotFoundError(root)
    files = sorted(item for item in root.rglob("*") if item.is_file())
    if not files:
        raise ValueError("fingerprinted directory is empty")
    digest = hashlib.sha256()
    for item in files:
        digest.update(item.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def safetensors_parameter_count(path: Path) -> int:
    root = Path(path)
    files = sorted(root.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError("no safetensors files under %s" % root)
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError("parameter counting requires safetensors") from exc
    count = 0
    seen = set()
    for file_path in files:
        with safe_open(file_path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in seen:
                    continue
                seen.add(key)
                shape = handle.get_slice(key).get_shape()
                product = 1
                for dimension in shape:
                    product *= int(dimension)
                count += product
    return count


def model_fingerprint(
    *,
    model_id: str,
    stage: str,
    base_model_path: Path,
    adapter_path: Path | None,
) -> EvaluatedModel:
    base = Path(base_model_path)
    adapter = Path(adapter_path) if adapter_path is not None else None
    base_count = safetensors_parameter_count(base)
    adapter_count = safetensors_parameter_count(adapter) if adapter else 0
    model = EvaluatedModel(
        model_id=model_id,
        stage=stage,
        base_model_path=base.as_posix(),
        adapter_path=adapter.as_posix() if adapter else None,
        model_tree_sha256=sha256_tree(base),
        adapter_tree_sha256=sha256_tree(adapter) if adapter else None,
        parameter_count=base_count + adapter_count,
        base_parameter_count=base_count,
        adapter_parameter_count=adapter_count,
    )
    model.validate()
    return model


def initialize_registry(path: Path) -> dict[str, Any]:
    path = Path(path)
    if path.exists():
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("schema_version") not in {
            REGISTRY_SCHEMA_VERSION,
            V1_1_REGISTRY_SCHEMA_VERSION,
        }:
            raise ValueError("model registry schema mismatch")
        return value
    schema_version = (
        V1_1_REGISTRY_SCHEMA_VERSION
        if "v1_1" in path.name else REGISTRY_SCHEMA_VERSION
    )
    value = {
        "schema_version": schema_version,
        "registered_models": {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return value


def register_model(path: Path, model: EvaluatedModel) -> dict[str, Any]:
    model.validate()
    path = Path(path)
    registry = initialize_registry(path)
    models = dict(registry["registered_models"])
    existing = models.get(model.model_id)
    current = model.to_dict()
    if existing is not None and existing != current:
        raise RuntimeError(
            "registered model fingerprint is immutable: %s" % model.model_id
        )
    models[model.model_id] = current
    updated = {**registry, "registered_models": models}
    temporary = path.with_name(path.name + ".tmp-%d" % os.getpid())
    temporary.write_text(
        json.dumps(updated, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return updated


def registered_models(path: Path) -> dict[str, EvaluatedModel]:
    value = initialize_registry(path)
    return {
        key: EvaluatedModel.from_dict(model)
        for key, model in value["registered_models"].items()
    }


def registry_hashes(path: Path) -> dict[str, str]:
    return {
        model_id: hashlib.sha256(json.dumps(
            model.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")).hexdigest()
        for model_id, model in registered_models(path).items()
    }


def validate_registered_model(
    registry_path: Path,
    model_id: str,
    configured: Mapping[str, Any],
) -> EvaluatedModel:
    models = registered_models(registry_path)
    if model_id not in models:
        raise ValueError("model is not registered: %s" % model_id)
    model = models[model_id]
    if model.stage != configured["stage"]:
        raise ValueError("registered model stage differs from config")
    registry_path = Path(registry_path).resolve()
    project_root = registry_path.parents[2]
    registered_base = Path(model.base_model_path)
    configured_base = Path(configured["base_model_path"])
    if not registered_base.is_absolute():
        registered_base = project_root / registered_base
    if not configured_base.is_absolute():
        configured_base = project_root / configured_base
    if registered_base.resolve() != configured_base.resolve():
        raise ValueError("registered base model path differs from config")
    expected_adapter = configured.get("adapter_path")
    registered_adapter = (
        Path(model.adapter_path) if model.adapter_path else None
    )
    configured_adapter = (
        Path(expected_adapter) if expected_adapter else None
    )
    if registered_adapter is not None and not registered_adapter.is_absolute():
        registered_adapter = project_root / registered_adapter
    if configured_adapter is not None and not configured_adapter.is_absolute():
        configured_adapter = project_root / configured_adapter
    if (
        registered_adapter.resolve() if registered_adapter else None
    ) != (
        configured_adapter.resolve() if configured_adapter else None
    ):
        raise ValueError("registered Adapter path differs from config")
    if sha256_tree(registered_base) != model.model_tree_sha256:
        raise RuntimeError("registered base model fingerprint changed")
    if registered_adapter:
        if sha256_tree(registered_adapter) != model.adapter_tree_sha256:
            raise RuntimeError("registered Adapter fingerprint changed")
    return model
