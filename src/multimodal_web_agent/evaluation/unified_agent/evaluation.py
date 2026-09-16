from __future__ import annotations

import gc
from dataclasses import replace
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from multimodal_web_agent.training.sft.checkpoint import (
    load_adapter_into_model,
)
from multimodal_web_agent.training.sft.config import load_config
from multimodal_web_agent.training.sft.model_factory import load_qwen_base
from multimodal_web_agent.training.sft.raw_vs_sft_format import (
    assert_raw_model_has_no_adapter,
    assert_sft_adapter_loaded,
)

from .agent_metrics import aggregate_agent_metrics
from .environment import FrozenToolEnvironment
from .fingerprints import validate_registered_model
from .runner import UnifiedAgentRunner
from .schema import UnifiedEvalExample


def read_examples(path: Path, expected_count: int) -> list[UnifiedEvalExample]:
    examples = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                examples.append(UnifiedEvalExample.from_dict(json.loads(line)))
    if len(examples) != expected_count:
        raise ValueError(
            "%s has %d examples; expected %d"
            % (path, len(examples), expected_count)
        )
    if len({item.eval_id for item in examples}) != len(examples):
        raise ValueError("duplicate Unified Eval IDs")
    return examples


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, values: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for value in values:
            handle.write(json.dumps(
                dict(value), ensure_ascii=False, sort_keys=True
            ) + "\n")


def load_runtime_model(
    project_root: Path,
    runner_config: Mapping[str, Any],
    model_config: Mapping[str, Any],
    model_id: str,
) -> tuple[Any, Any, Any]:
    configured = model_config["models"][model_id]
    if configured.get("enabled") is not True:
        raise ValueError("model is disabled: %s" % model_id)
    registry_path = project_root / runner_config["model_registry"]
    registered = validate_registered_model(
        registry_path, model_id, configured
    )
    training_config = load_config(
        project_root / runner_config["training_config"], project_root
    )
    configured_base = Path(configured["base_model_path"])
    if configured_base != training_config.model.path:
        training_config = replace(
            training_config,
            model=replace(training_config.model, path=configured_base),
        )
    model, processor = load_qwen_base(training_config)
    if model_id == "raw" or not configured.get("adapter_path"):
        assert_raw_model_has_no_adapter(model)
    else:
        adapter = Path(str(configured["adapter_path"]))
        if not adapter.is_absolute():
            adapter = project_root / adapter
        model = load_adapter_into_model(model, adapter)
        assert_sft_adapter_loaded(model)
    model.config.use_cache = True
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None:
        generation_config.do_sample = False
        generation_config.num_beams = 1
        generation_config.temperature = None
        generation_config.top_p = None
        generation_config.top_k = None
    return model, processor, registered


def clear_runtime_memory() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def evaluate_loaded_model(
    *,
    project_root: Path,
    runner_config: Mapping[str, Any],
    model_id: str,
    model: Any,
    processor: Any,
    examples: Sequence[UnifiedEvalExample],
    split: str,
    output_dir: Path,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError("refusing to overwrite model evaluation output")
    temporary = output_dir.with_name(
        ".%s.tmp-%d" % (output_dir.name, os.getpid())
    )
    if temporary.exists():
        raise FileExistsError(temporary)
    temporary.mkdir(parents=True)
    environment = FrozenToolEnvironment(
        project_root / runner_config["environment_dir"],
        image_search_top_k=5,
        text_search_top_k=5,
    )
    runner = UnifiedAgentRunner(
        model=model,
        processor=processor,
        environment=environment,
        model_id=model_id,
        max_new_tokens_per_turn=int(
            runner_config["generation"]["max_new_tokens_per_turn"]
        ),
    )
    rows = []
    for index, example in enumerate(examples, 1):
        result = runner.run_path(
            example, project_root / runner_config["dataset_dir"]
        )
        rows.append(result.to_dict())
        if index % 25 == 0 or index == len(examples):
            print(
                "%s %s: %d/%d"
                % (model_id, split, index, len(examples)),
                flush=True,
            )
    metrics, diagnostics = aggregate_agent_metrics(rows)
    manifest = {
        "schema_version": "unified-agent-evaluation-run-v1",
        "model_id": model_id,
        "split": split,
        "episode_count": len(rows),
        "dev_results_are_not_final_benchmark": split == "dev",
        "test_accessed": split == "test",
        "training_performed": False,
        "dynamic_internet_accessed": False,
        "natural_language_fallback_used": False,
        "generation": dict(runner_config["generation"]),
        "budgets": dict(runner_config["budgets"]),
        "renderer": runner.renderer.manifest_metadata(),
    }
    write_jsonl(temporary / "episodes.jsonl", rows)
    write_json(temporary / "metrics.json", metrics)
    write_json(temporary / "diagnostics.json", diagnostics)
    write_json(temporary / "run_manifest.json", manifest)
    os.replace(temporary, output_dir)
    return {
        "rows": rows,
        "metrics": metrics,
        "diagnostics": diagnostics,
        "manifest": manifest,
    }
