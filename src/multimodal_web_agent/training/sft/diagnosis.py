from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from multimodal_web_agent.agent import parse_action
from multimodal_web_agent.data.protocol_sft.schema import StateActionExample

from .generation import GenerationRecord


KNOWN_CHECKPOINTS = (
    "checkpoint-epoch-1",
    "checkpoint-epoch-2",
    "checkpoint-epoch-3",
    "best_adapter",
    "final_adapter",
)


def checkpoint_label(path: Path) -> str:
    name = path.name
    if name.startswith("checkpoint-epoch-"):
        return "epoch_" + name.rsplit("-", 1)[-1]
    return name


def discover_adapter_directories(root: Path) -> list[Dict[str, Any]]:
    root = Path(root)
    entries: Dict[str, Dict[str, Any]] = {}
    for name in KNOWN_CHECKPOINTS:
        path = root / name
        entries[str(path.resolve())] = {
            "name": checkpoint_label(path),
            "path": str(path).replace("\\", "/"),
            "status": (
                "ready"
                if (path / "adapter_config.json").is_file()
                else "missing_adapter_config"
            ),
        }
    if root.is_dir():
        for config_path in sorted(root.rglob("adapter_config.json")):
            path = config_path.parent
            key = str(path.resolve())
            entries[key] = {
                "name": checkpoint_label(path),
                "path": str(path).replace("\\", "/"),
                "status": "ready",
            }
    return sorted(entries.values(), key=lambda item: (item["name"], item["path"]))


def _question(example: StateActionExample) -> str:
    for message in example.state:
        if message.role == "user" and "<image>" in message.content:
            return message.content.replace("<image>", "", 1).strip()
    return ""


def transition_prediction_table(
    examples: Sequence[StateActionExample],
    records: Sequence[GenerationRecord],
) -> Dict[str, Any]:
    if len(examples) != len(records):
        raise ValueError("examples and predictions must have equal lengths")
    table: Dict[str, Counter[str]] = defaultdict(Counter)
    for example, record in zip(examples, records):
        parsed = parse_action(record.generated_text)
        predicted = (
            parsed.action_type.value
            if parsed.valid and parsed.action_type is not None
            else "invalid"
        )
        table[example.transition]["target_count"] += 1
        table[example.transition][predicted] += 1
    labels = ("answer", "image_search", "text_search", "invalid")
    return {
        transition: {
            "target_count": counts["target_count"],
            **{label: counts[label] for label in labels},
        }
        for transition, counts in sorted(table.items())
    }


def failure_examples(
    examples: Sequence[StateActionExample],
    records: Sequence[GenerationRecord],
) -> Dict[str, list[Dict[str, Any]]]:
    limits: Mapping[str, int | None] = {
        "initial_to_direct_answer": 10,
        "initial_to_text_search": 10,
        "image_information_to_text_search": None,
    }
    result: Dict[str, list[Dict[str, Any]]] = {
        transition: [] for transition in limits
    }
    for example, record in zip(examples, records):
        if example.transition not in limits:
            continue
        target = parse_action(example.target)
        parsed = parse_action(record.generated_text)
        if parsed.valid and parsed.action_type == target.action_type:
            continue
        limit = limits[example.transition]
        if limit is not None and len(result[example.transition]) >= limit:
            continue
        result[example.transition].append({
            "sample_id": example.example_id,
            "question": _question(example),
            "state_type": example.transition,
            "target": example.target,
            "generated_text": record.generated_text,
            "parsed_action": (
                parsed.action_type.value
                if parsed.valid and parsed.action_type is not None
                else "invalid"
            ),
        })
    return result

