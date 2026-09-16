#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from multimodal_web_agent.evaluation.unified_agent.embargo import assert_dev_authorized  # noqa: E402
from multimodal_web_agent.evaluation.unified_agent.evaluation import read_examples  # noqa: E402
from multimodal_web_agent.evaluation.unified_agent.fingerprints import sha256_tree  # noqa: E402
from run_online_web_agent_v1 import _select_examples  # noqa: E402


def _write_exact(path: Path, value) -> None:
    data = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.is_file():
        if path.read_text(encoding="utf-8") != data:
            raise RuntimeError("frozen O1 artifact changed: %s" % path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/evaluation/online_web_agent_o1_100.yaml"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/online_web_agent_v1_o1_100"),
    )
    args = parser.parse_args()
    root = args.project_root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    output = args.output_root if args.output_root.is_absolute() else root / args.output_root
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert_dev_authorized(root / config["test_embargo"])
    if config["boundaries"] != {
        "frozen_test_access": False,
        "training_allowed": False,
        "reward_changes_allowed": False,
        "pilot_is_model_selection": False,
    }:
        raise RuntimeError("O1 boundaries changed")

    examples = read_examples(root / config["dataset"], 200)
    selected, composition = _select_examples(
        examples,
        config["sampling"]["task_types"],
        int(config["sampling"]["o1_per_task_type"]),
    )
    expected = {
        "search_free": 25,
        "visual_search_required": 25,
        "text_search_required": 25,
        "mixed_search_required": 25,
    }
    actual = {
        task_type: sum(item.task_type == task_type for item in selected)
        for task_type in expected
    }
    if actual != expected or len(selected) != 100:
        raise RuntimeError("O1 composition is not 25/25/25/25")

    manifest = {
        "schema_version": "online-web-agent-o1-100-sample-manifest-v1",
        "online_pilot_is_development_eval": True,
        "source_split": "dev",
        "selection": "eval_id_ascending_within_task_type",
        "fixed_seed": None,
        "composition": actual,
        "episode_count": len(selected),
        "frozen_test_accessed": False,
        "episodes": [
            {
                "episode_id": item.eval_id,
                "eval_id": item.eval_id,
                "source": item.source_dataset,
                "route": item.task_type,
                "question_id": item.source_data_id,
                "question_sha256": hashlib.sha256(item.question.encode("utf-8")).hexdigest(),
                "image_identifier": item.image_path,
                "image_sha256": item.image_sha256,
            }
            for item in selected
        ],
    }
    _write_exact(output / "sample_selection/online_pilot_sample_ids.json", manifest)

    runner_manifest = {
        "schema_version": "online-web-agent-sample-ids-v1",
        "source_split": "dev",
        "selection": "eval_id_ascending_within_task_type",
        "per_task_type": 25,
        "composition": composition,
        "ordered_eval_ids": [item.eval_id for item in selected],
        "frozen_test_accessed": False,
    }
    _write_exact(output / "online_pilot/sample_ids.json", runner_manifest)

    o1 = yaml.safe_load(
        (root / config["search_configs"]["live"]).read_text(encoding="utf-8")
    )
    if o1["online_budget"].get("allow_paid_overage") is not False:
        raise RuntimeError("paid overage must remain disabled")

    checkpoint_audit = {}
    for model_id, value in config["models"].items():
        path = root / value["adapter_path"]
        if not path.is_dir():
            raise FileNotFoundError(path)
        checkpoint_audit[model_id] = {
            "stage": value["stage"],
            "adapter_path": value["adapter_path"],
            "resolved_path": str(path.resolve()),
            "adapter_tree_sha256": sha256_tree(path),
        }
    audit = {
        "schema_version": "online-web-agent-o1-100-preflight-v1",
        "config": str(config_path.relative_to(root)),
        "sample_manifest": "sample_selection/online_pilot_sample_ids.json",
        "composition": actual,
        "checkpoints": checkpoint_audit,
        "backend_matches_cost_revision_v2": True,
        "round_robin_order": ["sft", "reward_v21"],
        "online_pilot_is_development_eval": True,
        "frozen_test_accessed": False,
        "new_model_training_started": False,
    }
    _write_exact(output / "sample_selection/preflight_audit.json", audit)
    print("ONLINE_O1_100_SAMPLE_SELECTION_FROZEN")
    print("ONLINE_O1_100_PREFLIGHT_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
