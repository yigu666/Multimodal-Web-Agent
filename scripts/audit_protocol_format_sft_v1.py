#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from multimodal_web_agent.agent import ActionType, parse_action
from multimodal_web_agent.data.protocol_format_sft import (
    EXPECTED_TRANSITIONS,
    SCHEMA,
)
from multimodal_web_agent.data.protocol_sft.builder import resolve_project_path
from multimodal_web_agent.data.protocol_sft.schema import validate_example_dict
from multimodal_web_agent.data.quality.pool_manifest import read_jsonl, write_json


def audit_format_view(data_dir: Path) -> dict:
    manifest = json.loads(
        (data_dir / "manifest.json").read_text(encoding="utf-8")
    )
    splits = {
        name: read_jsonl(data_dir / ("%s.jsonl" % name))
        for name in ("train", "dev")
    }
    rows = splits["train"] + splits["dev"]
    counts = Counter()
    actions = Counter()
    transitions = Counter()
    source_splits = defaultdict(set)
    trajectory_splits = defaultdict(set)
    entity_splits = defaultdict(set)
    near_splits = defaultdict(set)
    for split, values in splits.items():
        for row in values:
            if row.get("schema_version") != SCHEMA:
                counts["example_schema_mismatch_count"] += 1
            source_splits[str(row["source_data_id"])].add(split)
            trajectory_splits[str(row["trajectory_id"])].add(split)
            entity_splits[str(row["source"]["entity_group_id"])].add(split)
            near_splits[
                str(row["source"]["near_duplicate_group_id"])
            ].add(split)
            transitions[str(row["transition"])] += 1
            try:
                validate_example_dict(row)
            except Exception:
                counts["strict_parser_invalid_targets"] += 1
                continue
            target = str(row["target"])
            parsed = parse_action(target)
            if not parsed.valid:
                counts["strict_parser_invalid_targets"] += 1
                code = parsed.error_code.value if parsed.error_code else ""
                if code == "empty_reason":
                    counts["empty_reason_target_count"] += 1
                if code in {"empty_query", "empty_answer"}:
                    counts["empty_action_payload_target_count"] += 1
                if code == "multiple_actions":
                    counts["multiple_action_target_count"] += 1
                if code == "forged_information":
                    counts["forged_information_target_count"] += 1
                continue
            actions[parsed.action_type.value] += 1
            if not (parsed.reason or "").strip():
                counts["empty_reason_target_count"] += 1
            if (
                parsed.action_type != ActionType.IMAGE_SEARCH
                and not (parsed.content or "").strip()
            ):
                counts["empty_action_payload_target_count"] += 1
            if "<information" in target.casefold():
                counts["forged_information_target_count"] += 1
            if len(target.split()) > 96:
                counts["target_truncation_count"] += 1
    counts["source_split_leak_count"] = sum(
        len(value) > 1 for value in source_splits.values()
    )
    counts["trajectory_split_leak_count"] = sum(
        len(value) > 1 for value in trajectory_splits.values()
    )
    counts["entity_group_split_leak_count"] = sum(
        len(value) > 1 for value in entity_splits.values()
    )
    counts["near_duplicate_split_leak_count"] = sum(
        len(value) > 1 for value in near_splits.values()
    )
    zero_metrics = (
        "example_schema_mismatch_count",
        "strict_parser_invalid_targets",
        "empty_reason_target_count",
        "empty_action_payload_target_count",
        "multiple_action_target_count",
        "forged_information_target_count",
        "target_truncation_count",
        "source_split_leak_count",
        "trajectory_split_leak_count",
        "entity_group_split_leak_count",
        "near_duplicate_split_leak_count",
    )
    report = {
        "schema_version": SCHEMA + "-audit",
        "dataset_schema": manifest.get("schema_version"),
        "state_action_examples": len(rows),
        "split_counts": {name: len(value) for name, value in splits.items()},
        "answer_target_count": actions["answer"],
        "image_search_target_count": actions["image_search"],
        "text_search_target_count": actions["text_search"],
        "transition_counts": dict(sorted(transitions.items())),
        "actual_transition_set": sorted(transitions),
        "initial_to_text_search_count": transitions[
            "initial_to_text_search"
        ],
        "test_file_present": (data_dir / "test.jsonl").exists(),
        **{name: counts[name] for name in zero_metrics},
    }
    report["passed"] = (
        report["dataset_schema"] == SCHEMA
        and report["state_action_examples"] == 1000
        and report["split_counts"] == {"train": 900, "dev": 100}
        and all(report[name] > 0 for name in (
            "answer_target_count",
            "image_search_target_count",
            "text_search_target_count",
        ))
        and set(report["actual_transition_set"]) == EXPECTED_TRANSITIONS
        and report["initial_to_text_search_count"] == 0
        and not report["test_file_present"]
        and all(report[name] == 0 for name in zero_metrics)
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument(
        "--data-dir", default="data/processed/protocol_format_sft_v1"
    )
    parser.add_argument(
        "--output", default="data/manifests/protocol_format_sft_v1_audit.json"
    )
    args = parser.parse_args()
    root = args.project_root.resolve()
    data_dir = resolve_project_path(root, args.data_dir)
    output = resolve_project_path(root, args.output)
    report = audit_format_view(data_dir)
    write_json(output, report)
    write_json(data_dir / "audit.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
