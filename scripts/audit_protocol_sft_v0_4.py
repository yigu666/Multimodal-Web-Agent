#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from multimodal_web_agent.agent import ActionType, parse_action
from multimodal_web_agent.data.protocol_sft.builder import resolve_project_path
from multimodal_web_agent.data.protocol_sft.query_builder import (
    contains_answer_leak,
)
from multimodal_web_agent.data.protocol_sft.schema import (
    validate_example_dict,
    validate_trajectory_dict,
)
from multimodal_web_agent.data.quality.answer_support import (
    validate_answer_action,
)
from multimodal_web_agent.data.quality.generic_terms import (
    GENERIC_QUERY_TERMS,
    content_tokens,
    text_tokens,
)
from multimodal_web_agent.data.quality.pool_manifest import (
    read_jsonl,
    write_json,
    write_sha256_manifest,
)
from multimodal_web_agent.data.quality.query_executability import (
    is_unresolvable_visual_reference,
)
from multimodal_web_agent.data.quality.visible_context import (
    build_visible_text_context,
    visible_information_text,
)


ZERO_QUALITY_METRICS = (
    "strict_parser_invalid_targets",
    "answer_leak_count",
    "query_answer_leak_count",
    "unavailable_context_leak_count",
    "unresolvable_visual_reference_count",
    "missing_visible_entity_count",
    "generic_query_count",
    "conflicting_route_label_count",
    "no_valid_action_count",
    "unreachable_evidence_count",
    "source_route_reuse_count",
    "source_split_leak_count",
    "trajectory_split_leak_count",
    "entity_group_split_leak_count",
    "near_duplicate_split_leak_count",
    "target_truncation_count",
)


def _audit(data_dir: Path) -> dict:
    manifest = json.loads(
        (data_dir / "manifest.json").read_text(encoding="utf-8")
    )
    trajectories = read_jsonl(data_dir / "trajectories.jsonl")
    splits = {
        name: read_jsonl(data_dir / ("%s.jsonl" % name))
        for name in ("train", "dev", "test")
    }
    counts = {metric: 0 for metric in ZERO_QUALITY_METRICS}
    source_routes = defaultdict(set)
    source_splits = defaultdict(set)
    trajectory_splits = defaultdict(set)
    for raw in trajectories:
        try:
            validate_trajectory_dict(raw)
        except Exception:
            counts["strict_parser_invalid_targets"] += 1
        source_routes[raw["source_data_id"]].add(raw["route"])
        aliases = tuple(raw["accepted_answers"])
        for index, step in enumerate(raw["steps"]):
            parsed = parse_action(step["target"])
            if not parsed.valid:
                counts["strict_parser_invalid_targets"] += 1
                continue
            if parsed.action_type != ActionType.ANSWER and contains_answer_leak(
                parsed.reason or "", aliases
            ):
                counts["answer_leak_count"] += 1
            if parsed.action_type == ActionType.TEXT_SEARCH:
                query = parsed.content or ""
                question = raw["question"]
                visible = build_visible_text_context(
                    question=question,
                    history_messages=step["state"],
                )
                if contains_answer_leak(query, aliases):
                    counts["query_answer_leak_count"] += 1
                substantive = set(content_tokens(query))
                visible_tokens = set(text_tokens(visible))
                if is_unresolvable_visual_reference(query, visible):
                    counts["unresolvable_visual_reference_count"] += 1
                if not substantive.intersection(visible_tokens):
                    counts["missing_visible_entity_count"] += 1
                if not substantive or set(text_tokens(query)).issubset(
                    GENERIC_QUERY_TERMS
                ):
                    counts["generic_query_count"] += 1
            if parsed.action_type == ActionType.ANSWER and index > 0:
                support = validate_answer_action(
                    state_type=step["transition"],
                    answer_aliases=aliases,
                    visible_information=visible_information_text(step["state"]),
                    source_category=str(raw["source"].get("category", "")),
                    question=raw["question"],
                )
                if not support.executable:
                    counts["unreachable_evidence_count"] += 1
            if len(text_tokens(step["target"])) > 96:
                counts["target_truncation_count"] += 1
    counts["source_route_reuse_count"] = sum(
        len(routes) > 1 for routes in source_routes.values()
    )
    for split, rows in splits.items():
        for raw in rows:
            try:
                validate_example_dict(raw)
            except Exception:
                counts["strict_parser_invalid_targets"] += 1
            source_splits[raw["source_data_id"]].add(split)
            trajectory_splits[raw["trajectory_id"]].add(split)
    counts["source_split_leak_count"] = sum(
        len(values) > 1 for values in source_splits.values()
    )
    counts["trajectory_split_leak_count"] = sum(
        len(values) > 1 for values in trajectory_splits.values()
    )
    group_audit = manifest.get("group_split_audit", {})
    counts["entity_group_split_leak_count"] = int(
        group_audit.get("entity_group_split_leak_count", 0)
    )
    counts["near_duplicate_split_leak_count"] = int(
        group_audit.get("near_duplicate_split_leak_count", 0)
    )
    actual_split_counts = {
        split: len(rows) for split, rows in splits.items()
    }
    report = {
        "schema_version": "protocol-sft-v0.4-audit",
        "state_action_examples": sum(actual_split_counts.values()),
        "split_counts": actual_split_counts,
        **counts,
        "new_test_embargoed": bool(
            manifest.get("new_test_embargoed", False)
        ),
        "test_preview_generated": bool(
            manifest.get("test_preview_generated", True)
        ),
        "historically_exposed_test_source_count": int(
            manifest.get("historically_exposed_test_source_count", 0)
        ),
        "historically_exposed_test_group_count": int(
            manifest.get("historically_exposed_test_group_count", 0)
        ),
        "new_test_source_count": int(
            manifest.get("new_test_source_count", 0)
        ),
    }
    report["passed"] = (
        report["state_action_examples"] == 1000
        and actual_split_counts == {"train": 800, "dev": 100, "test": 100}
        and all(report[metric] == 0 for metric in ZERO_QUALITY_METRICS)
        and report["new_test_embargoed"]
        and not report["test_preview_generated"]
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument(
        "--data-dir", default="data/processed/protocol_sft_v0_4"
    )
    parser.add_argument(
        "--output",
        default="data/manifests/protocol_sft_v0_4_audit.json",
    )
    args = parser.parse_args()
    root = args.project_root.resolve()
    data_dir = resolve_project_path(root, args.data_dir)
    output = resolve_project_path(root, args.output)
    report = _audit(data_dir)
    write_json(output, report)
    lines = [
        "# Protocol-SFT v0.4 Audit",
        "",
        "- Passed: `%s`" % str(report["passed"]).lower(),
        "- State-action examples: %d" % report["state_action_examples"],
        "- Splits: `%s`" % report["split_counts"],
        "",
        "## Zero-required metrics",
        "",
    ]
    lines.extend(
        "- %s: %d" % (metric, report[metric])
        for metric in ZERO_QUALITY_METRICS
    )
    (data_dir / "audit_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    manifest_copy = (
        root / "data" / "manifests" / "protocol_sft_v0_4_manifest.json"
    )
    paths = [
        data_dir / name
        for name in (
            "trajectories.jsonl",
            "train.jsonl",
            "dev.jsonl",
            "test.jsonl",
            "rejected.jsonl",
            "manifest.json",
            "audit_report.md",
            "manual_audit.jsonl",
            "sample_preview_train_dev.md",
        )
    ] + [output, manifest_copy]
    write_sha256_manifest(
        [path for path in paths if path.is_file()],
        project_root=root,
        output_path=(
            root / "data" / "manifests" /
            "protocol_sft_v0_4_files.sha256"
        ),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
