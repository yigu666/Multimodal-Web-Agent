#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

try:
    from audit_protocol_sft_v0_4 import (
        ZERO_QUALITY_METRICS,
        _audit as base_audit,
    )
except ModuleNotFoundError:
    from scripts.audit_protocol_sft_v0_4 import (
        ZERO_QUALITY_METRICS,
        _audit as base_audit,
    )
from multimodal_web_agent.data.protocol_sft.builder import resolve_project_path
from multimodal_web_agent.data.protocol_sft.v0_5_manifest import (
    GATE_SCHEMA,
    POOL_SCHEMA,
    PROTOCOL_SCHEMA,
)
from multimodal_web_agent.data.quality.pool_manifest import read_jsonl, write_json


def audit_protocol_sft_v0_5(data_dir: Path) -> dict:
    manifest = json.loads(
        (data_dir / "manifest.json").read_text(encoding="utf-8")
    )
    report = base_audit(data_dir)
    anchors = read_jsonl(data_dir / "initial_text_anchor_audit.jsonl")
    report.update(
        {
            "schema_version": PROTOCOL_SCHEMA + "-audit",
            "protocol_sft_schema": manifest.get("schema_version"),
            "source_pool_schema": manifest.get("source_pool_schema"),
            "master_pool_schema": manifest.get("master_pool_schema"),
            "gate_schema": manifest.get("gate_schema_version"),
            "category_only_initial_text_count": sum(
                bool(row.get("category_only")) for row in anchors
            ),
            "missing_anchor_provenance_count": sum(
                not bool(row.get("anchor")) or not bool(row.get("passed"))
                for row in anchors
            ),
            "initial_text_anchor_audit_count": len(anchors),
            "test_used_for_selection": bool(
                manifest.get("test_used_for_selection", True)
            ),
            "historically_exposed_source_count": int(
                manifest.get("historically_exposed_source_count", 0)
            ),
            "historically_exposed_group_count": int(
                manifest.get("historically_exposed_group_count", 0)
            ),
            "historically_exposed_dev_source_count": int(
                manifest.get("historically_exposed_dev_source_count", 0)
            ),
            "historically_exposed_test_source_count": int(
                manifest.get("historically_exposed_test_source_count", 0)
            ),
        }
    )
    report["passed"] = (
        report["protocol_sft_schema"] == PROTOCOL_SCHEMA
        and report["source_pool_schema"] == POOL_SCHEMA
        and report["master_pool_schema"] == POOL_SCHEMA
        and report["gate_schema"] == GATE_SCHEMA
        and report["state_action_examples"] == 1000
        and report["split_counts"] == {
            "train": 800, "dev": 100, "test": 100
        }
        and all(report[name] == 0 for name in ZERO_QUALITY_METRICS)
        and report["category_only_initial_text_count"] == 0
        and report["missing_anchor_provenance_count"] == 0
        and report["new_test_embargoed"]
        and not report["test_preview_generated"]
        and not report["test_used_for_selection"]
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument(
        "--data-dir", default="data/processed/protocol_sft_v0_5"
    )
    parser.add_argument(
        "--output", default="data/manifests/protocol_sft_v0_5_audit.json"
    )
    args = parser.parse_args()
    root = args.project_root.resolve()
    data_dir = resolve_project_path(root, args.data_dir)
    output = resolve_project_path(root, args.output)
    report = audit_protocol_sft_v0_5(data_dir)
    write_json(output, report)
    lines = [
        "# Protocol-SFT v0.5 Audit",
        "",
        "- Passed: `%s`" % str(report["passed"]).lower(),
        "- State-action examples: %d" % report["state_action_examples"],
        "- Splits: `%s`" % report["split_counts"],
        "- Initial Text anchor records: %d"
        % report["initial_text_anchor_audit_count"],
        "",
        "## Zero-required metrics",
        "",
    ]
    lines.extend(
        "- %s: %d" % (name, report[name])
        for name in (
            *ZERO_QUALITY_METRICS,
            "category_only_initial_text_count",
            "missing_anchor_provenance_count",
        )
    )
    (data_dir / "audit_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
