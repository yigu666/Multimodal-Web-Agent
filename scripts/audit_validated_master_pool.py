#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from multimodal_web_agent.data.protocol_sft.builder import resolve_project_path
from multimodal_web_agent.data.quality.pool_manifest import read_jsonl, write_json


SUSPICIOUS = re.compile(
    r"\b(this (?:goat|animal|man|vegetable|company|car|building)|"
    r"man in image|woman in image|root vegetable|star logo|the name of|"
    r"shown in image|depicted in image)\b",
    re.IGNORECASE,
)


def audit_master_pool(
    pool: Path,
    *,
    expected_schema: str,
    expected_gate_schema: str,
) -> dict[str, Any]:
    manifest = json.loads((pool / "manifest.json").read_text(encoding="utf-8"))
    accepted = read_jsonl(pool / "accepted_candidates.jsonl")
    rejected = read_jsonl(pool / "rejected_candidates.jsonl")
    validity = read_jsonl(pool / "action_validity.jsonl")
    accepted_ids = {row["candidate_id"] for row in accepted}
    accepted_validity = [
        row for row in validity if row["candidate_id"] in accepted_ids
    ]
    reasons = Counter(
        reason for row in rejected for reason in row.get("rejection_reasons", ())
    )
    accepted_reasons = Counter(
        reason
        for row in accepted_validity
        for reason in row.get("rejection_reasons", ())
    )
    text_rows = [
        row for row in accepted
        if row["route"] in {"text_search_answer", "image_text_search_answer"}
    ]
    anchor_rows = [
        anchor
        for row in text_rows
        for anchor in row.get("query_anchor_records", ())
    ]
    category_only = sum(not row.get("query_anchor_records") for row in text_rows)
    gate_mismatch = sum(
        row.get("gate_schema_version") != expected_gate_schema
        for row in accepted
    )
    suspicious = []
    for row in text_rows:
        queries = [
            validation.get("metadata", {}).get("target_query", "")
            for validation in next(
                (
                    value for value in accepted_validity
                    if value["candidate_id"] == row["candidate_id"]
                ),
                {},
            ).get("state_validations", ())
        ]
        if any(SUSPICIOUS.search(query) for query in queries):
            suspicious.append(row["candidate_id"])
    report = {
        "schema_version": expected_schema + "-audit",
        "master_pool_schema": manifest.get("schema_version"),
        "protocol_sft_schema": manifest.get("protocol_sft_schema"),
        "gate_schema": manifest.get("gate_schema_version"),
        "accepted_candidate_count": len(accepted),
        "rejected_candidate_count": len(rejected),
        "initial_text_candidate_count": len(text_rows),
        "initial_text_named_anchor_count": sum(
            row.get("query_anchor_type") == "named_entity"
            for row in anchor_rows
        ),
        "initial_text_unique_anchor_count": len(anchor_rows),
        "initial_text_category_only_count": category_only,
        "initial_text_unresolvable_reference_count": accepted_reasons[
            "unresolvable_visual_reference"
        ],
        "initial_text_gate_schema_mismatch_count": gate_mismatch,
        "suspicious_visual_reference_candidate_count": len(suspicious),
        "suspicious_visual_reference_candidate_ids": suspicious,
        "no_valid_action_count": accepted_reasons["no_valid_action"],
        "unresolvable_visual_reference_count": accepted_reasons[
            "unresolvable_visual_reference"
        ],
        "missing_visible_entity_count": accepted_reasons[
            "missing_visible_entity"
        ],
        "generic_query_count": accepted_reasons["generic_query"],
        "unreachable_evidence_count": accepted_reasons["unreachable_evidence"],
        "rejection_reason_distribution": dict(sorted(reasons.items())),
        "source_boundary_fvqa_train_only": (
            manifest.get("source_kind") == "FVQA Train"
            and not manifest.get("fvqa_test_included", True)
            and not manifest.get("eval_sources_included", True)
        ),
        "count_match": (
            len(accepted) == manifest["counts"]["accepted_candidates"]
            and len(rejected) == manifest["counts"]["rejected_candidates"]
        ),
    }
    zero_metrics = (
        "initial_text_category_only_count",
        "initial_text_unresolvable_reference_count",
        "initial_text_gate_schema_mismatch_count",
        "no_valid_action_count",
        "unresolvable_visual_reference_count",
        "missing_visible_entity_count",
        "generic_query_count",
        "unreachable_evidence_count",
    )
    report["passed"] = (
        bool(accepted)
        and report["master_pool_schema"] == expected_schema
        and report["gate_schema"] == expected_gate_schema
        and (
            expected_schema != "validated-master-pool-v0.2"
            or report["protocol_sft_schema"] == "protocol-sft-v0.5"
        )
        and report["source_boundary_fvqa_train_only"]
        and report["count_match"]
        and all(report[name] == 0 for name in zero_metrics)
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--config")
    parser.add_argument("--pool-dir")
    parser.add_argument("--output")
    args = parser.parse_args()
    root = args.project_root.resolve()
    if args.config:
        import yaml
        raw = yaml.safe_load(
            resolve_project_path(root, args.config).read_text(encoding="utf-8")
        )
        pool = resolve_project_path(
            root, args.pool_dir or raw["output"]["directory"]
        )
        output = resolve_project_path(
            root, args.output or raw["output"]["audit_file"]
        )
        expected_schema = raw["schema_version"]
        expected_gate = raw["gate"]["schema_version"]
    else:
        pool = resolve_project_path(
            root, args.pool_dir or "data/processed/validated_master_pool_v0_1"
        )
        output = resolve_project_path(
            root,
            args.output
            or "data/manifests/validated_master_pool_v0_1_audit.json",
        )
        manifest = json.loads(
            (pool / "manifest.json").read_text(encoding="utf-8")
        )
        expected_schema = manifest["schema_version"]
        expected_gate = manifest.get(
            "gate_schema_version",
            manifest.get("shared_executability_gate_schema", ""),
        )
    report = audit_master_pool(
        pool,
        expected_schema=expected_schema,
        expected_gate_schema=expected_gate,
    )
    write_json(output, report)
    lines = [
        "# %s Audit" % expected_schema,
        "",
        "- Passed: `%s`" % str(report["passed"]).lower(),
        "- Accepted: %d" % report["accepted_candidate_count"],
        "- Rejected: %d" % report["rejected_candidate_count"],
        "- Initial Text candidates: %d" % report["initial_text_candidate_count"],
        "- Category-only Initial Text: %d"
        % report["initial_text_category_only_count"],
        "- Missing Initial Text anchors: %d"
        % (
            report["initial_text_candidate_count"]
            - report["initial_text_unique_anchor_count"]
        ),
        "",
    ]
    (pool / "audit_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
