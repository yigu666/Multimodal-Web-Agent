#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from multimodal_web_agent.data.quality.pool_manifest import read_jsonl
from multimodal_web_agent.data.quality.stage_policy import (
    GRPOExecutabilityPolicy,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = read_jsonl(args.input)
    policy = GRPOExecutabilityPolicy()
    executable = [row for row in rows if policy.accepts(row)]
    rejected = [row for row in rows if not policy.accepts(row)]
    distribution = Counter(
        tuple(sorted(row.get("valid_action_set", ()))) for row in executable
    )
    report = {
        "grpo_candidate_count": len(rows),
        "grpo_executable_count": len(executable),
        "grpo_rejected_count": len(rejected),
        "no_valid_action_count": sum(
            not row.get("valid_action_set") for row in executable
        ),
        "unresolvable_visual_reference_count": sum(
            "unresolvable_visual_reference"
            in {
                reason
                for reasons in row.get("blocked_actions", {}).values()
                for reason in reasons
            }
            for row in rejected
        ),
        "unreachable_evidence_count": sum(
            not row.get("evidence_reachable", False) for row in rejected
        ),
        "ambiguous_but_valid_multi_action_count": sum(
            len(row.get("valid_action_set", ())) > 1 for row in executable
        ),
        "valid_action_set_distribution": {
            "|".join(actions): count
            for actions, count in sorted(distribution.items())
        },
    }
    report["passed"] = report["no_valid_action_count"] == 0
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
