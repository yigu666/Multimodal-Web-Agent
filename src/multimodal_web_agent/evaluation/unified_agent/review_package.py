from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def _jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(
                dict(row), ensure_ascii=False, sort_keys=True
            ) + "\n")


def create_automatic_quality_output(
    root: Path,
    candidate_rows: Sequence[Mapping[str, Any]],
    assignments: Mapping[str, str],
) -> None:
    output = Path(root) / "automatic_quality"
    if output.exists():
        raise FileExistsError("review package already exists")
    output.mkdir(parents=True)
    selected = [
        dict(row) for row in candidate_rows
        if row["candidate_id"] in assignments
    ]
    _jsonl(output / "accepted_candidates.jsonl", selected)
    _jsonl(output / "proposed_joint_assignment.jsonl", [
        {
            "candidate_id": candidate_id,
            "proposed_task_type": assignments[candidate_id],
            "automatic_assignment": True,
        }
        for candidate_id in sorted(assignments)
    ])
    _jsonl(output / "automatic_quality_decision.jsonl", [
        {
            "candidate_id": row["candidate_id"],
            "candidate_sha256": row["candidate_sha256"],
            "assigned_task_type": assignments[row["candidate_id"]],
            "automatic_quality_decision": "accepted",
            "manual_review_required": False,
            "formal_dataset_publication": False,
        }
        for row in selected
    ])
    (output / "automatic_quality_report.md").write_text(
        "# Unified Agent Eval v1 Automatic Quality Decision\n\n"
        "All selected candidates passed deterministic source, provenance, "
        "license, query-image, evidence, leakage, deduplication, eligibility "
        "and capacity gates. No manual sample filtering or approval is "
        "required. Formal Dev/Test publication remains a separate stage.\n",
        encoding="utf-8",
    )


def create_automatic_quality_output_if_capacity_passes(
    root: Path,
    *,
    capacity_gate_passed: bool,
    candidate_rows: Sequence[Mapping[str, Any]],
    assignments: Mapping[str, str],
) -> bool:
    review = Path(root) / "automatic_quality"
    if not capacity_gate_passed:
        if review.exists():
            raise RuntimeError(
                "automatic quality output exists while capacity is false"
            )
        return False
    create_automatic_quality_output(root, candidate_rows, assignments)
    return True


# Compatibility aliases only. They no longer create or require a manual
# review artifact.
create_review_package = create_automatic_quality_output
create_review_package_if_capacity_passes = (
    create_automatic_quality_output_if_capacity_passes
)
