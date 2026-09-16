from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from .checkpoint_selection import CheckpointCandidate
from .format_metrics import check_format_gate


def _epoch(name: str) -> int:
    match = re.search(r"epoch[-_]?(\d+)", name)
    return int(match.group(1)) if match else 1_000_000


def format_candidate_score(candidate: CheckpointCandidate) -> tuple[Any, ...]:
    if candidate.split != "dev":
        raise ValueError("Format checkpoint selection uses Dev only")
    metrics = candidate.metrics
    gate_passed = all(check_format_gate(metrics).values())
    return (
        int(gate_passed),
        float(metrics.get("protocol_valid_rate", 0.0)),
        float(metrics.get("exactly_one_action_rate", 0.0)),
        float(metrics.get("tag_closure_rate", 0.0)),
        float(metrics.get("nonempty_action_payload_rate", 0.0)),
        -float(metrics.get("malformed_rate", 1.0)),
        -float(metrics.get("extra_text_rate", 1.0)),
        -int(metrics.get("forged_information_count", 1)),
        -float(candidate.eval_loss),
        -_epoch(candidate.name),
    )


def select_format_checkpoint(
    candidates: Sequence[CheckpointCandidate],
) -> tuple[CheckpointCandidate, Dict[str, Any]]:
    if not candidates:
        raise ValueError("no Format checkpoint candidates were provided")
    ranked = sorted(candidates, key=format_candidate_score, reverse=True)
    rows = []
    for rank, candidate in enumerate(ranked, 1):
        checks = check_format_gate(candidate.metrics)
        rows.append({
            "rank": rank,
            "name": candidate.name,
            "path": str(candidate.path).replace("\\", "/"),
            "split": candidate.split,
            "format_gate_passed": all(checks.values()),
            "format_gate_checks": checks,
            "protocol_valid_rate": candidate.metrics.get("protocol_valid_rate"),
            "exactly_one_action_rate": candidate.metrics.get("exactly_one_action_rate"),
            "tag_closure_rate": candidate.metrics.get("tag_closure_rate"),
            "nonempty_action_payload_rate": candidate.metrics.get("nonempty_action_payload_rate"),
            "malformed_rate": candidate.metrics.get("malformed_rate"),
            "extra_text_rate": candidate.metrics.get("extra_text_rate"),
            "forged_information_count": candidate.metrics.get("forged_information_count"),
            "eval_loss": candidate.eval_loss,
        })
    selected = ranked[0]
    selected_passed = all(check_format_gate(selected.metrics).values())
    return selected, {
        "objective": "protocol_format_only",
        "selection_split": "dev",
        "test_metrics_used": False,
        "selected_name": selected.name,
        "selected_path": str(selected.path).replace("\\", "/"),
        "selected_format_gate_passed": selected_passed,
        "any_epoch_format_gate_passed": any(
            row["format_gate_passed"] for row in rows
        ),
        "policy_metrics_used": False,
        "routing_collapse_used": False,
        "selection_reason": (
            "Dev-only Format Gate, protocol validity, one action, tag closure, "
            "payload presence, malformed/extra/forged rates, eval loss, epoch."
        ),
        "ranking": rows,
    }


def materialize_format_selection(
    selected: CheckpointCandidate,
    report: Mapping[str, Any],
    output_root: Path,
    *,
    copy_adapter: bool = True,
) -> Path:
    output_root = Path(output_root)
    destination = output_root / "selected_adapter"
    if copy_adapter:
        if destination.exists():
            raise FileExistsError("refusing to overwrite selected_adapter")
        shutil.copytree(
            selected.path,
            destination,
            ignore=shutil.ignore_patterns("optimizer.pt"),
        )
    (output_root / "checkpoint_selection.json").write_text(
        json.dumps(dict(report), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Protocol Format SFT v1 Checkpoint Selection",
        "",
        "Selection split: Format Dev (not an independent benchmark)",
        "",
        "Policy metrics and routing collapse are not used.",
        "",
        "| Rank | Checkpoint | Gate | Protocol valid | One action | Tag closure | Payload | Malformed | Extra text | Forged | Eval loss |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["ranking"]:
        lines.append(
            "| {rank} | {name} | {format_gate_passed} | {protocol_valid_rate} | "
            "{exactly_one_action_rate} | {tag_closure_rate} | "
            "{nonempty_action_payload_rate} | {malformed_rate} | "
            "{extra_text_rate} | {forged_information_count} | {eval_loss} |".format(**row)
        )
    (output_root / "checkpoint_selection.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return destination
