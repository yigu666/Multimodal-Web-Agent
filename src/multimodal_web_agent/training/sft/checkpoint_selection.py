from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from .metrics import check_format_gates


class NoNonCollapsedCheckpointError(RuntimeError):
    pass


@dataclass(frozen=True)
class CheckpointCandidate:
    name: str
    path: Path
    metrics: Mapping[str, Any]
    eval_loss: float
    split: str = "dev"


def _routing_collapsed(metrics: Mapping[str, Any]) -> bool:
    if "routing_collapsed" in metrics:
        return bool(metrics["routing_collapsed"])
    initial = metrics.get("initial_transition_recall")
    if isinstance(initial, Mapping):
        if any(float(value) == 0.0 for value in initial.values()):
            return True
        recall = metrics.get("action_recall_by_type", {})
        if isinstance(recall, Mapping) and float(
            recall.get("text_search", 0.0) or 0.0
        ) == 0.0:
            return True
    # Legacy metrics did not include the Router-fix collapse contract.
    return False


def _epoch_number(name: str) -> int:
    match = re.search(r"(?:epoch[-_]?)(\d+)", name)
    return int(match.group(1)) if match else 1_000_000


def candidate_score(candidate: CheckpointCandidate) -> tuple[Any, ...]:
    if candidate.split != "dev":
        raise ValueError("Checkpoint selection may only use Dev metrics")
    gates = check_format_gates(candidate.metrics)
    gate_count = sum(gates.values())
    return (
        int(not _routing_collapsed(candidate.metrics)),
        int(all(gates.values())),
        gate_count,
        float(
            candidate.metrics.get(
                "minimum_initial_transition_recall", 0.0
            )
        ),
        float(candidate.metrics.get("macro_action_f1", 0.0)),
        float(candidate.metrics.get("action_type_accuracy", 0.0)),
        min(
            float(candidate.metrics.get("text_query_nonempty_rate", 0.0)),
            float(candidate.metrics.get("answer_finish_rate", 0.0)),
        ),
        float(candidate.metrics.get("protocol_valid_rate", 0.0)),
        -float(candidate.eval_loss),
        -_epoch_number(candidate.name),
    )


def select_checkpoint(
    candidates: Sequence[CheckpointCandidate],
    *,
    require_non_collapsed: bool = False,
) -> tuple[CheckpointCandidate, Dict[str, Any]]:
    if not candidates:
        raise ValueError("no checkpoint candidates were provided")
    collapsed = [_routing_collapsed(candidate.metrics) for candidate in candidates]
    if require_non_collapsed and all(collapsed):
        raise NoNonCollapsedCheckpointError(
            "NO_NON_COLLAPSED_CHECKPOINT"
        )
    ranked = sorted(candidates, key=candidate_score, reverse=True)
    selected = ranked[0]
    rows = []
    for rank, candidate in enumerate(ranked, start=1):
        gates = check_format_gates(candidate.metrics)
        rows.append({
            "rank": rank,
            "name": candidate.name,
            "path": str(candidate.path).replace("\\", "/"),
            "split": candidate.split,
            "gate_results": gates,
            "gate_pass_count": sum(gates.values()),
            "all_gates_passed": all(gates.values()),
            "routing_collapsed": _routing_collapsed(candidate.metrics),
            "minimum_initial_transition_recall": candidate.metrics.get(
                "minimum_initial_transition_recall", 0.0
            ),
            "macro_action_f1": candidate.metrics.get(
                "macro_action_f1", 0.0
            ),
            "action_type_accuracy": candidate.metrics.get(
                "action_type_accuracy"
            ),
            "routing_floor": min(
                float(candidate.metrics.get("text_query_nonempty_rate", 0.0)),
                float(candidate.metrics.get("answer_finish_rate", 0.0)),
            ),
            "protocol_valid_rate": candidate.metrics.get(
                "protocol_valid_rate"
            ),
            "eval_loss": candidate.eval_loss,
        })
    return selected, {
        "selection_split": "dev",
        "test_metrics_used": False,
        "selected_name": selected.name,
        "selected_path": str(selected.path).replace("\\", "/"),
        "selected_all_gates_passed": all(
            check_format_gates(selected.metrics).values()
        ),
        "selected_routing_collapsed": _routing_collapsed(
            selected.metrics
        ),
        "dev_gate_passed": bool(
            all(check_format_gates(selected.metrics).values())
            and not _routing_collapsed(selected.metrics)
            and float(
                selected.metrics.get(
                    "minimum_initial_transition_recall", 0.0
                )
            ) > 0.0
        ),
        "selection_reason": (
            "Highest Dev-only non-collapsed rank by gate count, minimum "
            "Initial transition recall, macro Action F1, Action accuracy, "
            "routing floor, protocol validity, eval loss, then earlier epoch."
        ),
        "ranking": rows,
    }


def materialize_selection(
    selected: CheckpointCandidate,
    report: Mapping[str, Any],
    output_root: Path,
) -> Path:
    output_root = Path(output_root)
    destination = output_root / "selected_adapter"
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(
        selected.path,
        destination,
        ignore=shutil.ignore_patterns("optimizer.pt"),
    )
    (output_root / "checkpoint_selection.json").write_text(
        json.dumps(dict(report), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Protocol-SFT Checkpoint Selection",
        "",
        "Selection split: Dev",
        "",
        "Reason: " + str(report.get("selection_reason", "")),
        "",
        "| Rank | Checkpoint | Collapsed | Gates | Min initial recall | Macro F1 | Action accuracy | Routing floor | Protocol valid | Eval loss |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["ranking"]:
        lines.append(
            "| {rank} | {name} | {routing_collapsed} | {gate_pass_count}/7 | "
            "{minimum_initial_transition_recall} | {macro_action_f1} | "
            "{action_type_accuracy} | {routing_floor} | "
            "{protocol_valid_rate} | {eval_loss} |".format(
                **row
            )
        )
    (output_root / "checkpoint_selection.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    return destination
