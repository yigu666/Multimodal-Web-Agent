from __future__ import annotations

import csv
from io import StringIO
from pathlib import Path
from typing import Any, Mapping, Sequence

from .schema import TASK_TYPES


MODEL_LABELS = {
    "raw": "Raw",
    "sft": "Protocol Format SFT",
    "reward_v21": "Reward-v2.1 GRPO",
    "stage2": "Stage2 S2-A step 16",
}

PUBLIC_MODEL_ORDER = ("raw", "sft", "reward_v21", "stage2")


def _format(value: Any) -> str:
    if value in {None, "not_available", "not_applicable"}:
        return "N/A"
    if isinstance(value, float):
        return "%.6f" % value
    return str(value)


def _markdown(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---:" if index else "---" for index in range(len(headers))) + "|",
    ]
    lines.extend(
        "| " + " | ".join(_format(value) for value in row) + " |"
        for row in rows
    )
    return "\n".join(lines) + "\n"


def _csv(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    output = StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(headers)
    writer.writerows(rows)
    return output.getvalue()


def build_main_table(
    model_metrics: Mapping[str, Mapping[str, Any]],
) -> tuple[list[str], list[list[Any]]]:
    headers = [
        "Model", "EM ↑", "Token F1 ↑", "Agent Success@Budget ↑",
        "Search Decision F1 ↑", "Avg Tool Calls ↓", "Protocol Valid ↑",
    ]
    rows = []
    for model_id in PUBLIC_MODEL_ORDER:
        if model_id not in model_metrics:
            continue
        value = model_metrics[model_id]
        rows.append([
            MODEL_LABELS[model_id],
            value["normalized_em"],
            value["token_f1"],
            value["agent_success_at_budget"],
            value["search_decision_f1"],
            value["avg_tool_calls"],
            value["protocol_valid"],
        ])
    return headers, rows


def build_task_type_table(
    model_metrics: Mapping[str, Mapping[str, Any]],
) -> tuple[list[str], list[list[Any]]]:
    if not model_metrics:
        raise ValueError("task-type table requires model metrics")
    first = next(iter(model_metrics.values()))["task_type_em"]
    labels = {
        "search_free": "Search-free EM",
        "visual_search_required": "Visual-search EM",
        "text_search_required": "Text-search EM",
        "mixed_search_required": "Mixed-search EM",
    }
    headers = ["Model"] + [
        "%s (n=%s)" % (labels[name], first[name]["n"])
        for name in TASK_TYPES
    ] + ["Overall EM"]
    rows = []
    for model_id in PUBLIC_MODEL_ORDER:
        if model_id not in model_metrics:
            continue
        value = model_metrics[model_id]
        for task_type in TASK_TYPES:
            if value["task_type_em"][task_type]["n"] != first[task_type]["n"]:
                raise ValueError("task-type sample counts differ by model")
        rows.append([
            MODEL_LABELS[model_id],
            *[value["task_type_em"][name]["em"] for name in TASK_TYPES],
            value["normalized_em"],
        ])
    return headers, rows


def build_efficiency_table(
    registrations: Mapping[str, Mapping[str, Any]],
    efficiency: Mapping[str, Mapping[str, Any]],
) -> tuple[list[str], list[list[Any]]]:
    headers = [
        "Model", "Params", "Avg Output Tokens ↓", "Avg Agent Turns ↓",
        "Avg Tool Calls ↓", "Throughput ↑",
    ]
    rows = []
    for model_id in PUBLIC_MODEL_ORDER:
        if model_id not in registrations:
            continue
        value = efficiency.get(model_id, {})
        rows.append([
            MODEL_LABELS[model_id],
            registrations[model_id]["parameter_count"],
            value.get("avg_output_tokens", "not_available"),
            value.get("avg_agent_turns", "not_available"),
            value.get("avg_tool_calls", "not_available"),
            value.get("throughput_episodes_per_minute", "not_available"),
        ])
    return headers, rows


def write_three_tables(
    output_dir: Path,
    *,
    model_metrics: Mapping[str, Mapping[str, Any]],
    registrations: Mapping[str, Mapping[str, Any]],
    efficiency: Mapping[str, Mapping[str, Any]],
) -> dict[str, str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    built = {
        "main_table": build_main_table(model_metrics),
        "task_type_table": build_task_type_table(model_metrics),
        "efficiency_table": build_efficiency_table(
            registrations, efficiency
        ),
    }
    rendered = {}
    for name, (headers, rows) in built.items():
        markdown = _markdown(headers, rows)
        csv_text = _csv(headers, rows)
        (output / (name + ".md")).write_text(markdown, encoding="utf-8")
        (output / (name + ".csv")).write_text(csv_text, encoding="utf-8")
        rendered[name] = markdown
    return rendered
