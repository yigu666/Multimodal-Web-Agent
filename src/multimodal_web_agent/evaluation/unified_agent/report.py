from __future__ import annotations

from pathlib import Path
from typing import Mapping


REPORT_SECTIONS = (
    "1. Evaluation Contract",
    "2. Main Table",
    "3. Task-type EM Table",
    "4. Efficiency Table",
    "5. Brief Conclusions",
    "6. Limitations",
)


def render_report(
    tables: Mapping[str, str],
    *,
    split: str,
    model_ids: list[str],
    evaluation_label: str = "v1",
) -> str:
    if split not in {"dev", "test"}:
        raise ValueError("report split must be dev or test")
    dev = split == "dev"
    lines = [
        "# Unified Agent Evaluation %s" % evaluation_label,
        "",
        "## 1. Evaluation Contract",
        "",
        "- Split: `%s`." % split,
        "- Models: `%s`." % ", ".join(model_ids),
        "- Shared dataset, frozen offline tools, Agent Runner, prompt, parser and budgets.",
        "- Dev results are not a final benchmark: `%s`." % str(dev).lower(),
        "- Frozen Test accessed: `%s`." % str(not dev).lower(),
        "",
        "## 2. Main Table",
        "",
        tables["main_table"].rstrip(),
        "",
        "## 3. Task-type EM Table",
        "",
        tables["task_type_table"].rstrip(),
        "",
        "## 4. Efficiency Table",
        "",
        tables["efficiency_table"].rstrip(),
        "",
        "## 5. Brief Conclusions",
        "",
        "The three tables provide the stage-independent comparison under the frozen Unified Agent contract.",
        "",
        "## 6. Limitations",
        "",
        (
            "- Eval Dev may be used for future checkpoint selection and is not "
            "an independent final benchmark."
            if dev else
            "- Frozen Eval Test was opened once for the four pre-registered models."
        ),
        "- Search quality is measured through end-to-end answers, not Query overlap or an online judge.",
        "- No dynamic internet access is used.",
        "",
    ]
    return "\n".join(lines)


def write_report(
    output_dir: Path,
    tables: Mapping[str, str],
    *,
    split: str,
    model_ids: list[str],
    evaluation_label: str = "v1",
) -> Path:
    text = render_report(
        tables,
        split=split,
        model_ids=model_ids,
        evaluation_label=evaluation_label,
    )
    path = Path(output_dir) / "report.md"
    path.write_text(text, encoding="utf-8")
    return path
