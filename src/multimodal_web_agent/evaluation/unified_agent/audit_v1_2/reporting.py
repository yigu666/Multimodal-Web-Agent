from __future__ import annotations

from collections import Counter
import json
from typing import Any, Mapping, Sequence


def _rate(value: float | None) -> str:
    return "N/A" if value is None else f"{100.0 * value:.2f}%"


def _number(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _matrix_table(values: Mapping[str, int]) -> str:
    lines = ["| Transition | Count |", "|---|---:|"]
    lines.extend(f"| `{key}` | {value} |" for key, value in values.items())
    return "\n".join(lines)


def evidence_metrics_markdown(metrics: Mapping[str, Any]) -> str:
    lines = [
        "# Evidence Utilization Metrics",
        "",
        "Primary correctness is frozen v1 normalized EM. v2 strict is reported "
        "separately and never overwrites v1.",
        "",
        "| Model | Successful-search episodes | Evidence hit | Hit rate | "
        "Hit + correct | Hit + wrong | Hit + no answer | Utilization | Completion given evidence |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model, dimensions in metrics["models"].items():
        row = dimensions["overall"]["overall"]
        lines.append(
            "| {model} | {searched} | {hits} | {hit_rate} | {correct} | "
            "{wrong} | {none} | {util} | {completion} |".format(
                model=model.upper(),
                searched=row["successful_search_episode_count"],
                hits=row["evidence_hit_count"],
                hit_rate=_rate(row["evidence_hit_rate"]),
                correct=row["evidence_hit_and_correct_count"],
                wrong=row["evidence_hit_but_wrong_count"],
                none=row["evidence_hit_but_no_answer_count"],
                util=_rate(row["evidence_utilization_rate"]),
                completion=_rate(row["answer_completion_given_evidence_rate"]),
            )
        )
    for model, dimensions in metrics["models"].items():
        lines.extend(["", f"## {model.upper()} by task type", ""])
        lines.extend([
            "| Task type | Searched | Evidence hit | Hit rate | Utilization | "
            "Finish failure | Entity-copy heuristic | Repeat search |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for task, row in dimensions["task_type"].items():
            lines.append(
                f"| `{task}` | {row['successful_search_episode_count']} | "
                f"{row['evidence_hit_count']} | {_rate(row['evidence_hit_rate'])} | "
                f"{_rate(row['evidence_utilization_rate'])} | "
                f"{_rate(row['evidence_hit_finish_failure_rate'])} | "
                f"{row['entity_copy_failure_count']} | "
                f"{row['repeat_search_failure_count']} |"
            )
    lines.extend([
        "",
        "Rates use successful-search episodes as denominator unless the column "
        "explicitly says ‘given evidence’. Entity-copy is a conservative heuristic; "
        "termination and duplicate-search flags are deterministic.",
    ])
    return "\n".join(lines) + "\n"


def answer_comparison_markdown(metrics: Mapping[str, Any]) -> str:
    lines = [
        "# Answer Evaluator v2 Comparison",
        "",
        "`unified-answer-evaluator-v2` composes the frozen v1 normalizer with "
        "deterministic numeric, unit and alias rules. Closed semantic matches are "
        "diagnostic only because `strict_semantic_enabled=false`.",
        "",
        "| Model | EM v1 | Token F1 v1 | EM v2 strict | Token F1 v2 | "
        "v1 wrong → v2 strict correct | Semantic diagnostic | Numeric | Unit | Alias |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model, row in metrics["models"].items():
        lines.append(
            f"| {model.upper()} | {row['em_v1']:.6f} | "
            f"{row['token_f1_v1']:.6f} | {row['em_v2_strict']:.6f} | "
            f"{row['token_f1_v2']:.6f} | "
            f"{row['v1_wrong_v2_strict_correct_count']} | "
            f"{row['semantic_equivalence_count']} | "
            f"{row['numeric_equivalence_count']} | "
            f"{row['unit_equivalence_count']} | "
            f"{row['alias_equivalence_count']} |"
        )
    lines.extend([
        "",
        "Strict v2 counts v1 normalized exact matches plus question-conditioned "
        "numeric equivalence, bounded unit conversion and deterministic alias rules. "
        "It does not use an LLM, exchange rates or per-example hard-coded aliases.",
    ])
    return "\n".join(lines) + "\n"


def paired_behavior_markdown(audit: Mapping[str, Any]) -> str:
    identity = audit["retrieval_trace_identity"]
    return "\n".join([
        "# SFT / GRPO Paired Behavior Audit",
        "",
        f"Strictly paired episodes: {audit['paired_episode_count']}.",
        "",
        "## Correctness transitions (v1 EM)",
        "",
        _matrix_table(audit["answer_correctness_v1"]),
        "",
        "## Correctness transitions (v2 strict)",
        "",
        _matrix_table(audit["answer_correctness_v2"]),
        "",
        "## Search transitions",
        "",
        _matrix_table(audit["search_behavior"]),
        "",
        "## First-tool transitions",
        "",
        _matrix_table(audit["tool_choice"]),
        "",
        "## Identity checks",
        "",
        "| Comparison | Same | Different |",
        "|---|---:|---:|",
        f"| Full search trace | {identity['same_search_trace_count']} | "
        f"{identity['different_search_trace_count']} |",
        f"| Action sequence | {identity['same_action_sequence_count']} | "
        f"{identity['different_action_sequence_count']} |",
        f"| Query sequence | {identity['same_query_sequence_count']} | "
        f"{identity['different_query_sequence_count']} |",
        f"| Returned Information | {identity['same_returned_information_count']} | "
        f"{identity['different_returned_information_count']} |",
        f"| Final answer | {identity['same_final_answer_count']} | "
        f"{identity['different_final_answer_count']} |",
        "",
        f"Effective-change cases written to the casebook: "
        f"{audit['effective_change_case_count']}.",
    ]) + "\n"


def _render_calls(calls: Sequence[Mapping[str, Any]]) -> list[str]:
    lines = []
    for index, call in enumerate(calls, 1):
        query = call.get("query") or {}
        query_text = query.get("text") or (
            f"query_image sha256={query.get('image_sha256')}"
        )
        lines.extend([
            f"- Tool {index}: `{call.get('tool')}`, turn={call.get('turn')}, "
            f"status=`{call.get('status')}`, query=`{query_text}`",
            f"  - Generated action: `{call.get('generated_action')}`",
            f"  - Next action: `{call.get('next_action')}`",
        ])
        for result in call.get("results", []):
            text = str(result.get("text", "")).replace("\n", " ")
            lines.append(
                f"  - Rank {result.get('rank')} (`{result.get('document_id')}`): {text}"
            )
    return lines or ["- No successful or attempted search call."]


def paired_casebook_markdown(audit: Mapping[str, Any]) -> str:
    lines = [
        "# Paired Behavior Casebook",
        "",
        "Cases are selected by deterministic transition categories, not by manual "
        "outcome preference. Full reconstructed traces remain in the evidence JSONL.",
    ]
    for case in audit["cases"]:
        lines.extend([
            "",
            f"## {case['episode_id']}",
            "",
            f"- Categories: `{', '.join(case['categories'])}`",
            f"- Question: {case['question']}",
            f"- Accepted: {json.dumps(case['accepted_answers'], ensure_ascii=False)}",
            f"- SFT: answer=`{case['sft']['answer']}`, EM v1="
            f"{case['sft']['em_v1']}, evidence_hit={case['sft']['evidence_hit_any']}, "
            f"finished={case['sft']['finished_with_answer']}",
            f"- GRPO: answer=`{case['grpo']['answer']}`, EM v1="
            f"{case['grpo']['em_v1']}, evidence_hit={case['grpo']['evidence_hit_any']}, "
            f"finished={case['grpo']['finished_with_answer']}",
            f"- SFT trace: `{json.dumps(case['sft']['search_trace'], ensure_ascii=False)}`",
            f"- GRPO trace: `{json.dumps(case['grpo']['search_trace'], ensure_ascii=False)}`",
        ])
        lines.append("- SFT reconstructed retrieval:")
        lines.extend(_render_calls(case["sft"]["retrieved_information"]))
        lines.append("- GRPO reconstructed retrieval:")
        lines.extend(_render_calls(case["grpo"]["retrieved_information"]))
    return "\n".join(lines) + "\n"


def evidence_casebook_markdown(
    records: Mapping[str, Sequence[Mapping[str, Any]]],
) -> str:
    keywords = (
        "raita", "waverley abbey", "peccary", "waffle", "trionychidae",
        "yann lecun", "world labs", "klieg light", "boeing 777x",
    )
    selected = []
    for model in ("sft", "grpo"):
        for row in records[model]:
            corpus = " ".join([
                row["question"], str(row.get("final_answer") or ""),
                " ".join(row["accepted_answers"]),
                " ".join(
                    str(result.get("text") or "")
                    for call in row["retrieved_information"]
                    for result in call.get("results", [])
                ),
            ]).casefold()
            known = any(keyword in corpus for keyword in keywords)
            diagnostic = any(bool(row[field]) for field in (
                "entity_copy_failure", "relation_extraction_failure",
                "answer_span_copy_failure", "termination_failure",
                "repeat_search_failure", "tool_budget_exceeded",
            ))
            if known or diagnostic:
                selected.append(row)
    lines = [
        "# Evidence Utilization Casebook",
        "",
        "Includes every SFT/GRPO episode flagged by the uniform extraction, "
        "termination, duplicate-search, budget, or known-case selection rules.",
    ]
    for row in selected:
        flags = [field for field in (
            "entity_copy_failure", "relation_extraction_failure",
            "answer_span_copy_failure", "termination_failure",
            "repeat_search_failure", "tool_budget_exceeded",
        ) if row[field]]
        lines.extend([
            "",
            f"## {row['model'].upper()} — {row['episode_id']}",
            "",
            f"- Question: {row['question']}",
            f"- Accepted: {json.dumps(row['accepted_answers'], ensure_ascii=False)}",
            f"- Final answer: `{row['final_answer']}`; EM v1={row['em_v1']}; "
            f"EM v2 strict={row['em_v2_strict']}",
            f"- Evidence hit: {row['evidence_hit_any']} at tool="
            f"`{row['evidence_hit_tool']}`, rank={row['evidence_hit_rank']}, "
            f"turn={row['evidence_hit_turn']}",
            f"- Action sequence: `{row['action_sequence']}`",
            f"- Flags: `{flags}`",
            f"- Termination: {'final answer' if row['finished_with_answer'] else 'no final answer'}",
        ])
        lines.extend(_render_calls(row["retrieved_information"]))
    return "\n".join(lines) + "\n"


def text_query_markdown(audit: Mapping[str, Any]) -> str:
    lines = [
        "# Text Search Query Audit",
        "",
        "Deterministic hard failures and heuristic quality failures are reported "
        "separately. No online search was used.",
        "",
        "| Model | Attempts | Successful | Evidence hit | Empty | Ellipsis | "
        "Question copy | Generic (heuristic) | Entity missing (heuristic) | "
        "Relation missing (heuristic) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model, row in audit["models"].items():
        lines.append(
            f"| {model.upper()} | {row['text_search_attempt_count']} | "
            f"{row['successful_execution_count']} | {row['evidence_hit_count']} | "
            f"{row['empty_query_count']} | {row['ellipsis_query_count']} | "
            f"{row['question_copy_count']} | {row['generic_query_count']} | "
            f"{row['entity_missing_count']} | {row['relation_missing_count']} |"
        )
    return "\n".join(lines) + "\n"


def text_casebook_markdown(audit: Mapping[str, Any]) -> str:
    lines = [
        "# Text Search Casebook",
        "",
        "All SFT and GRPO Text Search attempts are included below.",
    ]
    for row in audit["queries"]:
        lines.extend([
            "",
            f"## {row['model'].upper()} — {row['episode_id']}",
            "",
            f"- Question: {row['question']}",
            f"- Image-identified entities before query: "
            f"{json.dumps(row['identified_image_entities'], ensure_ascii=False)}",
            f"- Query: `{row['query']}`",
            f"- Status: `{row['status']}`; Evidence hit: {row['evidence_hit_any']}",
            f"- Final answer: `{row['final_answer']}`",
            f"- Failure types: `{row['failure_types']}`",
            "- Top-5:",
        ])
        for index, text in enumerate(row["top_5"], 1):
            lines.append(f"  {index}. {str(text).replace(chr(10), ' ')}")
    return "\n".join(lines) + "\n"


def image_truncation_markdown(audit: Mapping[str, Any]) -> str:
    overall = audit["dimensions"]["overall"]["overall"]
    lines = [
        "# Image Search Truncation Audit",
        "",
        "The frozen 512-character policy was not changed. Full candidate evidence "
        "was compared with the exact returned text.",
        "",
        "| Scope | Calls | Full source available | Full-source hits | Returned hits | "
        "Lost by truncation | Loss rate given full hit |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| Overall | {overall['call_count']} | "
        f"{overall['full_source_available_count']} | "
        f"{overall['full_source_evidence_hit_count']} | "
        f"{overall['returned_information_evidence_hit_count']} | "
        f"{overall['answer_lost_by_truncation_count']} | "
        f"{_rate(overall['answer_lost_by_truncation_rate_given_full_hit'])} |",
    ]
    for dimension in ("model", "question_relation", "source_dataset", "task_type"):
        lines.extend(["", f"## By {dimension}", "", "| Value | Calls | Full hits | Returned hits | Lost | Loss rate |", "|---|---:|---:|---:|---:|---:|"])
        for name, row in audit["dimensions"][dimension].items():
            lines.append(
                f"| `{name}` | {row['call_count']} | "
                f"{row['full_source_evidence_hit_count']} | "
                f"{row['returned_information_evidence_hit_count']} | "
                f"{row['answer_lost_by_truncation_count']} | "
                f"{_rate(row['answer_lost_by_truncation_rate_given_full_hit'])} |"
            )
    return "\n".join(lines) + "\n"


def task_label_markdown(audit: Mapping[str, Any]) -> str:
    sources = Counter(row["source_dataset"] for row in audit["provenance"])
    source_splits = Counter(
        f"{row['source_dataset']}::{row['source_split']}"
        for row in audit["provenance"]
    )
    lines = [
        "# Task Type / Search-required Label Audit",
        "",
        "The formal Dev labels were not modified. They are release-allocation labels, "
        "not direct source truth: answer reachability creates eligible search roles, "
        "maximum-flow assigns balanced search roles, and unassigned examples populate "
        "`search_free`. `search_required` is then derived as task type != search_free.",
        "",
        f"Episodes: {audit['episode_count']}; suspect cases: "
        f"{audit['suspect_case_count']}.",
        "",
        "| Source | Count |",
        "|---|---:|",
    ]
    lines.extend(f"| `{name}` | {count} |" for name, count in sorted(sources.items()))
    lines.extend(["", "## Source splits", "", "| Source and split | Count |", "|---|---:|"])
    lines.extend(
        f"| `{name}` | {count} |" for name, count in sorted(source_splits.items())
    )
    lines.extend([
        "",
        "## Suspect handling",
        "",
        _matrix_table(audit["suggested_handling_counts"]),
        "",
        "Because source task truth is absent for many FVQA/MMSearch examples and "
        "the metadata explicitly requests manual review, current route metrics must "
        "not be interpreted as accuracy against an independently annotated route truth.",
    ])
    return "\n".join(lines) + "\n"


def frozen_test_semantics_markdown(
    label_audit: Mapping[str, Any],
) -> str:
    splits = Counter(
        (row["source_dataset"], row["source_split"])
        for row in label_audit["provenance"]
    )
    lines = [
        "# Frozen Test Access Marker Semantics",
        "",
        "`FROZEN_TEST_NOT_ACCESSED` in v1.1 means the Unified Agent Eval v1.1 "
        "frozen `test.jsonl` partition was not opened or evaluated. The run manifests "
        "also record `test_accessed=false`.",
        "",
        "It does not mean that a Dev example cannot originate from a historical "
        "upstream split whose dataset name contains `test`. `fvqa_test` is an upstream "
        "source name; Visual InfoSeek `test`/`human` labels are upstream source splits. "
        "Those examples were selected into the already frozen Unified Dev before this "
        "audit and are therefore allowed Dev inputs here.",
        "",
        "| Upstream source | Upstream split label | Unified Dev count |",
        "|---|---|---:|",
    ]
    lines.extend(
        f"| `{source}` | `{split}` | {count} |"
        for (source, split), count in sorted(splits.items())
    )
    lines.extend([
        "",
        "The historical marker therefore has naming ambiguity but not evidence of "
        "Unified Test access. A clearer future marker is "
        "`UNIFIED_EVAL_FROZEN_TEST_NOT_ACCESSED`. Historical logs are unchanged.",
        "",
        "This audit did not read, hash, enumerate, or open the Unified frozen test "
        "partition.",
    ])
    return "\n".join(lines) + "\n"


def statistical_tests_markdown(tests: Mapping[str, Any]) -> str:
    lines = [
        "# Paired Statistical Tests",
        "",
        f"Comparison: {tests['comparison']}; seed={tests['seed']}; "
        f"bootstrap samples={tests['bootstrap_samples']}; confidence="
        f"{tests['confidence_level']:.2f}.",
        "",
        "| Group | n | Metric | Difference | 95% CI | p-value | Significant |",
        "|---|---:|---|---:|---|---:|---|",
    ]
    for group, values in tests["groups"].items():
        for metric in ("em_v1", "token_f1_v1", "em_v2_strict", "tool_calls"):
            row = values[metric]
            low, high = row["confidence_interval"]
            lines.append(
                f"| `{group}` | {values['episode_count']} | `{metric}` | "
                f"{row['difference_right_minus_left']:.6f} | "
                f"[{low:.6f}, {high:.6f}] | "
                f"{row['two_sided_bootstrap_p_value']:.6f} | "
                f"{row['statistically_significant']} |"
            )
        row = values["mcnemar_v1"]
        lines.append(
            f"| `{group}` | {values['episode_count']} | `mcnemar_v1` | N/A | N/A | "
            f"{row['exact_two_sided_p_value']:.6f} | "
            f"{row['statistically_significant']} |"
        )
    lines.extend([
        "",
        "A result is described as statistically significant only when the paired "
        "bootstrap confidence interval excludes zero or the exact McNemar p-value is "
        "below 0.05.",
    ])
    return "\n".join(lines) + "\n"


def reward_design_markdown(
    records: Mapping[str, Sequence[Mapping[str, Any]]],
    metrics: Mapping[str, Any],
    query_audit: Mapping[str, Any],
    truncation: Mapping[str, Any],
    label_audit: Mapping[str, Any],
) -> str:
    policy_rows = [*records["sft"], *records["grpo"]]
    searched = [row for row in policy_rows if row["successful_search_count"]]
    failures = {
        "Retriever coverage failure": sum(not row["evidence_hit_any"] for row in searched),
        "Information truncation failure": truncation["dimensions"]["overall"]["overall"]["answer_lost_by_truncation_count"],
        "Query generation failure": sum(
            value["text_search_attempt_count"] - value["evidence_hit_count"]
            for value in query_audit["models"].values()
        ),
        "Evidence utilization failure": sum(
            row["evidence_available_final_wrong_v2"] for row in policy_rows
        ),
        "Entity copy failure": sum(row["entity_copy_failure"] for row in policy_rows),
        "Termination failure": sum(row["termination_failure"] for row in policy_rows),
        "Duplicate search failure": sum(row["repeat_search_failure"] for row in policy_rows),
        "Answer evaluator mismatch": sum(
            not row["em_v1"] and row["em_v2_strict"] for row in policy_rows
        ),
        "Route label noise": int(label_audit["suspect_case_count"]),
    }
    ranking = sorted(failures.items(), key=lambda item: (-item[1], item[0]))
    lines = [
        "# Reward v2 Design Inputs",
        "",
        "This is a design-input report only. No reward was implemented, modified or run. "
        "Counts have different audit scopes, so ranking is triage—not a causal estimate.",
        "",
        "| Triage rank | Failure family | Observed count | Scope |",
        "|---:|---|---:|---|",
    ]
    scopes = {
        "Retriever coverage failure": "successful SFT+GRPO search episodes",
        "Information truncation failure": "successful Raw+SFT+GRPO image calls",
        "Query generation failure": "SFT+GRPO text calls without answer evidence",
        "Evidence utilization failure": "SFT+GRPO evidence-hit episodes with wrong v2 answer",
        "Entity copy failure": "SFT+GRPO episodes; heuristic",
        "Termination failure": "SFT+GRPO evidence-hit episodes",
        "Duplicate search failure": "SFT+GRPO episodes",
        "Answer evaluator mismatch": "SFT+GRPO episodes",
        "Route label noise": "unique Unified Dev examples; heuristic suspects",
    }
    for rank, (name, count) in enumerate(ranking, 1):
        lines.append(f"| {rank} | {name} | {count} | {scopes[name]} |")
    lines.extend([
        "",
        "## Intervention ownership",
        "",
        "- Reward can target evidence-conditioned answer correctness, completion after "
        "sufficient evidence, and duplicate/budget behavior, but only with masks that "
        "prevent rewarding unavailable evidence.",
        "- Retriever changes are required for no-hit searches and corpus coverage; a "
        "policy reward cannot manufacture missing evidence.",
        "- Evaluator changes are required for deterministic numeric/unit/alias mismatches. "
        "v2 should remain separately versioned and audited.",
        "- Data-label work is required before route labels can be used as trusted reward "
        "targets. Suspects should be excluded or manually reviewed, not silently relabeled.",
        "- Evidence-use SFT may be appropriate for extraction/entity-copy errors, but "
        "this audit neither starts nor selects that process.",
        "",
        "## Metrics that must not be direct reward signals",
        "",
        "- `task_type` / `search_required` as source truth; they are derived allocation labels.",
        "- Heuristic `entity_copy_failure`, `generic_query`, `entity_missing`, or "
        "`relation_missing` booleans without human precision validation.",
        "- Raw evidence substring hit alone; it can reward copying irrelevant spans.",
        "- v2 closed semantic diagnostics as strict correctness unless explicitly enabled.",
        "",
        "## Minimal candidate components and risks",
        "",
        "| Component | Definition | Main risk |",
        "|---|---|---|",
        "| Evidence-use | Reward correct final answer only when returned evidence hits an accepted answer | Overfitting to string hit; discouraging valid no-hit reasoning |",
        "| Completion | Reward valid final answer after evidence; penalize exhaustion/budget failure | Premature answers before enough evidence |",
        "| Duplicate control | Penalize repeated identical tool/query signatures | Preventing justified retries after transient failure |",
        "| Query utility | Reward text queries only from deterministic execution + evidence outcome | Sparse/noisy credit with very few text calls |",
        "| Evaluator v2 | Use audited strict numeric/unit/alias equivalence for terminal correctness | False positives from overly broad parsing; requires fixture gates |",
        "",
        "## Minimal ablation",
        "",
        "Keep the model, frozen environment, prompts and Dev inputs fixed. Compare: "
        "(A) current reward; (B) +completion/duplicate only; (C) +evidence-use only; "
        "(D) B+C; (E) D scored with v1 and separately with v2 strict. Report paired "
        "bootstrap/McNemar, retrieval coverage, evidence utilization, termination, and "
        "protocol validity. Do not tune or select on Unified Frozen Test.",
    ])
    return "\n".join(lines) + "\n"


def final_report_markdown(
    records: Mapping[str, Sequence[Mapping[str, Any]]],
    metrics: Mapping[str, Any],
    answer_metrics: Mapping[str, Any],
    paired: Mapping[str, Any],
    query_audit: Mapping[str, Any],
    truncation: Mapping[str, Any],
    label_audit: Mapping[str, Any],
    tests: Mapping[str, Any],
) -> str:
    sft = metrics["models"]["sft"]["overall"]["overall"]
    grpo = metrics["models"]["grpo"]["overall"]["overall"]
    sft_query = query_audit["models"]["sft"]
    grpo_query = query_audit["models"]["grpo"]
    trunc = truncation["dimensions"]["overall"]["overall"]
    identity = paired["retrieval_trace_identity"]
    overall_stats = tests["groups"]["overall"]
    significant_em = overall_stats["em_v1"]["statistically_significant"]
    lines = [
        "# Unified Agent Eval v1.2 Evidence Utilization and Evaluation Audit",
        "",
        "This report audits saved v1.1 Frozen Dev episodes and a hash-locked offline "
        "environment. It does not report new model inference.",
        "",
        "## Answers to the 16 required questions",
        "",
        "1. **Did SFT and GRPO receive search results?** Yes. SFT completed "
        f"{sft['successful_search_episode_count']} successful-search episodes and GRPO "
        f"{grpo['successful_search_episode_count']}; complete reconstructed Information "
        "is stored per call in the evidence JSONL.",
        "2. **What search behavior did GRPO add?** The paired search matrix records "
        f"`{paired['search_behavior']}`. Full traces differ for "
        f"{identity['different_search_trace_count']}/200 pairs; the increase is primarily "
        "Image Search, while Text Search attempts changed from "
        f"{sft_query['text_search_attempt_count']} to {grpo_query['text_search_attempt_count']}.",
        "3. **Did Text Search materially improve?** The saved behavior contains only "
        f"{sft_query['text_search_attempt_count']} SFT and "
        f"{grpo_query['text_search_attempt_count']} GRPO attempts, with evidence hits "
        f"{sft_query['evidence_hit_count']} and {grpo_query['evidence_hit_count']}. This "
        "small observational sample does not establish a material improvement.",
        "4. **How often did returned evidence contain an accepted answer?** Among "
        f"successful-search episodes: SFT {_rate(sft['evidence_hit_rate'])}; GRPO "
        f"{_rate(grpo['evidence_hit_rate'])}.",
        "5. **How often was answer evidence present but the final answer wrong?** "
        f"SFT {_rate(sft['evidence_hit_but_wrong_rate'])} and GRPO "
        f"{_rate(grpo['evidence_hit_but_wrong_rate'])} of successful-search episodes "
        "under v1 EM.",
        "6. **How often was answer evidence present but the episode did not finish?** "
        f"SFT {_rate(sft['evidence_hit_but_no_answer_rate'])}; GRPO "
        f"{_rate(grpo['evidence_hit_but_no_answer_rate'])}.",
        "7. **How frequent was entity copying?** The conservative heuristic flagged "
        f"{sft['entity_copy_failure_count']}/200 SFT and "
        f"{grpo['entity_copy_failure_count']}/200 GRPO episodes. This is diagnostic, "
        "not ground truth.",
        "8. **What was the Image Search truncation loss?** Full frozen candidate records "
        f"were available for {trunc['full_source_available_count']}/"
        f"{trunc['call_count']} audited calls. {trunc['answer_lost_by_truncation_count']} "
        f"calls lost an answer hit after 512-character truncation "
        f"({_rate(trunc['answer_lost_by_truncation_rate_given_full_hit'])} of full-source hits).",
        "9. **What were the main Text Query failure modes?** SFT counts are "
        f"`{sft_query['failure_type_counts']}` and GRPO counts are "
        f"`{grpo_query['failure_type_counts']}`. Generic/entity/relation flags are "
        "heuristics; empty/ellipsis/punctuation/too-short are deterministic.",
        "10. **How did paired correctness change?** v1 transitions are "
        f"`{paired['answer_correctness_v1']}`; v2 strict transitions are "
        f"`{paired['answer_correctness_v2']}`.",
        "11. **How did paired search change?** "
        f"`{paired['search_behavior']}`; first-tool transitions are "
        f"`{paired['tool_choice']}`.",
        "12. **How different are v1 and v2 evaluation?** v1-wrong/v2-strict-correct "
        "counts are " + ", ".join(
            f"{name.upper()}={row['v1_wrong_v2_strict_correct_count']}"
            for name, row in answer_metrics["models"].items()
        ) + ". Closed semantic equivalence remains diagnostic-only.",
        "13. **Are current task labels reliable ground truth?** No. They are derived "
        f"release-allocation labels; {label_audit['suspect_case_count']}/200 examples "
        "are flagged for conservative review. Formal labels were not changed.",
        "14. **What does the Frozen Test marker mean?** It guarantees that the Unified "
        "Eval frozen test partition was not accessed; it does not ban upstream source "
        "names/splits such as `fvqa_test` from the already frozen Unified Dev.",
        "15. **What is the present bottleneck?** The audit separates no-hit retrieval, "
        "hit-but-wrong policy use, termination, evaluator mismatch and label noise. "
        "Their counts are reported separately; no single cause is assumed. The observed "
        f"GRPO-minus-SFT EM difference is {'statistically significant' if significant_em else 'not statistically significant'} "
        "under the specified paired bootstrap.",
        "16. **What is the minimum reasonable Reward v2 design?** First evaluate "
        "completion/duplicate control and evidence-conditioned terminal correctness as "
        "separate ablations, use v2 strict only behind fixtures, and exclude noisy route "
        "labels/heuristic diagnostics from direct reward. Retriever and label failures "
        "require their own interventions.",
        "",
        "## Boundary",
        "",
        "No Raw/SFT/GRPO inference, training, reward update, checkpoint selection, online "
        "access, frozen-environment mutation, Prompt Pool mutation, or Unified "
        "Frozen Test access occurred.",
    ]
    return "\n".join(lines) + "\n"
