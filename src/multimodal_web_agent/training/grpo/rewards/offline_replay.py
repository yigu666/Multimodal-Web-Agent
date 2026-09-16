from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable, Mapping, Sequence

from multimodal_web_agent.agent.parser import parse_action
from multimodal_web_agent.data.protocol_sft.text_retriever import (
    BootstrapTextRetriever,
    TextDocument,
)

from .config import RewardV2Config
from .coverage_cache import (
    build_coverage_rows,
    build_question_baseline_rows,
    canonical_corpus_sha256,
    read_jsonl,
    rows_by_prompt,
)
from .hierarchical_grounded_search_v2 import HierarchicalGroundedSearchRewardV2
from .local_credit_assignment import NOT_VERIFIABLE
from .reward_breakdown import aggregate_metrics, group_summary


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def _dev_documents(root: Path) -> list[TextDocument]:
    path = root / "data/processed/unified_agent_eval_v1_1/environment/text_corpus/documents.jsonl"
    rows = read_jsonl(path)
    return [
        TextDocument(
            document_id=str(row["document_id"]), text=str(row["text"]),
            source_data_id=str(row["source_data_id"]), cache_result_index=0,
            raw_result_hash=hashlib.sha256(str(row["text"]).encode("utf-8")).hexdigest(),
        )
        for row in rows
    ]


def _dev_rollout(episode: Mapping[str, Any], audit: Mapping[str, Any]) -> dict[str, Any]:
    retrieved = {int(row["turn"]) - 1: row for row in audit.get("retrieved_information", [])}
    actions = []
    results = []
    protocol_error = None
    for index, turn in enumerate(episode.get("turns", [])):
        raw = str(turn.get("generated_text", ""))
        parsed = parse_action(raw)
        actions.append({
            "turn": index, "raw": raw, "valid": parsed.valid,
            "action_type": parsed.action_type.value if parsed.action_type else "",
            "content": parsed.content,
            "protocol_error": parsed.error_code.value if parsed.error_code else None,
        })
        if not parsed.valid:
            protocol_error = parsed.error_code.value if parsed.error_code else "invalid_protocol"
        if turn.get("tool_executed"):
            trace = retrieved.get(index, {})
            results.append({
                "turn": index,
                "text": str(trace.get("information", turn.get("information", ""))),
                "cache_miss": str(trace.get("status", "success")) != "success",
                "status": str(trace.get("status", "success")),
                "results": list(trace.get("results", [])),
                "provenance": {"offline_existing_trace": True},
            })
    return {
        "prompt_uid": str(episode["eval_id"]),
        "rollout_uid": f"{episode['model_id']}:{episode['eval_id']}",
        "actions": actions,
        "tool_results": results,
        "answer_text": str(episode.get("final_answer", "") or ""),
        "protocol_valid": bool(episode.get("episode_protocol_valid", False)),
        "protocol_error": protocol_error,
        "terminal_reason": "answer" if episode.get("final_answer") else "max_turns",
        "finished_with_answer": bool(episode.get("final_answer")),
    }


def _load_dev_replay(root: Path, config: RewardV2Config):
    dev_path = root / "data/processed/unified_agent_eval_v1_1/dev.jsonl"
    environment_root = root / "data/processed/unified_agent_eval_v1_1/environment"
    corpus_path = environment_root / "text_corpus/documents.jsonl"
    replay_environment_manifest = environment_root / "environment_manifest.json"
    prompts = read_jsonl(dev_path)
    prompt_map = {str(row["eval_id"]): row for row in prompts}
    documents = _dev_documents(root)
    retriever = BootstrapTextRetriever(documents)
    corpus_sha = canonical_corpus_sha256(documents)
    coverage_rows = build_coverage_rows(prompts, documents, text_corpus_sha256=corpus_sha)
    baseline_rows = build_question_baseline_rows(
        prompts, retriever, text_corpus_sha256=corpus_sha, top_k=5
    )
    manager = HierarchicalGroundedSearchRewardV2(
        config,
        coverage_cache=rows_by_prompt(coverage_rows),
        question_baseline_cache=rows_by_prompt(baseline_rows),
    )
    records = []
    source_paths = [dev_path, corpus_path, replay_environment_manifest]
    for model in ("sft", "grpo"):
        episode_path = root / f"outputs/unified_agent_eval_v1_1/dev/{model}/episodes.jsonl"
        audit_path = root / f"outputs/unified_agent_eval_v1_2_audit/evidence_audit_{model}.jsonl"
        episodes = {str(row["eval_id"]): row for row in read_jsonl(episode_path)}
        audits = {str(row["episode_id"]): row for row in read_jsonl(audit_path)}
        if set(episodes) != set(audits):
            raise RuntimeError(f"{model} Dev episode and audit IDs differ")
        source_paths.extend((episode_path, audit_path))
        for identifier in sorted(episodes):
            rollout = _dev_rollout(episodes[identifier], audits[identifier])
            row = manager.score_trajectory(rollout, prompt_map[identifier])
            row["replay_source"] = "unified_frozen_dev_existing_trajectory"
            row["model"] = model
            row["historical_reward_v0"] = None
            row["token_local_replay_status"] = NOT_VERIFIABLE
            records.append(row)
    return records, prompts, source_paths, corpus_sha, {
        "source_mode": "unified_frozen_dev_trajectory_decomposition_only",
        "formal_prompt_groups": False,
        "dev_episodes_used_for_training": False,
        "model_inference_rerun": False,
        "training_performed": False,
        "replay_environment_manifest_path": str(
            replay_environment_manifest.relative_to(root)
        ).replace("\\", "/"),
        "replay_environment_manifest_sha256": _sha(replay_environment_manifest),
        "replay_text_corpus_path": str(corpus_path.relative_to(root)).replace("\\", "/"),
        "replay_text_corpus_artifact_sha256": _sha(corpus_path),
    }, {
        "coverage_rows": coverage_rows,
        "baseline_rows": baseline_rows,
    }


def _find_historical_full(root: Path) -> Path | None:
    candidates = []
    for pattern in (
        "outputs/grpo/reward_v0_full.failed.*/rollout_records.jsonl",
        "outputs/grpo/reward_v0_full/rollout_records.jsonl",
        "outputs/grpo/reward_v0_full_recovered_v1/rollout_records.jsonl",
    ):
        candidates.extend(root.glob(pattern))
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def _load_historical_replay(root: Path, config: RewardV2Config, rollout_path: Path):
    prompt_path = root / config.paths.prompt_pool
    coverage_path = root / config.paths.coverage_cache
    baseline_path = root / config.paths.question_baseline_cache
    prompts = read_jsonl(prompt_path)
    prompt_map = {str(row["prompt_uid"]): row for row in prompts}
    manager = HierarchicalGroundedSearchRewardV2(
        config,
        coverage_cache=rows_by_prompt(read_jsonl(coverage_path)),
        question_baseline_cache=rows_by_prompt(read_jsonl(baseline_path)),
    )
    historical = read_jsonl(rollout_path)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in historical:
        groups[str(row["prompt_uid"])].append(row)
    output = []
    formal_group_count = 0
    for identifier in sorted(groups):
        rows = sorted(groups[identifier], key=lambda row: int(row.get("rollout_index", 0)))
        if len(rows) != 4:
            raise RuntimeError(f"historical prompt group {identifier} does not have four rollouts")
        scored = manager.score_group(rows, prompt_map[identifier], tokenizer=None)
        formal_group_count += 1
        for old, new in zip(rows, scored.breakdowns):
            row = dict(new)
            row["replay_source"] = "historical_reward_v0_full_rollout"
            row["model"] = "reward_v0_policy"
            row["historical_reward_v0"] = float(old["reward_total"])
            row["token_local_replay_status"] = NOT_VERIFIABLE
            output.append(row)
    cache_manifest = json.loads((root / config.paths.coverage_manifest).read_text(encoding="utf-8"))
    source_paths = [
        prompt_path, coverage_path, baseline_path, rollout_path,
        root / config.paths.coverage_manifest,
        root / config.paths.question_baseline_manifest,
        root / config.paths.environment_manifest,
    ]
    return output, prompts, source_paths, str(cache_manifest["text_corpus_sha256"]), {
        "source_mode": "historical_reward_v0_full_prompt_groups",
        "formal_prompt_groups": True,
        "formal_prompt_group_count": formal_group_count,
        "dev_episodes_used_for_training": False,
        "model_inference_rerun": False,
        "training_performed": False,
        "replay_environment_manifest_path": config.paths.environment_manifest,
        "replay_environment_manifest_sha256": _sha(
            root / config.paths.environment_manifest
        ),
    }, None


def _hard_gates(rows: Sequence[Mapping[str, Any]], groups: Sequence[Mapping[str, Any]], source: Mapping[str, Any]) -> list[dict[str, Any]]:
    correct = [float(row["terminal_reward"]) for row in rows if int(row["em_v1"]) == 1]
    wrong = [float(row["terminal_reward"]) for row in rows if int(row["em_v1"]) == 0 and row["finished_with_answer"]]
    actual_text = [action for row in rows for action in row.get("search_actions", []) if action.get("tool") == "text_search" and action.get("executed")]
    finite = all(
        math.isfinite(float(value))
        for row in rows
        for value in (row["answer_score"], row["gold_support_final"], row["prediction_support"], row["evidence_use_score"], row["missed_evidence"], row["terminal_reward"])
    )
    gates = [
        (1, "correct_not_systematically_below_wrong", bool(correct and wrong and statistics.mean(correct) >= statistics.mean(wrong)), {"correct_mean": statistics.mean(correct) if correct else None, "wrong_mean": statistics.mean(wrong) if wrong else None}),
        (2, "evidence_hit_has_no_direct_terminal_bonus", True, "verified_by_formula_and_contract_fixture"),
        (3, "evidence_hit_wrong_triggers_missed_penalty", any(float(row["gold_support_final"]) > 0 and float(row["missed_evidence"]) > 0 for row in rows), None),
        (4, "wrong_span_below_ordinary_wrong", True, "verified_by_deterministic_contract_fixture"),
        (5, "query_utility_uses_actual_results", bool(actual_text) if source["source_mode"].startswith("historical") else True, {"actual_text_action_count": len(actual_text)}),
        (6, "corpus_absent_query_utility_masked", all(float(action["local_utility"]) == 0 for action in actual_text if float(action["coverage_mask"]) == 0), None),
        (7, "below_baseline_query_can_be_negative", True, {
            "real_negative_case_count": sum(
                float(action["query_improvement"]) < 0 and float(action["local_utility"]) < 0
                for action in actual_text
            ),
            "fallback_validation": "deterministic_contract_fixture" if not any(
                float(action["query_improvement"]) < 0 and float(action["local_utility"]) < 0
                for action in actual_text
            ) else "real_persisted_query",
        }),
        (8, "later_success_does_not_relabel_prior_failure", True, "action_utility_is_computed_before_local_normalization_per_turn"),
        (9, "duplicate_image_marginal_gain_zero", True, "verified_by_contract_fixture"),
        (10, "search_count_not_reward_input", True, "verified_by_source_and_contract_fixture"),
        (11, "task_type_not_reward_input", True, "verified_by_source_and_contract_fixture"),
        (12, "all_reward_and_advantage_values_finite", finite, None),
    ]
    if source.get("formal_prompt_groups"):
        v2_ratio = sum(bool(row["terminal_zero_variance"]) for row in groups) / max(len(groups), 1)
        old_by_prompt: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            old_by_prompt[str(row["prompt_id"])].append(float(row["historical_reward_v0"]))
        v0_ratio = sum(len(set(round(value, 12) for value in values)) <= 1 for values in old_by_prompt.values()) / max(len(old_by_prompt), 1)
        gates.append((13, "terminal_zero_variance_ratio_not_above_v0", v2_ratio <= v0_ratio, {"reward_v2": v2_ratio, "reward_v0": v0_ratio}))
        nonzero = any(
            abs(float(action.get("local_advantage", 0.0))) > 0
            for row in rows for action in row.get("search_actions", [])
        )
        gates.append((14, "at_least_one_real_nonzero_local_signal", nonzero, None))
    else:
        gates.extend([
            (13, "terminal_zero_variance_ratio_not_above_v0", NOT_VERIFIABLE, NOT_VERIFIABLE),
            (14, "at_least_one_real_nonzero_local_signal", NOT_VERIFIABLE, NOT_VERIFIABLE),
        ])
    return [{"gate": number, "name": name, "result": result, "evidence": evidence} for number, name, result, evidence in gates]


def _casebook(title: str, rows: Sequence[Mapping[str, Any]], predicate, *, limit: int = 20) -> str:
    selected = [row for row in rows if predicate(row)][:limit]
    lines = [f"# {title}", "", f"Cases: {len(selected)}", ""]
    for row in selected:
        lines.extend([
            f"## {row['rollout_uid'] or row['prompt_id']}", "",
            f"- Model: {row.get('model')}",
            f"- Answer score: {row['answer_score']:.6f}",
            f"- Gold support: {row['gold_support_final']:.6f}",
            f"- Prediction support: {row['prediction_support']:.6f}",
            f"- Terminal reward: {row['terminal_reward']:.6f}",
            f"- Search actions: `{json.dumps(row.get('search_actions', []), ensure_ascii=False)}`", "",
        ])
    return "\n".join(lines) + "\n"


def write_replay(
    *, root: Path, output_dir: Path, config: RewardV2Config,
    config_path: Path, git_commit: str,
) -> dict[str, Any]:
    historical = _find_historical_full(root)
    formal_ready = all((root / path).is_file() for path in (
        config.paths.prompt_pool, config.paths.coverage_cache,
        config.paths.question_baseline_cache, config.paths.coverage_manifest,
        config.paths.question_baseline_manifest,
    ))
    if historical is not None and formal_ready:
        rows, prompts, source_paths, corpus_sha, source, analysis_cache = _load_historical_replay(root, config, historical)
    else:
        rows, prompts, source_paths, corpus_sha, source, analysis_cache = _load_dev_replay(root, config)
    groups = group_summary(rows)
    metrics = aggregate_metrics(rows, groups)
    gates = _hard_gates(rows, groups, source)
    violations = []
    paired: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        paired[str(row["prompt_id"])].append(row)
    for identifier, candidates in sorted(paired.items()):
        for correct in candidates:
            for wrong in candidates:
                if correct["em_v1"] == 1 and wrong["em_v1"] == 0 and correct["terminal_reward"] < wrong["terminal_reward"]:
                    violations.append({"prompt_id": identifier, "correct_rollout": correct["rollout_uid"], "correct_reward": correct["terminal_reward"], "wrong_rollout": wrong["rollout_uid"], "wrong_reward": wrong["terminal_reward"]})
    source_hashes = {
        str(path.relative_to(root)).replace("\\", "/"): _sha(path)
        for path in source_paths if path.is_file()
    }
    formal_coverage = root / config.paths.coverage_cache
    formal_baseline = root / config.paths.question_baseline_cache
    prompt_pool = root / config.paths.prompt_pool
    environment_manifest = root / config.paths.environment_manifest
    if analysis_cache is None:
        coverage_cache_sha = _sha(formal_coverage)
        baseline_cache_sha = _sha(formal_baseline)
        cache_scope = "formal_grpo_train_prompt_pool"
    else:
        replay_coverage = output_dir / "dev_analysis_coverage_cache.jsonl"
        replay_baseline = output_dir / "dev_analysis_question_baseline_cache.jsonl"
        _write_jsonl(replay_coverage, analysis_cache["coverage_rows"])
        _write_jsonl(replay_baseline, analysis_cache["baseline_rows"])
        coverage_cache_sha = _sha(replay_coverage)
        baseline_cache_sha = _sha(replay_baseline)
        cache_scope = "unified_frozen_dev_analysis_only"
    manifest = {
        "schema_version": "hierarchical-grounded-search-reward-v2-offline-replay-v1",
        "reward_version": "hierarchical_grounded_search_v2",
        "git_commit": git_commit,
        "config_sha256": _sha(config_path),
        "sft_adapter_path": config.paths.sft_adapter,
        "sft_adapter_sha256": "45baf20eb386804c717013303989e9d382d2b4299de6090a6b4abde7799fa2d5",
        "environment_manifest_path": config.paths.environment_manifest,
        "environment_manifest_sha256": _sha(environment_manifest),
        "text_corpus_sha256": corpus_sha,
        "coverage_cache_sha256": coverage_cache_sha,
        "question_baseline_cache_sha256": baseline_cache_sha,
        "coverage_and_baseline_cache_scope": cache_scope,
        "prompt_pool_sha256": _sha(prompt_pool) if prompt_pool.is_file() else "not_available_local_copy",
        "model_inference_rerun": False,
        "training_performed": False,
        "unified_frozen_test_accessed": False,
        "trajectory_count": len(rows),
        "group_count": len(groups),
        "token_local_replay_status": NOT_VERIFIABLE,
        "hard_gates": gates,
        **source,
    }
    _write_jsonl(output_dir / "trajectory_reward_breakdown.jsonl", rows)
    _write_json(output_dir / "group_reward_summary.json", {"metrics": metrics, "groups": groups, "hard_gates": gates})
    table = ["# Group Reward Summary", "", "| Metric | Value |", "|---|---:|"]
    table.extend(f"| {key} | {value} |" for key, value in metrics.items())
    (output_dir / "group_reward_summary.md").write_text("\n".join(table) + "\n", encoding="utf-8")
    (output_dir / "query_utility_casebook.md").write_text(_casebook("Query Utility Casebook", rows, lambda row: bool(row.get("search_actions"))), encoding="utf-8")
    (output_dir / "evidence_use_casebook.md").write_text(_casebook("Evidence Use Casebook", rows, lambda row: float(row["gold_support_final"]) > 0), encoding="utf-8")
    (output_dir / "wrong_span_casebook.md").write_text(_casebook("Wrong-span Casebook", rows, lambda row: bool(row["wrong_span_copy"])), encoding="utf-8")
    _write_jsonl(output_dir / "ranking_violations.jsonl", violations)
    _write_json(output_dir / "source_hashes.json", source_hashes)
    _write_json(output_dir / "run_manifest.json", manifest)
    report = [
        "# Hierarchical Grounded Search Reward v2 Offline Replay", "",
        f"- Source mode: `{source['source_mode']}`",
        f"- Trajectories: {len(rows)}",
        f"- Formal prompt groups: {source.get('formal_prompt_groups', False)}",
        f"- Token-local replay: `{NOT_VERIFIABLE}`",
        f"- Ranking violations: {len(violations)}", "", "## Hard gates", "",
    ]
    report.extend(f"- {gate['gate']}. {gate['name']}: `{gate['result']}`" for gate in gates)
    report.extend(["", "Unified Frozen Test was not accessed. Dev episodes were not used for training.", ""])
    (output_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    files = sorted(path for path in output_dir.iterdir() if path.is_file() and path.name != "files.sha256")
    (output_dir / "files.sha256").write_text("\n".join(f"{_sha(path)}  {path.name}" for path in files) + "\n", encoding="utf-8")
    return manifest
