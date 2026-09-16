from __future__ import annotations

from collections import Counter, defaultdict
import math
from typing import Any, Mapping, Sequence


SMOKE_CONTRACT_V2_SCHEMA = "grpo-reward-v0-smoke-contract-v2"
FORMAL_ENVIRONMENT = {
    "image_top_k": 3,
    "text_top_k": 3,
    "image_context_policy": "canonical_cached_top3",
}
RANK1_DIAGNOSTIC_FIELDS = {
    "rank1_probe_role": "historical_diagnostic_only",
    "rank1_probe_used_for_training": False,
    "rank1_probe_used_for_gate": False,
    "rank1_probe_used_for_selection": False,
}
BEHAVIOR_CONTRACT_FIELDS = {
    "text_search_exploration_required_for_smoke": False,
    "three_turn_exploration_required_for_smoke": False,
    "behavior_metrics_are_diagnostic_only": True,
}
FIXTURE_BOUNDARY_FIELDS = {
    "fixture_driven": True,
    "on_policy": False,
    "used_for_training": False,
    "used_for_reward": False,
    "used_for_advantage": False,
    "used_for_smoke_behavior_metrics": False,
}
ALLOWED_REWARD_VALUES = (0.00, 0.10, 0.81, 0.90, 0.91, 1.00)


def formal_chain_manifest_fields() -> dict[str, Any]:
    return {
        "formal_environment": dict(FORMAL_ENVIRONMENT),
        **RANK1_DIAGNOSTIC_FIELDS,
    }


def historical_config_hash_diagnostic(
    current_sha256: str,
    historical_hashes: Mapping[str, str],
) -> dict[str, Any]:
    """Report byte drift without invalidating semantically verified artifacts.

    Acceptance/configuration files are expected to change during this contract
    correction. Historical rollout provenance and frozen math/replay hashes,
    rather than byte identity with the corrected config, decide reuse.
    """
    matches = {
        str(stage): str(value) == str(current_sha256)
        for stage, value in historical_hashes.items()
    }
    return {
        "current_sha256": str(current_sha256),
        "historical_sha256": dict(historical_hashes),
        "byte_hash_matches": matches,
        "all_byte_hashes_match": all(matches.values()) if matches else None,
        "diagnostic_only": True,
        "blocks_historical_artifact_reuse": False,
    }


def _value(record: Any, key: str, default: Any = None) -> Any:
    if isinstance(record, Mapping):
        return record.get(key, default)
    return getattr(record, key, default)


def behavior_diagnostics(
    records: Sequence[Any],
    group_statistics: Sequence[Mapping[str, Any]] = (),
    *,
    reward_metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return policy-behavior measurements that never decide stage success."""
    count = len(records)
    denominator = count or 1
    first_actions = Counter(
        (
            _value(record, "actions", [])[0].get("action_type", "none")
            if _value(record, "actions", [])
            else "none"
        )
        for record in records
    )
    answer_only_count = sum(
        int(_value(record, "search_count", 0)) == 0 for record in records
    )
    image_search_count = sum(
        bool(_value(record, "used_image_search", False)) for record in records
    )
    text_search_count = sum(
        bool(_value(record, "used_text_search", False)) for record in records
    )
    three_turn_count = sum(
        len(_value(record, "actions", [])) == 3 for record in records
    )
    image_to_text_count = sum(
        any(
            left.get("action_type") == "image_search"
            and right.get("action_type") == "text_search"
            for left, right in zip(
                _value(record, "actions", []),
                _value(record, "actions", [])[1:],
            )
        )
        for record in records
    )
    searches = [
        int(_value(record, "search_count", 0)) for record in records
    ]
    turns = [len(_value(record, "actions", [])) for record in records]
    zero_variance = sum(
        bool(row.get("zero_variance_group")) for row in group_statistics
    )
    group_count = len(group_statistics) or 1
    diagnostics = {
        "answer_only_count": answer_only_count,
        "image_search_rollout_count": image_search_count,
        "text_search_rollout_count": text_search_count,
        "three_turn_rollout_count": three_turn_count,
        "image_to_text_route_count": image_to_text_count,
        "first_action_distribution": dict(sorted(first_actions.items())),
        "image_search_rate": image_search_count / denominator,
        "text_search_rate": text_search_count / denominator,
        "image_to_text_rate": image_to_text_count / denominator,
        "average_turns": sum(turns) / denominator,
        "average_search_calls": sum(searches) / denominator,
        "zero_variance_group_ratio": zero_variance / group_count,
        **BEHAVIOR_CONTRACT_FIELDS,
    }
    if reward_metrics:
        for key in (
            "search_free_search_rate",
            "search_required_search_rate",
            "unnecessary_search_rate",
            "search_required_no_search_rate",
            "all_wrong_group_ratio",
            "all_correct_group_ratio",
        ):
            if key in reward_metrics:
                diagnostics[key] = reward_metrics[key]
        diagnostics["search_required_without_search_rate"] = float(
            reward_metrics.get("search_required_no_search_rate", 0.0)
        )
    markers = []
    if text_search_count == 0:
        markers.append("TEXT_SEARCH_EXPLORATION_ABSENT")
    if three_turn_count == 0:
        markers.append("THREE_TURN_EXPLORATION_ABSENT")
    diagnostics["diagnostic_markers"] = markers
    return diagnostics


def validate_smoke_engineering_contract(
    *,
    records: Sequence[Any],
    group_statistics: Sequence[Mapping[str, Any]],
    update_metrics: Sequence[Mapping[str, Any]],
    prompt_count: int,
    rollout_count: int,
    optimizer_steps: int,
    group_size: int,
    expected_processor_hash: str,
    lora_parameter_change_count: int,
    visual_trainable_parameter_count: int,
    visual_parameter_change_count: int,
    checkpoint_saved: bool,
    checkpoint_reload_success: bool,
    reload_generation_success: bool,
    protocol_metrics_computed: bool,
    reward_metrics_computed: bool,
    test_accessed: bool,
) -> dict[str, Any]:
    """Validate only engineering correctness; action coverage is absent by design."""
    failures: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            failures.append(message)

    require(prompt_count == 32, "prompt_count must equal 32")
    require(group_size == 4, "group_size must equal 4")
    require(rollout_count == 128, "rollout_count must equal 128")
    require(optimizer_steps == 8, "optimizer_steps must equal 8")
    require(len(records) == rollout_count, "record count mismatch")
    require(len(update_metrics) == optimizer_steps, "update metric count mismatch")

    groups: dict[str, list[Any]] = defaultdict(list)
    for record in records:
        groups[str(_value(record, "prompt_uid", ""))].append(record)
    require(len(groups) == prompt_count, "prompt group count mismatch")
    require(
        all(len(group) == group_size for group in groups.values()),
        "each prompt must have exactly four rollouts",
    )
    rollout_uids = [str(_value(record, "rollout_uid", "")) for record in records]
    require(
        len(rollout_uids) == len(set(rollout_uids))
        and all(rollout_uids),
        "rollout_uid values must be nonempty and unique",
    )
    require(
        all(
            len(
                {
                    int(_value(record, "generation_seed", -1))
                    for record in group
                }
            )
            == group_size
            for group in groups.values()
        ),
        "generation_seed values must differ within each group",
    )
    require(
        all(
            min(
                abs(float(_value(record, "reward_total", -99.0)) - allowed)
                for allowed in ALLOWED_REWARD_VALUES
            )
            <= 1e-6
            for record in records
        ),
        "Reward v0 emitted a value outside its frozen domain",
    )
    require(
        all(
            str(_value(record, "processor_hash", ""))
            == expected_processor_hash
            for record in records
        ),
        "processor hash mismatch",
    )

    finite_group_fields = (
        "group_reward_mean",
        "group_reward_std",
        "advantage_mean",
        "advantage_std",
    )
    require(
        len(group_statistics) == prompt_count,
        "group statistic count mismatch",
    )
    require(
        all(
            math.isfinite(float(row.get(key, math.nan)))
            for row in group_statistics
            for key in finite_group_fields
        ),
        "group advantage statistics must be finite",
    )
    require(
        all(
            not row.get("zero_variance_group")
            or (
                abs(float(row.get("advantage_mean", math.inf))) <= 1e-7
                and abs(float(row.get("advantage_std", math.inf))) <= 1e-7
            )
            for row in group_statistics
        ),
        "zero-variance group must receive zero advantages",
    )

    update_finite_fields = (
        "loss",
        "gradient_norm",
        "clip_fraction",
        "policy_ratio_mean",
        "policy_ratio_max",
        "max_logprob_alignment_error",
    )
    require(
        all(
            math.isfinite(float(row.get(key, math.nan)))
            for row in update_metrics
            for key in update_finite_fields
        ),
        "loss, gradients, ratios, and alignment metrics must be finite",
    )
    require(
        sum(int(row.get("input_identity_failure_count", 1)) for row in update_metrics)
        == 0,
        "input identity failure detected",
    )
    require(
        sum(int(row.get("response_token_mismatch_count", 1)) for row in update_metrics)
        == 0,
        "response token mismatch detected",
    )
    require(
        sum(int(row.get("processor_hash_mismatch_count", 1)) for row in update_metrics)
        == 0,
        "processor hash mismatch detected",
    )
    require(
        sum(int(row.get("target_truncation_count", 1)) for row in update_metrics)
        == 0,
        "target truncation detected",
    )
    require(
        sum(
            int(row.get("information_token_train_mask_sum", 1))
            for row in update_metrics
        )
        == 0,
        "information tokens leaked into the train mask",
    )
    require(
        sum(int(row.get("policy_action_token_count", 0)) for row in update_metrics)
        > 0,
        "policy action token count must be positive",
    )
    require(
        max(
            (
                float(row.get("max_logprob_alignment_error", math.inf))
                for row in update_metrics
            ),
            default=math.inf,
        )
        < 1e-3,
        "max log-prob alignment error must be below 1e-3",
    )
    require(
        all(bool(row.get("gradient_finite", False)) for row in update_metrics),
        "non-finite gradient detected",
    )
    require(lora_parameter_change_count > 0, "LoRA parameters did not change")
    require(
        visual_trainable_parameter_count == 0,
        "visual parameters became trainable",
    )
    require(
        visual_parameter_change_count == 0,
        "visual parameters changed",
    )
    require(checkpoint_saved, "checkpoint was not saved")
    require(checkpoint_reload_success, "checkpoint reload failed")
    require(reload_generation_success, "generation after checkpoint reload failed")
    require(protocol_metrics_computed, "protocol metrics were not computed")
    require(reward_metrics_computed, "reward metrics were not computed")
    require(test_accessed is False, "Test data was accessed")
    if failures:
        raise RuntimeError(
            "Smoke contract v2 engineering gate failed: " + "; ".join(failures)
        )
    return {
        "schema_version": SMOKE_CONTRACT_V2_SCHEMA,
        "engineering_hard_gates_passed": True,
        "engineering_gate_failures": [],
        "prompt_count": prompt_count,
        "group_size": group_size,
        "rollout_count": rollout_count,
        "optimizer_steps": optimizer_steps,
        "input_identity_failure_count": 0,
        "response_token_mismatch_count": 0,
        "processor_hash_mismatch_count": 0,
        "target_truncation_count": 0,
        "information_token_train_mask_sum": 0,
        "max_logprob_alignment_error": max(
            float(row["max_logprob_alignment_error"])
            for row in update_metrics
        ),
        "lora_parameter_change_count": lora_parameter_change_count,
        "visual_trainable_parameter_count": visual_trainable_parameter_count,
        "visual_parameter_change_count": visual_parameter_change_count,
        "checkpoint_saved": checkpoint_saved,
        "checkpoint_reload_success": checkpoint_reload_success,
        "reload_generation_success": reload_generation_success,
        "protocol_metrics_computed": protocol_metrics_computed,
        "reward_metrics_computed": reward_metrics_computed,
        "test_accessed": False,
    }
