from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from multimodal_web_agent.training.grpo.rewards.text_search_exploration import (
    exploration_schedule_values,
)


SCHEMA = "grpo-stage2-continued-v1"
CONTRACT_MARKER = "GRPO_STAGE2_CONTRACT_READY"
V21_COMPLETE_MARKER = "GRPO_REWARD_V21_FULL_COMPLETE"
STRATEGIES = {"S2_A"}
V21_LR = 5.0e-7
STAGE2_LR = 2.5e-7


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(path: Path) -> str:
    root = Path(path)
    files = sorted(item for item in root.rglob("*") if item.is_file())
    if not files:
        raise RuntimeError(f"cannot fingerprint empty checkpoint: {root}")
    digest = hashlib.sha256()
    for item in files:
        digest.update(item.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def resolve_v21_selected_checkpoint(
    project_root: Path,
) -> tuple[Path, dict[str, Any]]:
    root = Path(project_root).resolve()
    manifest_path = root / "outputs/grpo_reward_v21_full/run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("marker") != V21_COMPLETE_MARKER:
        raise RuntimeError("Reward v2.1 Full completion marker is invalid")
    if manifest.get("experiment_id") != "reward_v21":
        raise RuntimeError("Reward v2.1 manifest identity differs")
    if manifest.get("continuous_engineering_contract_passed") is not True:
        raise RuntimeError("Reward v2.1 engineering contract did not pass")
    if manifest.get("unified_frozen_test_accessed") is not False:
        raise RuntimeError("Reward v2.1 accessed Frozen Test")
    selected_value = manifest.get("selected_checkpoint")
    if not isinstance(selected_value, str) or not selected_value:
        raise RuntimeError("Reward v2.1 manifest has no selected checkpoint")
    selected = Path(selected_value)
    manifest_absolute = selected.is_absolute() or selected_value.startswith("/")
    if manifest_absolute:
        try:
            selected.resolve().relative_to(root)
        except ValueError:
            # Formal artifacts may have been copied back from the temporary
            # /hy-tmp project to /dataB.  Rebase only the exact manifest-owned
            # outputs/... suffix; never infer or search for a checkpoint name.
            parts = selected.parts
            try:
                outputs_index = parts.index("outputs")
            except ValueError as exc:
                raise RuntimeError(
                    "Reward v2.1 absolute checkpoint has no project output suffix"
                ) from exc
            selected = root.joinpath(*parts[outputs_index:])
        else:
            selected = selected.resolve()
    else:
        selected = root / selected
    selected = selected.resolve()
    try:
        selected.relative_to(root)
    except ValueError as exc:
        raise RuntimeError("Reward v2.1 selected checkpoint escapes project") from exc
    formal_root = (root / "outputs/grpo_reward_v21_full").resolve()
    try:
        selected.relative_to(formal_root)
    except ValueError as exc:
        raise RuntimeError(
            "Reward v2.1 selected checkpoint is outside its formal Full output"
        ) from exc
    actual = sha256_tree(selected)
    expected = str(manifest.get("selected_checkpoint_tree_sha256", ""))
    if actual != expected:
        raise RuntimeError("Reward v2.1 selected checkpoint hash mismatch")
    return selected, manifest


def v21_terminal_exploration_state(project_root: Path) -> dict[str, float]:
    path = Path(project_root) / "configs/grpo/reward_v2_1_answer_dominant_positive.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    epsilon, bias = exploration_schedule_values(
        raw["text_search_exploration"], schedule="full",
        update_id=511, total_updates=512,
    )
    return {"epsilon": float(epsilon), "logit_bias": float(bias)}


def load_stage2_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if value.get("schema_version") != SCHEMA:
        raise ValueError("Stage-2 config schema mismatch")
    strategy = str(value.get("strategy", ""))
    if strategy not in STRATEGIES:
        raise ValueError("Stage-2 strategy is invalid")
    mode = str(value.get("mode", ""))
    if mode != "short":
        raise ValueError("The public Stage-2 release supports only the successful short run")
    expected_scale = {
        "prompt_count": 256, "group_size": 4, "total_rollouts": 1024,
        "prompt_groups_per_update": 4, "trajectories_per_update": 16,
        "scheduled_updates": 64,
    }
    if dict(value.get("scale", {})) != expected_scale:
        raise ValueError("Stage-2 scale differs")
    expected_checkpoints = [16, 32, 48, 64]
    if list(value.get("checkpoint_updates", [])) != expected_checkpoints:
        raise ValueError("Stage-2 checkpoint schedule differs")
    initialization = dict(value.get("initialization", {}))
    if initialization != {
        "source": "reward_v21_selected_checkpoint",
        "stage1_manifest": "outputs/grpo_reward_v21_full/run_manifest.json",
        "load_optimizer_state": False,
        "load_scheduler_state": False,
    }:
        raise ValueError("Stage-2 initialization contract differs")
    optimizer = dict(value.get("optimizer", {}))
    if optimizer != {
        "stage1_learning_rate": V21_LR,
        "learning_rate_scale": 0.5,
        "actor_learning_rate": STAGE2_LR,
        "optimizer": "paged_adamw_8bit",
        "scheduler": "none_as_v21_fresh_state",
        "max_grad_norm": 1.0,
        "weight_decay": 0.0,
        "ppo_epochs": 1,
        "clip_ratio": 0.2,
    }:
        raise ValueError("Stage-2 optimizer contract differs")
    exploration = dict(value.get("exploration", {}))
    if exploration != {
        "inherit_v21_terminal_state": True,
        "enabled": False,
        "epsilon": 0.0,
        "logit_bias": 0.0,
        "restart_schedule": False,
    }:
        raise ValueError("Stage-2 exploration contract differs")
    if value.get("frozen_test_access_allowed") is not False:
        raise ValueError("Stage-2 Frozen Test must remain embargoed")
    return value


def rank_checkpoint_rows(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            float(row["overall_em_v1"]),
            float(row["overall_token_f1_v1"]),
            float(row["search_required_em_v1"]),
            float(row["search_required_token_f1_v1"]),
        ),
        reverse=True,
    )


def judge_short(
    step0: Mapping[str, Any], post_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not post_rows:
        raise ValueError("Stage-2 has no post-training checkpoint metrics")
    best = dict(rank_checkpoint_rows(post_rows)[0])
    safety = (
        float(best["protocol_valid_rate"]) >= 0.90
        and float(best["missing_answer_rate"]) <= 0.12
    )
    strong = (
        float(best["overall_em_v1"]) >= 0.385
        and float(best["overall_token_f1_v1"]) >= 0.427
        and float(best["search_required_em_v1"]) >= 0.470
        and safety
    )
    improvements = {
        key: float(best[key]) > float(step0[key])
        for key in (
            "overall_token_f1_v1", "search_required_em_v1",
            "search_required_token_f1_v1",
        )
    }
    passed = (
        float(best["overall_em_v1"]) >= float(step0["overall_em_v1"])
        and any(improvements.values()) and safety
    )
    all_below_fail = all(float(row["overall_em_v1"]) < 0.360 for row in post_rows)
    status = "STRONG_PASS" if strong else "PASS" if passed else "FAIL"
    return {
        "status": status,
        "strong_pass": strong,
        "pass": strong or passed,
        "fail_all_post_em_below_0_360": all_below_fail,
        "protocol_missing_safety_passed": safety,
        "improvements_vs_step0": improvements,
        "best_checkpoint": best,
    }


def early_stop_signal(
    step0: Mapping[str, Any], rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: int(row["step"]))
    bad = [
        (
            float(row["overall_em_v1"])
            <= float(step0["overall_em_v1"]) - 0.025
            and float(row["search_required_em_v1"])
            < float(step0["search_required_em_v1"])
        )
        for row in ordered
    ]
    recommend = len(bad) >= 2 and bad[-1] and bad[-2]
    return {
        "recommend_stop_training": recommend,
        "automatic_process_termination": False,
        "consecutive_bad_checkpoint_count": (
            2 if recommend else 1 if bad and bad[-1] else 0
        ),
    }

