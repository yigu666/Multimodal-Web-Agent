"""Dependency-light entry point for the frozen-protocol GRPO package."""

__all__ = [
    "MMSearchLikeRewardV0",
    "compute_group_advantages",
    "compute_policy_loss",
    "score_reward_v0",
    "trajectory_balanced_policy_loss",
]


def __getattr__(name):
    if name == "compute_group_advantages":
        from .advantages import compute_group_advantages

        return compute_group_advantages
    if name in {"compute_policy_loss", "trajectory_balanced_policy_loss"}:
        from .policy_loss import (
            compute_policy_loss,
            trajectory_balanced_policy_loss,
        )

        return {
            "compute_policy_loss": compute_policy_loss,
            "trajectory_balanced_policy_loss": trajectory_balanced_policy_loss,
        }[name]
    if name in {"MMSearchLikeRewardV0", "score_reward_v0"}:
        from .reward_v0_mmsearch_like import (
            MMSearchLikeRewardV0,
            score_reward_v0,
        )

        return {
            "MMSearchLikeRewardV0": MMSearchLikeRewardV0,
            "score_reward_v0": score_reward_v0,
        }[name]
    raise AttributeError(name)
