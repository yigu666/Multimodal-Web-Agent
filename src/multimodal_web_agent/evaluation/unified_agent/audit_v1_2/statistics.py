from __future__ import annotations

import math
import random
from statistics import mean
from typing import Mapping, Sequence


BOOTSTRAP_SEED = 20260803
BOOTSTRAP_SAMPLES = 10_000
CONFIDENCE_LEVEL = 0.95


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("percentile requires values")
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def paired_bootstrap(
    left: Sequence[float],
    right: Sequence[float],
    *,
    seed: int = BOOTSTRAP_SEED,
    samples: int = BOOTSTRAP_SAMPLES,
    confidence_level: float = CONFIDENCE_LEVEL,
) -> dict:
    if len(left) != len(right) or not left:
        raise ValueError("paired bootstrap requires equal non-empty inputs")
    differences = [float(r) - float(l) for l, r in zip(left, right)]
    generator = random.Random(seed)
    n = len(differences)
    estimates = [
        sum(differences[generator.randrange(n)] for _ in range(n)) / n
        for _ in range(samples)
    ]
    alpha = (1.0 - confidence_level) / 2.0
    lower = _percentile(estimates, alpha)
    upper = _percentile(estimates, 1.0 - alpha)
    negative = sum(value <= 0.0 for value in estimates) / samples
    positive = sum(value >= 0.0 for value in estimates) / samples
    return {
        "paired_count": n,
        "left_mean": mean(float(value) for value in left),
        "right_mean": mean(float(value) for value in right),
        "difference_right_minus_left": mean(differences),
        "confidence_level": confidence_level,
        "confidence_interval": [lower, upper],
        "bootstrap_samples": samples,
        "seed": seed,
        "two_sided_bootstrap_p_value": min(1.0, 2.0 * min(negative, positive)),
        "statistically_significant": lower > 0.0 or upper < 0.0,
    }


def mcnemar_exact(
    left_correct: Sequence[bool],
    right_correct: Sequence[bool],
) -> dict:
    if len(left_correct) != len(right_correct) or not left_correct:
        raise ValueError("McNemar test requires equal non-empty inputs")
    left_only = sum(l and not r for l, r in zip(left_correct, right_correct))
    right_only = sum(not l and r for l, r in zip(left_correct, right_correct))
    discordant = left_only + right_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(
            math.comb(discordant, index)
            for index in range(0, min(left_only, right_only) + 1)
        ) / (2 ** discordant)
        p_value = min(1.0, 2.0 * tail)
    return {
        "paired_count": len(left_correct),
        "sft_correct_grpo_wrong": left_only,
        "sft_wrong_grpo_correct": right_only,
        "discordant_pair_count": discordant,
        "exact_two_sided_p_value": p_value,
        "statistically_significant": p_value < 0.05,
    }


def paired_statistical_audit(
    sft: Sequence[Mapping],
    grpo: Sequence[Mapping],
) -> dict:
    if [row["episode_id"] for row in sft] != [
        row["episode_id"] for row in grpo
    ]:
        raise ValueError("statistical audit inputs are not paired")
    groups = {"overall": list(range(len(sft)))}
    for task_type in sorted({str(row["task_type"]) for row in sft}):
        groups[task_type] = [
            index for index, row in enumerate(sft)
            if row["task_type"] == task_type
        ]
    result = {}
    for name, indices in groups.items():
        def values(rows: Sequence[Mapping], key: str) -> list[float]:
            return [float(rows[index][key]) for index in indices]

        seed = BOOTSTRAP_SEED
        result[name] = {
            "episode_count": len(indices),
            "em_v1": paired_bootstrap(
                values(sft, "em_v1"), values(grpo, "em_v1"), seed=seed
            ),
            "token_f1_v1": paired_bootstrap(
                values(sft, "token_f1_v1"),
                values(grpo, "token_f1_v1"),
                seed=seed,
            ),
            "em_v2_strict": paired_bootstrap(
                values(sft, "em_v2_strict"),
                values(grpo, "em_v2_strict"),
                seed=seed,
            ),
            "tool_calls": paired_bootstrap(
                values(sft, "successful_search_count"),
                values(grpo, "successful_search_count"),
                seed=seed,
            ),
            "mcnemar_v1": mcnemar_exact(
                [bool(sft[index]["em_v1"]) for index in indices],
                [bool(grpo[index]["em_v1"]) for index in indices],
            ),
            "mcnemar_v2": mcnemar_exact(
                [bool(sft[index]["em_v2_strict"]) for index in indices],
                [bool(grpo[index]["em_v2_strict"]) for index in indices],
            ),
        }
    return {
        "schema_version": "unified-agent-eval-v1-2-statistical-tests-v1",
        "seed": BOOTSTRAP_SEED,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "confidence_level": CONFIDENCE_LEVEL,
        "comparison": "GRPO minus SFT",
        "groups": result,
    }
