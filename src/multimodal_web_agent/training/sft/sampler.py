from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Sequence

from multimodal_web_agent.data.protocol_sft.schema import (
    StateActionExample,
    TransitionType,
)


TRANSITIONS = tuple(item.value for item in TransitionType)
FORMAT_TRANSITIONS = (
    "initial_to_direct_answer",
    "initial_to_image_search",
    "image_information_to_answer",
    "image_information_to_text_search",
    "text_information_to_answer",
)
EXPECTED_TRAIN_TRANSITION_COUNTS: Mapping[str, int] = {
    "initial_to_direct_answer": 192,
    "initial_to_image_search": 246,
    "image_information_to_answer": 234,
    "initial_to_text_search": 52,
    "image_information_to_text_search": 12,
    "text_information_to_answer": 64,
}
EXPECTED_V0_4_TRAIN_TRANSITION_COUNTS: Mapping[str, int] = {
    "initial_to_direct_answer": 248,
    "initial_to_image_search": 240,
    "image_information_to_answer": 232,
    "initial_to_text_search": 32,
    "image_information_to_text_search": 8,
    "text_information_to_answer": 40,
}


@dataclass(frozen=True)
class TransitionSamplingConfig:
    epoch_size: int
    balanced_fraction: float
    seed: int


@dataclass(frozen=True)
class InitialRouterSamplingConfig:
    epoch_size: int
    transition_quotas: Mapping[str, int]
    seed: int


@dataclass(frozen=True)
class SampledEpoch:
    epoch_index: int
    indices: tuple[int, ...]
    audit: Mapping[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "epoch_index": self.epoch_index,
            "indices": list(self.indices),
            "audit": dict(self.audit),
        }


class DeterministicMixedTransitionSampler:
    def __init__(
        self,
        examples: Sequence[StateActionExample],
        config: TransitionSamplingConfig,
        *,
        expected_counts: Mapping[str, int] | None = None,
    ):
        self.examples = list(examples)
        self.config = config
        self.groups: Dict[str, list[int]] = defaultdict(list)
        for index, example in enumerate(self.examples):
            self.groups[example.transition].append(index)
        actual = {name: len(self.groups[name]) for name in TRANSITIONS}
        if set(actual) != set(TRANSITIONS) or any(value <= 0 for value in actual.values()):
            raise ValueError("all six Protocol-SFT transitions must be present")
        if expected_counts is not None and actual != dict(expected_counts):
            raise ValueError(
                "Train transition counts differ from the frozen contract: %s"
                % actual
            )
        if config.epoch_size <= 0:
            raise ValueError("epoch_size must be positive")
        if not 0.0 <= config.balanced_fraction <= 1.0:
            raise ValueError("balanced_fraction must be between zero and one")
        self.balanced_count = round(
            config.epoch_size * config.balanced_fraction
        )
        self.natural_count = config.epoch_size - self.balanced_count
        if self.balanced_count % len(TRANSITIONS):
            raise ValueError("balanced sample count must be divisible by six")
        if self.natural_count > len(self.examples):
            raise ValueError("natural sample count exceeds the Train split")

    def sample_epoch(self, epoch_index: int) -> SampledEpoch:
        if epoch_index < 0:
            raise ValueError("epoch_index cannot be negative")
        rng = random.Random(self.config.seed + epoch_index)
        natural = rng.sample(range(len(self.examples)), self.natural_count)
        per_transition = self.balanced_count // len(TRANSITIONS)
        balanced: list[int] = []
        balanced_counts: Dict[str, int] = {}
        for transition in TRANSITIONS:
            values = self.groups[transition]
            selected = [rng.choice(values) for _ in range(per_transition)]
            balanced.extend(selected)
            balanced_counts[transition] = len(selected)
        indices = natural + balanced
        rng.shuffle(indices)
        if len(indices) != self.config.epoch_size:
            raise AssertionError("sampled epoch has the wrong size")
        if any(index < 0 or index >= len(self.examples) for index in indices):
            raise AssertionError("sampled index is outside Train")
        repeats = Counter(indices)
        sampled_counts = Counter(
            self.examples[index].transition for index in indices
        )
        repeat_by_transition = {
            transition: {
                self.examples[index].example_id: repeats.get(index, 0)
                for index in self.groups[transition]
            }
            for transition in TRANSITIONS
        }
        audit: Dict[str, Any] = {
            "config": asdict(self.config),
            "epoch_index": epoch_index,
            "sampled_transition_counts": {
                name: sampled_counts[name] for name in TRANSITIONS
            },
            "balanced_transition_counts": balanced_counts,
            "natural_sample_count": len(natural),
            "natural_unique_count": len(set(natural)),
            "unique_sample_count": len(repeats),
            "duplicate_sample_count": len(indices) - len(repeats),
            "max_repeat_count": max(repeats.values(), default=0),
            "repeat_distribution_by_transition": repeat_by_transition,
            "rare_transition_repeat_distribution": repeat_by_transition[
                "image_information_to_text_search"
            ],
        }
        return SampledEpoch(
            epoch_index=epoch_index,
            indices=tuple(indices),
            audit=audit,
        )


class DeterministicInitialRouterSampler:
    def __init__(
        self,
        examples: Sequence[StateActionExample],
        config: InitialRouterSamplingConfig,
        *,
        expected_counts: Mapping[str, int] | None = None,
    ):
        self.examples = list(examples)
        self.config = config
        self.groups: Dict[str, list[int]] = defaultdict(list)
        for index, example in enumerate(self.examples):
            self.groups[example.transition].append(index)
        actual = {name: len(self.groups[name]) for name in TRANSITIONS}
        if expected_counts is not None and actual != dict(expected_counts):
            raise ValueError(
                "Train transition counts differ from the frozen contract: %s"
                % actual
            )
        quotas = {
            str(key): int(value)
            for key, value in config.transition_quotas.items()
        }
        if set(quotas) != set(TRANSITIONS):
            raise ValueError(
                "Router-focused quotas must cover exactly six transitions"
            )
        if any(value <= 0 for value in quotas.values()):
            raise ValueError("Router-focused quotas must be positive")
        if sum(quotas.values()) != config.epoch_size:
            raise ValueError("Router-focused quotas do not sum to epoch_size")
        if any(not self.groups[name] for name in TRANSITIONS):
            raise ValueError("all six Train transitions must be present")
        self.quotas = quotas

    def sample_epoch(self, epoch_index: int) -> SampledEpoch:
        if epoch_index < 0:
            raise ValueError("epoch_index cannot be negative")
        rng = random.Random(self.config.seed + epoch_index)
        indices = []
        for transition in TRANSITIONS:
            indices.extend(
                rng.choice(self.groups[transition])
                for _ in range(self.quotas[transition])
            )
        rng.shuffle(indices)
        if len(indices) != self.config.epoch_size:
            raise AssertionError("Router-focused epoch has the wrong size")
        if any(index < 0 or index >= len(self.examples) for index in indices):
            raise AssertionError("Router-focused index is outside Train")
        sampled_counts = Counter(
            self.examples[index].transition for index in indices
        )
        if any(
            sampled_counts[name] != self.quotas[name]
            for name in TRANSITIONS
        ):
            raise AssertionError("Router-focused sampled quotas differ")
        repeats = Counter(indices)
        repeat_by_transition = {
            transition: {
                self.examples[index].example_id: repeats.get(index, 0)
                for index in self.groups[transition]
            }
            for transition in TRANSITIONS
        }
        audit: Dict[str, Any] = {
            "config": {
                "epoch_size": self.config.epoch_size,
                "transition_quotas": dict(self.quotas),
                "seed": self.config.seed,
            },
            "epoch_index": epoch_index,
            "sampled_transition_counts": {
                name: sampled_counts[name] for name in TRANSITIONS
            },
            "unique_sample_count": len(repeats),
            "duplicate_sample_count": len(indices) - len(repeats),
            "max_repeat_count": max(repeats.values(), default=0),
            "repeat_distribution_by_transition": repeat_by_transition,
            "initial_text_repeat_distribution": repeat_by_transition[
                "initial_to_text_search"
            ],
        }
        return SampledEpoch(
            epoch_index=epoch_index,
            indices=tuple(indices),
            audit=audit,
        )


class DeterministicFormatExposureSampler:
    """Balance syntax exposure over the five transitions that actually exist.

    Sampling is with replacement by design.  Its output is an optimization
    schedule, never an estimate of the real policy distribution.
    """

    def __init__(
        self,
        examples: Sequence[StateActionExample],
        config: InitialRouterSamplingConfig,
    ):
        self.examples = list(examples)
        self.config = config
        self.groups: Dict[str, list[int]] = defaultdict(list)
        for index, example in enumerate(self.examples):
            self.groups[example.transition].append(index)
        self.quotas = {
            str(key): int(value)
            for key, value in config.transition_quotas.items()
        }
        if set(self.quotas) != set(FORMAT_TRANSITIONS):
            raise ValueError(
                "Format exposure quotas must cover exactly five transitions"
            )
        if any(value <= 0 for value in self.quotas.values()):
            raise ValueError("Format exposure quotas must be positive")
        if sum(self.quotas.values()) != config.epoch_size:
            raise ValueError("Format exposure quotas do not sum to epoch_size")
        absent = [name for name in FORMAT_TRANSITIONS if not self.groups[name]]
        if absent:
            raise ValueError("Format Train transitions are absent: %s" % absent)
        if self.groups.get("initial_to_text_search"):
            raise ValueError("Format view must not contain initial_to_text_search")

    def sample_epoch(self, epoch_index: int) -> SampledEpoch:
        if epoch_index < 0:
            raise ValueError("epoch_index cannot be negative")
        rng = random.Random(self.config.seed + epoch_index)
        indices: list[int] = []
        for transition in FORMAT_TRANSITIONS:
            indices.extend(
                rng.choice(self.groups[transition])
                for _ in range(self.quotas[transition])
            )
        rng.shuffle(indices)
        sampled_counts = Counter(
            self.examples[index].transition for index in indices
        )
        if {
            name: sampled_counts[name] for name in FORMAT_TRANSITIONS
        } != self.quotas:
            raise AssertionError("Format exposure sampled quotas differ")
        repeats = Counter(indices)
        return SampledEpoch(
            epoch_index=epoch_index,
            indices=tuple(indices),
            audit={
                "config": {
                    "epoch_size": self.config.epoch_size,
                    "transition_quotas": dict(self.quotas),
                    "seed": self.config.seed,
                },
                "epoch_index": epoch_index,
                "sampler_purpose": "format_exposure_balance",
                "sampler_is_policy_distribution": False,
                "sampled_transition_counts": {
                    name: sampled_counts[name] for name in FORMAT_TRANSITIONS
                },
                "unique_sample_count": len(repeats),
                "duplicate_sample_count": len(indices) - len(repeats),
                "max_repeat_count": max(repeats.values(), default=0),
            },
        )
