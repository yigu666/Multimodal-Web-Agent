from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass(frozen=True)
class AnswerWeights:
    em_v1_weight: float
    token_f1_v1_weight: float
    deterministic_equivalence_v2_weight: float


@dataclass(frozen=True)
class EvidenceWeights:
    evidence_use_weight: float
    missed_evidence_weight: float
    wrong_span_copy_weight: float
    wrong_span_prediction_support_threshold: float
    wrong_span_answer_score_threshold: float
    wrong_span_gold_support_threshold: float


@dataclass(frozen=True)
class TerminalWeights:
    invalid_protocol_reward: float
    missing_answer_reward: float
    min_reward: float
    max_reward: float


@dataclass(frozen=True)
class TextQueryWeights:
    actual_rank_weight: float
    improvement_weight: float
    improvement_min: float
    improvement_max: float
    utility_min: float
    utility_max: float
    use_corpus_coverage_mask: bool
    use_question_only_baseline: bool


@dataclass(frozen=True)
class TextQueryAdvantageWeights:
    improvement_weight: float
    actual_rank_weight: float
    min_advantage: float
    max_advantage: float


@dataclass(frozen=True)
class LocalCreditWeights:
    text_terminal_weight: float
    text_local_weight: float
    image_terminal_weight: float
    image_local_weight: float
    min_valid_actions: int
    text_local_enabled: bool = True
    text_local_positive_only: bool = False
    advantage_epsilon: float = 1e-6
    variance_epsilon: float = 1e-12


@dataclass(frozen=True)
class RewardV2Paths:
    prompt_pool: str
    image_search_cache: str
    coverage_cache: str
    coverage_manifest: str
    question_baseline_cache: str
    question_baseline_manifest: str
    environment_manifest: str
    sft_adapter: str


@dataclass(frozen=True)
class RewardV2Config:
    name: str
    version: str
    mode: str
    answer: AnswerWeights
    evidence: EvidenceWeights
    terminal: TerminalWeights
    text_query: TextQueryWeights
    text_query_advantage: TextQueryAdvantageWeights
    local_credit: LocalCreditWeights
    paths: RewardV2Paths
    text_search_exploration: Mapping[str, Any]
    grounding: Mapping[str, Any]
    negative_shaping: Mapping[str, Any]
    query_credit: Mapping[str, Any]
    group_rank: Mapping[str, Any]
    raw: Mapping[str, Any]


_FROZEN_VALUES = {
    "answer.em_v1_weight": 0.60,
    "answer.token_f1_v1_weight": 0.25,
    "answer.deterministic_equivalence_v2_weight": 0.15,
    "evidence.evidence_use_weight": 0.25,
    "evidence.missed_evidence_weight": 0.30,
    "evidence.wrong_span_copy_weight": 0.20,
    "evidence.wrong_span_prediction_support_threshold": 0.80,
    "evidence.wrong_span_answer_score_threshold": 0.15,
    "evidence.wrong_span_gold_support_threshold": 0.50,
    "terminal.invalid_protocol_reward": -1.00,
    "terminal.missing_answer_reward": -0.60,
    "terminal.min_reward": -1.00,
    "terminal.max_reward": 1.00,
    "text_query.actual_rank_weight": 0.70,
    "text_query.improvement_weight": 0.30,
    "text_query.improvement_min": -0.50,
    "text_query.improvement_max": 1.00,
    "text_query.utility_min": -0.15,
    "text_query.utility_max": 1.00,
    "text_query_advantage.improvement_weight": 0.70,
    "text_query_advantage.actual_rank_weight": 0.30,
    "text_query_advantage.min_advantage": -0.25,
    "text_query_advantage.max_advantage": 1.00,
    "local_credit.text_terminal_weight": 0.35,
    "local_credit.text_local_weight": 0.65,
    "local_credit.image_terminal_weight": 1.00,
    "local_credit.image_local_weight": 0.00,
    "local_credit.min_valid_actions": 2,
}


def _section(raw: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = raw.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"reward config section is missing: {name}")
    return value


def _validate_frozen_values(reward: Mapping[str, Any]) -> None:
    for dotted, expected in _FROZEN_VALUES.items():
        section, key = dotted.split(".", 1)
        actual = _section(reward, section).get(key)
        if isinstance(expected, int):
            valid = int(actual) == expected
        else:
            valid = abs(float(actual) - expected) <= 1e-12
        if not valid:
            raise ValueError(
                f"Reward v2 frozen value differs: {dotted}="
                f"{actual!r}, expected={expected!r}"
            )
    text_query = _section(reward, "text_query")
    text_query_advantage = _section(reward, "text_query_advantage")
    if text_query.get("use_corpus_coverage_mask") is not True:
        raise ValueError("Reward v2 requires the corpus coverage mask")
    if text_query.get("use_question_only_baseline") is not True:
        raise ValueError("Reward v2 requires the question-only baseline")


def load_reward_v2_config(path: str | Path) -> RewardV2Config:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    reward = _section(raw, "reward")
    mode = str(reward.get("mode", "hierarchical_grounded_search_v2"))
    allowed = {
        "hierarchical_grounded_search_v2": "v2",
        "answer_dominant_positive": "v2.1",
    }
    if mode not in allowed or str(reward.get("version")) != allowed[mode]:
        raise ValueError("unexpected configured Reward mode/version")
    if mode == "hierarchical_grounded_search_v2":
        if reward.get("name") != "hierarchical_grounded_search_v2":
            raise ValueError("unexpected Reward v2 name")
        _validate_frozen_values(reward)
    answer = _section(reward, "answer")
    evidence = _section(reward, "evidence")
    terminal = _section(reward, "terminal")
    text_query = _section(reward, "text_query")
    text_query_advantage = _section(reward, "text_query_advantage")
    local = _section(reward, "local_credit")
    paths = _section(raw, "paths")
    exploration = _section(raw, "text_search_exploration")
    stage2 = dict(raw.get("stage2", {}))
    stage2_variant = str(stage2.get("variant", ""))
    if stage2_variant and stage2_variant != "S2_A":
        raise ValueError("unknown Stage-2 Reward variant")
    exploration_enabled = exploration.get("enabled_train_only") is True
    if not exploration_enabled and not stage2_variant:
        raise ValueError("Text Search exploration must be training-only")
    expected_trajectories = 1 if exploration_enabled else 0
    if int(exploration.get("trajectories_per_group", -1)) != expected_trajectories:
        raise ValueError("Text Search exploration trajectory count is inconsistent")
    for key in (
        "use_gold_conditioning", "use_task_type_conditioning",
        "use_search_required_conditioning", "use_coverage_conditioning",
    ):
        if exploration.get(key) is not False:
            raise ValueError(f"Text Search exploration forbids {key}")
    if exploration.get("use_source_dataset_conditioning", False) is not False:
        raise ValueError(
            "Text Search exploration forbids use_source_dataset_conditioning"
        )
    config = RewardV2Config(
        name=str(reward["name"]),
        version=str(reward["version"]),
        mode=mode,
        answer=AnswerWeights(**{k: float(answer[k]) for k in AnswerWeights.__annotations__}),
        evidence=EvidenceWeights(**{k: float(evidence[k]) for k in EvidenceWeights.__annotations__}),
        terminal=TerminalWeights(**{k: float(terminal[k]) for k in TerminalWeights.__annotations__}),
        text_query=TextQueryWeights(
            **{
                key: bool(text_query[key])
                if key.startswith("use_")
                else float(text_query[key])
                for key in TextQueryWeights.__annotations__
            }
        ),
        text_query_advantage=TextQueryAdvantageWeights(
            **{
                key: float(text_query_advantage[key])
                for key in TextQueryAdvantageWeights.__annotations__
            }
        ),
        local_credit=LocalCreditWeights(
            text_terminal_weight=float(local["text_terminal_weight"]),
            text_local_weight=float(local["text_local_weight"]),
            image_terminal_weight=float(local["image_terminal_weight"]),
            image_local_weight=float(local["image_local_weight"]),
            min_valid_actions=int(local["min_valid_actions"]),
            text_local_enabled=bool(local.get("text_local_enabled", True)),
            text_local_positive_only=bool(
                local.get("text_local_positive_only", False)
            ),
            advantage_epsilon=float(local.get("advantage_epsilon", 1e-6)),
            variance_epsilon=float(local.get("variance_epsilon", 1e-12)),
        ),
        paths=RewardV2Paths(**{k: str(paths[k]) for k in RewardV2Paths.__annotations__}),
        text_search_exploration=exploration,
        grounding=dict(reward.get("grounding", {})),
        negative_shaping=dict(reward.get("negative_shaping", {})),
        query_credit=dict(reward.get("query_credit", {})),
        group_rank=dict(reward.get("group_rank", {})),
        raw=raw,
    )
    if abs(sum((
        config.answer.em_v1_weight, config.answer.token_f1_v1_weight,
        config.answer.deterministic_equivalence_v2_weight,
    )) - 1.0) > 1e-12:
        raise ValueError("answer weights must sum to one")
    if mode == "answer_dominant_positive":
        expected_answer = (0.70, 0.30, 0.0)
        actual_answer = (
            config.answer.em_v1_weight,
            config.answer.token_f1_v1_weight,
            config.answer.deterministic_equivalence_v2_weight,
        )
        if any(abs(actual - expected) > 1e-12 for actual, expected in zip(
            actual_answer, expected_answer
        )):
            raise ValueError("configured Reward mode has incorrect answer weights")
        expected = (0.80, 0.20, True, True, 0.15)
        actual = (
            config.local_credit.text_terminal_weight,
            config.local_credit.text_local_weight,
            config.local_credit.text_local_enabled,
            config.local_credit.text_local_positive_only,
            float(config.grounding.get("weight", 0.0)),
        )
        if any(
            (a != b if isinstance(b, bool) else abs(float(a) - b) > 1e-12)
            for a, b in zip(actual, expected)
        ):
            raise ValueError("configured Reward mode has incorrect core weights")
        if float(config.negative_shaping.get("missed_evidence_weight", 0.0)) != 0.0:
            raise ValueError("new Reward modes forbid missed-evidence shaping")
        if float(config.negative_shaping.get("wrong_span_copy_weight", 0.0)) != 0.0:
            raise ValueError("new Reward modes forbid wrong-span shaping")
        if bool(config.grounding.get("enabled")) is not True:
            raise ValueError("configured Reward mode has incorrect grounding switch")
        if abs(float(config.grounding.get("answer_em_weight", 0.0)) - 0.70) > 1e-12:
            raise ValueError("grounding answer EM weight differs")
        if abs(float(config.grounding.get("answer_token_f1_weight", 0.0)) - 0.30) > 1e-12:
            raise ValueError("grounding answer F1 weight differs")
        if stage2_variant:
            if mode != "answer_dominant_positive":
                raise ValueError("Stage-2 must reuse Reward v2.1 terminal mode")
            expected_mix = (0.80, 0.20)
            actual_mix = (
                config.local_credit.text_terminal_weight,
                config.local_credit.text_local_weight,
            )
            if actual_mix != expected_mix:
                raise ValueError("Stage-2 Text Search credit mix differs")
            if exploration_enabled:
                raise ValueError("Stage-2 must inherit zero terminal exploration")
            if exploration.get("inherited_from") != "reward_v21_terminal_state":
                raise ValueError("Stage-2 exploration provenance differs")
            if float(exploration.get("epsilon", -1.0)) != 0.0:
                raise ValueError("Stage-2 exploration epsilon must be zero")
            if float(exploration.get("logit_bias", -1.0)) != 0.0:
                raise ValueError("Stage-2 exploration logit bias must be zero")
            forbidden = {
                "success_gated_query_credit": False,
                "group_rank_reward": False,
                "success_filtered_group_updates": False,
                "negative_evidence_shaping": False,
            }
            for key, expected_value in forbidden.items():
                if stage2.get(key) is not expected_value:
                    raise ValueError(f"Stage-2 forbidden mechanism differs: {key}")
    return config
