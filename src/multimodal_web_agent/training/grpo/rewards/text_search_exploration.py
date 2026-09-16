from __future__ import annotations

import hashlib
import math
from typing import Any, Mapping, Sequence

import torch
from transformers.generation.logits_process import LogitsProcessor


EXPLORATION_VERSION = "training-only-prefix-text-search-v2"
EXPLORATION_METHOD = "prefix_logit_bias"
TEXT_SEARCH_OPENING_TAG = "<text_search>"


def action_epsilon_mixture_logprob(
    *, base_probability: float, exploration_probability: float,
    epsilon: float,
) -> float:
    """Auditable reference formula; the current free-generation runner does not use it."""
    mixture = (
        (1.0 - float(epsilon)) * float(base_probability)
        + float(epsilon) * float(exploration_probability)
    )
    if not 0.0 < mixture <= 1.0:
        raise ValueError("action epsilon mixture probability must be in (0, 1]")
    return math.log(mixture)


def _uniform_hash(*parts: object) -> float:
    payload = "\0".join(str(part) for part in parts).encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return value / float(1 << 64)


def deterministic_exploration_slot(
    *, group_size: int, seed: int, prompt_group_id: str, update_id: int
) -> int:
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    return int(
        _uniform_hash("slot", seed, prompt_group_id, update_id) * group_size
    ) % group_size


def deterministic_boundary_enabled(
    *, epsilon: float, seed: int, prompt_group_id: str,
    update_id: int, trajectory_index: int, turn: int,
) -> bool:
    if not 0.0 <= float(epsilon) <= 1.0:
        raise ValueError("epsilon must be in [0, 1]")
    return _uniform_hash(
        "boundary", seed, prompt_group_id, update_id, trajectory_index, turn
    ) < float(epsilon)


def exploration_schedule_values(
    config: Mapping[str, Any], *, schedule: str,
    update_id: int, total_updates: int,
) -> tuple[float, float]:
    """Resolve the frozen train-only exploration schedule for one update."""
    if config.get("enabled_train_only") is not True:
        return 0.0, 0.0
    if schedule == "smoke":
        values = dict(config.get("smoke", {}))
        if values.get("mode") != "fixed":
            raise ValueError("Smoke exploration schedule must be fixed")
        return float(values["epsilon"]), float(values["logit_bias"])
    if schedule != "full":
        raise ValueError(f"unknown exploration schedule: {schedule}")
    values = dict(config.get("full", {}))
    if values.get("mode") != "linear_decay":
        raise ValueError("Full exploration schedule must use linear decay")
    if total_updates <= 1 or not 0 <= int(update_id) < int(total_updates):
        raise ValueError("Full exploration update is out of range")
    start = float(values["decay_start_fraction"])
    end = float(values["decay_end_fraction"])
    if not 0.0 <= start < end <= 1.0:
        raise ValueError("invalid Full exploration decay interval")
    fraction = int(update_id) / float(int(total_updates) - 1)
    if fraction <= start:
        progress = 0.0
    elif fraction >= end:
        progress = 1.0
    else:
        progress = (fraction - start) / (end - start)
    epsilon = (
        float(values["initial_epsilon"]) * (1.0 - progress)
        + float(values["final_epsilon"]) * progress
    )
    bias = (
        float(values["initial_logit_bias"]) * (1.0 - progress)
        + float(values["final_logit_bias"]) * progress
    )
    if not 0.0 <= epsilon <= 1.0 or bias < 0.0:
        raise ValueError("Full exploration schedule emitted invalid values")
    return epsilon, bias


def _encode(tokenizer: Any, text: str) -> list[int]:
    try:
        values = tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        values = tokenizer.encode(text)
    return [int(value) for value in values]


def _decode(tokenizer: Any, token_ids: Sequence[int]) -> str:
    try:
        return str(tokenizer.decode(
            list(token_ids), skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ))
    except TypeError:
        return str(tokenizer.decode(list(token_ids), skip_special_tokens=True))


class PrefixTextSearchLogitsProcessor(LogitsProcessor):
    """Bias only the opening-tag prefix after a complete top-level reason."""

    def __init__(
        self, *, tokenizer: Any, prompt_length: int,
        logit_bias: float, enabled: bool,
    ):
        self.tokenizer = tokenizer
        self.prompt_length = int(prompt_length)
        self.logit_bias = float(logit_bias)
        self.enabled = bool(enabled)
        self.target_ids = _encode(tokenizer, TEXT_SEARCH_OPENING_TAG)
        if not self.target_ids:
            raise RuntimeError("Tokenizer produced no Text Search opening-tag tokens")
        self.target_prefixes = [
            _decode(tokenizer, self.target_ids[:index]).strip()
            for index in range(1, len(self.target_ids) + 1)
        ]
        if self.target_prefixes[-1] != TEXT_SEARCH_OPENING_TAG:
            raise RuntimeError(
                "EXACT_TEXT_SEARCH_EXPLORATION_NOT_SUPPORTED: opening tag "
                "does not round-trip through the active tokenizer"
            )
        self.decisions: list[dict[str, Any]] = []
        self._closed = False

    def _progress(self, generated: Sequence[int]) -> int | None:
        decoded = _decode(self.tokenizer, generated)
        reason_end = decoded.rfind("</reason>")
        if reason_end < 0:
            return None
        tail = decoded[reason_end + len("</reason>"):].strip()
        if not tail:
            return 0
        for index, prefix in enumerate(self.target_prefixes, start=1):
            if tail == prefix:
                return index
        return None

    def _finalize_previous_selection(self, generated: Sequence[int]) -> None:
        if not self.decisions:
            return
        previous = self.decisions[-1]
        if "selected_token_id" in previous:
            return
        step = int(previous["generation_step"])
        if step >= len(generated):
            return
        selected = int(generated[step])
        previous["selected_token_id"] = selected
        previous["selected_target_token"] = bool(
            selected == int(previous["target_token_id"])
        )
        selected_text = _decode(self.tokenizer, [selected])
        previous["selected_whitespace_token"] = not selected_text.strip()
        if (
            not previous["selected_target_token"]
            and not previous["selected_whitespace_token"]
        ):
            self._closed = True

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        generated = [
            int(value) for value in input_ids[0, self.prompt_length:].tolist()
        ]
        self._finalize_previous_selection(generated)
        if self._closed:
            return scores
        progress = self._progress(generated)
        if progress is None or progress >= len(self.target_ids):
            return scores
        target = int(self.target_ids[progress])
        row = {
            "generation_step": len(generated),
            "target_prefix_index": progress,
            "target_token_id": target,
            "exploration_applied": self.enabled,
            "exploration_method": EXPLORATION_METHOD,
            "logit_bias": self.logit_bias if self.enabled else 0.0,
        }
        if self.enabled:
            row["base_logits"] = scores[0].detach().float().cpu()
            scores = scores.clone()
            scores[:, target] += self.logit_bias
        else:
            self._closed = True
        self.decisions.append(row)
        return scores

    def finalize(
        self, *, generated_ids: Sequence[int], output_scores: Sequence[torch.Tensor],
        temperature: float,
    ) -> list[dict[str, Any]]:
        generated = [int(value) for value in generated_ids]
        self._finalize_previous_selection(generated)
        result = []
        for raw in self.decisions:
            row = dict(raw)
            base_logits = row.pop("base_logits", None)
            step = int(row["generation_step"])
            if step >= len(generated) or step >= len(output_scores):
                raise RuntimeError(
                    "EXACT_TEXT_SEARCH_EXPLORATION_NOT_SUPPORTED: generation "
                    "score/token length mismatch"
                )
            selected = int(generated[step])
            behavior_scores = output_scores[step][0].detach().float().cpu()
            behavior_log_probs = torch.log_softmax(behavior_scores, dim=-1)
            row["selected_token_id"] = selected
            row["selected_target_token"] = selected == int(row["target_token_id"])
            row["behavior_policy_logprob"] = float(behavior_log_probs[selected])
            row["behavior_target_probability"] = float(
                behavior_log_probs[int(row["target_token_id"])].exp()
            )
            if base_logits is None:
                base_scores = behavior_scores
            else:
                base_scores = base_logits / float(temperature)
            base_log_probs = torch.log_softmax(base_scores, dim=-1)
            row["base_policy_logprob"] = float(base_log_probs[selected])
            row["base_target_probability"] = float(
                base_log_probs[int(row["target_token_id"])].exp()
            )
            result.append(row)
        return result


def behavior_log_probs_from_raw_logits(
    *, raw_logits: torch.Tensor, selected_ids: torch.Tensor,
    bias_by_position: Mapping[int, tuple[int, float]], temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return base π_old and modified μ_old selected-token log-probabilities."""
    if raw_logits.ndim != 2 or selected_ids.ndim != 1:
        raise ValueError("raw logits and selected IDs must be rank 2 and rank 1")
    if raw_logits.shape[0] != selected_ids.shape[0]:
        raise ValueError("raw logits and selected IDs differ in sequence length")
    scaled = raw_logits / float(temperature)
    base = torch.log_softmax(scaled, dim=-1).gather(
        -1, selected_ids.unsqueeze(-1)
    ).squeeze(-1)
    behavior = base.clone()
    for position, (target_token_id, bias) in bias_by_position.items():
        position = int(position)
        modified = raw_logits[position].clone()
        modified[int(target_token_id)] += float(bias)
        selected = int(selected_ids[position])
        behavior[position] = torch.log_softmax(
            modified / float(temperature), dim=-1
        )[selected]
    if not torch.isfinite(base).all() or not torch.isfinite(behavior).all():
        raise RuntimeError("non-finite exploration behavior log-probability")
    return base, behavior


def restore_exact_generated_behavior_log_probs(
    behavior_log_probs: torch.Tensor,
    exact_by_position: Mapping[int, float],
) -> torch.Tensor:
    """Restore exact generation-time μ_old values at modified positions.

    A cached autoregressive NF4 forward and a later full-sequence forward can
    differ slightly.  The distribution returned by ``generate`` is the policy
    that actually sampled the token, so its selected-token log-probability is
    authoritative for the PPO denominator at every biased position.
    """
    if behavior_log_probs.ndim != 1:
        raise ValueError("behavior log-probabilities must be rank 1")
    restored = behavior_log_probs.clone()
    for raw_position, raw_value in exact_by_position.items():
        position = int(raw_position)
        value = float(raw_value)
        if not 0 <= position < restored.numel():
            raise ValueError("exact behavior log-probability position is out of range")
        if not math.isfinite(value):
            raise ValueError("exact behavior log-probability must be finite")
        restored[position] = value
    if not torch.isfinite(restored).all():
        raise RuntimeError("non-finite restored behavior log-probability")
    return restored


def exploration_config_audit(config: Mapping[str, Any]) -> dict[str, Any]:
    forbidden = {
        key: bool(config.get(key))
        for key in (
            "use_gold_conditioning", "use_task_type_conditioning",
            "use_search_required_conditioning", "use_coverage_conditioning",
            "use_source_dataset_conditioning",
        )
    }
    return {
        "exploration_version": EXPLORATION_VERSION,
        "exploration_method": EXPLORATION_METHOD,
        "exploration_train_only": bool(config.get("enabled_train_only")),
        "trajectories_per_group": int(config.get("trajectories_per_group", 0)),
        "forbidden_conditioning": forbidden,
        "gold_or_route_conditioning_used": any(forbidden.values()),
        "all_values_finite": all(
            math.isfinite(float(value))
            for section in ("smoke", "full")
            for key, value in dict(config.get(section, {})).items()
            if key not in {"mode"}
        ),
    }
