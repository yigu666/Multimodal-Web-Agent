from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import torch

from multimodal_web_agent.agent.parser import parse_action
from multimodal_web_agent.agent.schema import ActionType
from multimodal_web_agent.training.grpo.schema import (
    PromptPoolItem,
    RolloutRecord,
    validate_rollout_group,
)
from multimodal_web_agent.training.grpo.server_runner import (
    GenerationSettings,
    TransformersTrajectoryRunner,
    _move_batch,
)

from .text_search_exploration import (
    EXPLORATION_METHOD,
    EXPLORATION_VERSION,
    PrefixTextSearchLogitsProcessor,
    behavior_log_probs_from_raw_logits,
    deterministic_boundary_enabled,
    deterministic_exploration_slot,
    exploration_schedule_values,
    restore_exact_generated_behavior_log_probs,
)
from .local_credit_assignment import _verified_character_token_span


def _decoded(tokenizer: Any, token_ids: Sequence[int]) -> str:
    try:
        return str(tokenizer.decode(
            list(token_ids),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ))
    except TypeError:
        return str(tokenizer.decode(
            list(token_ids), skip_special_tokens=True
        ))


def _generated_content_token_span(
    tokenizer: Any,
    generated_ids: Sequence[int],
    decoded_text: str,
) -> tuple[int, int]:
    """Locate exact generated content without decode/strip/re-encode.

    Generation may append EOS/chat-control IDs which disappear under
    ``skip_special_tokens=True``.  Only such zero-decoding edge IDs are
    removed.  Whitespace is deliberately preserved because stripping it can
    change BPE tokenization and invalidate the behavior-policy alignment.
    """
    ids = [int(value) for value in generated_ids]
    expected = str(decoded_text)
    if _decoded(tokenizer, ids) != expected:
        raise RuntimeError(
            "EXACT_TEXT_SEARCH_EXPLORATION_NOT_SUPPORTED: generated token "
            "decode differs from returned assistant text"
        )
    start, end = 0, len(ids)
    changed = True
    while changed and start < end:
        changed = False
        if _decoded(tokenizer, ids[start + 1:end]) == expected:
            start += 1
            changed = True
        if start < end and _decoded(tokenizer, ids[start:end - 1]) == expected:
            end -= 1
            changed = True
    if _decoded(tokenizer, ids[start:end]) != expected:
        raise RuntimeError(
            "EXACT_TEXT_SEARCH_EXPLORATION_NOT_SUPPORTED: cannot isolate "
            "generated assistant content tokens"
        )
    return start, end


class TextSearchExplorationTrajectoryRunner(TransformersTrajectoryRunner):
    """Training-only runner with auditable opening-tag prefix exploration."""

    def __init__(
        self, *args, exploration_config: Mapping[str, Any],
        global_seed: int, exploration_schedule: str = "smoke",
        total_updates: int = 8, **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.exploration_config = dict(exploration_config)
        self.global_seed = int(global_seed)
        self.exploration_schedule = str(exploration_schedule)
        self.total_updates = int(total_updates)
        exploration_schedule_values(
            self.exploration_config, schedule=self.exploration_schedule,
            update_id=0, total_updates=self.total_updates,
        )
        self._context: dict[str, Any] = {}
        self._turn_traces: list[dict[str, Any]] = []

    @property
    def tokenizer(self) -> Any:
        return getattr(self.processor, "tokenizer", self.processor)

    def _protocol_allows_text_search(
        self, messages: Sequence[Mapping[str, Any]]
    ) -> bool:
        tool_call_count = 0
        text_search_count = 0
        for message in messages:
            if str(message.get("role", "")) != "assistant":
                continue
            content = message.get("content", "")
            if not isinstance(content, str):
                continue
            parsed = parse_action(content)
            if not parsed.valid:
                continue
            if parsed.action_type in {ActionType.IMAGE_SEARCH, ActionType.TEXT_SEARCH}:
                tool_call_count += 1
            if parsed.action_type == ActionType.TEXT_SEARCH:
                text_search_count += 1
        return tool_call_count < self.max_turns - 1 and text_search_count == 0

    def _generate_turn(
        self,
        messages: Sequence[Mapping[str, Any]],
        image: Any,
        *,
        seed: int,
        settings: GenerationSettings,
    ) -> tuple[str, torch.Tensor]:
        selected = bool(self._context.get("exploration_selected", False))
        allowed = selected and self._protocol_allows_text_search(messages)
        if not allowed:
            return super()._generate_turn(
                messages, image, seed=seed, settings=settings
            )
        if not settings.do_sample:
            raise RuntimeError("EXACT_TEXT_SEARCH_EXPLORATION_NOT_SUPPORTED")
        turn = sum(
            str(message.get("role", "")) == "assistant"
            for message in messages
        )
        epsilon = float(self._context["exploration_epsilon"])
        logit_bias = float(self._context["exploration_logit_bias"])
        boundary_enabled = deterministic_boundary_enabled(
            epsilon=epsilon,
            seed=self.global_seed,
            prompt_group_id=str(self._context["prompt_group_id"]),
            update_id=int(self._context["update_id"]),
            trajectory_index=int(self._context["trajectory_index"]),
            turn=int(turn),
        )
        batch = _move_batch(
            self._encode(messages, image, add_generation_prompt=True),
            self.device,
        )
        prompt_length = int(batch["input_ids"].shape[-1])
        processor = PrefixTextSearchLogitsProcessor(
            tokenizer=self.tokenizer,
            prompt_length=prompt_length,
            logit_bias=logit_bias,
            enabled=boundary_enabled,
        )
        kwargs: dict[str, Any] = {
            "max_new_tokens": settings.max_new_tokens,
            "do_sample": True,
            "num_beams": settings.num_beams,
            "use_cache": True,
            "return_dict_in_generate": True,
            "output_scores": True,
            "temperature": settings.temperature,
            # Exploration rollouts use the complete modified softmax. This
            # makes μ_old exact and leaves unmodified query tokens at π_old.
            "top_p": 1.0,
            "top_k": 0,
            "logits_processor": [processor],
            "pad_token_id": getattr(self.tokenizer, "pad_token_id", None),
        }
        cuda_devices: list[int] = []
        if self.device.type == "cuda":
            cuda_devices = [
                self.device.index
                if self.device.index is not None
                else torch.cuda.current_device()
            ]
        with torch.random.fork_rng(devices=cuda_devices, enabled=True):
            torch.manual_seed(seed)
            if cuda_devices:
                torch.cuda.manual_seed_all(seed)
            output = self.model.generate(**batch, **kwargs)
        generated = output.sequences[:, prompt_length:].detach().cpu()[0]
        decoder = getattr(self.processor, "batch_decode", None)
        if decoder is None:
            decoder = self.tokenizer.batch_decode
        text = decoder(
            generated.unsqueeze(0), skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        content_start, content_end = _generated_content_token_span(
            self.tokenizer, generated.tolist(), str(text)
        )
        behavior_selected = []
        for step, scores in enumerate(output.scores):
            token = int(generated[step])
            behavior_selected.append(float(
                torch.log_softmax(scores[0].detach().float(), dim=-1)[token]
            ))
        decisions = processor.finalize(
            generated_ids=generated.tolist(), output_scores=output.scores,
            temperature=settings.temperature,
        )
        self._turn_traces.append({
            "turn": int(turn),
            "generated_ids": [int(value) for value in generated.tolist()],
            "generated_behavior_logprobs": behavior_selected,
            "raw": str(text),
            "content_generated_start": content_start,
            "content_generated_end": content_end,
            "epsilon": epsilon,
            "logit_bias": logit_bias,
            "temperature": float(settings.temperature),
            "boundary_enabled": boundary_enabled,
            "action_boundary_count": len(decisions),
            "decisions": decisions,
        })
        return str(text), generated

    def _raw_next_token_logits(self, record: RolloutRecord) -> torch.Tensor:
        input_ids = torch.as_tensor(
            record.full_input_ids, device=self.device
        ).unsqueeze(0)
        attention_mask = torch.as_tensor(
            record.attention_mask, device=self.device
        ).unsqueeze(0)
        kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "use_cache": False,
        }
        if record.pixel_values is not None:
            kwargs["pixel_values"] = torch.as_tensor(
                record.pixel_values, device=self.device
            )
        if record.image_grid_thw is not None:
            kwargs["image_grid_thw"] = torch.as_tensor(
                record.image_grid_thw, device=self.device
            )
        with torch.no_grad():
            return self.model(**kwargs).logits[0, :-1, :]

    def _attach_behavior_policy(self, record: RolloutRecord) -> None:
        base_existing = torch.as_tensor(record.old_log_probs).detach().cpu()
        if not self._turn_traces:
            record.base_policy_log_probs = base_existing.clone()
            record.behavior_policy_log_probs = base_existing.clone()
            return
        full_ids = [int(value) for value in torch.as_tensor(
            record.full_input_ids
        ).tolist()]
        bias_by_position: dict[int, tuple[int, float]] = {}
        exact_behavior_by_position: dict[int, float] = {}
        public_trace = []
        generation_alignment_errors = []
        for turn_trace in self._turn_traces:
            turn = int(turn_trace["turn"])
            if turn >= len(record.actions) or turn >= len(record.assistant_turn_spans):
                raise RuntimeError("exploration turn/action alignment failed")
            raw = str(record.actions[turn]["raw"])
            if raw != str(turn_trace["raw"]):
                raise RuntimeError(
                    "EXACT_TEXT_SEARCH_EXPLORATION_NOT_SUPPORTED: assistant "
                    "text changed between generation and rollout assembly"
                )
            generated_ids = list(turn_trace["generated_ids"])
            generated_offset = int(turn_trace["content_generated_start"])
            generated_end = int(turn_trace["content_generated_end"])
            raw_ids = generated_ids[generated_offset:generated_end]
            if _decoded(self.tokenizer, raw_ids) != raw:
                raise RuntimeError(
                    "EXACT_TEXT_SEARCH_EXPLORATION_NOT_SUPPORTED: saved "
                    "generated content does not decode to the action"
                )
            assistant_start, assistant_end = map(
                int, record.assistant_turn_spans[turn]
            )
            assistant_ids = full_ids[assistant_start:assistant_end]
            assistant_text = _decoded(self.tokenizer, assistant_ids)
            raw_offsets = [
                index
                for index in range(len(assistant_text) - len(raw) + 1)
                if assistant_text[index:index + len(raw)] == raw
            ]
            if len(raw_offsets) != 1:
                raise RuntimeError(
                    "EXACT_TEXT_SEARCH_EXPLORATION_NOT_SUPPORTED: canonical "
                    "assistant text/action alignment is not unique: "
                    f"matches={raw_offsets}"
                )
            raw_char_offset = raw_offsets[0]
            decision_positions: dict[int, int] = {}
            decision_char_spans: dict[int, tuple[int, int]] = {}
            for decision in turn_trace["decisions"]:
                generated_step = int(decision["generation_step"])
                content_step = generated_step - generated_offset
                if not 0 <= content_step < len(raw_ids):
                    raise RuntimeError(
                        "EXACT_TEXT_SEARCH_EXPLORATION_NOT_SUPPORTED: biased "
                        "decision token was not preserved in assistant text"
                    )
                before_text = _decoded(
                    self.tokenizer, raw_ids[:content_step]
                )
                through_text = _decoded(
                    self.tokenizer, raw_ids[:content_step + 1]
                )
                if not through_text.startswith(before_text):
                    raise RuntimeError(
                        "EXACT_TEXT_SEARCH_EXPLORATION_NOT_SUPPORTED: "
                        "generated token decode prefix is unstable"
                    )
                char_start = raw_char_offset + len(before_text)
                char_end = raw_char_offset + len(through_text)
                try:
                    canonical_start, canonical_end = (
                        _verified_character_token_span(
                            tokenizer=self.tokenizer,
                            token_ids=assistant_ids,
                            text=assistant_text,
                            char_start=char_start,
                            char_end=char_end,
                        )
                    )
                except ValueError as exc:
                    raise RuntimeError(
                        "EXACT_TEXT_SEARCH_EXPLORATION_NOT_SUPPORTED: "
                        "decision character/token alignment failed: "
                        f"{exc}"
                    ) from exc
                if canonical_end - canonical_start != 1:
                    raise RuntimeError(
                        "EXACT_TEXT_SEARCH_EXPLORATION_NOT_SUPPORTED: biased "
                        "generation token maps to multiple canonical tokens: "
                        f"generated_step={generated_step} "
                        f"canonical_width={canonical_end - canonical_start}"
                    )
                full_position = assistant_start + canonical_start
                response_position = full_position - 1
                decision_positions[generated_step] = full_position
                decision_char_spans[generated_step] = (
                    char_start - raw_char_offset,
                    char_end - raw_char_offset,
                )
                selected = int(full_ids[full_position])
                if bool(decision["exploration_applied"]):
                    if selected != int(decision["selected_token_id"]):
                        raise RuntimeError(
                            "EXACT_TEXT_SEARCH_EXPLORATION_NOT_SUPPORTED: "
                            "biased token changed during canonical rendering: "
                            f"generated={decision['selected_token_id']} "
                            f"canonical={selected} "
                            f"target={decision['target_token_id']}"
                        )
                    bias_by_position[response_position] = (
                        int(decision["target_token_id"]),
                        float(decision["logit_bias"]),
                    )
                    exact_behavior_by_position[response_position] = float(
                        decision["behavior_policy_logprob"]
                    )
                else:
                    observed = float(turn_trace[
                        "generated_behavior_logprobs"
                    ][generated_step])
                    generation_alignment_errors.append((
                        response_position,
                        abs(observed - float(base_existing[response_position])),
                    ))
            selected_action = str(record.actions[turn].get("action_type", ""))
            query = (
                str(record.actions[turn].get("content", ""))
                if selected_action == ActionType.TEXT_SEARCH.value else None
            )
            for decision in turn_trace["decisions"]:
                row = {
                    key: value for key, value in decision.items()
                    if key != "base_logits"
                }
                generated_step = int(row["generation_step"])
                char_start, char_end = decision_char_spans[generated_step]
                row.update({
                    "prompt_group_id": record.prompt_uid,
                    "trajectory_index": record.rollout_index,
                    "update_id": int(self._context["update_id"]),
                    "exploration_selected": True,
                    "action_boundary_index": decision_positions[generated_step],
                    "generated_text_character_span": [char_start, char_end],
                    "canonical_alignment_method": (
                        "decision_character_to_canonical_token"
                    ),
                    "protocol_state": "top_level_action",
                    "epsilon": float(turn_trace["epsilon"]),
                    "selected_action": selected_action,
                    "query": query,
                    "turn": turn,
                })
                row["base_text_search_probability"] = float(
                    row["base_target_probability"]
                )
                row["behavior_text_search_probability"] = float(
                    row["behavior_target_probability"]
                )
                row["logit_bias_or_epsilon"] = float(row["logit_bias"])
                public_trace.append(row)
        raw_logits = self._raw_next_token_logits(record)
        temperatures = {
            float(row["temperature"]) for row in self._turn_traces
        }
        if len(temperatures) != 1:
            raise RuntimeError(
                "EXACT_TEXT_SEARCH_EXPLORATION_NOT_SUPPORTED: generation "
                "temperature changed within one trajectory"
            )
        behavior_temperature = next(iter(temperatures))
        selected_ids = torch.as_tensor(
            record.full_input_ids[1:], device=raw_logits.device
        )
        base, behavior = behavior_log_probs_from_raw_logits(
            raw_logits=raw_logits,
            selected_ids=selected_ids,
            bias_by_position=bias_by_position,
            temperature=behavior_temperature,
        )
        behavior = restore_exact_generated_behavior_log_probs(
            behavior, exact_behavior_by_position
        )
        base_cpu = base.detach().cpu()
        behavior_cpu = behavior.detach().cpu()
        base_error = float((base_cpu - base_existing).abs().max())
        if base_error >= 1e-3:
            raise RuntimeError(
                "EXACT_TEXT_SEARCH_EXPLORATION_NOT_SUPPORTED: base replay "
                f"error={base_error}"
            )
        for row in public_trace:
            position = row.get("action_boundary_index")
            if position is None or not row.get("exploration_applied"):
                continue
            response_position = int(position) - 1
            row["recomputed_behavior_policy_logprob"] = float(
                behavior_cpu[response_position]
            )
            row["behavior_logprob_abs_error"] = abs(
                float(row["behavior_policy_logprob"])
                - float(behavior_cpu[response_position])
            )
        bias_positions = set(bias_by_position)
        nonbias_errors = [
            error for position, error in generation_alignment_errors
            if position not in bias_positions
        ]
        record.base_policy_log_probs = base_cpu
        record.behavior_policy_log_probs = behavior_cpu
        record.old_log_probs = behavior_cpu.clone()
        record.exploration_trace = public_trace
        record.exploration_metadata.update({
            "exploration_version": EXPLORATION_VERSION,
            "exploration_method": EXPLORATION_METHOD,
            "exploration_selected": True,
            "bias_token_count": len(bias_by_position),
            "behavior_logprob_error_max": max(
                [float(row.get("behavior_logprob_abs_error", 0.0)) for row in public_trace],
                default=0.0,
            ),
            "base_replay_error_max": base_error,
            "unmodified_token_generation_error_max": max(
                nonbias_errors, default=0.0
            ),
            "behavior_temperature": behavior_temperature,
        })

    def rollout(
        self, prompt: PromptPoolItem, *, run_seed: int, pool_pass: int,
        rollout_index: int, settings: GenerationSettings,
    ) -> RolloutRecord:
        self._turn_traces = []
        record = super().rollout(
            prompt, run_seed=run_seed, pool_pass=pool_pass,
            rollout_index=rollout_index, settings=settings,
        )
        record.exploration_metadata = {
            **self._context,
            "exploration_version": EXPLORATION_VERSION,
            "exploration_method": EXPLORATION_METHOD,
            "exploration_train_only": True,
        }
        if bool(self._context.get("exploration_selected", False)):
            self._attach_behavior_policy(record)
        else:
            base = torch.as_tensor(record.old_log_probs).detach().cpu()
            record.base_policy_log_probs = base.clone()
            record.behavior_policy_log_probs = base.clone()
        return record

    def replay_behavior_log_probs(self, record: RolloutRecord) -> torch.Tensor:
        raw_logits = self._raw_next_token_logits(record)
        bias_by_position = {
            int(row["action_boundary_index"]) - 1: (
                int(row["target_token_id"]), float(row["logit_bias"])
            )
            for row in record.exploration_trace
            if row.get("exploration_applied")
            and row.get("action_boundary_index") is not None
        }
        _, behavior = behavior_log_probs_from_raw_logits(
            raw_logits=raw_logits,
            selected_ids=torch.as_tensor(
                record.full_input_ids[1:], device=raw_logits.device
            ),
            bias_by_position=bias_by_position,
            temperature=float(
                record.exploration_metadata["behavior_temperature"]
            ),
        )
        exact_by_position = {
            int(row["action_boundary_index"]) - 1: float(
                row["behavior_policy_logprob"]
            )
            for row in record.exploration_trace
            if row.get("exploration_applied")
            and row.get("action_boundary_index") is not None
        }
        return restore_exact_generated_behavior_log_probs(
            behavior, exact_by_position
        )

    def rollout_group_with_exploration(
        self, prompt: PromptPoolItem, *, run_seed: int, pool_pass: int,
        group_size: int, settings: GenerationSettings,
        group_index: int, update_id: int,
    ) -> list[RolloutRecord]:
        epsilon, logit_bias = exploration_schedule_values(
            self.exploration_config,
            schedule=self.exploration_schedule,
            update_id=update_id,
            total_updates=self.total_updates,
        )
        exploration_enabled = (
            self.exploration_config.get("enabled_train_only") is True
        )
        slot = (
            deterministic_exploration_slot(
                group_size=group_size, seed=self.global_seed,
                prompt_group_id=prompt.prompt_uid, update_id=update_id,
            )
            if exploration_enabled else -1
        )
        records = []
        for index in range(group_size):
            self._context = {
                "prompt_group_id": prompt.prompt_uid,
                "group_index": int(group_index),
                "update_id": int(update_id),
                "trajectory_index": int(index),
                "exploration_slot": int(slot),
                "exploration_selected": bool(
                    exploration_enabled and index == slot
                ),
                "exploration_schedule": self.exploration_schedule,
                "exploration_epsilon": float(epsilon),
                "exploration_logit_bias": float(logit_bias),
            }
            records.append(self.rollout(
                prompt, run_seed=run_seed, pool_pass=pool_pass,
                rollout_index=index, settings=settings,
            ))
        validate_rollout_group(records, group_size)
        selected = sum(
            bool(row.exploration_metadata["exploration_selected"])
            for row in records
        )
        expected = 1 if exploration_enabled else 0
        if selected != expected:
            raise RuntimeError("exploration selection count differs from config")
        return records
