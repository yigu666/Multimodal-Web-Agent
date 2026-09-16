from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Sequence

from PIL import Image
from transformers import LogitsProcessor, LogitsProcessorList

from multimodal_web_agent.agent import ActionType, parse_action
from multimodal_web_agent.data.protocol_sft.schema import Message
from multimodal_web_agent.training.sft.generation import generate_for_example
from multimodal_web_agent.training.sft.renderer import ProtocolRenderer

from .answer_metrics import (
    maximum_alias_token_f1,
    normalized_exact_match,
)
from .environment import FrozenToolEnvironment
from .episode import EpisodeResult, EpisodeTurn
from .protocol import render_prompt
from .schema import UnifiedEvalExample


class ForbiddenActionOpeningLogitsProcessor(LogitsProcessor):
    """Block only the token that would complete a forbidden XML opening.

    This string-aware completion guard is needed because byte-level BPE may
    merge the opening ``<`` with its preceding character.  It does not ban any
    constituent token in ordinary reasoning text.
    """

    def __init__(self, tokenizer: Any, openings: Sequence[str]) -> None:
        self.tokenizer = tokenizer
        self.stems = tuple(opening[:-1] for opening in openings if opening.endswith(">"))
        vocabulary_size = len(tokenizer)
        self.completion_token_ids = tuple(
            token_id
            for token_id in range(vocabulary_size)
            if tokenizer.decode(
                [token_id],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            ).startswith(">")
        )
        if self.stems and not self.completion_token_ids:
            raise ValueError("tokenizer has no action-opening completion token")

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        for batch_index in range(int(input_ids.shape[0])):
            tail = self.tokenizer.decode(
                input_ids[batch_index, -24:].tolist(),
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            if any(tail.endswith(stem) for stem in self.stems):
                scores[batch_index, list(self.completion_token_ids)] = -float("inf")
        return scores


class UnifiedAgentRunner:
    def __init__(
        self,
        *,
        model: Any,
        processor: Any,
        environment: FrozenToolEnvironment,
        model_id: str,
        max_new_tokens_per_turn: int = 128,
        forbidden_action_openings: Sequence[str] = (),
    ):
        self.model = model
        self.processor = processor
        self.environment = environment
        self.model_id = model_id
        self.max_new_tokens_per_turn = max_new_tokens_per_turn
        self.renderer = ProtocolRenderer(processor)
        self.forbidden_action_openings = tuple(str(value) for value in forbidden_action_openings)
        tokenizer = getattr(processor, "tokenizer", processor)
        # Byte-level BPE can merge the opening ``<`` with an immediately
        # preceding space or ``>`` (for example ``</reason><search>``), so the
        # standalone encoding alone is not context-complete.  Ban the three
        # exact complete-opening variants that cover protocol output: at the
        # start/after a newline, after whitespace, and directly after a tag.
        forbidden_text_sequences = tuple(
            prefix + opening
            for opening in self.forbidden_action_openings
            for prefix in ("", " ", ">")
        )
        self.forbidden_action_token_ids = tuple(
            tuple(int(token) for token in tokenizer.encode(sequence, add_special_tokens=False))
            for sequence in forbidden_text_sequences
        )
        if any(not tokens for tokens in self.forbidden_action_token_ids):
            raise ValueError("forbidden action opening must tokenize to at least one token")
        self.forbidden_action_logits_processor = (
            ForbiddenActionOpeningLogitsProcessor(tokenizer, self.forbidden_action_openings)
            if self.forbidden_action_openings else None
        )

    def _token_count(self, text: str) -> int:
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        if hasattr(tokenizer, "encode"):
            return len(tokenizer.encode(text, add_special_tokens=False))
        return len(str(text).split())

    def run(self, example: UnifiedEvalExample, image: Any) -> EpisodeResult:
        example.validate()
        begin_episode = getattr(self.environment, "begin_episode", None)
        if callable(begin_episode):
            begin_episode(example, image)
        history: list[Message] = []
        turns = []
        final_answer = None
        tool_calls = image_calls = text_calls = 0
        protocol_valid = True
        within_budget = True
        tool_failure = False
        max_turn_exhausted = False
        initial_prompt_hash = ""
        started = time.perf_counter()
        self.model.eval()
        for turn_index in range(1, example.maximum_agent_turns + 1):
            state, prompt_hash = render_prompt(
                self.renderer, example, history
            )
            if turn_index == 1:
                initial_prompt_hash = prompt_hash
            generation_kwargs = {
                "do_sample": False,
                "num_beams": 1,
            }
            if self.forbidden_action_token_ids:
                # Transformers applies bad_words_ids as exact multi-token sequence
                # constraints. This masks only the complete action openings and
                # does not ban constituent tokens from ordinary reasoning text.
                generation_kwargs["bad_words_ids"] = [
                    list(tokens) for tokens in self.forbidden_action_token_ids
                ]
                generation_kwargs["logits_processor"] = LogitsProcessorList([
                    self.forbidden_action_logits_processor
                ])
            generated = generate_for_example(
                self.model,
                self.processor,
                self.renderer,
                state,
                image,
                max_new_tokens=self.max_new_tokens_per_turn,
                generation_kwargs=generation_kwargs,
            ).generated_text
            parsed = parse_action(generated)
            action = parsed.action_type.value if parsed.action_type else None
            executed = False
            parse_error = (
                parsed.error_code.value if parsed.error_code else None
            )
            if not parsed.valid:
                protocol_valid = False
                turns.append(EpisodeTurn(
                    turn_index, generated, False, action, False,
                    self._token_count(generated), prompt_hash, parse_error,
                ))
                break
            if parsed.action_type == ActionType.ANSWER:
                final_answer = parsed.content
                turns.append(EpisodeTurn(
                    turn_index, generated, True, action, False,
                    self._token_count(generated), prompt_hash,
                ))
                break
            if parsed.action_type not in {
                ActionType.IMAGE_SEARCH, ActionType.TEXT_SEARCH
            }:
                protocol_valid = False
                turns.append(EpisodeTurn(
                    turn_index, generated, False, action, False,
                    self._token_count(generated), prompt_hash,
                    "unknown_action",
                ))
                break
            if self.forbidden_action_openings:
                raise AssertionError(
                    "NO_SEARCH_ACTION_MASK_BYPASS: %r" % generated
                )
            next_image_calls = image_calls + int(
                parsed.action_type == ActionType.IMAGE_SEARCH
            )
            next_text_calls = text_calls + int(
                parsed.action_type == ActionType.TEXT_SEARCH
            )
            if (
                tool_calls + 1 > example.maximum_tool_calls
                or next_image_calls > example.maximum_image_search_calls
                or next_text_calls > example.maximum_text_search_calls
            ):
                within_budget = False
                turns.append(EpisodeTurn(
                    turn_index, generated, True, action, False,
                    self._token_count(generated), prompt_hash,
                    "tool_budget_exceeded",
                ))
                break
            try:
                if parsed.action_type == ActionType.IMAGE_SEARCH:
                    information = self.environment.image_search(
                        example.image_sha256
                    )
                    image_calls = next_image_calls
                else:
                    information = self.environment.text_search(
                        parsed.content or ""
                    )
                    text_calls = next_text_calls
            except (LookupError, ValueError) as exc:
                tool_failure = True
                error_code = getattr(exc, "code", "tool_execution_failure")
                turns.append(EpisodeTurn(
                    turn_index, generated, True, action, False,
                    self._token_count(generated), prompt_hash,
                    error_code,
                ))
                break
            tool_calls += 1
            executed = True
            turns.append(EpisodeTurn(
                turn_index, generated, True, action, executed,
                self._token_count(generated), prompt_hash,
            ))
            history.extend([
                Message("assistant", generated),
                Message("tool", information),
            ])
        else:
            max_turn_exhausted = True
            within_budget = False
        em = normalized_exact_match(final_answer, example.answer_aliases)
        f1 = maximum_alias_token_f1(final_answer, example.answer_aliases)
        success = bool(
            em == 1 and protocol_valid and within_budget and not tool_failure
        )
        return EpisodeResult(
            eval_id=example.eval_id,
            model_id=self.model_id,
            task_type=example.task_type,
            search_required=example.search_required,
            turns=tuple(turns),
            final_answer=final_answer,
            normalized_em=em,
            token_f1=f1,
            tool_call_count=tool_calls,
            image_search_call_count=image_calls,
            text_search_call_count=text_calls,
            agent_turn_count=len(turns),
            episode_protocol_valid=protocol_valid,
            within_budget=within_budget,
            agent_success_at_budget=success,
            tool_execution_failure=tool_failure,
            max_turn_exhausted=max_turn_exhausted,
            wall_clock_seconds=time.perf_counter() - started,
            initial_prompt_sha256=initial_prompt_hash,
        )

    def run_path(
        self,
        example: UnifiedEvalExample,
        dataset_root: Path,
    ) -> EpisodeResult:
        path = Path(dataset_root) / example.image_path
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as opened:
            image = opened.convert("RGB")
        return self.run(example, image)
