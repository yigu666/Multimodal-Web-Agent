from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import gc
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import statistics
import time
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

import torch
import yaml

from multimodal_web_agent.agent.parser import parse_action
from multimodal_web_agent.agent.schema import ActionType
from multimodal_web_agent.data.protocol_sft.schema import Message
from multimodal_web_agent.data.protocol_sft.templates import SYSTEM_PROMPT
from multimodal_web_agent.training.sft.dataset import ImageStore
from multimodal_web_agent.training.sft.renderer import ProtocolRenderer

from .advantages import compute_group_advantages, group_advantage_statistics
from .formal_contract import (
    ALLOWED_REWARD_VALUES,
    BEHAVIOR_CONTRACT_FIELDS,
    FORMAL_ENVIRONMENT,
    RANK1_DIAGNOSTIC_FIELDS,
    SMOKE_CONTRACT_V2_SCHEMA,
    behavior_diagnostics,
    formal_chain_manifest_fields,
    validate_smoke_engineering_contract,
)
from .input_identity import assert_input_identity
from .logprob_alignment import (
    assert_logprob_alignment,
    selected_token_log_probs,
)
from .optimizer import build_paged_adamw_8bit
from .policy_loss import policy_loss_metrics, trajectory_balanced_policy_loss
from .policy_mask import build_grpo_masks
from .prompt_pool import POOL_SCHEMA
from .reward_v0_mmsearch_like import REWARD_NAME, score_reward_v0
from .rollout_engine import CachedSandbox, derive_generation_seed
from .schema import PromptPoolItem, RolloutRecord, validate_rollout_group
from .smoke_selection import (
    select_prompts as _select_prompts,
    select_smoke_prompts as _select_smoke_prompts,
)


TERMINAL_REASONS = {
    "answer",
    "protocol_error",
    "max_turns",
    "max_new_tokens",
    "cache_miss_then_answer",
    "environment_error",
}
ANSWER_PAYLOAD_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(path: Path) -> str:
    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    for file in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(file.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with file.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def sha256_code_tree(path: Path, suffixes: tuple[str, ...]) -> str:
    root = Path(path)
    digest = hashlib.sha256()
    files = sorted(
        item
        for item in root.rglob("*")
        if item.is_file() and item.suffix in suffixes
    )
    for file in files:
        digest.update(file.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(file.read_bytes())
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
            )


def _model_device(model: Any) -> torch.device:
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    return torch.device("cuda")


def _move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _record_field(record: RolloutRecord, name: str) -> Any:
    return getattr(record, name)


def _public_rollout(record: RolloutRecord) -> dict[str, Any]:
    tensor_keys = (
        "full_input_ids",
        "attention_mask",
        "position_ids",
        "response_ids",
        "policy_action_mask",
        "information_mask",
        "old_log_probs",
        "base_policy_log_probs",
        "behavior_policy_log_probs",
        "pixel_values",
        "image_grid_thw",
    )
    value = {
        key: getattr(record, key)
        for key in record.__dataclass_fields__
        if key not in tensor_keys
    }
    for key in tensor_keys:
        raw = getattr(record, key)
        if raw is None:
            value[key] = None
        elif torch.is_tensor(raw):
            value[key] = {
                "shape": list(raw.shape),
                "dtype": str(raw.dtype),
                "stored_in_tensor_record": True,
            }
    return value


class PromptImageStore:
    def __init__(self, source_parquet: Path):
        self.store = ImageStore(source_parquet)

    def load(self, prompt: PromptPoolItem) -> Any:
        proxy = SimpleNamespace(
            image_refs=[prompt.image_ref],
            source={
                "source_row_index": prompt.metadata.get(
                    "row_index", prompt.image_ref.get("row_index")
                )
            },
            source_data_id=prompt.source_data_id,
            example_id=prompt.prompt_uid,
        )
        return self.store.load(proxy)


@dataclass(frozen=True)
class GenerationSettings:
    do_sample: bool
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 0
    max_new_tokens: int = 96
    num_beams: int = 1


class TransformersTrajectoryRunner:
    """Sequential Qwen-VL rollout that persists exact update tensors."""

    def __init__(
        self,
        *,
        model: Any,
        processor: Any,
        environment: CachedSandbox,
        image_store: PromptImageStore,
        max_seq_len: int = 1536,
        max_turns: int = 3,
        base_model_hash: str = "",
        adapter_hash: str = "",
    ):
        self.model = model
        self.processor = processor
        self.renderer = ProtocolRenderer(processor)
        self.environment = environment
        self.image_store = image_store
        self.max_seq_len = max_seq_len
        self.max_turns = max_turns
        self.device = _model_device(model)
        self.base_model_hash = base_model_hash
        self.adapter_hash = adapter_hash
        self.processor_hash = hashlib.sha256(
            json.dumps(
                {
                    "processor": type(processor).__name__,
                    "image_processor": str(
                        getattr(processor, "image_processor", "")
                    ),
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        self.chat_template_hash = self.renderer.chat_template_sha256
        self._visual_cache: dict[str, tuple[torch.Tensor | None, torch.Tensor | None]] = {}

    def release_visual(self, prompt_uid: str) -> None:
        self._visual_cache.pop(prompt_uid, None)

    def _messages(
        self,
        prompt: PromptPoolItem,
        history: Sequence[tuple[str, str]],
    ) -> list[dict[str, Any]]:
        messages = [
            self.renderer._message(Message(role="system", content=SYSTEM_PROMPT)),
            self.renderer._message(
                Message(
                    role="user",
                    content=f"<image>\nQuestion: {prompt.question}",
                )
            ),
        ]
        for role, content in history:
            messages.append(
                self.renderer._message(Message(role=role, content=content))
            )
        return messages

    def _render(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        add_generation_prompt: bool,
    ) -> str:
        return str(
            self.processor.apply_chat_template(
                list(messages),
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        )

    def _encode(
        self,
        messages: Sequence[Mapping[str, Any]],
        image: Any,
        *,
        add_generation_prompt: bool,
    ) -> dict[str, Any]:
        text = self._render(
            messages, add_generation_prompt=add_generation_prompt
        )
        batch = self.processor(
            text=[text],
            images=[image],
            padding=False,
            truncation=False,
            return_tensors="pt",
        )
        if int(batch["input_ids"].shape[-1]) > self.max_seq_len:
            raise RuntimeError(
                f"trajectory exceeds max_seq_len={self.max_seq_len}"
            )
        return dict(batch)

    def _generate_turn(
        self,
        messages: Sequence[Mapping[str, Any]],
        image: Any,
        *,
        seed: int,
        settings: GenerationSettings,
    ) -> tuple[str, torch.Tensor]:
        batch = _move_batch(
            self._encode(messages, image, add_generation_prompt=True),
            self.device,
        )
        kwargs: dict[str, Any] = {
            "max_new_tokens": settings.max_new_tokens,
            "do_sample": settings.do_sample,
            "num_beams": settings.num_beams,
            "use_cache": True,
            "return_dict_in_generate": True,
            "pad_token_id": getattr(
                getattr(self.processor, "tokenizer", self.processor),
                "pad_token_id",
                None,
            ),
        }
        if settings.do_sample:
            kwargs.update(
                {
                    "temperature": settings.temperature,
                    "top_p": settings.top_p,
                    "top_k": settings.top_k,
                }
            )
        cuda_devices: list[int] = []
        if self.device.type == "cuda":
            cuda_devices = [
                self.device.index
                if self.device.index is not None
                else torch.cuda.current_device()
            ]
        # Transformers 4.49 validates and rejects ``generator`` as an unused
        # model kwarg for Qwen2.5-VL through PEFT.  Forking RNG state gives the
        # same deterministic per-rollout schedule without leaking RNG changes
        # into other rollouts.
        with torch.random.fork_rng(devices=cuda_devices, enabled=True):
            torch.manual_seed(seed)
            if cuda_devices:
                torch.cuda.manual_seed_all(seed)
            output = self.model.generate(**batch, **kwargs)
        prompt_length = int(batch["input_ids"].shape[-1])
        generated_ids = output.sequences[:, prompt_length:].detach().cpu()
        decoder = getattr(self.processor, "batch_decode", None)
        if decoder is None:
            decoder = self.processor.tokenizer.batch_decode
        text = decoder(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return str(text).strip(), generated_ids[0]

    def _token_length(
        self,
        messages: Sequence[Mapping[str, Any]],
        image: Any,
        *,
        add_generation_prompt: bool,
    ) -> tuple[int, torch.Tensor]:
        batch = self._encode(
            messages, image, add_generation_prompt=add_generation_prompt
        )
        return int(batch["input_ids"].shape[-1]), batch["input_ids"][0]

    def _spans_and_final_batch(
        self,
        messages: Sequence[Mapping[str, Any]],
        image: Any,
    ) -> tuple[dict[str, Any], list[tuple[int, int]], list[tuple[int, int]]]:
        final_batch = self._encode(
            messages, image, add_generation_prompt=False
        )
        final_ids = final_batch["input_ids"][0]
        assistant_spans: list[tuple[int, int]] = []
        information_spans: list[tuple[int, int]] = []
        for index, message in enumerate(messages):
            role = str(message.get("role", ""))
            content = message.get("content", "")
            is_information = (
                isinstance(content, str)
                and content.lstrip().startswith("<information>")
            )
            if role != "assistant" and not is_information:
                continue
            before_messages = messages[:index]
            before, before_ids = self._token_length(
                before_messages,
                image,
                add_generation_prompt=(role == "assistant"),
            )
            after, after_ids = self._token_length(
                messages[: index + 1],
                image,
                add_generation_prompt=False,
            )
            if not torch.equal(final_ids[:before], before_ids):
                raise RuntimeError("chat-template prefix mismatch before span")
            if not torch.equal(final_ids[:after], after_ids):
                raise RuntimeError("chat-template prefix mismatch after span")
            span = (before, after)
            if role == "assistant":
                assistant_spans.append(span)
            else:
                information_spans.append(span)
        return final_batch, assistant_spans, information_spans

    def _forward_log_probs(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor | None,
        pixel_values: torch.Tensor | None,
        image_grid_thw: torch.Tensor | None,
        grad: bool,
    ) -> torch.Tensor:
        kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "use_cache": False,
        }
        if position_ids is not None:
            kwargs["position_ids"] = position_ids
        if pixel_values is not None:
            kwargs["pixel_values"] = pixel_values
        if image_grid_thw is not None:
            kwargs["image_grid_thw"] = image_grid_thw
        context = torch.enable_grad() if grad else torch.no_grad()
        with context:
            output = self.model(**kwargs)
            return selected_token_log_probs(
                output.logits[:, :-1, :],
                input_ids[:, 1:],
                temperature=0.7,
            )

    def replay_log_probs(
        self, record: RolloutRecord, *, grad: bool
    ) -> torch.Tensor:
        kwargs = {
            "input_ids": torch.as_tensor(
                record.full_input_ids, device=self.device
            ).unsqueeze(0)
            if torch.as_tensor(record.full_input_ids).ndim == 1
            else torch.as_tensor(record.full_input_ids, device=self.device),
            "attention_mask": torch.as_tensor(
                record.attention_mask, device=self.device
            ).unsqueeze(0)
            if torch.as_tensor(record.attention_mask).ndim == 1
            else torch.as_tensor(record.attention_mask, device=self.device),
            "position_ids": (
                torch.as_tensor(record.position_ids, device=self.device)
                if record.position_ids is not None
                else None
            ),
            "pixel_values": (
                torch.as_tensor(record.pixel_values, device=self.device)
                if record.pixel_values is not None
                else None
            ),
            "image_grid_thw": (
                torch.as_tensor(record.image_grid_thw, device=self.device)
                if record.image_grid_thw is not None
                else None
            ),
            "grad": grad,
        }
        return self._forward_log_probs(**kwargs)[0]

    def rollout(
        self,
        prompt: PromptPoolItem,
        *,
        run_seed: int,
        pool_pass: int,
        rollout_index: int,
        settings: GenerationSettings,
    ) -> RolloutRecord:
        seed = derive_generation_seed(
            run_seed, prompt.prompt_uid, pool_pass, rollout_index
        )
        generation_hash = hashlib.sha256(
            json.dumps(asdict(settings), sort_keys=True).encode("utf-8")
        ).hexdigest()
        record = RolloutRecord(
            prompt_uid=prompt.prompt_uid,
            rollout_uid=f"{prompt.prompt_uid}:pass{pool_pass}:r{rollout_index}",
            data_id=prompt.data_id,
            rollout_index=rollout_index,
            generation_seed=seed,
            image_sha256=prompt.image_sha256,
            category=prompt.category,
            base_model_hash=self.base_model_hash,
            adapter_hash=self.adapter_hash,
            processor_hash=self.processor_hash,
            chat_template_hash=self.chat_template_hash,
            generation_config_hash=generation_hash,
        )
        image = self.image_store.load(prompt)
        history: list[tuple[str, str]] = []
        cache_miss = False
        seen_queries: set[str] = set()
        generated_ids_by_turn: list[torch.Tensor] = []
        for turn in range(self.max_turns):
            messages = self._messages(prompt, history)
            try:
                raw, generated_ids = self._generate_turn(
                    messages,
                    image,
                    seed=seed + turn,
                    settings=settings,
                )
            except RuntimeError as exc:
                record.protocol_error = f"generation_error:{type(exc).__name__}"
                record.terminal_reason = "environment_error"
                break
            generated_ids_by_turn.append(generated_ids)
            parsed = parse_action(raw)
            record.actions.append(
                {
                    "turn": turn,
                    "raw": raw,
                    "valid": parsed.valid,
                    "action_type": (
                        parsed.action_type.value if parsed.action_type else ""
                    ),
                    "content": parsed.content,
                    "protocol_error": (
                        parsed.error_code.value if parsed.error_code else None
                    ),
                }
            )
            history.append(("assistant", raw))
            if not parsed.valid:
                answer_match = ANSWER_PAYLOAD_RE.search(raw)
                if answer_match is not None:
                    record.answer_text = answer_match.group(1).strip()
                record.protocol_error = (
                    parsed.error_code.value
                    if parsed.error_code
                    else "invalid_protocol"
                )
                record.terminal_reason = (
                    "max_new_tokens"
                    if int(generated_ids.numel()) >= settings.max_new_tokens
                    else "protocol_error"
                )
                break
            if parsed.action_type == ActionType.ANSWER:
                record.answer_text = parsed.content or ""
                record.protocol_valid = True
                record.terminal_reason = (
                    "cache_miss_then_answer" if cache_miss else "answer"
                )
                break
            record.search_count += 1
            if parsed.action_type == ActionType.IMAGE_SEARCH:
                record.used_image_search = True
                result = self.environment.image_search(prompt.image_cache_key)
            else:
                record.used_text_search = True
                query = (parsed.content or "").strip()
                normalized_query = query.casefold()
                if normalized_query in seen_queries:
                    record.duplicate_query = True
                seen_queries.add(normalized_query)
                result = self.environment.text_search(
                    query, exclude_source_data_id=prompt.source_data_id
                )
            cache_miss = cache_miss or result.cache_miss
            record.tool_results.append(
                {
                    "turn": turn,
                    "text": result.text,
                    "cache_miss": result.cache_miss,
                    "provenance": result.provenance or {},
                }
            )
            history.append(("tool", result.text))
        else:
            record.terminal_reason = "max_turns"
        if not record.terminal_reason:
            record.terminal_reason = "max_turns"
        if record.terminal_reason not in TERMINAL_REASONS:
            raise AssertionError("unsupported terminal reason")

        final_messages = self._messages(prompt, history)
        final_batch, assistant_spans, information_spans = (
            self._spans_and_final_batch(final_messages, image)
        )
        full_ids = final_batch["input_ids"][0].detach().cpu()
        attention = final_batch.get(
            "attention_mask", torch.ones_like(final_batch["input_ids"])
        )[0].detach().cpu()
        policy_mask, information_mask = build_grpo_masks(
            int(full_ids.shape[0]),
            assistant_spans,
            information_spans,
            attention,
        )
        if int(policy_mask.sum()) <= 0:
            raise RuntimeError("rollout has no policy action tokens")
        if int((policy_mask * information_mask).sum()) != 0:
            raise RuntimeError("information tokens leaked into policy mask")
        pixel_values = final_batch.get("pixel_values")
        image_grid_thw = final_batch.get("image_grid_thw")
        pixel_cpu = (
            pixel_values.detach().cpu() if pixel_values is not None else None
        )
        grid_cpu = (
            image_grid_thw.detach().cpu()
            if image_grid_thw is not None
            else None
        )
        cached_visual = self._visual_cache.get(prompt.prompt_uid)
        if cached_visual is None:
            self._visual_cache[prompt.prompt_uid] = (pixel_cpu, grid_cpu)
        else:
            cached_pixel, cached_grid = cached_visual
            if pixel_cpu is not None and not torch.equal(pixel_cpu, cached_pixel):
                raise RuntimeError("visual tensor mismatch within prompt group")
            if grid_cpu is not None and not torch.equal(grid_cpu, cached_grid):
                raise RuntimeError("image_grid_thw mismatch within prompt group")
            pixel_cpu, grid_cpu = cached_visual

        input_gpu = full_ids.unsqueeze(0).to(self.device)
        attention_gpu = attention.unsqueeze(0).to(self.device)
        old = self._forward_log_probs(
            input_ids=input_gpu,
            attention_mask=attention_gpu,
            position_ids=None,
            pixel_values=(
                pixel_cpu.to(self.device) if pixel_cpu is not None else None
            ),
            image_grid_thw=(
                grid_cpu.to(self.device) if grid_cpu is not None else None
            ),
            grad=False,
        )[0].detach().cpu()
        record.full_input_ids = full_ids
        record.attention_mask = attention
        record.position_ids = None
        record.response_ids = full_ids[1:].clone()
        record.policy_action_mask = policy_mask[1:].detach().cpu()
        record.information_mask = information_mask[1:].detach().cpu()
        record.old_log_probs = old
        record.pixel_values = pixel_cpu
        record.image_grid_thw = grid_cpu
        record.assistant_turn_spans = assistant_spans
        record.information_spans = information_spans
        record.generated_token_ids_by_turn = [
            value.detach().cpu() for value in generated_ids_by_turn
        ]
        reward = score_reward_v0(record, prompt.to_dict())
        record.answer_em = reward.answer_em
        record.answer_f1 = reward.answer_f1
        record.reward_components = reward.to_dict()
        record.reward_total = reward.reward_total
        return record

    def rollout_group(
        self,
        prompt: PromptPoolItem,
        *,
        run_seed: int,
        pool_pass: int,
        group_size: int,
        settings: GenerationSettings,
    ) -> list[RolloutRecord]:
        records = [
            self.rollout(
                prompt,
                run_seed=run_seed,
                pool_pass=pool_pass,
                rollout_index=index,
                settings=settings,
            )
            for index in range(group_size)
        ]
        validate_rollout_group(records, group_size)
        return records


def _reward_summary(
    records: Sequence[RolloutRecord],
    group_statistics: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rewards = [record.reward_total for record in records]
    groups: dict[str, list[RolloutRecord]] = defaultdict(list)
    for record in records:
        groups[record.prompt_uid].append(record)
    n = len(records) or 1
    group_count = len(groups) or 1
    first_actions = Counter(
        record.actions[0]["action_type"] if record.actions else "none"
        for record in records
    )
    zero_variance = sum(
        bool(row.get("zero_variance_group")) for row in group_statistics
    )
    all_wrong = sum(
        all(record.answer_em == 0 for record in group)
        for group in groups.values()
    )
    all_correct = sum(
        all(record.answer_em == 1 for record in group)
        for group in groups.values()
    )
    search_free = [
        record for record in records if record.category == "search_free"
    ]
    search_required = [
        record for record in records if record.category == "search_required"
    ]
    def mean(values: Sequence[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    image_to_text = sum(
        any(
            left.get("action_type") == "image_search"
            and right.get("action_type") == "text_search"
            for left, right in zip(record.actions, record.actions[1:])
        )
        for record in records
    )
    parsed_turns = [
        action for record in records for action in record.actions
    ]
    return {
        "reward_mean": statistics.mean(rewards) if rewards else 0.0,
        "reward_std": (
            statistics.pstdev(rewards) if len(rewards) > 1 else 0.0
        ),
        "reward_min": min(rewards) if rewards else 0.0,
        "reward_max": max(rewards) if rewards else 0.0,
        "reward_distribution": dict(
            sorted(Counter(f"{value:.2f}" for value in rewards).items())
        ),
        "zero_variance_group_count": zero_variance,
        "zero_variance_group_ratio": zero_variance / group_count,
        "all_wrong_group_ratio": all_wrong / group_count,
        "all_correct_group_ratio": all_correct / group_count,
        "protocol_valid_rate": sum(record.protocol_valid for record in records)
        / n,
        "malformed_rate": sum(not record.protocol_valid for record in records)
        / n,
        "finish_rate": sum(
            record.terminal_reason in {"answer", "cache_miss_then_answer"}
            for record in records
        )
        / n,
        "answer_em": sum(record.answer_em for record in records) / n,
        "answer_f1": sum(record.answer_f1 for record in records) / n,
        "overall_em": sum(record.answer_em for record in records) / n,
        "overall_f1": sum(record.answer_f1 for record in records) / n,
        "search_free_em": mean(
            [record.answer_em for record in search_free]
        ),
        "search_free_f1": mean(
            [record.answer_f1 for record in search_free]
        ),
        "search_required_em": mean(
            [record.answer_em for record in search_required]
        ),
        "search_required_f1": mean(
            [record.answer_f1 for record in search_required]
        ),
        "search_free_search_rate": (
            sum(record.search_count > 0 for record in search_free)
            / len(search_free)
            if search_free
            else 0.0
        ),
        "search_required_search_rate": (
            sum(record.search_count > 0 for record in search_required)
            / len(search_required)
            if search_required
            else 0.0
        ),
        "search_required_search_recall": (
            sum(record.search_count > 0 for record in search_required)
            / len(search_required)
            if search_required
            else 0.0
        ),
        "search_free_direct_answer_rate": (
            sum(record.search_count == 0 for record in search_free)
            / len(search_free)
            if search_free
            else 0.0
        ),
        "unnecessary_search_rate": sum(
            record.category == "search_free" and record.search_count > 0
            for record in records
        )
        / n,
        "search_required_no_search_rate": sum(
            record.category == "search_required" and record.search_count == 0
            for record in records
        )
        / n,
        "first_action_distribution": dict(sorted(first_actions.items())),
        "image_search_rate": sum(record.used_image_search for record in records)
        / n,
        "text_search_rate": sum(record.used_text_search for record in records)
        / n,
        "image_to_text_rate": image_to_text / n,
        "search_call_ratio": sum(record.search_count for record in records)
        / max(sum(len(record.actions) for record in records), 1),
        "answer_only_rate": sum(record.search_count == 0 for record in records)
        / n,
        "average_search_calls": sum(record.search_count for record in records)
        / n,
        "average_turns": sum(len(record.actions) for record in records) / n,
        "duplicate_query_rate": sum(record.duplicate_query for record in records)
        / n,
        "forged_information_count": sum(
            record.protocol_error == "forged_information"
            for record in records
        ),
        "exactly_one_action_rate": (
            sum(bool(action.get("valid")) for action in parsed_turns)
            / len(parsed_turns)
            if parsed_turns
            else 0.0
        ),
        "nonempty_query_rate": (
            sum(
                bool(str(action.get("content", "")).strip())
                for action in parsed_turns
                if action.get("action_type") == "text_search"
            )
            / max(
                sum(
                    action.get("action_type") == "text_search"
                    for action in parsed_turns
                ),
                1,
            )
        ),
        "nonempty_answer_rate": (
            sum(bool(record.answer_text.strip()) for record in records)
            / n
        ),
        "max_turn_exhaustion_rate": sum(
            record.terminal_reason == "max_turns" for record in records
        )
        / n,
        "cache_miss_rate": sum(
            any(bool(result.get("cache_miss")) for result in record.tool_results)
            for record in records
        )
        / n,
    }


def _compute_group_values(
    records: Sequence[RolloutRecord],
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    rewards = torch.stack(
        [torch.tensor(record.reward_total, dtype=torch.float32) for record in records]
    )
    group_ids = [record.prompt_uid for record in records]
    advantages = compute_group_advantages(rewards, group_ids)
    return advantages, group_advantage_statistics(
        rewards, group_ids, advantages
    )


def _set_rollout_mode(model: Any) -> None:
    model.eval()
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    if hasattr(model, "config"):
        model.config.use_cache = True


def _set_update_mode(model: Any) -> None:
    if hasattr(model, "config"):
        model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            model.gradient_checkpointing_enable()
    model.train()


def _identity_mapping(record: RolloutRecord) -> dict[str, Any]:
    return {
        "input_ids": record.full_input_ids,
        "response_ids": record.response_ids,
        "policy_action_mask": record.policy_action_mask,
        "information_mask": record.information_mask,
        "image_sha256": record.image_sha256,
    }


def update_records(
    *,
    model: Any,
    trajectory_runner: TransformersTrajectoryRunner,
    optimizer: Any,
    records: Sequence[RolloutRecord],
    clip_ratio: float = 0.2,
    max_grad_norm: float = 1.0,
    verify_alignment: bool = False,
) -> dict[str, Any]:
    advantages, group_stats = _compute_group_values(records)
    if not bool(torch.isfinite(advantages).all()):
        raise RuntimeError("non-finite group advantage")
    input_identity_failure_count = 0
    response_token_mismatch_count = 0
    processor_hash_mismatch_count = 0
    target_truncation_count = 0
    information_token_train_mask_sum = 0
    policy_action_token_count = 0
    maximum_alignment_error = 0.0
    for record in records:
        input_ids = torch.as_tensor(record.full_input_ids)
        response_ids = torch.as_tensor(record.response_ids)
        if not torch.equal(input_ids[1:], response_ids):
            response_token_mismatch_count += 1
        if record.processor_hash != trajectory_runner.processor_hash:
            processor_hash_mismatch_count += 1
        if int(input_ids.numel()) > trajectory_runner.max_seq_len:
            target_truncation_count += 1
        policy_mask = torch.as_tensor(record.policy_action_mask)
        information_mask = torch.as_tensor(record.information_mask)
        information_token_train_mask_sum += int(
            (policy_mask * information_mask).sum().item()
        )
        policy_action_token_count += int(policy_mask.sum().item())
        replay_mapping = {
            "input_ids": input_ids,
            "response_ids": input_ids[1:],
            "policy_action_mask": policy_mask,
            "information_mask": information_mask,
            "image_sha256": record.image_sha256,
        }
        try:
            assert_input_identity(_identity_mapping(record), replay_mapping)
        except AssertionError:
            input_identity_failure_count += 1
    if any(
        (
            input_identity_failure_count,
            response_token_mismatch_count,
            processor_hash_mismatch_count,
            target_truncation_count,
            information_token_train_mask_sum,
        )
    ):
        raise RuntimeError(
            "rollout/update identity contract failed: "
            f"identity={input_identity_failure_count} "
            f"response={response_token_mismatch_count} "
            f"processor={processor_hash_mismatch_count} "
            f"truncation={target_truncation_count} "
            f"information_train_mask={information_token_train_mask_sum}"
        )
    if policy_action_token_count <= 0:
        raise RuntimeError("update has no policy action tokens")
    if verify_alignment:
        _set_rollout_mode(model)
        for record in records:
            recomputed = trajectory_runner.replay_log_probs(
                record, grad=False
            ).detach().cpu()
            error = assert_logprob_alignment(
                torch.as_tensor(record.old_log_probs),
                recomputed,
                mask=torch.as_tensor(record.policy_action_mask),
                tolerance=1e-3,
            )
            maximum_alignment_error = max(maximum_alignment_error, error)
    _set_update_mode(model)
    optimizer.zero_grad(set_to_none=True)
    loss_values = []
    ratio_rows = []
    for record, advantage in zip(records, advantages):
        new_log_probs = trajectory_runner.replay_log_probs(record, grad=True)
        old_log_probs = torch.as_tensor(
            record.old_log_probs, device=new_log_probs.device
        )
        mask = torch.as_tensor(
            record.policy_action_mask, device=new_log_probs.device
        )
        advantage_tokens = torch.ones_like(new_log_probs) * advantage.to(
            new_log_probs.device
        )
        loss = trajectory_balanced_policy_loss(
            old_log_probs,
            new_log_probs,
            advantage_tokens,
            mask,
            clip_ratio=clip_ratio,
        )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("non-finite GRPO policy loss")
        (loss / len(records)).backward()
        loss_values.append(float(loss.detach()))
        ratio_rows.append(
            policy_loss_metrics(
                old_log_probs.detach(),
                new_log_probs.detach(),
                mask,
                clip_ratio=clip_ratio,
            )
        )
        del new_log_probs, old_log_probs, mask, advantage_tokens, loss
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    gradient_finite = all(
        parameter.grad is None
        or bool(torch.isfinite(parameter.grad).all())
        for parameter in trainable
    )
    if not gradient_finite:
        raise RuntimeError("non-finite GRPO gradient")
    gradient_norm = float(
        torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
    )
    if not math.isfinite(gradient_norm):
        raise RuntimeError("non-finite GRPO gradient norm")
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    metrics = {
        "loss": statistics.mean(loss_values) if loss_values else 0.0,
        "gradient_norm": gradient_norm,
        "clip_fraction": statistics.mean(
            row["clip_fraction"] for row in ratio_rows
        )
        if ratio_rows
        else 0.0,
        "policy_ratio_mean": statistics.mean(
            row["policy_ratio_mean"] for row in ratio_rows
        )
        if ratio_rows
        else 1.0,
        "policy_ratio_max": max(
            (row["policy_ratio_max"] for row in ratio_rows), default=1.0
        ),
        "gradient_finite": gradient_finite,
        "input_identity_failure_count": input_identity_failure_count,
        "response_token_mismatch_count": response_token_mismatch_count,
        "processor_hash_mismatch_count": processor_hash_mismatch_count,
        "target_truncation_count": target_truncation_count,
        "information_token_train_mask_sum": information_token_train_mask_sum,
        "policy_action_token_count": policy_action_token_count,
        "max_logprob_alignment_error": maximum_alignment_error,
        "group_statistics": group_stats,
    }
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


def _load_runtime(project_root: Path) -> tuple[Any, Any, dict[str, Any]]:
    from multimodal_web_agent.training.sft.config import load_config
    from multimodal_web_agent.training.sft.model_factory import load_qwen_lora

    config = load_config(
        project_root / "configs/protocol_sft/train_full_format_v1.yaml",
        project_root=project_root,
    )
    adapter_path = (
        project_root
        / "outputs/protocol_format_sft_v1_full/selected_adapter"
    )
    print("[runtime] validating frozen SFT adapter fingerprint", flush=True)
    expected_adapter_hash = (
        "45baf20eb386804c717013303989e9d382d2b4299de6090a6b4abde7799fa2d5"
    )
    actual_adapter_hash = sha256_tree(adapter_path)
    if actual_adapter_hash != expected_adapter_hash:
        raise RuntimeError(
            "selected SFT adapter hash mismatch: "
            f"expected={expected_adapter_hash} actual={actual_adapter_hash}"
        )
    print("[runtime] loading Base NF4 + selected SFT adapter", flush=True)
    model, processor, audit = load_qwen_lora(
        config, adapter_path=adapter_path
    )
    if audit.trainable_visual_parameter_count != 0:
        raise RuntimeError("visual parameters are trainable")
    print("[runtime] model ready; visual modules frozen", flush=True)
    return model, processor, {
        "adapter_hash": actual_adapter_hash,
        "base_model_hash": sha256_tree(config.model.path),
        "model_audit": audit.to_dict(),
        "base_model_path": str(config.model.path),
    }


def _validate_pool(
    project_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = json.loads(
        (
            project_root
            / "data/processed/grpo_prompt_pool_v1/manifest.json"
        ).read_text(encoding="utf-8")
    )
    audit = json.loads(
        (
            project_root / "data/processed/grpo_prompt_pool_v1/audit.json"
        ).read_text(encoding="utf-8")
    )
    if manifest.get("schema_version") != POOL_SCHEMA:
        raise RuntimeError("GRPO Prompt Pool schema mismatch")
    if not audit.get("passed"):
        raise RuntimeError("GRPO Prompt Pool audit is not passing")
    if any(
        int(value) != 0
        for pair in manifest.get("group_isolation", {}).values()
        for value in pair.values()
    ):
        raise RuntimeError("GRPO Prompt Pool group isolation is non-zero")
    if manifest.get("total_unique_prompts") != 2560:
        raise RuntimeError("GRPO Prompt Pool total count is not 2560")
    return manifest, audit


def _load_environment(project_root: Path) -> CachedSandbox:
    from multimodal_web_agent.data.protocol_sft.cache_reader import (
        ImageSearchCache,
    )

    config_path = project_root / "configs/grpo/common_v1_server.yaml"
    common = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    environment_config = common.get("environment", {})
    image_top_k = int(environment_config.get("image_top_k", 0))
    text_top_k = int(environment_config.get("text_top_k", 0))
    image_context_policy = str(
        environment_config.get("image_context_policy", "")
    )
    if image_top_k != 3:
        raise RuntimeError(
            "Formal GRPO Image Search must use canonical cached top-3"
        )
    if text_top_k != 3:
        raise RuntimeError("Formal GRPO Text Search must use top_k=3")
    if image_context_policy != "canonical_cached_top3":
        raise RuntimeError(
            "Formal GRPO image_context_policy must be canonical_cached_top3"
        )
    cache = ImageSearchCache.load(
        project_root
        / "data/raw/fvqa/fvqa_train_image_search_results_cache.pkl",
        label="fvqa_train_official_cache",
    )
    return CachedSandbox(
        cache, image_top_k=image_top_k, text_top_k=text_top_k
    )


def _environment_manifest(
    project_root: Path,
    environment: CachedSandbox,
) -> dict[str, Any]:
    config_path = project_root / "configs/grpo/common_v1_server.yaml"
    return {
        "image_top_k": environment.image_top_k,
        "text_top_k": environment.text_top_k,
        "image_context_policy": "canonical_cached_top3",
        "common_config_hash": sha256_file(config_path),
    }


def _load_rank1_diagnostic_environment(project_root: Path) -> CachedSandbox:
    """Historical route probe only; never used by a formal training stage."""
    from multimodal_web_agent.data.protocol_sft.cache_reader import (
        ImageSearchCache,
    )

    cache = ImageSearchCache.load(
        project_root
        / "data/raw/fvqa/fvqa_train_image_search_results_cache.pkl",
        label="fvqa_train_official_cache",
    )
    return CachedSandbox(cache, image_top_k=1, text_top_k=3)


def _save_tensor_group(
    output_dir: Path,
    group_index: int,
    records: Sequence[RolloutRecord],
) -> None:
    path = output_dir / "rollout_records" / f"group_{group_index:06d}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(list(records), path)


def _code_provenance(project_root: Path, output_dir: Path) -> dict[str, Any]:
    files = [
        project_root
        / "src/multimodal_web_agent/training/grpo/server_runner.py",
        project_root
        / "src/multimodal_web_agent/training/grpo/prompt_pool.py",
        project_root
        / "src/multimodal_web_agent/training/grpo/advantages.py",
        project_root
        / "src/multimodal_web_agent/training/grpo/input_identity.py",
        project_root
        / "src/multimodal_web_agent/training/grpo/old_logprobs.py",
        project_root
        / "src/multimodal_web_agent/training/grpo/rollout_record.py",
        project_root
        / "src/multimodal_web_agent/training/grpo/policy_mask.py",
        project_root
        / "src/multimodal_web_agent/training/grpo/policy_loss.py",
        project_root
        / "src/multimodal_web_agent/training/grpo/reward_v0_mmsearch_like.py",
        project_root
        / "src/multimodal_web_agent/training/grpo/rollout_engine.py",
        project_root
        / "src/multimodal_web_agent/training/grpo/formal_contract.py",
        project_root
        / "src/multimodal_web_agent/training/grpo/text_search_integration.py",
        project_root
        / "src/multimodal_web_agent/training/grpo/smoke_selection.py",
        project_root / "scripts/run_grpo_v1_stage.py",
        project_root / "scripts/run_grpo_reward_v0_audit.py",
        project_root / "scripts/check_grpo_smoke_selection.py",
        project_root / "scripts/check_grpo_smoke_contract_v2.py",
        project_root
        / "scripts/validate_grpo_reward_v0_top3_prerequisites.py",
        project_root
        / "scripts/run_grpo_text_search_path_integration.py",
        project_root / "scripts/run_grpo_text_search_route_probe.py",
        project_root
        / "scripts/run_grpo_v1_text_search_route_probe_server.sh",
        project_root
        / "scripts/run_grpo_reward_v0_top3_prerequisites_server.sh",
        project_root
        / "scripts/run_grpo_text_search_path_integration_server.sh",
        project_root
        / "scripts/run_grpo_v1_reward_v0_smoke_v2_server.sh",
        project_root / "configs/grpo/common_v1_server.yaml",
        project_root / "configs/grpo/reward_v0_mmsearch_like.yaml",
    ]
    lines = [
        f"{sha256_file(file)}  {file.relative_to(project_root).as_posix()}"
        for file in files
    ]
    (output_dir / "build_code_files.sha256").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return {
        "code_provenance_mode": "file_hash_manifest",
        "src_tree_sha256": sha256_code_tree(
            project_root / "src", (".py",)
        ),
        "scripts_tree_sha256": sha256_code_tree(
            project_root / "scripts", (".py", ".sh")
        ),
        "configs_tree_sha256": sha256_code_tree(
            project_root / "configs", (".yaml", ".yml")
        ),
    }


def run_text_search_route_probe(
    *,
    project_root: Path,
    output_dir: Path,
    run_seed: int = 20260730,
) -> dict[str, Any]:
    """Run four rank-1 route probes without updating the frozen policy."""
    _validate_pool(project_root)
    train_rows = read_jsonl(
        project_root / "data/processed/grpo_prompt_pool_v1/train.jsonl"
    )
    prompts, selection = _select_smoke_prompts(
        train_rows, project_root=project_root
    )
    prompts = prompts[:4]
    model, processor, runtime = _load_runtime(project_root)
    environment = _load_rank1_diagnostic_environment(project_root)
    runtime["environment"] = {
        "image_top_k": 1,
        "text_top_k": 3,
        "image_context_policy": "historical_rank1_route_diagnostic",
    }
    runner = TransformersTrajectoryRunner(
        model=model,
        processor=processor,
        environment=environment,
        image_store=PromptImageStore(
            project_root / "data/raw/fvqa/fvqa_train.parquet"
        ),
        base_model_hash=runtime["base_model_hash"],
        adapter_hash=runtime["adapter_hash"],
    )
    runtime["processor_hash"] = runner.processor_hash
    runtime["chat_template_hash"] = runner.chat_template_hash
    _set_rollout_mode(model)
    settings = GenerationSettings(do_sample=True)
    records: list[RolloutRecord] = []
    for group_index, prompt in enumerate(prompts):
        group = runner.rollout_group(
            prompt,
            run_seed=run_seed,
            pool_pass=1,
            group_size=4,
            settings=settings,
        )
        _save_tensor_group(output_dir, group_index, group)
        runner.release_visual(prompt.prompt_uid)
        records.extend(group)
    diagnostics = {
        "stage": "diagnostic_text_search_route_probe",
        "prompt_group_count": len(prompts),
        "rollout_count": len(records),
        "update_model": False,
        "selection": selection,
        "image_search_rollout_count": sum(
            record.used_image_search for record in records
        ),
        "text_search_rollout_count": sum(
            record.used_text_search for record in records
        ),
        "three_turn_rollout_count": sum(
            len(record.actions) == 3 for record in records
        ),
        "first_action_distribution": dict(
            sorted(
                Counter(
                    (
                        record.actions[0]["action_type"]
                        if record.actions
                        else "none"
                    )
                    for record in records
                ).items()
            )
        ),
        "test_accessed": False,
        **formal_chain_manifest_fields(),
        **RANK1_DIAGNOSTIC_FIELDS,
        **runtime,
    }
    write_jsonl(
        output_dir / "rollout_records.jsonl",
        [_public_rollout(record) for record in records],
    )
    write_json(output_dir / "route_probe_diagnostics.json", diagnostics)
    write_json(output_dir / "run_manifest.json", diagnostics)
    _code_provenance(project_root, output_dir)
    print(f"[text_search_route_probe] {diagnostics}", flush=True)
    if diagnostics["text_search_rollout_count"] == 0:
        raise RuntimeError(
            "Rank-1 route probe produced zero Text Search rollouts; "
            "frozen policy remains route-collapsed"
        )
    return diagnostics


def run_reward_audit(
    *,
    project_root: Path,
    output_dir: Path,
    run_seed: int = 20260730,
) -> dict[str, Any]:
    pool_manifest, _ = _validate_pool(project_root)
    prompts = [
        PromptPoolItem.from_dict(row)
        for row in read_jsonl(
            project_root
            / "data/processed/grpo_prompt_pool_v1/reward_audit.jsonl"
        )
    ]
    if len(prompts) != 256:
        raise RuntimeError("Reward Audit requires exactly 256 prompts")
    model, processor, runtime = _load_runtime(project_root)
    environment = _load_environment(project_root)
    runtime["environment"] = _environment_manifest(
        project_root, environment
    )
    runner = TransformersTrajectoryRunner(
        model=model,
        processor=processor,
        environment=environment,
        image_store=PromptImageStore(
            project_root / "data/raw/fvqa/fvqa_train.parquet"
        ),
        base_model_hash=runtime["base_model_hash"],
        adapter_hash=runtime["adapter_hash"],
    )
    runtime["processor_hash"] = runner.processor_hash
    runtime["chat_template_hash"] = runner.chat_template_hash
    settings = GenerationSettings(do_sample=True)
    _set_rollout_mode(model)
    component_rows: list[dict[str, Any]] = []
    public_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    all_records: list[RolloutRecord] = []
    started = time.monotonic()
    for group_index, prompt in enumerate(prompts):
        records = runner.rollout_group(
            prompt,
            run_seed=run_seed,
            pool_pass=1,
            group_size=4,
            settings=settings,
        )
        _save_tensor_group(output_dir, group_index, records)
        runner.release_visual(prompt.prompt_uid)
        all_records.extend(records)
        public_rows.extend(_public_rollout(record) for record in records)
        component_rows.extend(
            dict(record.reward_components, prompt_uid=record.prompt_uid,
                 rollout_uid=record.rollout_uid)
            for record in records
        )
        _, stats = _compute_group_values(records)
        group_rows.extend(stats)
        for record in records:
            record.full_input_ids = None
            record.attention_mask = None
            record.response_ids = None
            record.policy_action_mask = None
            record.information_mask = None
            record.old_log_probs = None
            record.pixel_values = None
            record.image_grid_thw = None
        if (group_index + 1) % 8 == 0 or group_index + 1 == len(prompts):
            print(
                f"[reward_audit] prompt_groups={group_index + 1}/{len(prompts)} "
                f"rollouts={(group_index + 1) * 4}",
                flush=True,
            )
    if len(all_records) != 1024:
        raise RuntimeError("Reward Audit did not produce 1024 trajectories")
    summary = _reward_summary(all_records, group_rows)
    undefined = [
        value
        for value in summary["reward_distribution"]
        if min(abs(float(value) - expected) for expected in ALLOWED_REWARD_VALUES)
        > 1e-6
    ]
    if undefined:
        raise RuntimeError(f"undefined Reward v0 values: {undefined}")
    write_jsonl(output_dir / "rollout_records.jsonl", public_rows)
    write_jsonl(output_dir / "reward_components.jsonl", component_rows)
    write_jsonl(output_dir / "group_statistics.jsonl", group_rows)
    write_json(output_dir / "reward_distribution.json", summary)
    write_json(
        output_dir / "behavior_diagnostics.json",
        {
            key: value
            for key, value in summary.items()
            if key
            not in {
                "reward_mean",
                "reward_std",
                "reward_min",
                "reward_max",
                "reward_distribution",
            }
        },
    )
    resource = {
        "elapsed_seconds": time.monotonic() - started,
        "peak_vram": (
            int(torch.cuda.max_memory_allocated())
            if torch.cuda.is_available()
            else 0
        ),
    }
    write_json(output_dir / "resource_metrics.json", resource)
    run_manifest = {
        "stage": "reward_audit",
        "reward_name": REWARD_NAME,
        "prompt_count": 256,
        "rollout_count": 1024,
        "group_size": 4,
        "update_model": False,
        "test_accessed": False,
        "pool_manifest": pool_manifest,
        "prompt_pool_hash": sha256_file(
            project_root
            / "data/processed/grpo_prompt_pool_v1/reward_audit.jsonl"
        ),
        "reward_config_hash": sha256_file(
            project_root / "configs/grpo/reward_v0_mmsearch_like.yaml"
        ),
        "training_config_hash": sha256_file(
            project_root / "configs/grpo/reward_v0_audit_server.yaml"
        ),
        **formal_chain_manifest_fields(),
        **_code_provenance(project_root, output_dir),
        **runtime,
    }
    write_json(output_dir / "run_manifest.json", run_manifest)
    (output_dir / "audit_report.md").write_text(
        "# Reward v0 Audit\n\n```json\n"
        + json.dumps(summary, ensure_ascii=False, indent=2)
        + "\n```\n\nGRPO_REWARD_V0_AUDIT_READY\n",
        encoding="utf-8",
    )
    return summary


def _snapshot_trainable(model: Any) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def _trainable_changed(
    before: Mapping[str, torch.Tensor], model: Any
) -> bool:
    for name, parameter in model.named_parameters():
        if name in before and not torch.equal(
            before[name], parameter.detach().cpu()
        ):
            return True
    return False


def _trainable_change_count(
    before: Mapping[str, torch.Tensor], model: Any
) -> int:
    return sum(
        name in before
        and not torch.equal(before[name], parameter.detach().cpu())
        for name, parameter in model.named_parameters()
    )


def _is_visual_parameter(name: str) -> bool:
    lowered = name.casefold()
    return any(
        token in lowered for token in ("visual", "vision", "merger", "image")
    )


def _snapshot_visual_versions(model: Any) -> dict[str, int]:
    return {
        name: int(getattr(parameter, "_version", 0))
        for name, parameter in model.named_parameters()
        if _is_visual_parameter(name)
    }


def _snapshot_trainable_content_hashes(model: Any) -> dict[str, str]:
    """Hash trainable tensor values without relying on PyTorch version counters.

    Fused and bitsandbytes optimizers can mutate parameter storage without
    incrementing ``Parameter._version``. Content hashes therefore provide the
    optimizer-independent evidence required by the Full LoRA-change gate.
    """
    result: dict[str, str] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        value = parameter.detach().to(device="cpu").contiguous()
        digest = hashlib.sha256()
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(json.dumps(list(value.shape)).encode("utf-8"))
        digest.update(value.view(torch.uint8).numpy().tobytes())
        result[name] = digest.hexdigest()
    if not result:
        raise RuntimeError("no trainable parameters available for fingerprint")
    return result


def _content_hash_change_count(
    before: Mapping[str, str], after: Mapping[str, str]
) -> int:
    if set(before) != set(after):
        raise RuntimeError("trainable parameter set changed during Full")
    return sum(before[name] != after[name] for name in before)


def _content_hash_root(values: Mapping[str, str]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _visual_change_count(
    before: Mapping[str, int], model: Any
) -> int:
    return sum(
        name in before
        and int(getattr(parameter, "_version", 0)) != before[name]
        for name, parameter in model.named_parameters()
        if _is_visual_parameter(name)
    )


def _save_adapter(model: Any, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(path)


def _verify_adapter_checkpoint(path: Path) -> None:
    config = path / "adapter_config.json"
    tensors = sorted(path.glob("*.safetensors"))
    if not config.is_file() or not tensors:
        raise RuntimeError("adapter checkpoint is incomplete")
    from safetensors import safe_open

    key_count = 0
    for tensor_file in tensors:
        with safe_open(tensor_file, framework="pt", device="cpu") as handle:
            key_count += len(list(handle.keys()))
    if key_count <= 0:
        raise RuntimeError("adapter checkpoint contains no tensors")


def _reload_adapter_and_generate(
    *,
    model: Any,
    runner: TransformersTrajectoryRunner,
    checkpoint: Path,
    prompt: PromptPoolItem,
    run_seed: int,
) -> dict[str, Any]:
    """Actually load the saved adapter into PEFT, then execute one generation."""
    if not hasattr(model, "load_adapter") or not hasattr(model, "set_adapter"):
        raise RuntimeError("PEFT model does not support adapter reload")
    adapter_name = "grpo_smoke_v2_reload_validation"
    model.load_adapter(
        str(checkpoint),
        adapter_name=adapter_name,
        is_trainable=False,
    )
    model.set_adapter(adapter_name)
    _set_rollout_mode(model)
    record = runner.rollout(
        prompt,
        run_seed=run_seed,
        pool_pass=0,
        rollout_index=0,
        settings=GenerationSettings(do_sample=False),
    )
    runner.release_visual(prompt.prompt_uid)
    success = bool(record.actions) and int(
        torch.as_tensor(record.policy_action_mask).sum().item()
    ) > 0
    if not success:
        raise RuntimeError("generation after checkpoint reload failed")
    return {
        "checkpoint_reload_success": True,
        "reload_generation_success": True,
        "reload_generation_terminal_reason": record.terminal_reason,
        "reload_generation_action_count": len(record.actions),
    }


def _evaluate_prompts(
    runner: TransformersTrajectoryRunner,
    model: Any,
    prompts: Sequence[PromptPoolItem],
    *,
    run_seed: int,
) -> dict[str, Any]:
    _set_rollout_mode(model)
    settings = GenerationSettings(do_sample=False)
    records = []
    for prompt in prompts:
        records.append(
            runner.rollout(
            prompt,
            run_seed=run_seed,
            pool_pass=0,
            rollout_index=0,
            settings=settings,
        )
        )
        runner.release_visual(prompt.prompt_uid)
    _, groups = _compute_group_values(records)
    return _reward_summary(records, groups)


def run_training_stage(
    *,
    project_root: Path,
    output_dir: Path,
    stage: str,
    run_seed: int = 20260730,
) -> dict[str, Any]:
    if stage not in {"contract_smoke", "smoke", "smoke_v2", "full"}:
        raise ValueError(stage)
    pool_manifest, _ = _validate_pool(project_root)
    train_rows = read_jsonl(
        project_root / "data/processed/grpo_prompt_pool_v1/train.jsonl"
    )
    if stage == "contract_smoke":
        prompts = _select_prompts(
            train_rows, search_free=2, search_required=2
        )
    elif stage in {"smoke", "smoke_v2"}:
        prompts = _select_prompts(
            train_rows, search_free=12, search_required=20
        )
    else:
        prompts = [PromptPoolItem.from_dict(row) for row in train_rows]
        if len(prompts) != 2048:
            raise RuntimeError("Full requires exactly 2048 prompts")
    model, processor, runtime = _load_runtime(project_root)
    environment = _load_environment(project_root)
    runtime["environment"] = _environment_manifest(
        project_root, environment
    )
    runner = TransformersTrajectoryRunner(
        model=model,
        processor=processor,
        environment=environment,
        image_store=PromptImageStore(
            project_root / "data/raw/fvqa/fvqa_train.parquet"
        ),
        base_model_hash=runtime["base_model_hash"],
        adapter_hash=runtime["adapter_hash"],
    )
    runtime["processor_hash"] = runner.processor_hash
    runtime["chat_template_hash"] = runner.chat_template_hash
    optimizer = build_paged_adamw_8bit(
        (parameter for parameter in model.parameters()
         if parameter.requires_grad),
        learning_rate=5e-7,
        weight_decay=0.0,
    )
    before = (
        _snapshot_trainable(model)
        if stage in {"contract_smoke", "smoke_v2"}
        else {}
    )
    trainable_hashes_before = (
        _snapshot_trainable_content_hashes(model)
        if stage == "full"
        else {}
    )
    visual_versions_before = _snapshot_visual_versions(model)
    dev_rows = read_jsonl(
        project_root / "data/processed/grpo_prompt_pool_v1/dev.jsonl"
    )
    dev_prompts = [PromptPoolItem.from_dict(row) for row in dev_rows]
    if stage in {"smoke", "smoke_v2"}:
        dev_prompts = _select_prompts(
            dev_rows, search_free=8, search_required=8
        )
    settings = GenerationSettings(do_sample=True)
    all_public: list[dict[str, Any]] = []
    all_components: list[dict[str, Any]] = []
    all_group_stats: list[dict[str, Any]] = []
    update_rows: list[dict[str, Any]] = []
    dev_evaluations: list[dict[str, Any]] = []
    started = time.monotonic()
    expected_steps = {
        "contract_smoke": 1,
        "smoke": 8,
        "smoke_v2": 8,
        "full": 512,
    }[stage]
    if stage == "full":
        print("[full] running SFT Init Frozen Dev evaluation", flush=True)
        init_metrics = _evaluate_prompts(
            runner, model, dev_prompts, run_seed=run_seed
        )
        init_metrics["prompt_count_processed"] = 0
        dev_evaluations.append(init_metrics)
        write_json(
            output_dir / "dev_evaluations/sft_init.json", init_metrics
        )
    prompt_buffer: list[RolloutRecord] = []
    optimizer_step = 0
    group_index = 0
    for prompt_index, prompt in enumerate(prompts, start=1):
        _set_rollout_mode(model)
        records = runner.rollout_group(
            prompt,
            run_seed=run_seed,
            pool_pass=1,
            group_size=4,
            settings=settings,
        )
        _save_tensor_group(output_dir, group_index, records)
        runner.release_visual(prompt.prompt_uid)
        group_index += 1
        prompt_buffer.extend(records)
        all_public.extend(_public_rollout(record) for record in records)
        all_components.extend(
            dict(record.reward_components, prompt_uid=record.prompt_uid,
                 rollout_uid=record.rollout_uid)
            for record in records
        )
        _, stats = _compute_group_values(records)
        all_group_stats.extend(stats)
        if len(prompt_buffer) == 16:
            update = update_records(
                model=model,
                trajectory_runner=runner,
                optimizer=optimizer,
                records=prompt_buffer,
                verify_alignment=stage
                in {"contract_smoke", "smoke_v2", "full"},
            )
            optimizer_step += 1
            update_rows.append(
                {
                    key: value
                    for key, value in update.items()
                    if key != "group_statistics"
                }
                | {
                    "optimizer_step": optimizer_step,
                    "prompt_count_processed": prompt_index,
                }
            )
            print(
                f"[{stage}] optimizer_step={optimizer_step}/{expected_steps} "
                f"prompts={prompt_index}/{len(prompts)}",
                flush=True,
            )
            prompt_buffer = []
        if stage == "full" and prompt_index % 256 == 0:
            _save_adapter(
                model,
                output_dir
                / "checkpoints"
                / f"prompt_{prompt_index:04d}",
            )
        if stage == "full" and prompt_index % 512 == 0:
            print(
                f"[full] running Frozen Dev evaluation at prompt {prompt_index}",
                flush=True,
            )
            metrics = _evaluate_prompts(
                runner, model, dev_prompts, run_seed=run_seed
            )
            metrics["prompt_count_processed"] = prompt_index
            dev_evaluations.append(metrics)
            write_json(
                output_dir
                / "dev_evaluations"
                / f"prompt_{prompt_index:04d}.json",
                metrics,
            )
    if optimizer_step != expected_steps:
        raise RuntimeError(
            f"{stage} optimizer steps={optimizer_step}, expected={expected_steps}"
        )
    if stage == "contract_smoke":
        if not _trainable_changed(before, model):
            raise RuntimeError("LoRA parameters did not change")
        if any(
            parameter.requires_grad
            for name, parameter in model.named_parameters()
            if any(token in name.casefold() for token in ("visual", "vision", "merger", "image"))
        ):
            raise RuntimeError("visual parameter became trainable")
        miss = environment.image_search("__grpo_contract_cache_miss__")
        if not miss.cache_miss or "Cache Miss" not in miss.text:
            raise RuntimeError("deterministic Cache Miss contract failed")
        if any(
            min(
                abs(float(row["reward_total"]) - expected)
                for expected in ALLOWED_REWARD_VALUES
            )
            > 1e-6
            for row in all_public
        ):
            raise RuntimeError("Contract Smoke emitted undefined Reward v0")
        zero_fixture = compute_group_advantages(
            torch.tensor([0.5, 0.5]), ["zero", "zero"]
        )
        if not torch.equal(zero_fixture, torch.zeros_like(zero_fixture)):
            raise RuntimeError("zero-variance group contract failed")
        _save_adapter(model, output_dir / "checkpoint")
        _verify_adapter_checkpoint(output_dir / "checkpoint")
    reward_metrics = _reward_summary(
        [RolloutRecord.from_dict(row) for row in all_public],
        all_group_stats,
    )
    stage_behavior = behavior_diagnostics(
        all_public,
        all_group_stats,
        reward_metrics=reward_metrics,
    )
    if stage in {"smoke", "smoke_v2", "full"}:
        write_json(output_dir / "behavior_diagnostics.json", stage_behavior)
    engineering_contract: dict[str, Any] | None = None
    if stage in {"smoke", "smoke_v2"}:
        _save_adapter(model, output_dir / "checkpoint")
        _verify_adapter_checkpoint(output_dir / "checkpoint")
        dev_metrics = _evaluate_prompts(
            runner, model, dev_prompts, run_seed=run_seed
        )
        write_json(output_dir / "dev_format_regression.json", dev_metrics)
    if stage == "smoke_v2":
        lora_change_count = _trainable_change_count(before, model)
        visual_change_count = _visual_change_count(
            visual_versions_before, model
        )
        reload_validation = _reload_adapter_and_generate(
            model=model,
            runner=runner,
            checkpoint=output_dir / "checkpoint",
            prompt=dev_prompts[0],
            run_seed=run_seed,
        )
        engineering_contract = validate_smoke_engineering_contract(
            records=all_public,
            group_statistics=all_group_stats,
            update_metrics=update_rows,
            prompt_count=len(prompts),
            rollout_count=len(all_public),
            optimizer_steps=optimizer_step,
            group_size=4,
            expected_processor_hash=runner.processor_hash,
            lora_parameter_change_count=lora_change_count,
            visual_trainable_parameter_count=int(
                runtime["model_audit"]["trainable_visual_parameter_count"]
            ),
            visual_parameter_change_count=visual_change_count,
            checkpoint_saved=(output_dir / "checkpoint").is_dir(),
            checkpoint_reload_success=bool(
                reload_validation["checkpoint_reload_success"]
            ),
            reload_generation_success=bool(
                reload_validation["reload_generation_success"]
            ),
            protocol_metrics_computed=all(
                key in reward_metrics
                for key in ("protocol_valid_rate", "malformed_rate", "finish_rate")
            ),
            reward_metrics_computed=all(
                key in reward_metrics
                for key in ("reward_mean", "reward_std", "reward_distribution")
            ),
            test_accessed=False,
        )
        engineering_contract.update(reload_validation)
        write_json(
            output_dir / "engineering_contract_v2.json",
            engineering_contract,
        )
    # Persist the completed audit trail before evaluating final Full gates.
    # Atomic publication still happens only after every gate passes, while a
    # failed directory now retains the exact 8192/2048/512 evidence for audit.
    write_jsonl(output_dir / "rollout_records.jsonl", all_public)
    write_jsonl(output_dir / "reward_components.jsonl", all_components)
    write_jsonl(output_dir / "group_statistics.jsonl", all_group_stats)
    write_jsonl(output_dir / "update_metrics.jsonl", update_rows)
    full_engineering_contract: dict[str, Any] | None = None
    if stage == "full":
        full_failures = []
        if len(all_public) != 8192:
            full_failures.append("rollout_count")
        if optimizer_step != 512:
            full_failures.append("optimizer_steps")
        if len(all_group_stats) != 2048:
            full_failures.append("group_statistics")
        if len(dev_evaluations) != 5:
            full_failures.append("frozen_dev_evaluations")
        finite_fields = (
            "loss",
            "gradient_norm",
            "clip_fraction",
            "policy_ratio_mean",
            "policy_ratio_max",
            "max_logprob_alignment_error",
        )
        if not all(
            math.isfinite(float(row.get(key, math.nan)))
            for row in update_rows
            for key in finite_fields
        ):
            full_failures.append("finite_update_metrics")
        for key in (
            "input_identity_failure_count",
            "response_token_mismatch_count",
            "processor_hash_mismatch_count",
            "target_truncation_count",
            "information_token_train_mask_sum",
        ):
            if sum(int(row.get(key, 1)) for row in update_rows) != 0:
                full_failures.append(key)
        if max(
            (
                float(row.get("max_logprob_alignment_error", math.inf))
                for row in update_rows
            ),
            default=math.inf,
        ) >= 1e-3:
            full_failures.append("logprob_alignment")
        if not all(bool(row.get("gradient_finite")) for row in update_rows):
            full_failures.append("gradient_finite")
        trainable_hashes_after = _snapshot_trainable_content_hashes(model)
        trainable_change_count = _content_hash_change_count(
            trainable_hashes_before, trainable_hashes_after
        )
        visual_change_count = _visual_change_count(
            visual_versions_before, model
        )
        if trainable_change_count <= 0:
            full_failures.append("lora_change")
        if visual_change_count != 0:
            full_failures.append("visual_change")
        if any(
            parameter.requires_grad
            for name, parameter in model.named_parameters()
            if _is_visual_parameter(name)
        ):
            full_failures.append("visual_trainable")
        checkpoint_count = len(
            list((output_dir / "checkpoints").glob("prompt_*"))
        )
        if checkpoint_count != 8:
            full_failures.append("checkpoint_count")
        if full_failures:
            raise RuntimeError(
                "Full engineering contract failed: "
                + ", ".join(full_failures)
            )
        full_engineering_contract = {
            "engineering_hard_gates_passed": True,
            "rollout_count": 8192,
            "optimizer_steps": 512,
            "input_identity_failure_count": 0,
            "logprob_alignment_failure_count": 0,
            "information_mask_failure_count": 0,
            "target_truncation_count": 0,
            "loss_and_gradient_finite": True,
            "lora_change_detection_method": "per_parameter_content_sha256",
            "lora_parameter_change_count": trainable_change_count,
            "lora_parameter_fingerprint_before": _content_hash_root(
                trainable_hashes_before
            ),
            "lora_parameter_fingerprint_after": _content_hash_root(
                trainable_hashes_after
            ),
            "visual_trainable_parameter_count": 0,
            "visual_parameter_change_count": 0,
            "checkpoint_count": checkpoint_count,
            "frozen_dev_evaluation_count": len(dev_evaluations),
            "test_accessed": False,
        }
        write_json(
            output_dir / "full_engineering_contract.json",
            full_engineering_contract,
        )
    summary = {
        "stage": stage,
        "prompt_count": len(prompts),
        "rollout_count": len(all_public),
        "group_size": 4,
        "prompt_groups_per_update": 4,
        "optimizer_steps": optimizer_step,
        "update_model": True,
        "pool_passes": 1,
        "reward_name": REWARD_NAME,
        "test_accessed": False,
        "elapsed_seconds": time.monotonic() - started,
        "peak_vram": (
            int(torch.cuda.max_memory_allocated())
            if torch.cuda.is_available()
            else 0
        ),
        "pool_manifest_hash": sha256_file(
            project_root
            / "data/processed/grpo_prompt_pool_v1/manifest.json"
        ),
        "prompt_pool_hash": sha256_file(
            project_root / "data/processed/grpo_prompt_pool_v1/train.jsonl"
        ),
        "reward_config_hash": sha256_file(
            project_root / "configs/grpo/reward_v0_mmsearch_like.yaml"
        ),
        "training_config_hash": sha256_file(
            project_root
            / "configs/grpo"
            / {
                "contract_smoke": "contract_smoke_v1_server.yaml",
                "smoke": "smoke_reward_v0_server.yaml",
                "smoke_v2": "smoke_reward_v0_contract_v2_server.yaml",
                "full": "full_reward_v0_server.yaml",
            }[stage]
        ),
        **formal_chain_manifest_fields(),
        **_code_provenance(project_root, output_dir),
        **runtime,
    }
    if stage == "smoke_v2":
        summary["schema_version"] = SMOKE_CONTRACT_V2_SCHEMA
        summary["engineering_contract"] = engineering_contract
        summary["behavior_diagnostics"] = stage_behavior
        summary.update(BEHAVIOR_CONTRACT_FIELDS)
    if stage == "full":
        summary["behavior_diagnostics"] = stage_behavior
        summary["engineering_contract"] = full_engineering_contract
        summary.update(BEHAVIOR_CONTRACT_FIELDS)
    write_json(output_dir / "run_manifest.json", summary)
    if stage == "full":
        selection = select_reward_v0_checkpoint(
            output_dir=output_dir,
            evaluations=dev_evaluations,
        )
        summary["selected_reward_v0_checkpoint"] = selection[
            "selected_checkpoint"
        ]
        write_json(output_dir / "run_manifest.json", summary)
        _write_baseline_report(
            output_dir=output_dir,
            pool_manifest=pool_manifest,
            run_summary=summary,
            evaluations=dev_evaluations,
            selection=selection,
        )
    return summary


def select_reward_v0_checkpoint(
    *, output_dir: Path, evaluations: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    if not evaluations:
        raise RuntimeError("checkpoint selection has no Dev evaluations")
    init = evaluations[0]
    eligible = []
    for row in evaluations[1:]:
        prompt_count = int(row["prompt_count_processed"])
        checkpoint = output_dir / "checkpoints" / f"prompt_{prompt_count:04d}"
        passes = (
            float(row.get("protocol_valid_rate", 0.0))
            >= float(init.get("protocol_valid_rate", 0.0)) - 0.01
            and float(row.get("malformed_rate", 1.0))
            <= float(init.get("malformed_rate", 1.0)) + 0.01
            and float(row.get("finish_rate", 0.0))
            >= float(init.get("finish_rate", 0.0)) - 0.02
            and int(row.get("forged_information_count", 0)) == 0
        )
        if passes and checkpoint.is_dir():
            eligible.append((row, checkpoint))
    if not eligible:
        raise RuntimeError("no Reward v0 checkpoint passes protocol gates")
    eligible.sort(
        key=lambda pair: (
            -float(pair[0].get("overall_f1", 0.0)),
            -float(pair[0].get("search_required_em", 0.0)),
            float(pair[0].get("unnecessary_search_rate", 1.0)),
            -float(pair[0].get("protocol_valid_rate", 0.0)),
            float(pair[0].get("average_search_calls", 999.0)),
            int(pair[0]["prompt_count_processed"]),
        )
    )
    row, checkpoint = eligible[0]
    selected = output_dir / "selected_reward_v0_checkpoint"
    if selected.exists():
        raise FileExistsError(selected)
    shutil.copytree(checkpoint, selected)
    result = {
        "selected_checkpoint": checkpoint.relative_to(output_dir).as_posix(),
        "selected_prompt_count": int(row["prompt_count_processed"]),
        "metrics": dict(row),
        "eligible_checkpoint_count": len(eligible),
        "final_teacher_selected": False,
    }
    write_json(output_dir / "checkpoint_selection.json", result)
    (output_dir / "checkpoint_selection.md").write_text(
        "# Reward v0 Checkpoint Selection\n\n"
        f"Selected: `{checkpoint.name}`\n\n"
        "This is a Reward v0 baseline checkpoint, not a final Teacher.\n",
        encoding="utf-8",
    )
    return result


def _write_baseline_report(
    *,
    output_dir: Path,
    pool_manifest: Mapping[str, Any],
    run_summary: Mapping[str, Any],
    evaluations: Sequence[Mapping[str, Any]],
    selection: Mapping[str, Any],
) -> None:
    sections = [
        ("实验目标", "Lightweight Multimodal GRPO v1 Reward v0 baseline."),
        ("为什么称为 Lightweight", "Single TITAN RTX, Qwen2.5-VL-3B, NF4 QLoRA, FP16, frozen vision, sequential Transformers rollout."),
        ("当前服务器配置", json.dumps(run_summary, ensure_ascii=False, indent=2)),
        ("Prompt Pool 规模和分布", json.dumps(pool_manifest.get("counts"), ensure_ascii=False, indent=2)),
        ("Pool Group Isolation", json.dumps(pool_manifest.get("group_isolation"), ensure_ascii=False, indent=2)),
        ("SFT Init Fingerprint", str(run_summary.get("adapter_hash"))),
        ("Reward v0 精确定义", "0.90 × answer_score_after_penalty + 0.10 × format_score."),
        ("Full 训练步数与 Rollout 数", f"{run_summary.get('optimizer_steps')} steps; {run_summary.get('rollout_count')} rollouts."),
        ("Frozen Dev 曲线", json.dumps(list(evaluations), ensure_ascii=False, indent=2)),
        ("Selected Reward v0 Checkpoint", json.dumps(selection, ensure_ascii=False, indent=2)),
        ("为什么它还不是最终 Teacher", "Reward v1 and Reward v2 have not been run or compared."),
        ("下一步 Reward 优化建议", "Only after inspecting the measured Reward v0 behavior diagnostics."),
    ]
    lines = ["# Reward v0 Baseline Report", ""]
    for title, body in sections:
        lines.extend([f"## {title}", "", body, ""])
    lines.append("GRPO_REWARD_V0_FULL_COMPLETE")
    lines.append("GRPO_REWARD_V0_BASELINE_READY")
    (output_dir / "reward_v0_baseline_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
