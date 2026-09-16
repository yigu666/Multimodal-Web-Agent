from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from multimodal_web_agent.agent.parser import parse_action
from multimodal_web_agent.agent.schema import ActionType
from multimodal_web_agent.data.protocol_sft.cache_reader import ImageSearchCache
from multimodal_web_agent.data.protocol_sft.information_formatter import format_image_information, format_text_information
from multimodal_web_agent.data.protocol_sft.text_retriever import BootstrapTextRetriever, text_backend_provenance

from .schema import PromptPoolItem, RolloutRecord, validate_rollout_group


def derive_generation_seed(run_seed: int, prompt_uid: str, pool_pass: int, rollout_index: int) -> int:
    raw = f"{int(run_seed)}|{prompt_uid}|{int(pool_pass)}|{int(rollout_index)}".encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") % (2**31 - 1)


@dataclass(frozen=True)
class ToolResult:
    text: str
    cache_miss: bool = False
    provenance: dict[str, Any] | None = None


class CachedSandbox:
    """Offline, deterministic image-cache and BM25 text environment."""

    def __init__(self, cache: ImageSearchCache, *, image_top_k: int = 3, text_top_k: int = 3):
        self.cache = cache
        self.image_top_k = image_top_k
        self.text_top_k = text_top_k
        self.text_retriever = BootstrapTextRetriever.from_image_cache(cache)

    def image_search(self, data_id: str) -> ToolResult:
        entry = self.cache.get(data_id)
        if entry is None or not entry.usable_image_results:
            return ToolResult("<information>\nCache Miss\n</information>", True, {"cache_miss": True, "data_id": data_id})
        bundle = format_image_information(entry, self.image_top_k)
        return ToolResult(bundle.text, False, bundle.provenance)

    def text_search(self, query: str, *, exclude_source_data_id: str = "") -> ToolResult:
        hits = self.text_retriever.retrieve(query, self.text_top_k, exclude_source_data_id=exclude_source_data_id)
        if not hits:
            return ToolResult("<information>\nCache Miss\n</information>", True, {"cache_miss": True, "query": query})
        provenance = text_backend_provenance(self.cache, hits)
        return ToolResult(format_text_information(hits, provenance).text, False, provenance)


class RolloutEngine:
    def __init__(self, environment: CachedSandbox, *, max_turns: int = 3, max_new_tokens_per_turn: int = 96):
        self.environment = environment
        self.max_turns = max_turns
        self.max_new_tokens_per_turn = max_new_tokens_per_turn

    def rollout(
        self,
        prompt: PromptPoolItem | Mapping[str, Any],
        generate: Callable[..., str],
        *,
        run_seed: int,
        pool_pass: int,
        rollout_index: int,
    ) -> RolloutRecord:
        item = prompt if isinstance(prompt, PromptPoolItem) else PromptPoolItem.from_dict(prompt)
        seed = derive_generation_seed(run_seed, item.prompt_uid, pool_pass, rollout_index)
        rng = random.Random(seed)
        record = RolloutRecord(
            prompt_uid=item.prompt_uid, rollout_uid=f"{item.prompt_uid}:r{rollout_index}",
            data_id=item.data_id, rollout_index=rollout_index, generation_seed=seed,
            image_sha256=item.image_sha256,
        )
        observations: list[str] = []
        cache_miss = False
        seen_queries: set[str] = set()
        for turn in range(self.max_turns):
            try:
                try:
                    output = generate(item, tuple(observations), rng=rng, seed=seed, turn=turn, max_new_tokens=self.max_new_tokens_per_turn)
                except TypeError:
                    output = generate(item, tuple(observations), rng)
            except Exception as exc:
                record.protocol_error = f"generation_error:{type(exc).__name__}"
                record.terminal_reason = "environment_error"
                return record
            parsed = parse_action(str(output))
            record.actions.append({"turn": turn, "raw": str(output), "action_type": parsed.action_type.value if parsed.action_type else "", "content": parsed.content, "valid": parsed.valid, "error": parsed.error_code.value if parsed.error_code else None})
            if not parsed.valid:
                record.protocol_error = parsed.error_code.value if parsed.error_code else "invalid_protocol"
                record.terminal_reason = "protocol_error"
                return record
            action = parsed.action_type
            if action == ActionType.ANSWER:
                record.answer_text = parsed.content or ""
                record.protocol_valid = True
                record.terminal_reason = "answer"
                return record
            record.search_count += 1
            if action == ActionType.IMAGE_SEARCH:
                record.used_image_search = True
                result = self.environment.image_search(item.data_id)
            else:
                record.used_text_search = True
                query = parsed.content or ""
                if query.casefold() in seen_queries:
                    record.duplicate_query = True
                seen_queries.add(query.casefold())
                result = self.environment.text_search(query, exclude_source_data_id=item.data_id)
            cache_miss = cache_miss or result.cache_miss
            record.tool_results.append({"turn": turn, "text": result.text, "cache_miss": result.cache_miss, "provenance": result.provenance or {}})
            observations.append(result.text)
        record.terminal_reason = "max_turns"
        if cache_miss:
            record.terminal_reason = "max_turns"
        return record

    def rollout_group(self, prompt, generate, *, run_seed: int, pool_pass: int, group_size: int = 4) -> list[RolloutRecord]:
        records = [self.rollout(prompt, generate, run_seed=run_seed, pool_pass=pool_pass, rollout_index=index) for index in range(group_size)]
        validate_rollout_group(records, group_size)
        return records
