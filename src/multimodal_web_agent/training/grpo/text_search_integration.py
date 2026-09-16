from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import torch

from multimodal_web_agent.agent.parser import parse_action
from multimodal_web_agent.agent.schema import ActionType
from multimodal_web_agent.data.protocol_sft.cache_reader import ImageSearchCache
from multimodal_web_agent.data.protocol_sft.schema import Message
from multimodal_web_agent.data.protocol_sft.templates import SYSTEM_PROMPT
from multimodal_web_agent.training.sft.config import load_config
from multimodal_web_agent.training.sft.dataset import ImageStore
from multimodal_web_agent.training.sft.model_factory import load_processor
from multimodal_web_agent.training.sft.renderer import ProtocolRenderer

from .formal_contract import (
    FIXTURE_BOUNDARY_FIELDS,
    formal_chain_manifest_fields,
)
from .policy_mask import build_grpo_masks
from .rollout_engine import CachedSandbox
from .schema import PromptPoolItem


FIXTURE_ACTION = (
    "<reason>Additional factual evidence is required.</reason>"
    "<text_search>location of Mirador de Isabel II</text_search>"
)
FIXTURE_SUBSEQUENT_ACTION = (
    "<reason>The retrieved evidence identifies the location.</reason>"
    "<answer>ceuta</answer>"
)


def parse_text_search_fixture(text: str = FIXTURE_ACTION) -> str:
    parsed = parse_action(text)
    if not parsed.valid or parsed.action_type != ActionType.TEXT_SEARCH:
        raise RuntimeError("Text Search fixture failed Strict Parser contract")
    query = str(parsed.content or "").strip()
    if not query:
        raise RuntimeError("Text Search fixture query is empty")
    return query


def fixture_boundary_manifest() -> dict[str, Any]:
    return dict(FIXTURE_BOUNDARY_FIELDS)


def _encode(
    processor: Any,
    messages: Sequence[Mapping[str, Any]],
    image: Any,
    *,
    add_generation_prompt: bool,
) -> dict[str, Any]:
    text = str(
        processor.apply_chat_template(
            list(messages),
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    )
    return dict(
        processor(
            text=[text],
            images=[image],
            padding=False,
            truncation=False,
            return_tensors="pt",
        )
    )


def _spans_and_batch(
    processor: Any,
    messages: Sequence[Mapping[str, Any]],
    image: Any,
) -> tuple[dict[str, Any], list[tuple[int, int]], list[tuple[int, int]]]:
    final_batch = _encode(
        processor, messages, image, add_generation_prompt=False
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
        before_batch = _encode(
            processor,
            messages[:index],
            image,
            add_generation_prompt=(role == "assistant"),
        )
        after_batch = _encode(
            processor,
            messages[: index + 1],
            image,
            add_generation_prompt=False,
        )
        before_ids = before_batch["input_ids"][0]
        after_ids = after_batch["input_ids"][0]
        before = int(before_ids.shape[0])
        after = int(after_ids.shape[0])
        if not torch.equal(final_ids[:before], before_ids):
            raise RuntimeError("fixture prefix mismatch before span")
        if not torch.equal(final_ids[:after], after_ids):
            raise RuntimeError("fixture prefix mismatch after span")
        if role == "assistant":
            assistant_spans.append((before, after))
        else:
            information_spans.append((before, after))
    return final_batch, assistant_spans, information_spans


def validate_fixture_masks(
    *,
    sequence_length: int,
    assistant_spans: Sequence[tuple[int, int]],
    information_spans: Sequence[tuple[int, int]],
    attention_mask: Any = None,
) -> dict[str, int]:
    policy_mask, information_mask = build_grpo_masks(
        sequence_length,
        assistant_spans,
        information_spans,
        attention_mask,
    )
    information_token_train_mask_sum = int(
        (policy_mask * information_mask).sum().item()
    )
    if information_token_train_mask_sum != 0:
        raise RuntimeError("fixture Information tokens leaked into Policy Mask")
    if int(information_mask.sum().item()) <= 0:
        raise RuntimeError("fixture has no Information tokens")
    if int(policy_mask.sum().item()) <= 0:
        raise RuntimeError("fixture has no assistant action tokens")
    return {
        "information_token_train_mask_sum": information_token_train_mask_sum,
        "information_token_count": int(information_mask.sum().item()),
        "policy_action_token_count": int(policy_mask.sum().item()),
    }


def run_text_search_path_integration(
    *, project_root: Path, output_dir: Path
) -> dict[str, Any]:
    query = parse_text_search_fixture()
    cache = ImageSearchCache.load(
        project_root
        / "data/raw/fvqa/fvqa_train_image_search_results_cache.pkl",
        label="fvqa_train_official_cache",
    )
    environment = CachedSandbox(cache, image_top_k=3, text_top_k=3)
    first = environment.text_search(query)
    second = environment.text_search(query)
    if first.cache_miss:
        raise RuntimeError("Text Search fixture produced Cache Miss")
    if first.text != second.text or first.provenance != second.provenance:
        raise RuntimeError("BM25 Text Search fixture is not deterministic")
    document_ids = list((first.provenance or {}).get("document_ids", []))
    if len(document_ids) != 3:
        raise RuntimeError(
            f"Text Search fixture expected deterministic top-3, got {len(document_ids)}"
        )
    if not (
        first.text.startswith("<information>")
        and first.text.endswith("</information>")
        and "[Text Search Results]" in first.text
    ):
        raise RuntimeError("Text Search fixture Information format is invalid")

    rows = [
        json.loads(line)
        for line in (
            project_root
            / "data/processed/grpo_prompt_pool_v1/train.jsonl"
        ).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise RuntimeError("GRPO Train Pool is empty")
    prompt = PromptPoolItem.from_dict(rows[0])
    config = load_config(
        project_root / "configs/protocol_sft/train_full_format_v1.yaml",
        project_root=project_root,
    )
    processor = load_processor(config)
    renderer = ProtocolRenderer(processor)
    image_store = ImageStore(
        project_root / "data/raw/fvqa/fvqa_train.parquet"
    )
    image = image_store.load(
        SimpleNamespace(
            image_refs=[prompt.image_ref],
            source={
                "source_row_index": prompt.metadata.get(
                    "row_index", prompt.image_ref.get("row_index")
                )
            },
            source_data_id=prompt.source_data_id,
            example_id=prompt.prompt_uid,
        )
    )
    messages = [
        renderer._message(Message(role="system", content=SYSTEM_PROMPT)),
        renderer._message(
            Message(
                role="user",
                content=f"<image>\nQuestion: {prompt.question}",
            )
        ),
        renderer._message(Message(role="assistant", content=FIXTURE_ACTION)),
        renderer._message(Message(role="tool", content=first.text)),
        renderer._message(
            Message(role="assistant", content=FIXTURE_SUBSEQUENT_ACTION)
        ),
    ]
    subsequent = parse_action(FIXTURE_SUBSEQUENT_ACTION)
    if not subsequent.valid or subsequent.action_type != ActionType.ANSWER:
        raise RuntimeError("fixture subsequent-turn protocol is invalid")
    prefix_batch = _encode(
        processor, messages[:-1], image, add_generation_prompt=True
    )
    final_batch, assistant_spans, information_spans = _spans_and_batch(
        processor, messages, image
    )
    sequence_length = int(final_batch["input_ids"].shape[-1])
    prefix_length = int(prefix_batch["input_ids"].shape[-1])
    if sequence_length > 1536 or prefix_length >= 1536:
        raise RuntimeError("Text Search integration context exceeds 1536 tokens")
    if len(assistant_spans) != 2 or len(information_spans) != 1:
        raise RuntimeError("Text Search integration context spans are incomplete")
    if assistant_spans[-1][1] > sequence_length:
        raise RuntimeError("fixture subsequent target was truncated")
    attention = final_batch.get(
        "attention_mask", torch.ones_like(final_batch["input_ids"])
    )[0]
    mask_contract = validate_fixture_masks(
        sequence_length=sequence_length,
        assistant_spans=assistant_spans,
        information_spans=information_spans,
        attention_mask=attention,
    )
    processor_hash = hashlib.sha256(
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
    report = {
        "schema_version": "grpo-text-search-path-integration-v1",
        "stage": "text_search_path_integration",
        "passed": True,
        "marker": "GRPO_TEXT_SEARCH_PATH_INTEGRATION_PASS",
        "fixture_action": FIXTURE_ACTION,
        "fixture_query": query,
        "parser_contract_passed": True,
        "dispatcher_contract_passed": True,
        "bm25_deterministic": True,
        "bm25_top_k": len(document_ids),
        "bm25_document_ids": document_ids,
        "information_format_contract_passed": True,
        "context_assembly_contract_passed": True,
        "information_mask_contract_passed": True,
        "subsequent_turn_contract_passed": True,
        "sequence_length": sequence_length,
        "max_sequence_length": 1536,
        "target_truncation_count": 0,
        "processor_hash": processor_hash,
        "chat_template_hash": renderer.chat_template_sha256,
        "test_accessed": False,
        "mmsearch_accessed": False,
        **mask_contract,
        **fixture_boundary_manifest(),
        **formal_chain_manifest_fields(),
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "run_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "information.txt").write_text(
        first.text + "\n", encoding="utf-8"
    )
    return report
