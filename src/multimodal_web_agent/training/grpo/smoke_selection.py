from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence
import unicodedata

from multimodal_web_agent.data.protocol_sft.relation_mapper import (
    map_question_relation,
)

from .schema import PromptPoolItem


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def select_prompts(
    rows: Sequence[Mapping[str, Any]],
    *,
    search_free: int,
    search_required: int,
    priority_source_ranks: Mapping[str, int] | None = None,
) -> list[PromptPoolItem]:
    items = [PromptPoolItem.from_dict(row) for row in rows]
    priority_source_ranks = priority_source_ranks or {}
    default_rank = (
        max(priority_source_ranks.values(), default=-1) + 1
        if priority_source_ranks
        else 0
    )

    def category_items(category: str, count: int) -> list[PromptPoolItem]:
        candidates = [item for item in items if item.category == category]
        candidates.sort(
            key=lambda item: (
                priority_source_ranks.get(
                    item.source_data_id, default_rank
                ),
                item.prompt_uid,
            )
        )
        return candidates[:count]

    selected = (
        category_items("search_free", search_free)
        + category_items("search_required", search_required)
    )
    if sum(item.category == "search_free" for item in selected) != search_free:
        raise RuntimeError("insufficient search_free prompts for stage")
    if (
        sum(item.category == "search_required" for item in selected)
        != search_required
    ):
        raise RuntimeError("insufficient search_required prompts for stage")
    return selected


def normalize_probe_question(value: str) -> str:
    return " ".join(
        unicodedata.normalize("NFKC", str(value)).casefold().split()
    )


def _sft_user_question(row: Mapping[str, Any]) -> str:
    for message in row.get("state", []):
        if not isinstance(message, Mapping) or message.get("role") != "user":
            continue
        content = message.get("content", "")
        if isinstance(content, Sequence) and not isinstance(content, str):
            content = " ".join(
                str(part.get("text", ""))
                for part in content
                if isinstance(part, Mapping)
            )
        value = str(content).strip()
        marker = "Question:"
        if marker in value:
            value = value.split(marker, 1)[1].strip()
        if value:
            return value
    return str(row.get("question", "")).strip()


def frozen_sft_text_search_probe_signatures(
    project_root: Path,
) -> dict[str, set[str]]:
    """Read route signatures exclusively from frozen SFT Train."""
    path = (
        project_root
        / "data/processed/protocol_format_sft_v1/train.jsonl"
    )
    if not path.is_file():
        return {"source_ids": set(), "questions": set(), "relations": set()}
    source_ids: set[str] = set()
    questions: set[str] = set()
    relations: set[str] = set()
    for row in _read_jsonl(path):
        if str(row.get("target_action_type", "")).casefold() != "text_search":
            continue
        source_data_id = str(row.get("source_data_id", "")).strip()
        if source_data_id:
            source_ids.add(source_data_id)
        question = _sft_user_question(row)
        normalized_question = normalize_probe_question(question)
        if normalized_question:
            questions.add(normalized_question)
        relation = map_question_relation(question)
        if relation.matched:
            relations.add(relation.relation)
    return {
        "source_ids": source_ids,
        "questions": questions,
        "relations": relations,
    }


def select_smoke_prompts(
    rows: Sequence[Mapping[str, Any]],
    *,
    project_root: Path,
) -> tuple[list[PromptPoolItem], dict[str, int]]:
    # The frozen SFT view has only nine Text Search routes.  Exact sources can
    # be absent from GRPO Train because Prompt Pool group isolation deliberately
    # assigns different groups.  Rank legal GRPO Train prompts by frozen
    # SFT-train signatures: exact source, exact question template, then mapped
    # relation family.  Dev/test files are never opened.
    signatures = frozen_sft_text_search_probe_signatures(project_root)
    source_ids = signatures["source_ids"]
    questions = signatures["questions"]
    relations = signatures["relations"]

    def rank(prompt: PromptPoolItem) -> int:
        if prompt.source_data_id in source_ids:
            return 0
        if normalize_probe_question(prompt.question) in questions:
            return 1
        relation = map_question_relation(prompt.question)
        if relation.matched and relation.relation in relations:
            return 2
        return 3

    items = [PromptPoolItem.from_dict(row) for row in rows]
    source_ranks = {
        item.source_data_id: rank(item)
        for item in items
    }
    prompts = select_prompts(
        rows,
        search_free=12,
        search_required=20,
        priority_source_ranks=source_ranks,
    )
    # Put strongest route probes before any optimizer update.
    prompts.sort(
        key=lambda prompt: (
            source_ranks[prompt.source_data_id],
            prompt.prompt_uid,
        )
    )
    selected_ranks = [
        source_ranks[prompt.source_data_id] for prompt in prompts
    ]
    diagnostics = {
        "source_id": selected_ranks.count(0),
        "exact_question": selected_ranks.count(1),
        "relation_family": selected_ranks.count(2),
        "total": sum(value < 3 for value in selected_ranks),
    }
    if diagnostics["total"] < 4:
        raise RuntimeError(
            "Smoke selection requires at least four frozen SFT-train "
            "text-search route signature matches in GRPO Train; "
            f"found {diagnostics['total']}"
        )
    return prompts, diagnostics
