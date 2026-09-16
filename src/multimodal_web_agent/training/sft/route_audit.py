from __future__ import annotations

import html
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

from multimodal_web_agent.agent import ActionType, parse_action
from multimodal_web_agent.data.protocol_sft.schema import StateActionExample


INITIAL_TRANSITIONS = (
    "initial_to_direct_answer",
    "initial_to_image_search",
    "initial_to_text_search",
)
GENERIC_VISUAL_WORDS = frozenset({
    "a", "an", "the", "this", "image", "photo", "picture", "shown",
    "depicted", "visible", "object", "person", "character", "building",
    "government", "logo", "emblem", "located", "location", "name", "type",
    "event", "thing", "identify", "identification", "of", "in", "on", "at",
    "is", "are", "was", "were", "be", "being", "who", "what", "which",
    "where", "when", "how", "does", "do", "did", "can", "could", "please",
})
ENTITY_TYPE_WORDS = frozenset({
    "character", "object", "person", "building", "government", "logo",
    "emblem", "event", "animal", "vehicle", "plant", "landmark", "weapon",
    "instrument", "food", "flag", "monument",
})
IMAGE_WORDS = frozenset({"image", "photo", "picture"})
TOKEN_RE = re.compile(r"[^\W_]+", flags=re.UNICODE)


class InitialRouteLabelRepairRequired(RuntimeError):
    pass


def normalize_question(text: str) -> str:
    value = unicodedata.normalize("NFKC", html.unescape(str(text))).casefold()
    tokens = TOKEN_RE.findall(value)
    normalized = []
    for token in tokens:
        if token in IMAGE_WORDS:
            token = "image"
        if token in {"this", "the", "a", "an"}:
            token = "the"
        normalized.append(token)
    # Articles directly before image do not add routing information.
    collapsed = [
        token
        for index, token in enumerate(normalized)
        if not (
            token == "the"
            and index + 1 < len(normalized)
            and normalized[index + 1] == "image"
        )
    ]
    return " ".join(collapsed)


def normalize_question_template(text: str) -> str:
    tokens = normalize_question(text).split()
    return " ".join(
        "[entity_type]" if token in ENTITY_TYPE_WORDS else token
        for token in tokens
    )


def token_jaccard(left: str | Iterable[str], right: str | Iterable[str]) -> float:
    left_tokens = set(left.split() if isinstance(left, str) else left)
    right_tokens = set(right.split() if isinstance(right, str) else right)
    union = left_tokens | right_tokens
    return len(left_tokens & right_tokens) / len(union) if union else 1.0


def question_for_example(example: StateActionExample) -> str:
    fallback = ""
    for message in example.state:
        if message.role != "user":
            continue
        value = message.content.replace("<image>", "", 1).strip()
        fallback = fallback or value
        if "<image>" in message.content:
            return value
    return fallback


def image_path_for_example(example: StateActionExample) -> str:
    for reference in example.image_refs:
        if reference.get("path"):
            return str(reference["path"]).replace("\\", "/")
    row = example.source.get("source_row_index")
    return f"source_parquet#row={row}" if row is not None else ""


def _proper_entity_tokens(question: str) -> set[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9_-]*|\d+", html.unescape(question))
    entities = set()
    for index, word in enumerate(words):
        normalized = word.casefold()
        if normalized.isdigit():
            entities.add(normalized)
        elif (
            index > 0
            and word[:1].isupper()
            and normalized not in GENERIC_VISUAL_WORDS
        ):
            entities.add(normalized)
    return entities


def weak_initial_text_query(
    question: str,
    query: str,
) -> Dict[str, Any]:
    normalized_question = normalize_question(question)
    normalized_query = normalize_question(query)
    question_tokens = normalized_question.split()
    query_tokens = normalized_query.split()
    substantive = [
        token for token in query_tokens
        if token not in GENERIC_VISUAL_WORDS
    ]
    explicit_entities = _proper_entity_tokens(question)
    query_token_set = set(query_tokens)
    entity_missing = bool(explicit_entities - query_token_set)
    copy_rate = (
        len(set(query_tokens) & set(question_tokens))
        / len(set(query_tokens))
        if query_tokens else 0.0
    )
    query_question_copy = bool(
        query_tokens
        and copy_rate >= 0.80
        and set(query_tokens).issubset(set(question_tokens))
    )
    generic_only = not substantive
    flags = []
    if generic_only:
        flags.append("generic_visual_reference_only")
    if entity_missing:
        flags.append("visible_entity_missing")
    if query_question_copy:
        flags.append("query_question_copy")
    if generic_only and query_question_copy:
        flags.append("weak_initial_text_query")
    return {
        "generic_visual_reference_only": generic_only,
        "initial_text_visible_entity_missing": entity_missing,
        "initial_text_query_question_copy": query_question_copy,
        "query_question_copy_score": copy_rate,
        "query_tokens": query_tokens,
        "substantive_query_tokens": substantive,
        "explicit_question_entity_tokens": sorted(explicit_entities),
        "automatic_flags": flags,
    }


def _initial_rows(
    split_examples: Mapping[str, Sequence[StateActionExample]],
) -> list[Dict[str, Any]]:
    rows = []
    for split in ("train", "dev"):
        for example in split_examples[split]:
            if example.transition not in INITIAL_TRANSITIONS:
                continue
            parsed = parse_action(example.target)
            if not parsed.valid:
                raise ValueError(f"invalid target: {example.example_id}")
            question = question_for_example(example)
            rows.append({
                "sample_id": example.example_id,
                "trajectory_id": example.trajectory_id,
                "split": split,
                "route": example.route,
                "transition": example.transition,
                "image_path": image_path_for_example(example),
                "question": question,
                "target_reason": parsed.reason,
                "target_query": (
                    parsed.content
                    if parsed.action_type == ActionType.TEXT_SEARCH
                    else None
                ),
                "normalized_question": normalize_question(question),
                "normalized_question_template": (
                    normalize_question_template(question)
                ),
            })
    return rows


def build_initial_route_audit(
    train_examples: Sequence[StateActionExample],
    dev_examples: Sequence[StateActionExample],
) -> Dict[str, Any]:
    rows = _initial_rows({"train": train_examples, "dev": dev_examples})
    by_normalized: Dict[str, list[Dict[str, Any]]] = defaultdict(list)
    by_template: Dict[str, list[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_normalized[row["normalized_question"]].append(row)
        by_template[row["normalized_question_template"]].append(row)

    def collisions(
        groups: Mapping[str, Sequence[Mapping[str, Any]]],
        key_name: str,
    ) -> list[Dict[str, Any]]:
        result = []
        for key, values in sorted(groups.items()):
            routes = sorted({str(value["transition"]) for value in values})
            if len(routes) <= 1:
                continue
            result.append({
                key_name: key,
                "transitions": routes,
                "examples": [
                    {
                        "sample_id": value["sample_id"],
                        "split": value["split"],
                        "transition": value["transition"],
                        "question": value["question"],
                    }
                    for value in values
                ],
            })
        return result

    exact = collisions(by_normalized, "normalized_question")
    templates = collisions(
        by_template, "normalized_question_template"
    )
    cross_route_pairs = []
    nearest: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for left_index, left in enumerate(rows):
        for right in rows[left_index + 1:]:
            if left["transition"] == right["transition"]:
                continue
            similarity = token_jaccard(
                left["normalized_question"],
                right["normalized_question"],
            )
            if similarity >= 0.80:
                cross_route_pairs.append({
                    "left_sample_id": left["sample_id"],
                    "right_sample_id": right["sample_id"],
                    "left_split": left["split"],
                    "right_split": right["split"],
                    "left_transition": left["transition"],
                    "right_transition": right["transition"],
                    "left_question": left["question"],
                    "right_question": right["question"],
                    "jaccard": similarity,
                })
            for source, candidate in ((left, right), (right, left)):
                if source["transition"] != "initial_to_text_search":
                    continue
                if candidate["transition"] not in {
                    "initial_to_direct_answer",
                    "initial_to_image_search",
                }:
                    continue
                prior = nearest[source["sample_id"]].get(
                    candidate["transition"]
                )
                tie_key = (-similarity, candidate["sample_id"])
                if prior is None or tie_key < (
                    -prior["jaccard"], prior["sample_id"]
                ):
                    nearest[source["sample_id"]][candidate["transition"]] = {
                        "sample_id": candidate["sample_id"],
                        "split": candidate["split"],
                        "transition": candidate["transition"],
                        "question": candidate["question"],
                        "jaccard": similarity,
                    }
    initial_text_rows = []
    for row in rows:
        if row["transition"] != "initial_to_text_search":
            continue
        weak = weak_initial_text_query(
            row["question"], str(row["target_query"] or "")
        )
        initial_text_rows.append({
            **row,
            **weak,
            "nearest_cross_route_examples": [
                nearest[row["sample_id"]][transition]
                for transition in (
                    "initial_to_direct_answer",
                    "initial_to_image_search",
                )
                if transition in nearest[row["sample_id"]]
            ],
        })
    weak_rows = [
        row for row in initial_text_rows
        if row["automatic_flags"]
    ]
    nearest_rows = [
        {
            "sample_id": row["sample_id"],
            "split": row["split"],
            "question": row["question"],
            "nearest_cross_route_examples": row[
                "nearest_cross_route_examples"
            ],
        }
        for row in initial_text_rows
    ]
    return {
        "summary": {
            "train_examples_checked": len(train_examples),
            "dev_examples_checked": len(dev_examples),
            "initial_examples_checked": len(rows),
            "initial_text_train_count": sum(
                row["split"] == "train" for row in initial_text_rows
            ),
            "initial_text_dev_count": sum(
                row["split"] == "dev" for row in initial_text_rows
            ),
            "initial_text_total_count": len(initial_text_rows),
            "exact_normalized_question_cross_route_count": len(exact),
            "normalized_template_cross_route_count": len(templates),
            "high_similarity_cross_route_pair_count": len(cross_route_pairs),
            "initial_text_generic_query_count": sum(
                row["generic_visual_reference_only"]
                for row in initial_text_rows
            ),
            "initial_text_visible_entity_missing_count": sum(
                row["initial_text_visible_entity_missing"]
                for row in initial_text_rows
            ),
            "initial_text_query_question_copy_rate": (
                sum(
                    row["initial_text_query_question_copy"]
                    for row in initial_text_rows
                )
                / len(initial_text_rows)
                if initial_text_rows else 0.0
            ),
            "test_split_used": False,
        },
        "normalized_collisions": exact,
        "template_collisions": templates,
        "cross_route_nearest_neighbors": nearest_rows,
        "high_similarity_cross_route_pairs": cross_route_pairs,
        "weak_initial_text_queries": weak_rows,
        "all_initial_text": initial_text_rows,
    }


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(dict(row), ensure_ascii=False, sort_keys=True)
                + "\n"
            )


def validate_initial_route_approval(
    path: Path,
    *,
    dataset_manifest_hash: str,
    expected_reviewed_count: int,
) -> Dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Initial Route approval file is missing: {path}"
        )
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("dataset_schema") != "protocol-sft-v0.3":
        raise ValueError("Initial Route approval dataset schema mismatch")
    if value.get("dataset_manifest_hash") != dataset_manifest_hash:
        raise ValueError("Initial Route approval dataset Manifest hash mismatch")
    if int(value.get("confirmed_label_conflict_count", 0)) > 0:
        raise InitialRouteLabelRepairRequired(
            "INITIAL_ROUTE_LABEL_REPAIR_REQUIRED"
        )
    if value.get("approved") is not True:
        raise ValueError("Initial Route audit has not been approved")
    if int(value.get("reviewed_initial_text_count", -1)) != int(
        expected_reviewed_count
    ):
        raise ValueError(
            "Initial Route reviewed count does not cover all Train+Dev "
            "Initial Text samples"
        )
    if not str(value.get("reviewer_notes", "")).strip():
        raise ValueError("Initial Route approval requires reviewer_notes")
    return dict(value)
