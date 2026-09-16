from __future__ import annotations

import hashlib
import io
import json
import math
from collections import Counter, defaultdict
from html import unescape
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from multimodal_web_agent.agent import ActionType, parse_action

from .context_leak import detect_unavailable_context_leak
from .entity_extractor import entity_visible_in_information
from .information_formatter import FORBIDDEN_VISUAL_PLACEHOLDERS
from .query_builder import build_image_context_query, contains_answer_leak, query_tokens
from .relation_mapper import map_question_relation
from .rejection import rejection_distributions
from .schema import StateActionExample, TransitionType, validate_example_dict, validate_trajectory_dict
from .verifier import information_supports_answer


class AuditFailure(RuntimeError):
    pass


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise AuditFailure("%s:%d is not a JSON object" % (path, line_number))
            records.append(value)
    return records


def _count_information(messages: Sequence[Any]) -> int:
    return sum(
        message.content.count("<information>")
        for message in messages
        if message.role == "tool"
    )


def audit_records(
    trajectories_raw: Sequence[Mapping[str, Any]],
    examples_by_split_raw: Mapping[str, Sequence[Mapping[str, Any]]],
    expected_counts: Optional[Mapping[str, Any]] = None,
    min_unique_reasons: int = 1,
    reason_min_tokens: int = 8,
    reason_max_tokens: int = 48,
) -> Dict[str, Any]:
    errors: List[str] = []
    trajectories = []
    for index, raw in enumerate(trajectories_raw):
        try:
            trajectories.append(validate_trajectory_dict(raw))
        except Exception as exc:
            errors.append("trajectory[%d] schema: %s" % (index, exc))
    examples: List[StateActionExample] = []
    for split_name, raw_examples in examples_by_split_raw.items():
        for index, raw in enumerate(raw_examples):
            try:
                example = validate_example_dict(raw)
                examples.append(example)
                if example.split != split_name:
                    errors.append("%s[%d] carries split=%s" % (split_name, index, example.split))
            except Exception as exc:
                errors.append("%s[%d] schema: %s" % (split_name, index, exc))

    transition_counts = Counter(example.transition for example in examples)
    route_counts = Counter(trajectory.route for trajectory in trajectories)
    split_counts = Counter(example.split for example in examples)
    source_splits: Dict[str, set] = defaultdict(set)
    trajectory_splits: Dict[str, set] = defaultdict(set)
    source_routes: Dict[str, set] = defaultdict(set)
    examples_by_trajectory: Dict[str, List[StateActionExample]] = defaultdict(list)
    target_error_counts = Counter()
    query_leaks = []
    unavailable_context_leaks = []
    cache_provenance_failures = []
    forbidden_visual_placeholder_count = 0
    reasons_by_transition: Dict[str, Counter[str]] = defaultdict(Counter)
    reason_length_failures = []
    reason_answer_leaks = []
    chain_failures = []
    entity_not_visible = []
    image_answer_present_before_text = []
    text_answer_evidence_missing = []
    text_evidence_same_as_image_context_only = []
    new_image_answer_evidence_missing = []
    new_image_answer_top1_hits = 0
    new_image_answer_top3_hits = 0

    for example in examples:
        source_splits[example.source_data_id].add(example.split)
        source_routes[example.source_data_id].add(example.route)
        trajectory_splits[example.trajectory_id].add(example.split)
        examples_by_trajectory[example.trajectory_id].append(example)
        parsed = parse_action(example.target)
        if not parsed.valid:
            target_error_counts[str(parsed.error_code)] += 1
        if parsed.valid and parsed.action_type == ActionType.TEXT_SEARCH:
            if contains_answer_leak(parsed.content or "", example.accepted_answers):
                query_leaks.append(example.example_id)
        if parsed.valid and parsed.reason:
            reasons_by_transition[example.transition][parsed.reason] += 1
            reason_token_count = len(query_tokens(parsed.reason))
            if not reason_min_tokens <= reason_token_count <= min(reason_max_tokens, 48):
                reason_length_failures.append(example.example_id)
            if contains_answer_leak(parsed.reason, example.accepted_answers):
                reason_answer_leaks.append(example.example_id)
        forbidden_visual_placeholder_count += sum(
            message.content.count(token)
            for message in example.state
            for token in FORBIDDEN_VISUAL_PLACEHOLDERS
        )
        forbidden_visual_placeholder_count += sum(
            example.target.count(token) for token in FORBIDDEN_VISUAL_PLACEHOLDERS
        )
        transition = TransitionType(example.transition)
        information_count = _count_information(example.state)
        expected_information_count = {
            TransitionType.INITIAL_TO_DIRECT_ANSWER: 0,
            TransitionType.INITIAL_TO_IMAGE_SEARCH: 0,
            TransitionType.INITIAL_TO_TEXT_SEARCH: 0,
            TransitionType.IMAGE_INFORMATION_TO_ANSWER: 1,
            TransitionType.IMAGE_INFORMATION_TO_TEXT_SEARCH: 1,
            TransitionType.TEXT_INFORMATION_TO_ANSWER: (
                2 if example.route == "image_text_search_answer" else 1
            ),
        }[transition]
        post_search = expected_information_count > 0
        if information_count != expected_information_count:
            errors.append(
                "%s must contain exactly %d environment information block(s)"
                % (example.example_id, expected_information_count)
            )
        if not post_search and information_count:
            errors.append("%s initial state unexpectedly contains information" % example.example_id)
        if post_search:
            provenance = example.information_provenance or {}
            required = {"backend", "cache_source", "cache_version", "cache_file_sha256", "online_access"}
            if not required.issubset(provenance) or provenance.get("online_access") is not False:
                cache_provenance_failures.append(example.example_id)

    split_leaks = sorted(source for source, splits in source_splits.items() if len(splits) > 1)
    trajectory_split_leaks = sorted(
        trajectory for trajectory, splits in trajectory_splits.items() if len(splits) > 1
    )
    route_reuse = sorted(source for source, routes in source_routes.items() if len(routes) > 1)
    pair_failures = []
    trajectory_lookup = {trajectory.trajectory_id: trajectory for trajectory in trajectories}
    for trajectory_id, trajectory in trajectory_lookup.items():
        paired = sorted(examples_by_trajectory.get(trajectory_id, []), key=lambda item: item.example_id)
        expected_steps = len(trajectory.steps)
        if len(paired) != expected_steps:
            pair_failures.append(trajectory_id + ":wrong_step_count")
            if trajectory.route == "image_text_search_answer":
                chain_failures.append(trajectory_id + ":wrong_example_count")
            continue
        if len({item.split for item in paired}) != 1:
            pair_failures.append(trajectory_id + ":split_mismatch")
        for step_index, example in enumerate(paired):
            step = trajectory.steps[step_index]
            if example.transition != step.transition or example.target != step.target:
                pair_failures.append(trajectory_id + ":step_mismatch")
                break
        if expected_steps == 2:
            search_target = trajectory.steps[0].target
            assistant_history = [
                message.content
                for message in trajectory.steps[1].state
                if message.role == "assistant"
            ]
            if assistant_history != [search_target]:
                pair_failures.append(trajectory_id + ":search_history_mismatch")
        if (
            trajectory.route == "image_search_answer"
            and trajectory.source.get("v0_3_origin")
            == "new_image_search_answer"
            and expected_steps == 2
        ):
            image_blocks = [
                message.content
                for message in trajectory.steps[1].state
                if message.role == "tool"
            ]
            if len(image_blocks) != 1 or not information_supports_answer(
                image_blocks[0], trajectory.accepted_answers
            ):
                new_image_answer_evidence_missing.append(trajectory_id)
            provenance = trajectory.steps[1].information_provenance or {}
            support_ranks = {
                int(rank)
                for rank in provenance.get(
                    "answer_evidence_support_ranks", []
                )
            }
            if 1 in support_ranks:
                new_image_answer_top1_hits += 1
            if support_ranks.intersection({1, 2, 3}):
                new_image_answer_top3_hits += 1
        if trajectory.route in {"text_search", "text_search_answer"} and expected_steps == 2:
            parsed_query = parse_action(trajectory.steps[0].target)
            information = "\n".join(
                message.content
                for message in trajectory.steps[1].state
                if message.role == "tool"
            )
            context_result = detect_unavailable_context_leak(
                parsed_query.content or "",
                visible_texts=[trajectory.question],
                unavailable_texts=[("future_text_information", information)],
            )
            provenance = trajectory.steps[1].information_provenance or {}
            if context_result.leaked or provenance.get("unavailable_context_leak") is True:
                unavailable_context_leaks.append(trajectory_id)
        if trajectory.route == "image_text_search_answer":
            if expected_steps != 3:
                chain_failures.append(trajectory_id + ":wrong_step_count")
                continue
            first_target = trajectory.steps[0].target
            second_target = trajectory.steps[1].target
            first_history = [
                message.content
                for message in trajectory.steps[1].state
                if message.role == "assistant"
            ]
            final_history = [
                message.content
                for message in trajectory.steps[2].state
                if message.role == "assistant"
            ]
            if first_history != [first_target]:
                chain_failures.append(trajectory_id + ":step1_history")
            if final_history != [first_target, second_target]:
                chain_failures.append(trajectory_id + ":step2_history")
            image_blocks = [
                message.content
                for message in trajectory.steps[1].state
                if message.role == "tool"
            ]
            final_blocks = [
                message.content
                for message in trajectory.steps[2].state
                if message.role == "tool"
            ]
            if len(image_blocks) != 1 or len(final_blocks) != 2:
                chain_failures.append(trajectory_id + ":information_blocks")
                continue
            if image_blocks[0] != final_blocks[0]:
                chain_failures.append(trajectory_id + ":image_information_changed")
            if any(
                step.image_refs[:1] != trajectory.steps[0].image_refs[:1]
                for step in trajectory.steps[1:]
            ):
                chain_failures.append(trajectory_id + ":image_reference_changed")
            parsed_query = parse_action(second_target)
            image_provenance = trajectory.steps[1].information_provenance or {}
            text_provenance = trajectory.steps[2].information_provenance or {}
            entity = str(
                text_provenance.get(
                    "identified_entity",
                    image_provenance.get("identified_entity", ""),
                )
            )
            query_provenance = text_provenance.get("query_provenance", {})
            if query_provenance != {
                "relation_source": "question",
                "entity_source": "visible_image_information",
                "used_ground_truth": False,
                "used_future_text_result": False,
            }:
                chain_failures.append(trajectory_id + ":query_provenance")
            mapped_relation = map_question_relation(trajectory.question)
            if (
                not mapped_relation.matched
                or mapped_relation.relation != text_provenance.get("relation")
                or mapped_relation.query_prefix
                != text_provenance.get("relation_query_prefix")
            ):
                chain_failures.append(trajectory_id + ":relation_mismatch")
            try:
                expected_query = build_image_context_query(
                    trajectory.question, entity, mapped_relation
                )
            except ValueError:
                expected_query = ""
            if unescape(parsed_query.content or "") != expected_query:
                chain_failures.append(trajectory_id + ":query_construction_mismatch")
            if (
                not entity
                or not entity_visible_in_information(entity, image_blocks[0])
                or image_provenance.get("entity_visible_in_image_information") is not True
                or image_provenance.get("entity_source_doc_id")
                not in set(image_provenance.get("image_context_document_ids", []))
            ):
                entity_not_visible.append(trajectory_id)
            if not entity_visible_in_information(
                entity, parsed_query.content or ""
            ):
                chain_failures.append(trajectory_id + ":query_entity_mismatch")
            if information_supports_answer(
                image_blocks[0], trajectory.accepted_answers
            ):
                image_answer_present_before_text.append(trajectory_id)
            if not information_supports_answer(
                final_blocks[1], trajectory.accepted_answers
            ):
                text_answer_evidence_missing.append(trajectory_id)
            image_document_ids = set(
                text_provenance.get(
                    "image_context_document_ids",
                    image_provenance.get("image_context_document_ids", []),
                )
            )
            evidence_document_ids = set(
                text_provenance.get("answer_evidence_document_ids", [])
            )
            if image_document_ids != set(
                image_provenance.get("image_context_document_ids", [])
            ):
                chain_failures.append(trajectory_id + ":image_document_ids")
            if list(text_provenance.get("document_ids", [])) != list(
                text_provenance.get("text_result_document_ids", [])
            ):
                chain_failures.append(trajectory_id + ":text_document_ids")
            if (
                not evidence_document_ids
                or evidence_document_ids.issubset(image_document_ids)
                or text_provenance.get(
                    "text_evidence_hit_excluding_image_context_docs"
                ) is not True
            ):
                text_evidence_same_as_image_context_only.append(trajectory_id)
            context_result = detect_unavailable_context_leak(
                parsed_query.content or "",
                visible_texts=[
                    trajectory.question,
                    image_blocks[0],
                    "%s %s"
                    % (
                        text_provenance.get("relation_query_prefix", ""),
                        entity,
                    ),
                ],
                unavailable_texts=[("future_text_information", final_blocks[1])],
            )
            if context_result.leaked or text_provenance.get(
                "unavailable_context_leak"
            ) is True:
                unavailable_context_leaks.append(trajectory_id)

    route_split_trajectory_counts: Dict[str, Counter[str]] = defaultdict(Counter)
    for trajectory_id, trajectory in trajectory_lookup.items():
        splits = trajectory_splits.get(trajectory_id, set())
        if len(splits) == 1:
            route_split_trajectory_counts[trajectory.route][next(iter(splits))] += 1

    if split_leaks:
        errors.append("source split leaks: %s" % split_leaks[:10])
    if trajectory_split_leaks:
        errors.append("trajectory split leaks: %s" % trajectory_split_leaks[:10])
    if route_reuse:
        errors.append("source route reuse: %s" % route_reuse[:10])
    if query_leaks:
        errors.append("query answer leaks: %s" % query_leaks[:10])
    if unavailable_context_leaks:
        errors.append(
            "unavailable context leaks: %s" % unavailable_context_leaks[:10]
        )
    if forbidden_visual_placeholder_count:
        errors.append(
            "forbidden visual placeholders: %d" % forbidden_visual_placeholder_count
        )
    if cache_provenance_failures:
        errors.append("cache provenance failures: %s" % cache_provenance_failures[:10])
    if pair_failures:
        errors.append("trajectory pair failures: %s" % pair_failures[:10])
    if reason_length_failures:
        errors.append("reason length failures: %s" % reason_length_failures[:10])
    if reason_answer_leaks:
        errors.append("reason answer leaks: %s" % reason_answer_leaks[:10])
    if chain_failures:
        errors.append("trajectory chain failures: %s" % chain_failures[:10])
    if entity_not_visible:
        errors.append("image entities not visible: %s" % entity_not_visible[:10])
    if image_answer_present_before_text:
        errors.append(
            "answers present before text search: %s"
            % image_answer_present_before_text[:10]
        )
    if text_answer_evidence_missing:
        errors.append(
            "text answer evidence missing: %s"
            % text_answer_evidence_missing[:10]
        )
    if text_evidence_same_as_image_context_only:
        errors.append(
            "text evidence is not independent: %s"
            % text_evidence_same_as_image_context_only[:10]
        )
    if new_image_answer_evidence_missing:
        errors.append(
            "new image answer evidence missing: %s"
            % new_image_answer_evidence_missing[:10]
        )

    present_routes = {trajectory.route for trajectory in trajectories}
    present_transition_values = set()
    route_transitions = {
        "direct_answer": {TransitionType.INITIAL_TO_DIRECT_ANSWER.value},
        "image_search": {
            TransitionType.INITIAL_TO_IMAGE_SEARCH.value,
            TransitionType.IMAGE_INFORMATION_TO_ANSWER.value,
        },
        "image_search_answer": {
            TransitionType.INITIAL_TO_IMAGE_SEARCH.value,
            TransitionType.IMAGE_INFORMATION_TO_ANSWER.value,
        },
        "text_search": {
            TransitionType.INITIAL_TO_TEXT_SEARCH.value,
            TransitionType.TEXT_INFORMATION_TO_ANSWER.value,
        },
        "text_search_answer": {
            TransitionType.INITIAL_TO_TEXT_SEARCH.value,
            TransitionType.TEXT_INFORMATION_TO_ANSWER.value,
        },
        "image_text_search_answer": {
            TransitionType.INITIAL_TO_IMAGE_SEARCH.value,
            TransitionType.IMAGE_INFORMATION_TO_TEXT_SEARCH.value,
            TransitionType.TEXT_INFORMATION_TO_ANSWER.value,
        },
    }
    for route in present_routes:
        present_transition_values.update(route_transitions.get(route, set()))
    audited_transitions = [
        transition
        for transition in TransitionType
        if transition.value in present_transition_values
    ]
    unique_reason_count = {
        transition.value: len(reasons_by_transition[transition.value])
        for transition in audited_transitions
    }
    max_reason_frequency = {
        transition.value: (
            max(reasons_by_transition[transition.value].values())
            if reasons_by_transition[transition.value]
            else 0
        )
        for transition in audited_transitions
    }
    reason_entropy = {}
    for transition in audited_transitions:
        counts = reasons_by_transition[transition.value]
        total = sum(counts.values())
        reason_entropy[transition.value] = round(
            -sum(
                (count / total) * math.log2(count / total)
                for count in counts.values()
            ),
            6,
        ) if total else 0.0
    low_diversity = {
        transition: count
        for transition, count in unique_reason_count.items()
        if count < min_unique_reasons
    }
    if low_diversity:
        errors.append("insufficient reason diversity: %s" % low_diversity)

    if expected_counts:
        expected_trajectories = expected_counts.get("logical_trajectories")
        expected_examples = expected_counts.get("state_action_examples")
        if expected_trajectories is not None and len(trajectories_raw) != expected_trajectories:
            errors.append("logical trajectory count mismatch")
        if expected_examples is not None and len(examples) != expected_examples:
            errors.append("state-action example count mismatch")
        for name, expected in expected_counts.get("splits", {}).items():
            if split_counts[name] != expected:
                errors.append("split %s count mismatch" % name)
        for name, expected in expected_counts.get("routes", {}).items():
            if route_counts[name] != expected:
                errors.append("route %s count mismatch" % name)
        for name, expected in expected_counts.get("transitions", {}).items():
            if transition_counts[name] != expected:
                errors.append("transition %s count mismatch" % name)

    return {
        "passed": not errors,
        "errors": errors,
        "counts": {
            "logical_trajectories": len(trajectories_raw),
            "state_action_examples": len(examples),
            "routes": dict(sorted(route_counts.items())),
            "transitions": dict(sorted(transition_counts.items())),
            "splits": dict(sorted(split_counts.items())),
            "route_split_trajectories": {
                route: dict(sorted(counts.items()))
                for route, counts in sorted(route_split_trajectory_counts.items())
            },
        },
        "strict_parser_invalid_targets": sum(target_error_counts.values()),
        "strict_parser_error_counts": dict(sorted(target_error_counts.items())),
        "query_leak_count": len(query_leaks),
        "query_answer_leak_count": len(query_leaks),
        "answer_leak_count": len(query_leaks),
        "unavailable_context_leak_count": len(unavailable_context_leaks),
        "forbidden_visual_placeholder_count": forbidden_visual_placeholder_count,
        "source_split_leak_count": len(split_leaks),
        "trajectory_split_leak_count": len(trajectory_split_leaks),
        "source_route_reuse_count": len(route_reuse),
        "cache_provenance_failure_count": len(cache_provenance_failures),
        "pair_consistency_failure_count": len(pair_failures),
        "chain_consistency_failure_count": len(
            {failure.rsplit(":", 1)[0] for failure in chain_failures}
        ),
        "entity_not_visible_in_image_information_count": len(entity_not_visible),
        "image_answer_present_before_text_search_count": len(
            image_answer_present_before_text
        ),
        "text_answer_evidence_missing_count": len(text_answer_evidence_missing),
        "text_evidence_same_as_image_context_only_count": len(
            text_evidence_same_as_image_context_only
        ),
        "new_image_answer_evidence_missing_count": len(
            new_image_answer_evidence_missing
        ),
        "new_image_answer_top1_hit_count": new_image_answer_top1_hits,
        "new_image_answer_top3_hit_count": new_image_answer_top3_hits,
        "reason_length_failure_count": len(reason_length_failures),
        "reason_answer_leak_count": len(reason_answer_leaks),
        "unique_reason_count_by_transition": unique_reason_count,
        "max_reason_template_frequency": max_reason_frequency,
        "reason_template_entropy": reason_entropy,
    }


def run_processor_preflight_examples(
    examples: Sequence[StateActionExample],
    model_path: str,
    max_seq_len: int,
    visual_token_target: int,
    target_reserve: int,
    source_parquet_path: Optional[Path] = None,
) -> Dict[str, Any]:
    try:
        from transformers import AutoConfig, AutoProcessor
    except ImportError as exc:
        raise RuntimeError("Processor preflight requires transformers in the server environment") from exc
    model_config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    patch_size = int(model_config.vision_config.patch_size)
    spatial_merge_size = int(model_config.vision_config.spatial_merge_size)
    effective_patch = patch_size * spatial_merge_size
    pixel_budget = visual_token_target * effective_patch * effective_patch
    processor = AutoProcessor.from_pretrained(
        model_path,
        min_pixels=pixel_budget,
        max_pixels=pixel_budget,
        local_files_only=True,
        use_fast=False,
    )
    tokenizer = getattr(processor, "tokenizer", processor)
    original_visual_counts: Dict[str, int] = {}
    if source_parquet_path is not None and Path(source_parquet_path).suffix.casefold() in {".parquet", ".pq"}:
        try:
            import pyarrow.parquet as pq
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("Full Processor preflight requires pyarrow and Pillow") from exc
        wanted_rows = {
            int(example.source.get("source_row_index")): example.source_data_id
            for example in examples
        }
        parquet = pq.ParquetFile(source_parquet_path)
        current_row = 0
        image_token_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
        for batch in parquet.iter_batches(batch_size=32, columns=["data_id", "images"]):
            rows = batch.to_pylist()
            for offset, row in enumerate(rows):
                row_index = current_row + offset
                if row_index not in wanted_rows:
                    continue
                expected_data_id = wanted_rows[row_index]
                if str(row.get("data_id")) != expected_data_id:
                    raise RuntimeError("source row index/data_id mismatch during Processor preflight")
                images = row.get("images")
                if not isinstance(images, list) or not images or not isinstance(images[0], dict):
                    raise RuntimeError("FVQA image is missing during Processor preflight")
                image_record = images[0]
                if image_record.get("bytes"):
                    with Image.open(io.BytesIO(image_record["bytes"])) as opened:
                        image = opened.convert("RGB")
                elif image_record.get("path"):
                    with Image.open(image_record["path"]) as opened:
                        image = opened.convert("RGB")
                else:
                    raise RuntimeError("FVQA image has neither bytes nor path")
                visual_messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": "Inspect this image."},
                        ],
                    }
                ]
                visual_prompt = processor.apply_chat_template(
                    visual_messages, tokenize=False, add_generation_prompt=True
                )
                processed = processor(
                    text=[visual_prompt], images=[image], padding=True, return_tensors="pt"
                )
                original_visual_counts[expected_data_id] = int(
                    (processed["input_ids"] == image_token_id).sum().item()
                )
            current_row += len(rows)
        missing_images = sorted(set(wanted_rows.values()) - set(original_visual_counts))
        if missing_images:
            raise RuntimeError(
                "Processor preflight could not load %d selected FVQA images" % len(missing_images)
            )
    truncations = []
    max_total = 0
    max_target = 0
    total_token_counts = []
    forbidden_visual_placeholder_count = 0
    for example in examples:
        messages = [
            {"role": message.role, "content": message.content}
            for message in example.state
        ]
        try:
            prompt_ids = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True
            )
        except Exception:
            prompt_text = "\n".join(
                "%s: %s" % (message["role"], message["content"])
                for message in messages
            )
            prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=True)
        target_ids = tokenizer.encode(example.target, add_special_tokens=False)
        original_visual = original_visual_counts.get(
            example.source_data_id, visual_token_target
        )
        cached_result_images = sum(
            reference.get("kind") == "fvqa_image_cache_result"
            for reference in example.image_refs
        )
        total = (
            len(prompt_ids)
            + original_visual
            + visual_token_target * cached_result_images
            + len(target_ids)
        )
        total_token_counts.append(total)
        forbidden_visual_placeholder_count += sum(
            message.content.count(token)
            for message in example.state
            for token in FORBIDDEN_VISUAL_PLACEHOLDERS
        )
        forbidden_visual_placeholder_count += sum(
            example.target.count(token) for token in FORBIDDEN_VISUAL_PLACEHOLDERS
        )
        max_total = max(max_total, total)
        max_target = max(max_target, len(target_ids))
        if len(target_ids) > target_reserve or total > max_seq_len:
            truncations.append(example.example_id)
    sorted_totals = sorted(total_token_counts)
    p95_index = max(0, math.ceil(len(sorted_totals) * 0.95) - 1)
    return {
        "model_path": model_path,
        "max_seq_len": max_seq_len,
        "visual_token_target": visual_token_target,
        "target_reserve": target_reserve,
        "examples_checked": len(examples),
        "max_estimated_sequence_tokens": max_total,
        "mean_total_tokens": round(
            sum(total_token_counts) / len(total_token_counts), 3
        ) if total_token_counts else 0.0,
        "p95_total_tokens": sorted_totals[p95_index] if sorted_totals else 0,
        "max_total_tokens": max_total,
        "max_target_tokens": max_target,
        "target_truncation_count": len(truncations),
        "target_truncation_examples": truncations[:50],
        "forbidden_visual_placeholder_count": forbidden_visual_placeholder_count,
        "processor_used": True,
        "original_images_processed": len(original_visual_counts),
        "original_visual_token_min": min(original_visual_counts.values()) if original_visual_counts else None,
        "original_visual_token_max": max(original_visual_counts.values()) if original_visual_counts else None,
        "cached_thumbnail_token_accounting": "none; cached search-result titles are text-only",
        "pixel_budget": pixel_budget,
        "method": "Qwen Processor for selected original FVQA images and chat-template text; cached search results remain text-only",
    }


def audit_data_dir(data_dir: Path) -> Dict[str, Any]:
    data_dir = Path(data_dir)
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    trajectories = read_jsonl(data_dir / "trajectories.jsonl")
    examples_by_split = {
        split: read_jsonl(data_dir / (split + ".jsonl"))
        for split in ("train", "dev", "test")
    }
    min_unique_reasons = int(
        manifest.get("reason_templates", {}).get("min_unique_per_transition", 1)
    )
    report = audit_records(
        trajectories,
        examples_by_split,
        manifest.get("counts"),
        min_unique_reasons=min_unique_reasons,
        reason_min_tokens=8,
        reason_max_tokens=int(
            manifest.get("build_config", {}).get("reason_max_tokens", 48)
        ),
    )
    report["schema_version"] = manifest.get("schema_version")
    report["input_provenance"] = manifest.get("inputs")
    report["processor_preflight"] = manifest.get("processor_preflight", {"status": "not_requested"})
    report["target_truncation_count"] = report["processor_preflight"].get(
        "target_truncation_count", 0
    )
    if report["processor_preflight"].get("target_truncation_count", 0):
        report["errors"].append("processor target truncation detected")
    if report["processor_preflight"].get("forbidden_visual_placeholder_count", 0):
        report["errors"].append("processor preflight found visual placeholders")
    if report["schema_version"] in {
        "protocol-sft-v0.2",
        "protocol-sft-v0.3",
    }:
        targets = manifest.get("targets", {})
        for route, expected in targets.get("routes", {}).items():
            if report["counts"]["routes"].get(route, 0) != expected:
                report["errors"].append("Full route target mismatch: %s" % route)
        expected_logical = targets.get("logical_trajectories")
        if (
            expected_logical is not None
            and report["counts"]["logical_trajectories"] != expected_logical
        ):
            report["errors"].append("Full logical trajectory target mismatch")
        expected_examples = targets.get("state_action_examples")
        if (
            expected_examples is not None
            and report["counts"]["state_action_examples"] != expected_examples
        ):
            report["errors"].append("Full state-action target mismatch")
        if (
            expected_examples == 1000
            and report["processor_preflight"].get("processor_used") is not True
        ):
            report["errors"].append("Full dataset requires Processor preflight")
        for split, expected in targets.get("requested_splits", {}).items():
            if report["counts"]["splits"].get(split, 0) != expected:
                report["errors"].append("Full split target mismatch: %s" % split)
        for route, quotas in targets.get(
            "route_split_trajectory_quotas", {}
        ).items():
            actual = report["counts"]["route_split_trajectories"].get(
                route, {}
            )
            if any(actual.get(split, 0) != expected for split, expected in quotas.items()):
                report["errors"].append(
                    "Full route split quota mismatch: %s" % route
                )
        if (
            report["processor_preflight"].get("processor_used") is True
            and report["processor_preflight"].get("examples_checked")
            != report["counts"]["state_action_examples"]
        ):
                report["errors"].append("Processor preflight example count mismatch")
        if (
            report["schema_version"] == "protocol-sft-v0.3"
            and expected_examples == 1000
            and report["processor_preflight"].get("original_images_processed")
            != report["counts"]["logical_trajectories"]
        ):
            report["errors"].append(
                "Processor preflight original image count mismatch"
            )

    previous_dataset = manifest.get("previous_dataset", {})
    report["previous_dataset"] = previous_dataset
    report["reused_invalid_trajectory_count"] = int(
        previous_dataset.get("reused_invalid_trajectory_count", 0)
    )
    if (
        report["schema_version"] == "protocol-sft-v0.3"
        and report["reused_invalid_trajectory_count"] != 0
    ):
        report["errors"].append("v0.3 reused invalid trajectories")
    new_data = manifest.get("new_data", {})
    report["new_data"] = new_data
    if report["schema_version"] == "protocol-sft-v0.3":
        for metric in (
            "new_image_answer_evidence_missing_count",
            "new_image_answer_top1_hit_count",
            "new_image_answer_top3_hit_count",
        ):
            if int(new_data.get(metric, -1)) != int(report.get(metric, -2)):
                report["errors"].append(
                    "v0.3 new image evidence metric mismatch: %s" % metric
                )
        if report["new_image_answer_evidence_missing_count"]:
            report["errors"].append("v0.3 new image evidence is missing")

    rejected_path = data_dir / "rejected.jsonl"
    rejected = read_jsonl(rejected_path) if rejected_path.is_file() else []
    try:
        rejection_audit = rejection_distributions(rejected)
    except Exception as exc:
        rejection_audit = {
            "rejected_attempt_count": len(rejected),
            "rejection_reason_distribution": {},
            "rejection_reason_distribution_by_route": {},
        }
        report["errors"].append("invalid rejection audit: %s" % exc)
    report.update(rejection_audit)
    for key in (
        "rejected_attempt_count",
        "rejection_reason_distribution",
        "rejection_reason_distribution_by_route",
    ):
        if manifest.get(key) != rejection_audit.get(key):
            report["errors"].append("manifest rejection metric mismatch: %s" % key)

    manual_config = manifest.get("manual_audit", {})
    manual_path = data_dir / "manual_route_audit.jsonl"
    manual_markdown_path = data_dir / "manual_route_audit.md"
    manual_records = read_jsonl(manual_path) if manual_path.is_file() else []
    manual_counts = Counter(record.get("route") for record in manual_records)
    image_route = (
        "image_search_answer"
        if report["schema_version"] in {"protocol-sft-v0.2", "protocol-sft-v0.3"}
        else "image_search"
    )
    text_route = (
        "text_search_answer"
        if report["schema_version"] in {"protocol-sft-v0.2", "protocol-sft-v0.3"}
        else "text_search"
    )
    expected_manual_counts = {
        "direct_answer": min(
            int(manual_config.get("direct_count", 0)),
            int(report["counts"]["routes"].get("direct_answer", 0)),
        ),
        image_route: min(
            int(manual_config.get("image_count", 0)),
            int(report["counts"]["routes"].get(image_route, 0)),
        ),
        text_route: min(
            int(manual_config.get("text_count", 0)),
            int(report["counts"]["routes"].get(text_route, 0)),
        ),
    }
    if report["schema_version"] in {"protocol-sft-v0.2", "protocol-sft-v0.3"}:
        expected_manual_counts["image_text_search_answer"] = min(
            int(manual_config.get("image_text_count", 0)),
            int(
                report["counts"]["routes"].get(
                    "image_text_search_answer", 0
                )
            ),
        )
    if sum(expected_manual_counts.values()):
        if not manual_path.is_file() or not manual_markdown_path.is_file():
            report["errors"].append("manual route audit files are missing")
        if dict(manual_counts) != {
            key: value for key, value in expected_manual_counts.items() if value
        }:
            report["errors"].append("manual route audit count mismatch")
        manifest_actual_counts = manual_config.get("actual_counts")
        computed_actual_counts = {
            route: manual_counts.get(route, 0)
            for route in (
                "direct_answer",
                "image_search",
                "text_search",
                "image_search_answer",
                "text_search_answer",
                "image_text_search_answer",
            )
        }
        if (
            manifest_actual_counts is not None
            and manifest_actual_counts
            != {
                route: computed_actual_counts.get(route, 0)
                for route in manifest_actual_counts
            }
        ):
            report["errors"].append("manifest manual audit count mismatch")
        if len({record.get("trajectory_id") for record in manual_records}) != len(manual_records):
            report["errors"].append("manual route audit contains duplicates")
        if any(record.get("contains_unavailable_cache_information") for record in manual_records):
            report["errors"].append("manual audit contains unavailable-context leaks")
        if report["schema_version"] == "protocol-sft-v0.3":
            actual_composition = Counter(
                str(record.get("source_group", "legacy"))
                for record in manual_records
            )
            expected_composition = {
                "new_direct_answer": int(
                    manual_config.get("new_direct_count", 0)
                ),
                "new_image_search_answer": int(
                    manual_config.get("new_image_count", 0)
                ),
                "reused_v0_2": int(
                    manual_config.get("reused_image_count", 0)
                )
                + int(manual_config.get("text_count", 0))
                + int(manual_config.get("image_text_count", 0)),
            }
            expected_composition = {
                key: value for key, value in expected_composition.items() if value
            }
            if dict(actual_composition) != expected_composition:
                report["errors"].append(
                    "manual v0.3 audit composition mismatch"
                )
    report["manual_route_audit_counts"] = dict(sorted(manual_counts.items()))

    text_retrieval = manifest.get("text_retrieval", {})
    report["evidence_hit_normal"] = text_retrieval.get("evidence_hit_normal", 0)
    report["evidence_hit_leave_one_source_out"] = text_retrieval.get(
        "evidence_hit_leave_one_source_out", 0
    )
    report["shortfall"] = manifest.get("shortfall", {})
    image_text_metrics = manifest.get("image_text_route", {})
    report["text_evidence_hit_excluding_image_context_docs"] = (
        image_text_metrics.get(
            "text_evidence_hit_excluding_image_context_docs", 0
        )
    )
    report["text_evidence_cross_source_count"] = image_text_metrics.get(
        "text_evidence_cross_source_count", 0
    )
    report["passed"] = not report["errors"]
    return report


def write_audit_markdown(report: Mapping[str, Any], path: Path) -> None:
    processor = report.get("processor_preflight", {})
    lines = [
        "# Protocol-SFT Audit Report",
        "",
        "- Status: **%s**" % ("PASS" if report.get("passed") else "FAIL"),
        "- Logical trajectories: %s" % report.get("counts", {}).get("logical_trajectories"),
        "- State-action examples: %s" % report.get("counts", {}).get("state_action_examples"),
        "- Query leaks: %s" % report.get("query_leak_count"),
        "- Answer leaks: %s" % report.get("answer_leak_count"),
        "- Unavailable-context leaks: %s"
        % report.get("unavailable_context_leak_count"),
        "- Forbidden visual placeholders: %s"
        % report.get("forbidden_visual_placeholder_count"),
        "- Source split leaks: %s" % report.get("source_split_leak_count"),
        "- Trajectory split leaks: %s" % report.get("trajectory_split_leak_count"),
        "- Source route reuse: %s" % report.get("source_route_reuse_count"),
        "- Pair consistency failures: %s" % report.get("pair_consistency_failure_count"),
        "- Chain consistency failures: %s"
        % report.get("chain_consistency_failure_count"),
        "- Cache provenance failures: %s" % report.get("cache_provenance_failure_count"),
        "- Entities not visible in image information: %s"
        % report.get("entity_not_visible_in_image_information_count"),
        "- Answers present before text search: %s"
        % report.get("image_answer_present_before_text_search_count"),
        "- Text answer evidence missing: %s"
        % report.get("text_answer_evidence_missing_count"),
        "- Text evidence only from image context: %s"
        % report.get("text_evidence_same_as_image_context_only_count"),
        "- New image answer evidence missing: %s"
        % report.get("new_image_answer_evidence_missing_count", 0),
        "- New image answer Top-1 hits: %s"
        % report.get("new_image_answer_top1_hit_count", 0),
        "- New image answer Top-3 hits: %s"
        % report.get("new_image_answer_top3_hit_count", 0),
        "- Reused invalid trajectories: %s"
        % report.get("reused_invalid_trajectory_count", 0),
        "- Target truncations: %s"
        % processor.get("target_truncation_count", "not checked"),
        "- Mean total tokens: %s" % processor.get("mean_total_tokens", "not checked"),
        "- P95 total tokens: %s" % processor.get("p95_total_tokens", "not checked"),
        "- Max total tokens: %s" % processor.get("max_total_tokens", "not checked"),
        "- Max target tokens: %s" % processor.get("max_target_tokens", "not checked"),
        "- Evidence hits (normal): %s" % report.get("evidence_hit_normal"),
        "- Evidence hits (leave-one-source-out): %s"
        % report.get("evidence_hit_leave_one_source_out"),
        "- Independent image-context text evidence hits: %s"
        % report.get("text_evidence_hit_excluding_image_context_docs"),
        "- Cross-source text evidence hits: %s"
        % report.get("text_evidence_cross_source_count"),
        "",
        "## Reason templates",
        "",
        "- Unique reasons by transition: `%s`"
        % json.dumps(report.get("unique_reason_count_by_transition", {}), ensure_ascii=False, sort_keys=True),
        "- Maximum template frequency: `%s`"
        % json.dumps(report.get("max_reason_template_frequency", {}), ensure_ascii=False, sort_keys=True),
        "- Template entropy: `%s`"
        % json.dumps(report.get("reason_template_entropy", {}), ensure_ascii=False, sort_keys=True),
        "- Reason length failures: %s" % report.get("reason_length_failure_count"),
        "- Reason answer leaks: %s" % report.get("reason_answer_leak_count"),
        "",
        "## Rejections and manual audit",
        "",
        "- Rejected attempts: %s" % report.get("rejected_attempt_count"),
        "- Rejection reasons: `%s`"
        % json.dumps(report.get("rejection_reason_distribution", {}), ensure_ascii=False, sort_keys=True),
        "- Rejections by route: `%s`"
        % json.dumps(report.get("rejection_reason_distribution_by_route", {}), ensure_ascii=False, sort_keys=True),
        "- Manual route audit counts: `%s`"
        % json.dumps(report.get("manual_route_audit_counts", {}), ensure_ascii=False, sort_keys=True),
        "",
        "## Errors",
        "",
    ]
    errors = report.get("errors", [])
    lines.extend("- %s" % error for error in errors)
    if not errors:
        lines.append("- None")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_sha256_manifest(paths: Iterable[Path], root: Path, output: Path) -> None:
    lines = []
    for path in sorted((Path(path) for path in paths), key=lambda item: str(item)):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        try:
            label = path.relative_to(root).as_posix()
        except ValueError:
            label = path.name
        lines.append("%s  %s" % (digest, label))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
