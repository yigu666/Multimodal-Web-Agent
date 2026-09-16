from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from multimodal_web_agent.agent import ActionType, parse_action

from .context_leak import detect_unavailable_context_leak
from .schema import RouteType, Trajectory


@dataclass(frozen=True)
class ManualAuditConfig:
    direct_count: int = 30
    image_count: int = 30
    text_count: int = 50
    seed: int = 20260722
    image_text_count: int = 0
    new_direct_count: int = 0
    new_image_count: int = 0
    reused_image_count: int = 0


def _stable_sample(
    trajectories: Sequence[Trajectory],
    route: str,
    count: int,
    seed: int,
) -> List[Trajectory]:
    candidates = [trajectory for trajectory in trajectories if trajectory.route == route]
    candidates.sort(
        key=lambda trajectory: (
            hashlib.sha256(
                ("%d:%s:%s" % (seed, route, trajectory.trajectory_id)).encode("utf-8")
            ).hexdigest(),
            trajectory.trajectory_id,
        )
    )
    return candidates[:count]


def _stable_sample_origin(
    trajectories: Sequence[Trajectory],
    route: str,
    origin: str,
    count: int,
    seed: int,
) -> List[Trajectory]:
    candidates = [
        trajectory
        for trajectory in trajectories
        if trajectory.route == route
        and trajectory.source.get("v0_3_origin") == origin
    ]
    candidates.sort(
        key=lambda trajectory: (
            hashlib.sha256(
                (
                    "%d:%s:%s:%s"
                    % (seed, route, origin, trajectory.trajectory_id)
                ).encode("utf-8")
            ).hexdigest(),
            trajectory.trajectory_id,
        )
    )
    return candidates[:count]


def _tool_information_blocks(trajectory: Trajectory) -> List[str]:
    return [
        message.content
        for message in trajectory.steps[-1].state
        if message.role == "tool"
    ]


def _audit_record(trajectory: Trajectory) -> Dict[str, Any]:
    initial = trajectory.steps[0]
    information_blocks = _tool_information_blocks(trajectory)
    image_information = ""
    text_information = ""
    if trajectory.route == RouteType.IMAGE_TEXT_SEARCH_ANSWER.value:
        image_information = information_blocks[0] if information_blocks else ""
        text_information = information_blocks[1] if len(information_blocks) > 1 else ""
    elif trajectory.route in {
        RouteType.IMAGE_SEARCH.value,
        RouteType.IMAGE_SEARCH_ANSWER.value,
    }:
        image_information = information_blocks[0] if information_blocks else ""
    elif trajectory.route in {
        RouteType.TEXT_SEARCH.value,
        RouteType.TEXT_SEARCH_ANSWER.value,
    }:
        text_information = information_blocks[0] if information_blocks else ""
    information = "\n".join(information_blocks)
    query = None
    for step in trajectory.steps:
        parsed = parse_action(step.target)
        if parsed.action_type == ActionType.TEXT_SEARCH:
            query = parsed.content
            break
    visible_texts = [trajectory.question]
    unavailable_texts = []
    if trajectory.route == RouteType.IMAGE_TEXT_SEARCH_ANSWER.value:
        visible_texts.append(image_information)
        final_provenance = trajectory.steps[-1].information_provenance or {}
        visible_texts.append(
            "%s %s"
            % (
                final_provenance.get("relation_query_prefix", ""),
                final_provenance.get("identified_entity", ""),
            )
        )
        if text_information:
            unavailable_texts.append(("future_text_information", text_information))
    elif information:
        unavailable_texts.append(("future_search_information", information))
    leak = detect_unavailable_context_leak(
        query or "",
        visible_texts=visible_texts,
        unavailable_texts=unavailable_texts,
    )
    provenance = (
        trajectory.steps[-1].information_provenance or {}
        if len(trajectory.steps) > 1
        else {}
    )
    query_provenance = provenance.get("query_provenance", {})
    source_group = str(trajectory.source.get("v0_3_origin", "legacy"))
    is_new_v0_3 = source_group in {
        "new_direct_answer",
        "new_image_search_answer",
    }
    return {
        "trajectory_id": trajectory.trajectory_id,
        "source_data_id": trajectory.source_data_id,
        "route": trajectory.route,
        "source_group": source_group,
        "question": trajectory.question,
        "canonical_answer": trajectory.canonical_answer,
        "initial_target": initial.target,
        "search_information": information or None,
        "image_search_information": image_information or None,
        "text_search_information": text_information or None,
        "final_answer_target": trajectory.steps[-1].target,
        "query": query,
        "query_construction_source": provenance.get(
            "query_construction_sources",
            ["question", "visible_image_information"]
            if query_provenance
            else ["question"] if query else [],
        ),
        "contains_unavailable_cache_information": leak.leaked,
        "automatic_evidence_hit": provenance.get(
            "evidence_hit_normal",
            provenance.get(
                "text_evidence_hit_excluding_image_context_docs",
                trajectory.route
                not in {
                    RouteType.TEXT_SEARCH.value,
                    RouteType.TEXT_SEARCH_ANSWER.value,
                },
            ),
        ),
        "leave_one_source_out_evidence_hit": provenance.get(
            "evidence_hit_leave_one_source_out"
        ),
        "information_provenance": provenance or None,
        "human_annotation": ({
            "image_search_necessary": "",
            "image_information_identifies_correct_entity": "",
            "answer_already_in_image_information": "",
            "text_search_relation_correct": "",
            "query_entity_from_visible_information": "",
            "text_information_provides_answer": "",
            "route_reasonable": "",
            "notes": "",
        } if trajectory.route == RouteType.IMAGE_TEXT_SEARCH_ANSWER.value else ({
            "route_reasonable": "",
            "image_search_necessary": "",
            "image_information_supports_answer": "",
            "answer_consistent_with_evidence": "",
            "obvious_ambiguity": "",
            "notes": "",
        } if is_new_v0_3 else {
            "route_reasonable": "",
            "query_generatable_from_current_state": "",
            "evidence_sufficient": "",
            "answer_consistent_with_evidence": "",
            "notes": "",
        })),
    }


def build_manual_route_audit(
    trajectories: Sequence[Trajectory],
    config: ManualAuditConfig,
) -> List[Dict[str, Any]]:
    available_routes = {trajectory.route for trajectory in trajectories}
    image_route = (
        RouteType.IMAGE_SEARCH_ANSWER.value
        if RouteType.IMAGE_SEARCH_ANSWER.value in available_routes
        else RouteType.IMAGE_SEARCH.value
    )
    text_route = (
        RouteType.TEXT_SEARCH_ANSWER.value
        if RouteType.TEXT_SEARCH_ANSWER.value in available_routes
        else RouteType.TEXT_SEARCH.value
    )
    selected = []
    if any(
        (
            config.new_direct_count,
            config.new_image_count,
            config.reused_image_count,
        )
    ):
        selected.extend(
            _stable_sample_origin(
                trajectories,
                RouteType.DIRECT_ANSWER.value,
                "new_direct_answer",
                config.new_direct_count,
                config.seed,
            )
        )
        selected.extend(
            _stable_sample_origin(
                trajectories,
                image_route,
                "new_image_search_answer",
                config.new_image_count,
                config.seed,
            )
        )
        selected.extend(
            _stable_sample_origin(
                trajectories,
                image_route,
                "reused_v0_2",
                config.reused_image_count,
                config.seed,
            )
        )
    else:
        selected.extend(
            _stable_sample(
                trajectories,
                RouteType.DIRECT_ANSWER.value,
                config.direct_count,
                config.seed,
            )
        )
    selected.extend(
        _stable_sample(
            trajectories,
            RouteType.IMAGE_TEXT_SEARCH_ANSWER.value,
            config.image_text_count,
            config.seed,
        )
    )
    if not any(
        (
            config.new_direct_count,
            config.new_image_count,
            config.reused_image_count,
        )
    ):
        selected.extend(
            _stable_sample(
                trajectories, image_route, config.image_count, config.seed
            )
        )
    selected.extend(
        _stable_sample(
            trajectories, text_route, config.text_count, config.seed
        )
    )
    records = [_audit_record(trajectory) for trajectory in selected]
    if len({record["trajectory_id"] for record in records}) != len(records):
        raise ValueError("manual route audit contains duplicate trajectories")
    return records


def write_manual_route_audit(
    records: Sequence[Mapping[str, Any]],
    jsonl_path: Path,
    markdown_path: Path,
) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    version_label = (
        "v0.3 Full"
        if any(record.get("source_group") != "legacy" for record in records)
        else "v0.2 Full"
        if any(
            record.get("route") == RouteType.IMAGE_TEXT_SEARCH_ANSWER.value
            for record in records
        )
        else "v0.1"
    )
    lines = ["# Protocol-SFT %s Manual Route Audit" % version_label, ""]
    for index, record in enumerate(records, start=1):
        lines.extend(
            [
                "## %d. %s" % (index, record["trajectory_id"]),
                "",
                "- Source: `%s`" % record["source_data_id"],
                "- Route: `%s`" % record["route"],
                "- Source group: `%s`" % record.get("source_group", "legacy"),
                "- Question: %s" % record["question"],
                "- Canonical answer: %s" % record["canonical_answer"],
                "- Query: %s" % (record.get("query") or "N/A"),
                "- Query construction source: `%s`"
                % ", ".join(record.get("query_construction_source") or []),
                "- Unavailable-context leak: `%s`"
                % record.get("contains_unavailable_cache_information"),
                "- Evidence hit (normal): `%s`" % record.get("automatic_evidence_hit"),
                "- Evidence hit (leave-one-source-out): `%s`"
                % record.get("leave_one_source_out_evidence_hit"),
                "",
                "Initial target:",
                "",
                "```xml",
                str(record["initial_target"]),
                "```",
                "",
                "Search information:",
                "",
                "```text",
                str(record.get("search_information") or "N/A"),
                "```",
                "",
                "Final answer target:",
                "",
                "```xml",
                str(record["final_answer_target"]),
                "```",
                "",
                "### Human annotation",
                "",
                *(
                    [
                        "- Image Search 是否必要：yes / no / uncertain",
                        "- Image Information 是否识别出正确实体：yes / no",
                        "- Answer 是否已经出现在 Image Information：yes / no",
                        "- Text Search Relation 是否正确：yes / no",
                        "- Query Entity 是否来自当前可见信息：yes / no",
                        "- Text Information 是否真正提供 Answer：yes / no / partial",
                        "- Route 是否合理：yes / no / uncertain",
                        "- 备注：",
                    ]
                    if record["route"] == RouteType.IMAGE_TEXT_SEARCH_ANSWER.value
                    else [
                        "- Route 是否合理：yes / no / uncertain",
                        "- Image Search 是否必要：yes / no / uncertain",
                        "- Image Information 是否明确支持 Answer：yes / no / partial / N/A",
                        "- Answer 是否与 Evidence 一致：yes / no",
                        "- 是否存在明显歧义：yes / no",
                        "- 备注：",
                    ]
                    if record.get("source_group") in {
                        "new_direct_answer",
                        "new_image_search_answer",
                    }
                    else [
                        "- Route 合理：yes / no / uncertain",
                        "- Query 可由当前状态生成：yes / no",
                        "- Evidence 足够：yes / no / partial",
                        "- Answer 与 Evidence 一致：yes / no",
                        "- 备注：",
                    ]
                ),
                "",
            ]
        )
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
