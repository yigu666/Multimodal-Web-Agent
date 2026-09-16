from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from multimodal_web_agent.agent import ActionType, parse_action
from multimodal_web_agent.data.quality.pool_manifest import (
    read_jsonl,
    write_json,
    write_jsonl,
)
from multimodal_web_agent.data.quality.visible_context import (
    build_visible_text_context,
)

from .schema import StateActionExample, Trajectory
from .unfolder import group_examples_by_split, unfold_trajectories
from .v0_4_selector import (
    RouteRange,
    select_candidates,
    unique_group_availability,
)
from .v0_4_splitter import (
    group_split_leak_report,
    state_action_counts_by_split,
)
from .v0_5_selector import choose_v05_route_plan
from .v0_5_splitter import assign_v05_group_aware_splits


PROTOCOL_SCHEMA = "protocol-sft-v0.5"
POOL_SCHEMA = "validated-master-pool-v0.2"
GATE_SCHEMA = "shared-executability-gate-v0.2"


@dataclass(frozen=True)
class V05BuildResult:
    selected_candidates: list[dict[str, Any]]
    trajectories: list[Trajectory]
    examples_by_split: dict[str, list[StateActionExample]]
    rejected: list[dict[str, Any]]
    manual_audit: list[dict[str, Any]]
    anchor_audit: list[dict[str, Any]]
    manifest: dict[str, Any]


def _read_historical_sources(
    historical_paths: Mapping[str, Sequence[Path]],
) -> tuple[dict[str, set[str]], set[str]]:
    by_split = {"train": set(), "dev": set(), "test": set()}
    for split, paths in historical_paths.items():
        for path in paths:
            if not path.is_file():
                continue
            by_split[split].update(
                str(row["source_data_id"]) for row in read_jsonl(path)
            )
    return by_split, set().union(*by_split.values())


def _route_ranges(
    raw: Mapping[str, Mapping[str, int]],
) -> dict[str, RouteRange]:
    defaults = {
        "direct_answer": {"minimum": 200, "preferred": 310, "maximum": 500},
        "image_search_answer": {
            "minimum": 200, "preferred": 290, "maximum": 400
        },
        "text_search_answer": {
            "minimum": 0, "preferred": 40, "maximum": 80
        },
        "image_text_search_answer": {
            "minimum": 0, "preferred": 10, "maximum": 30
        },
    }
    return {
        route: RouteRange(
            minimum=int(raw.get(route, value).get("minimum", value["minimum"])),
            preferred=int(
                raw.get(route, value).get("preferred", value["preferred"])
            ),
            maximum=int(raw.get(route, value).get("maximum", value["maximum"])),
        )
        for route, value in defaults.items()
    }


def _promote_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    value = json.loads(json.dumps(candidate))
    old_id = str(value["candidate_id"])
    suffix = old_id.split(":", 1)[1] if ":" in old_id else old_id
    new_id = "protocol_sft_v0_5:%s" % suffix
    value["candidate_id"] = new_id
    value["trajectory"]["trajectory_id"] = new_id
    value["trajectory"]["schema_version"] = PROTOCOL_SCHEMA
    value["trajectory"].setdefault("source", {})[
        "validated_pool_origin"
    ] = "validated_master_pool_v0_2"
    return value


def _anchor_audit_rows(
    selected: Sequence[Mapping[str, Any]],
    assignments: Mapping[str, str],
) -> list[dict[str, Any]]:
    rows = []
    for candidate in selected:
        anchors = {
            value["state_type"]: value
            for value in candidate.get("query_anchor_records", ())
        }
        trajectory = candidate["trajectory"]
        for index, step in enumerate(trajectory["steps"]):
            parsed = parse_action(step["target"])
            if parsed.action_type != ActionType.TEXT_SEARCH:
                continue
            transition = step["transition"]
            anchor = anchors.get(transition, {})
            provenance = anchor.get("query_anchor_provenance", {})
            anchor_source = provenance.get("source", "")
            passed = bool(
                anchor
                and provenance.get("visible_in_current_text_state") is True
                and (
                    anchor_source == "question"
                    if transition == "initial_to_text_search"
                    else anchor_source in {"question", "information"}
                )
            )
            rows.append(
                {
                    "sample_id": "%s:step:%d"
                    % (candidate["candidate_id"], index),
                    "source_data_id": candidate["source_data_id"],
                    "split": assignments[candidate["candidate_id"]],
                    "state_type": transition,
                    "question": trajectory["question"],
                    "query": parsed.content or "",
                    "visible_text_context": build_visible_text_context(
                        question=trajectory["question"],
                        history_messages=step["state"],
                    ),
                    "anchor": anchor.get("query_anchor", ""),
                    "anchor_type": anchor.get("query_anchor_type", ""),
                    "anchor_source": anchor_source,
                    "anchor_visible": bool(
                        provenance.get("visible_in_current_text_state", False)
                    ),
                    "category_only": not bool(anchor),
                    "passed": passed,
                }
            )
    return rows


def build_protocol_sft_v0_5(
    *,
    master_pool_dir: Path,
    master_pool_audit_path: Path,
    historical_paths: Mapping[str, Sequence[Path]],
    reserved_paths: Sequence[Path] = (),
    historical_manifest_paths: Sequence[Path] = (),
    seed: int,
    route_targets: Mapping[str, Mapping[str, int]],
) -> V05BuildResult:
    pool_manifest = json.loads(
        (master_pool_dir / "manifest.json").read_text(encoding="utf-8")
    )
    pool_audit = json.loads(
        master_pool_audit_path.read_text(encoding="utf-8")
    )
    if pool_manifest.get("schema_version") != POOL_SCHEMA:
        raise ValueError("Protocol-SFT v0.5 requires Master Pool v0.2")
    if pool_manifest.get("gate_schema_version") != GATE_SCHEMA:
        raise ValueError("Protocol-SFT v0.5 requires Gate v0.2")
    if pool_audit.get("passed") is not True:
        raise ValueError("Master Pool v0.2 audit has not passed")

    accepted = read_jsonl(master_pool_dir / "accepted_candidates.jsonl")
    source_groups = read_jsonl(master_pool_dir / "source_groups.jsonl")
    historical_by_split, historical_sources = _read_historical_sources(
        historical_paths
    )
    reserved_sources = {
        str(row["source_data_id"])
        for path in reserved_paths
        if path.is_file()
        for row in read_jsonl(path)
    }
    excluded_sources = historical_sources | reserved_sources
    group_by_source = {
        str(row["source_data_id"]): (
            str(row["entity_group_id"]),
            str(row["near_duplicate_group_id"]),
        )
        for row in source_groups
    }
    historical_entity_groups = {
        group_by_source[source][0]
        for source in excluded_sources
        if source in group_by_source
    }
    historical_near_groups = {
        group_by_source[source][1]
        for source in excluded_sources
        if source in group_by_source
    }
    eligible = [
        row for row in accepted
        if str(row["near_duplicate_group_id"]) not in historical_near_groups
    ]
    ranges = _route_ranges(route_targets)
    availability = unique_group_availability(
        eligible,
        excluded_source_ids=excluded_sources,
        excluded_entity_group_ids=historical_entity_groups,
    )
    plan = choose_v05_route_plan(
        availability,
        direct_range=ranges["direct_answer"],
        image_range=ranges["image_search_answer"],
        text_range=ranges["text_search_answer"],
        image_text_range=ranges["image_text_search_answer"],
    )
    selected_raw = select_candidates(
        eligible,
        plan=plan,
        seed=seed,
        excluded_source_ids=excluded_sources,
        excluded_entity_group_ids=historical_entity_groups,
    )
    selected = [_promote_candidate(row) for row in selected_raw]
    assignments = assign_v05_group_aware_splits(selected, seed=seed)
    leaks = group_split_leak_report(selected, assignments)
    if any(leaks.values()):
        raise ValueError("group-aware split leak detected: %r" % leaks)
    split_counts = state_action_counts_by_split(selected, assignments)
    if split_counts != {"train": 800, "dev": 100, "test": 100}:
        raise ValueError("v0.5 split counts are not exact: %r" % split_counts)

    trajectories = [
        Trajectory.from_dict(candidate["trajectory"]) for candidate in selected
    ]
    for trajectory in trajectories:
        trajectory.validate()
    examples = unfold_trajectories(trajectories, assignments)
    grouped = group_examples_by_split(examples)
    anchors = _anchor_audit_rows(selected, assignments)
    if any(not row["passed"] for row in anchors):
        raise ValueError("Initial Text anchor audit failed")
    selected_old_ids = {row["candidate_id"] for row in selected_raw}
    rejected = [
        {
            "candidate_id": row["candidate_id"],
            "source_data_id": row["source_data_id"],
            "route": row["route"],
            "decision": "not_selected",
            "rejection_reasons": ["dynamic_route_or_group_selection"],
        }
        for row in accepted
        if row["candidate_id"] not in selected_old_ids
    ]
    manual = [
        {
            "candidate_id": row["candidate_id"],
            "source_data_id": row["source_data_id"],
            "route": row["route"],
            "split": assignments[row["candidate_id"]],
            "gate_decision": "accept",
            "repair_mode": "reject_only",
            "human_review": "",
            "notes": "",
        }
        for row in selected
        if assignments[row["candidate_id"]] != "test"
    ]
    route_counts = Counter(row["route"] for row in selected)
    manifest = {
        "schema_version": PROTOCOL_SCHEMA,
        "protocol_sft_schema": PROTOCOL_SCHEMA,
        "source_pool_schema": POOL_SCHEMA,
        "master_pool_schema": POOL_SCHEMA,
        "gate_schema": GATE_SCHEMA,
        "gate_schema_version": GATE_SCHEMA,
        "source_pool_dir": str(master_pool_dir).replace("\\", "/"),
        "policy": "reject_first",
        "repair": {
            "mode": "reject_only",
            "automatic_route_conversion": False,
            "automatic_query_rewrite": False,
        },
        "seed": seed,
        "counts": {
            "trajectories": len(trajectories),
            "state_action_examples": len(examples),
            "splits": split_counts,
            "routes": dict(sorted(route_counts.items())),
        },
        "dynamic_route_plan": plan.to_dict(),
        "route_equation": "D + 2I + 2T + 3M = 1000",
        "strict_candidate_availability": availability,
        "group_split_audit": leaks,
        "historically_exposed_source_count": len(historical_sources),
        "historically_exposed_group_count": (
            len(historical_entity_groups) + len(historical_near_groups)
        ),
        "historically_exposed_entity_group_count": len(
            historical_entity_groups
        ),
        "historically_exposed_near_duplicate_group_count": len(
            historical_near_groups
        ),
        "historically_exposed_dev_source_count": len(
            historical_by_split["dev"]
        ),
        "historically_exposed_test_source_count": len(
            historical_by_split["test"]
        ),
        "historically_reserved_test_source_count": len(reserved_sources),
        "historical_groups_excluded_from_new_selection": True,
        "historical_manifest_schemas": [
            json.loads(path.read_text(encoding="utf-8")).get(
                "schema_version", ""
            )
            for path in historical_manifest_paths
            if path.is_file()
        ],
        "new_test_embargoed": True,
        "test_preview_generated": False,
        "test_used_for_selection": False,
        "passed": True,
    }
    return V05BuildResult(
        selected_candidates=selected,
        trajectories=trajectories,
        examples_by_split=grouped,
        rejected=rejected,
        manual_audit=manual,
        anchor_audit=anchors,
        manifest=manifest,
    )


def _preview(result: V05BuildResult, per_transition: int = 2) -> str:
    lines = [
        "# Protocol-SFT v0.5 Train/Dev Preview",
        "",
        "Test is embargoed and intentionally omitted.",
        "",
    ]
    seen = Counter()
    for split in ("train", "dev"):
        for example in result.examples_by_split[split]:
            if seen[(split, example.transition)] >= per_transition:
                continue
            seen[(split, example.transition)] += 1
            lines += [
                "## %s / %s" % (split, example.transition),
                "",
                "- Example ID: `%s`" % example.example_id,
                "- Source ID: `%s`" % example.source_data_id,
                "- Target: `%s`" % " ".join(example.target.split()),
                "",
            ]
    return "\n".join(lines)


def _anchor_markdown(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# Protocol-SFT v0.5 Initial Text Anchor Audit",
        "",
        "- Records: %d" % len(rows),
        "- Passed: %d" % sum(bool(row["passed"]) for row in rows),
        "",
    ]
    for row in rows:
        lines += [
            "## %s" % row["sample_id"],
            "",
            "- State: `%s`" % row["state_type"],
            "- Split: `%s`" % row["split"],
            "- Query: `%s`" % row["query"],
            "- Anchor: `%s` (%s, %s)"
            % (row["anchor"], row["anchor_type"], row["anchor_source"]),
            "",
        ]
    return "\n".join(lines)


def write_protocol_sft_v0_5(
    result: V05BuildResult,
    output_dir: Path,
) -> None:
    if output_dir.exists():
        raise FileExistsError("refusing to overwrite output: %s" % output_dir)
    output_dir.mkdir(parents=True)
    write_jsonl(
        output_dir / "trajectories.jsonl",
        (trajectory.to_dict() for trajectory in result.trajectories),
    )
    for split in ("train", "dev", "test"):
        write_jsonl(
            output_dir / ("%s.jsonl" % split),
            (row.to_dict() for row in result.examples_by_split[split]),
        )
    write_jsonl(output_dir / "rejected.jsonl", result.rejected)
    write_jsonl(output_dir / "manual_audit.jsonl", result.manual_audit)
    write_jsonl(
        output_dir / "initial_text_anchor_audit.jsonl",
        result.anchor_audit,
    )
    write_json(output_dir / "manifest.json", result.manifest)
    (output_dir / "sample_preview_train_dev.md").write_text(
        _preview(result), encoding="utf-8"
    )
    (output_dir / "initial_text_anchor_audit.md").write_text(
        _anchor_markdown(result.anchor_audit), encoding="utf-8"
    )
