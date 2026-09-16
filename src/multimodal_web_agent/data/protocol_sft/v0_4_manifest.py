from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from multimodal_web_agent.data.quality.pool_manifest import (
    read_jsonl,
    write_json,
    write_jsonl,
)

from .schema import StateActionExample, Trajectory
from .unfolder import group_examples_by_split, unfold_trajectories
from .v0_4_selector import (
    RouteRange,
    choose_dynamic_route_plan,
    select_candidates,
    unique_group_availability,
)
from .v0_4_splitter import (
    assign_group_aware_splits,
    group_split_leak_report,
    state_action_counts_by_split,
)


@dataclass(frozen=True)
class V04BuildResult:
    selected_candidates: list[dict[str, Any]]
    trajectories: list[Trajectory]
    examples_by_split: dict[str, list[StateActionExample]]
    rejected: list[dict[str, Any]]
    manual_audit: list[dict[str, Any]]
    manifest: dict[str, Any]


def _historical_exposure(
    historical_test_path: Path,
    source_groups: Sequence[Mapping[str, Any]],
) -> tuple[set[str], set[str]]:
    source_ids = {
        str(row["source_data_id"])
        for row in read_jsonl(historical_test_path)
    }
    group_by_source = {
        str(row["source_data_id"]): str(row["entity_group_id"])
        for row in source_groups
    }
    entity_groups = {
        group_by_source[source]
        for source in source_ids
        if source in group_by_source
    }
    return source_ids, entity_groups


def _route_ranges(raw: Mapping[str, Mapping[str, int]]) -> dict[str, RouteRange]:
    defaults = {
        "direct_answer": {"minimum": 280, "preferred": 310, "maximum": 340},
        "image_search_answer": {
            "minimum": 250, "preferred": 315, "maximum": 380
        },
        "text_search_answer": {
            "minimum": 0, "preferred": 60, "maximum": 80
        },
        "image_text_search_answer": {
            "minimum": 0, "preferred": 20, "maximum": 30
        },
    }
    values = {}
    for route, default in defaults.items():
        configured = dict(raw.get(route, default))
        values[route] = RouteRange(
            minimum=int(configured.get("minimum", default["minimum"])),
            preferred=int(configured.get("preferred", default["preferred"])),
            maximum=int(configured.get("maximum", default["maximum"])),
        )
    return values


def build_protocol_sft_v0_4(
    *,
    master_pool_dir: Path,
    historical_test_path: Path,
    seed: int,
    route_targets: Mapping[str, Mapping[str, int]],
) -> V04BuildResult:
    accepted = read_jsonl(master_pool_dir / "accepted_candidates.jsonl")
    source_groups = read_jsonl(master_pool_dir / "source_groups.jsonl")
    exposed_sources, exposed_entities = _historical_exposure(
        historical_test_path, source_groups
    )
    ranges = _route_ranges(route_targets)
    availability = unique_group_availability(
        accepted,
        excluded_source_ids=exposed_sources,
        excluded_entity_group_ids=exposed_entities,
    )
    plan = choose_dynamic_route_plan(
        availability,
        direct_range=ranges["direct_answer"],
        image_range=ranges["image_search_answer"],
        text_range=ranges["text_search_answer"],
        image_text_range=ranges["image_text_search_answer"],
    )
    selected = select_candidates(
        accepted,
        plan=plan,
        seed=seed,
        excluded_source_ids=exposed_sources,
        excluded_entity_group_ids=exposed_entities,
    )
    assignments = assign_group_aware_splits(selected, seed=seed)
    leak_report = group_split_leak_report(selected, assignments)
    if any(leak_report.values()):
        raise ValueError("group-aware split leak detected: %r" % leak_report)
    split_counts = state_action_counts_by_split(selected, assignments)
    if split_counts != {"train": 800, "dev": 100, "test": 100}:
        raise ValueError(
            "v0.4 split state-action counts are not exact: %r"
            % split_counts
        )

    trajectories = [
        Trajectory.from_dict(candidate["trajectory"])
        for candidate in selected
    ]
    for trajectory in trajectories:
        trajectory.validate()
    examples = unfold_trajectories(trajectories, assignments)
    grouped = group_examples_by_split(examples)
    selected_ids = {row["candidate_id"] for row in selected}
    rejected = [
        {
            "candidate_id": row["candidate_id"],
            "source_data_id": row["source_data_id"],
            "route": row["route"],
            "decision": "not_selected",
            "rejection_reasons": ["dynamic_route_or_group_selection"],
        }
        for row in accepted
        if row["candidate_id"] not in selected_ids
    ]
    manual_audit = [
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
        "schema_version": "protocol-sft-v0.4",
        "source_pool_schema": "validated-master-pool-v0.1",
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
        "strict_candidate_availability": availability,
        "group_split_audit": leak_report,
        "historically_exposed_test_source_count": len(exposed_sources),
        "historically_exposed_test_group_count": len(exposed_entities),
        "new_test_source_count": len(
            {
                row["source_data_id"]
                for row in selected
                if assignments[row["candidate_id"]] == "test"
            }
        ),
        "new_test_embargoed": True,
        "test_preview_generated": False,
        "test_used_for_selection": False,
        "passed": True,
    }
    return V04BuildResult(
        selected_candidates=selected,
        trajectories=trajectories,
        examples_by_split=grouped,
        rejected=rejected,
        manual_audit=manual_audit,
        manifest=manifest,
    )


def _preview_train_dev(
    result: V04BuildResult,
    *,
    per_transition: int = 2,
) -> str:
    lines = [
        "# Protocol-SFT v0.4 Train/Dev Preview",
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
            lines.extend(
                [
                    "## %s / %s" % (split, example.transition),
                    "",
                    "- Example ID: `%s`" % example.example_id,
                    "- Source ID: `%s`" % example.source_data_id,
                    "- Target: `%s`"
                    % " ".join(example.target.split()),
                    "",
                ]
            )
    return "\n".join(lines)


def write_protocol_sft_v0_4(
    result: V04BuildResult,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(
        output_dir / "trajectories.jsonl",
        (trajectory.to_dict() for trajectory in result.trajectories),
    )
    for split in ("train", "dev", "test"):
        write_jsonl(
            output_dir / ("%s.jsonl" % split),
            (
                example.to_dict()
                for example in result.examples_by_split[split]
            ),
        )
    write_jsonl(output_dir / "rejected.jsonl", result.rejected)
    write_jsonl(output_dir / "manual_audit.jsonl", result.manual_audit)
    write_json(output_dir / "manifest.json", result.manifest)
    (output_dir / "sample_preview_train_dev.md").write_text(
        _preview_train_dev(result), encoding="utf-8"
    )
