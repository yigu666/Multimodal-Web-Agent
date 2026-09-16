from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from multimodal_web_agent.agent import ActionType, parse_action

from .protocol_sft.schema import StateActionExample, Trajectory
from .protocol_sft.unfolder import group_examples_by_split, unfold_trajectories
from .quality.pool_manifest import read_jsonl, write_json, write_jsonl


SCHEMA = "protocol-format-sft-v1"
ROUTE_TARGETS = {
    "direct_answer": {"train": 279, "dev": 31},
    "image_search_answer": {"train": 297, "dev": 33},
    "image_text_search_answer": {"train": 9, "dev": 1},
}
EXPECTED_TRANSITIONS = {
    "initial_to_direct_answer",
    "initial_to_image_search",
    "image_information_to_answer",
    "image_information_to_text_search",
    "text_information_to_answer",
}


@dataclass(frozen=True)
class ProtocolFormatBuildResult:
    candidates: list[dict[str, Any]]
    trajectories: list[Trajectory]
    examples_by_split: dict[str, list[StateActionExample]]
    format_examples_by_action: list[dict[str, Any]]
    manifest: dict[str, Any]


def _stable_key(seed: int, value: str) -> str:
    return hashlib.sha256(
        ("%d:%s" % (seed, value)).encode("utf-8")
    ).hexdigest()


def _source_ids(paths: Sequence[Path]) -> set[str]:
    return {
        str(row["source_data_id"])
        for path in paths
        if path.is_file()
        for row in read_jsonl(path)
    }


def _promote(candidate: Mapping[str, Any]) -> dict[str, Any]:
    value = json.loads(json.dumps(candidate))
    old = str(value["candidate_id"])
    suffix = old.split(":", 1)[1] if ":" in old else old
    new = "protocol_format_sft_v1:%s" % suffix
    value["candidate_id"] = new
    value["trajectory"]["trajectory_id"] = new
    value["trajectory"]["schema_version"] = SCHEMA
    source = value["trajectory"].setdefault("source", {})
    source["format_training_view"] = SCHEMA
    source["entity_group_id"] = str(value["entity_group_id"])
    source["near_duplicate_group_id"] = str(value["near_duplicate_group_id"])
    return value


def _select_route_split(
    candidates: Sequence[Mapping[str, Any]],
    *,
    route: str,
    split: str,
    count: int,
    preferred_sources: set[str],
    deprioritized_sources: set[str],
    forbidden_sources: set[str],
    forbidden_entities: set[str],
    forbidden_near: set[str],
    used_sources: set[str],
    used_entities: set[str],
    used_near: set[str],
    seed: int,
) -> list[dict[str, Any]]:
    eligible = [
        row for row in candidates
        if row["route"] == route
        and str(row["source_data_id"]) not in forbidden_sources
        and str(row["entity_group_id"]) not in forbidden_entities
        and str(row["near_duplicate_group_id"]) not in forbidden_near
    ]
    ordered = sorted(
        eligible,
        key=lambda row: (
            (
                0
                if str(row["source_data_id"]) in preferred_sources
                else (
                    2
                    if str(row["source_data_id"]) in deprioritized_sources
                    else 1
                )
            ),
            _stable_key(seed, "%s:%s" % (split, row["candidate_id"])),
        ),
    )
    selected = []
    for raw in ordered:
        source = str(raw["source_data_id"])
        entity = str(raw["entity_group_id"])
        near = str(raw["near_duplicate_group_id"])
        if (
            source in used_sources
            or entity in used_entities
            or near in used_near
        ):
            continue
        selected.append(_promote(raw))
        used_sources.add(source)
        used_entities.add(entity)
        used_near.add(near)
        if len(selected) == count:
            return selected
    raise ValueError(
        "%s/%s candidate shortfall: selected=%d target=%d"
        % (split, route, len(selected), count)
    )


def build_protocol_format_sft_v1(
    *,
    protocol_v0_5_dir: Path,
    master_pool_dir: Path,
    historical_train_paths: Sequence[Path],
    historical_dev_paths: Sequence[Path],
    seed: int = 20260728,
) -> ProtocolFormatBuildResult:
    v05_manifest = json.loads(
        (protocol_v0_5_dir / "manifest.json").read_text(encoding="utf-8")
    )
    pool_manifest = json.loads(
        (master_pool_dir / "manifest.json").read_text(encoding="utf-8")
    )
    if v05_manifest.get("schema_version") != "protocol-sft-v0.5":
        raise ValueError("format view requires Protocol-SFT v0.5")
    if pool_manifest.get("schema_version") != "validated-master-pool-v0.2":
        raise ValueError("format view requires Master Pool v0.2")
    if pool_manifest.get("gate_schema_version") != (
        "shared-executability-gate-v0.2"
    ):
        raise ValueError("format view requires Shared Gate v0.2")

    candidates = read_jsonl(master_pool_dir / "accepted_candidates.jsonl")
    if any(
        row.get("gate_schema_version") != "shared-executability-gate-v0.2"
        for row in candidates
    ):
        raise ValueError("accepted candidate Gate schema mismatch")
    # This is deliberately an allowlist.  Protected Test/evaluation files are
    # never opened: only sources proven to belong to Train/Dev may enter.
    v05_train = _source_ids([protocol_v0_5_dir / "train.jsonl"])
    v05_dev = _source_ids([protocol_v0_5_dir / "dev.jsonl"])
    prior_train = _source_ids(historical_train_paths)
    prior_dev = _source_ids(historical_dev_paths)
    historical_train = prior_train | v05_train
    historical_dev = prior_dev | v05_dev
    allowed_sources = historical_train | historical_dev
    candidate_sources = {
        str(row["source_data_id"]) for row in candidates
    }
    forbidden = candidate_sources - allowed_sources
    group_path = master_pool_dir / "source_groups.jsonl"
    group_rows = read_jsonl(group_path) if group_path.is_file() else candidates
    forbidden_entities = {
        str(row["entity_group_id"])
        for row in group_rows
        if str(row["source_data_id"]) in forbidden
    }
    forbidden_near = {
        str(row["near_duplicate_group_id"])
        for row in group_rows
        if str(row["source_data_id"]) in forbidden
    }
    historical_train -= forbidden
    historical_dev -= forbidden

    used_sources: set[str] = set()
    used_entities: set[str] = set()
    used_near: set[str] = set()
    selected_by_split = {"train": [], "dev": []}
    # Dev first protects its small fixed quota; Train then consumes the rest.
    for split in ("dev", "train"):
        preferred = historical_dev if split == "dev" else historical_train
        deprioritized = (
            historical_train if split == "dev" else historical_dev
        )
        for route in (
            "image_text_search_answer",
            "direct_answer",
            "image_search_answer",
        ):
            selected_by_split[split].extend(
                _select_route_split(
                    candidates,
                    route=route,
                    split=split,
                    count=ROUTE_TARGETS[route][split],
                    preferred_sources=preferred,
                    deprioritized_sources=deprioritized,
                    forbidden_sources=forbidden,
                    forbidden_entities=forbidden_entities,
                    forbidden_near=forbidden_near,
                    used_sources=used_sources,
                    used_entities=used_entities,
                    used_near=used_near,
                    seed=seed,
                )
            )
    selected = selected_by_split["train"] + selected_by_split["dev"]
    all_image_text_ids = {
        str(row["candidate_id"])
        for row in candidates
        if row["route"] == "image_text_search_answer"
    }
    # Candidate prefixes may differ across views, so source IDs are the
    # authoritative identity for the "all ten" contract.
    all_image_text_sources = {
        str(row["source_data_id"])
        for row in candidates
        if row["route"] == "image_text_search_answer"
    }
    selected_image_text_sources = {
        str(row["source_data_id"])
        for row in selected
        if row["route"] == "image_text_search_answer"
    }
    if len(all_image_text_ids) != 10 or selected_image_text_sources != all_image_text_sources:
        raise ValueError("Format view must include all ten Image→Text candidates")
    assignments = {
        row["candidate_id"]: split
        for split, rows in selected_by_split.items()
        for row in rows
    }
    for row in selected:
        if row["route"] != "image_text_search_answer":
            continue
        anchors = row.get("query_anchor_records", ())
        if not anchors:
            raise ValueError("Image→Text candidate lacks visible Anchor")
        if not any(
            isinstance(anchor.get("query_anchor_provenance"), Mapping)
            and anchor["query_anchor_provenance"].get(
                "visible_in_current_text_state"
            ) is True
            for anchor in anchors
        ):
            raise ValueError("Image→Text candidate Anchor is not visibly grounded")
        if any(
            reason in set(row.get("rejection_reasons", ()))
            for reason in (
                "missing_visible_entity",
                "unresolvable_visual_reference",
                "unreachable_evidence",
            )
        ):
            raise ValueError("Image→Text candidate violates executability Gate")
        for step in row["trajectory"]["steps"]:
            parsed = parse_action(step["target"])
            if (
                parsed.action_type == ActionType.TEXT_SEARCH
                and not (parsed.content or "").strip()
            ):
                raise ValueError("Image→Text candidate has empty query")

    trajectories = [
        Trajectory.from_dict(row["trajectory"]) for row in selected
    ]
    for trajectory in trajectories:
        trajectory.validate()
    examples = unfold_trajectories(trajectories, assignments)
    grouped = group_examples_by_split(examples)
    action_rows = []
    for example in examples:
        parsed = parse_action(example.target)
        action_rows.append(
            {
                "sample_id": example.example_id,
                "split": example.split,
                "transition": example.transition,
                "action_type": parsed.action_type.value,
            }
        )
    actions = Counter(row["action_type"] for row in action_rows)
    transitions = Counter(row["transition"] for row in action_rows)
    selected_sources = {str(row["source_data_id"]) for row in selected}
    manifest = {
        "schema_version": SCHEMA,
        "view_type": "training_view",
        "source_protocol_schema": "protocol-sft-v0.5",
        "source_master_pool_schema": "validated-master-pool-v0.2",
        "source_protocol_manifest_sha256": hashlib.sha256(
            (protocol_v0_5_dir / "manifest.json").read_bytes()
        ).hexdigest(),
        "source_master_pool_manifest_sha256": hashlib.sha256(
            (master_pool_dir / "manifest.json").read_bytes()
        ).hexdigest(),
        "gate_schema": "shared-executability-gate-v0.2",
        "objective": "protocol_format_only",
        "learns_protocol_syntax": True,
        "learns_action_serialization": True,
        "learns_optimal_routing": False,
        "learns_search_quality": False,
        "learns_answer_quality": False,
        "policy_metrics_are_diagnostic_only": True,
        "weighted_loss_enabled": False,
        "test_evaluation_performed": False,
        "dev_purpose": "format_checkpoint_selection",
        "dev_is_independent_benchmark": False,
        "test_split_created": False,
        "counts": {
            "trajectories": len(trajectories),
            "state_action_examples": len(examples),
            "splits": {name: len(rows) for name, rows in grouped.items()},
            "routes": dict(Counter(row["route"] for row in selected)),
            "actions": dict(actions),
            "transitions": dict(transitions),
        },
        "route_split_plan": ROUTE_TARGETS,
        "source_reuse_counts": {
            "protocol_v0_5_train": len(selected_sources & v05_train),
            "protocol_v0_5_dev": len(selected_sources & v05_dev),
            "historical_train": len(selected_sources & prior_train),
            "historical_dev": len(selected_sources & prior_dev),
        },
        "historical_source_policy": {
            "historical_train": "allowed_prefer_train",
            "historical_dev": "allowed_prefer_dev",
            "opened_test": "forbidden_by_train_dev_allowlist",
            "reserved_test": "forbidden_by_train_dev_allowlist",
            "external_eval": "forbidden_by_train_dev_allowlist",
            "protected_split_files_read": False,
            "test_accessed": False,
            "historical_train_dev_files_usage": (
                "source_data_id_governance_only"
            ),
        },
        "group_split_audit": {
            "source_split_leak_count": 0,
            "trajectory_split_leak_count": 0,
            "entity_group_split_leak_count": 0,
            "near_duplicate_split_leak_count": 0,
        },
        "forbidden_source_count": len(forbidden),
        "forbidden_entity_group_count": len(forbidden_entities),
        "forbidden_near_duplicate_group_count": len(forbidden_near),
        "passed": True,
    }
    if len(trajectories) != 650 or len(examples) != 1000:
        raise AssertionError("format view count contract failed")
    if {name: len(rows) for name, rows in grouped.items()} != {
        "train": 900, "dev": 100, "test": 0
    }:
        raise AssertionError("format view split contract failed")
    if set(transitions) != EXPECTED_TRANSITIONS:
        raise AssertionError("format view transition contract failed")
    if not all(actions[name] > 0 for name in ("answer", "image_search", "text_search")):
        raise AssertionError("format view action coverage failed")
    return ProtocolFormatBuildResult(
        candidates=selected,
        trajectories=trajectories,
        examples_by_split=grouped,
        format_examples_by_action=action_rows,
        manifest=manifest,
    )


def write_protocol_format_sft_v1(
    result: ProtocolFormatBuildResult,
    output_dir: Path,
) -> None:
    if output_dir.exists():
        raise FileExistsError("refusing to overwrite output: %s" % output_dir)
    output_dir.mkdir(parents=True)
    write_jsonl(
        output_dir / "trajectories.jsonl",
        (row.to_dict() for row in result.trajectories),
    )
    for split in ("train", "dev"):
        write_jsonl(
            output_dir / ("%s.jsonl" % split),
            (row.to_dict() for row in result.examples_by_split[split]),
        )
    write_json(output_dir / "manifest.json", result.manifest)
    write_jsonl(
        output_dir / "format_examples_by_action.jsonl",
        result.format_examples_by_action,
    )
    lines = [
        "# Protocol Format SFT v1 Train/Dev Preview",
        "",
        "No Test split exists in this training view.",
        "",
    ]
    seen = Counter()
    for split in ("train", "dev"):
        for example in result.examples_by_split[split]:
            key = (split, example.transition)
            if seen[key] >= 2:
                continue
            seen[key] += 1
            lines += [
                "## %s / %s" % key,
                "",
                "- Sample: `%s`" % example.example_id,
                "- Target: `%s`" % " ".join(example.target.split()),
                "",
            ]
    (output_dir / "sample_preview_train_dev.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
