from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from .schema import validate_trajectory_dict


V0_2_REUSED_ROUTE_COUNTS: Dict[str, int] = {
    "direct_answer": 202,
    "image_search_answer": 200,
    "text_search_answer": 64,
    "image_text_search_answer": 14,
}

V0_2_ALLOWED_SHORTFALL_ERRORS = {
    "Full route target mismatch: image_text_search_answer",
    "Full logical trajectory target mismatch",
    "Full state-action target mismatch",
    "Full split target mismatch: train",
    "Full split target mismatch: dev",
    "Full split target mismatch: test",
}

REQUIRED_ZERO_QUALITY_METRICS: Tuple[str, ...] = (
    "strict_parser_invalid_targets",
    "answer_leak_count",
    "query_leak_count",
    "unavailable_context_leak_count",
    "forbidden_visual_placeholder_count",
    "source_route_reuse_count",
    "source_split_leak_count",
    "trajectory_split_leak_count",
    "pair_consistency_failure_count",
    "chain_consistency_failure_count",
    "cache_provenance_failure_count",
    "target_truncation_count",
    "entity_not_visible_in_image_information_count",
    "image_answer_present_before_text_search_count",
    "text_answer_evidence_missing_count",
    "text_evidence_same_as_image_context_only_count",
)


@dataclass(frozen=True)
class PreviousDatasetValidation:
    valid: bool
    trajectory_count: int
    state_action_count: int
    route_counts: Dict[str, int]
    invalid_trajectory_ids: Tuple[str, ...]
    errors: Tuple[str, ...]
    schema_version: str
    manifest_hash: str
    audit_hash: str
    global_passed: bool
    global_failure_allowed_reason: Optional[str]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Mapping[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError("expected a JSON object: %s" % path)
    return value


def _allowed_shortfall_failure(errors: Sequence[object]) -> bool:
    normalized = {str(error) for error in errors}
    return bool(normalized) and normalized.issubset(V0_2_ALLOWED_SHORTFALL_ERRORS)


def load_and_validate_previous_trajectories(
    dataset_dir: Path,
    audit_path: Path,
    *,
    expected_trajectory_count: int = 480,
    expected_state_action_count: int = 772,
    expected_route_counts: Optional[Mapping[str, int]] = None,
) -> tuple[List[dict], PreviousDatasetValidation]:
    """Load v0.2 while allowing only its documented global count shortfall.

    The global audit status is not trusted on its own.  Every trajectory is
    schema-validated and every required quality metric must independently be
    zero before any data can be reused by v0.3.
    """

    dataset_dir = Path(dataset_dir)
    audit_path = Path(audit_path)
    trajectories_path = dataset_dir / "trajectories.jsonl"
    manifest_path = dataset_dir / "manifest.json"
    for required in (trajectories_path, manifest_path, audit_path):
        if not required.is_file():
            raise FileNotFoundError(required)

    manifest = _read_json(manifest_path)
    audit = _read_json(audit_path)
    expected_routes = dict(expected_route_counts or V0_2_REUSED_ROUTE_COUNTS)
    errors: List[str] = []
    raw_trajectories: List[dict] = []
    invalid_ids: List[str] = []
    route_counts: Counter[str] = Counter()
    source_ids = set()
    state_action_count = 0

    with trajectories_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = None
            try:
                raw = json.loads(line)
                if not isinstance(raw, dict):
                    raise TypeError("trajectory record is not an object")
                trajectory = validate_trajectory_dict(raw)
            except Exception as exc:
                invalid_id = "line:%d" % line_number
                if isinstance(raw, dict):
                    invalid_id = str(raw.get("trajectory_id", invalid_id))
                invalid_ids.append(invalid_id)
                errors.append("invalid previous trajectory %s: %s" % (invalid_id, exc))
                continue
            raw_trajectories.append(raw)
            route_counts[trajectory.route] += 1
            state_action_count += len(trajectory.steps)
            if trajectory.source_data_id in source_ids:
                errors.append(
                    "previous source_data_id is reused: %s"
                    % trajectory.source_data_id
                )
            source_ids.add(trajectory.source_data_id)

    schema_version = str(manifest.get("schema_version", ""))
    if schema_version != "protocol-sft-v0.2":
        errors.append("previous dataset schema must be protocol-sft-v0.2")
    if str(audit.get("schema_version", "")) != "protocol-sft-v0.2":
        errors.append("previous audit schema must be protocol-sft-v0.2")
    if len(raw_trajectories) != expected_trajectory_count:
        errors.append(
            "previous trajectory count is %d; expected %d"
            % (len(raw_trajectories), expected_trajectory_count)
        )
    if state_action_count != expected_state_action_count:
        errors.append(
            "previous state-action count is %d; expected %d"
            % (state_action_count, expected_state_action_count)
        )
    if dict(route_counts) != expected_routes:
        errors.append(
            "previous route counts are %s; expected %s"
            % (dict(sorted(route_counts.items())), dict(sorted(expected_routes.items())))
        )

    for label, counts in (
        ("manifest", manifest.get("counts", {})),
        ("audit", audit.get("counts", {})),
    ):
        if not isinstance(counts, Mapping):
            errors.append("previous %s counts are missing" % label)
            continue
        if int(counts.get("logical_trajectories", -1)) != expected_trajectory_count:
            errors.append("previous %s logical trajectory count mismatch" % label)
        if int(counts.get("state_action_examples", -1)) != expected_state_action_count:
            errors.append("previous %s state-action count mismatch" % label)
        recorded_routes = counts.get("routes", {})
        if not isinstance(recorded_routes, Mapping) or {
            str(key): int(value) for key, value in recorded_routes.items()
        } != expected_routes:
            errors.append("previous %s route counts mismatch" % label)

    for metric in REQUIRED_ZERO_QUALITY_METRICS:
        if metric not in audit:
            errors.append("previous audit is missing quality metric: %s" % metric)
        elif int(audit.get(metric, -1)) != 0:
            errors.append(
                "previous audit quality metric is nonzero: %s=%s"
                % (metric, audit.get(metric))
            )

    global_passed = bool(audit.get("passed", False))
    audit_errors = audit.get("errors", [])
    if not isinstance(audit_errors, list):
        errors.append("previous audit errors must be a list")
        audit_errors = []
    allowed_shortfall = not global_passed and _allowed_shortfall_failure(audit_errors)
    if not global_passed and not allowed_shortfall:
        errors.append("previous global failure is not limited to the documented shortfall")
    failure_reason = (
        "v0.2 global passed=false only because Full route/count/split targets shortfall"
        if allowed_shortfall
        else None
    )

    validation = PreviousDatasetValidation(
        valid=not errors and not invalid_ids,
        trajectory_count=len(raw_trajectories),
        state_action_count=state_action_count,
        route_counts=dict(sorted(route_counts.items())),
        invalid_trajectory_ids=tuple(invalid_ids),
        errors=tuple(errors),
        schema_version=schema_version,
        manifest_hash=_sha256_file(manifest_path),
        audit_hash=_sha256_file(audit_path),
        global_passed=global_passed,
        global_failure_allowed_reason=failure_reason,
    )
    return raw_trajectories, validation
