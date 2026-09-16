from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Dict, Iterable, Mapping, Sequence

from multimodal_web_agent.agent import ActionType, ProtocolError, parse_action
from multimodal_web_agent.data.protocol_sft.answer_normalizer import (
    normalize_answer,
)
from multimodal_web_agent.data.protocol_sft.schema import StateActionExample

from .config import sha256_file
from .dataset import FORMAT_TRANSITIONS
from .format_metrics import (
    ACTION_LABELS,
    evaluate_format_generation_records,
    has_exactly_one_action,
    has_valid_tag_closure,
)
from .generation import GenerationRecord
from .test_embargo import sha256_directory


POLICY_BOUNDARY = {
    "policy_diagnostic_only": True,
    "used_for_model_selection": False,
    "used_for_sft_gate": False,
    "used_for_training_decision": False,
}
ALL_TRANSITIONS = (*FORMAT_TRANSITIONS, "initial_to_text_search")
CORE_HIGHER_IS_BETTER = (
    "protocol_valid_rate",
    "exactly_one_action_rate",
    "nonempty_reason_rate",
    "nonempty_action_payload_rate",
    "tag_closure_rate",
)
CORE_LOWER_IS_BETTER = (
    "malformed_rate",
    "extra_text_rate",
    "forged_information_rate",
)
ERROR_CATEGORIES = (
    "valid",
    "malformed_xml",
    "multiple_actions",
    "extra_text",
    "forged_information",
    "empty_reason",
    "empty_payload",
    "tag_not_closed",
    "unknown_action",
)
ACTION_MATRIX_LABELS = (
    "invalid",
    "answer",
    "image_search",
    "text_search",
)


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def selected_epoch_from_name(name: str) -> int:
    match = re.fullmatch(r"checkpoint-epoch-(\d+)", str(name))
    if match is None:
        raise ValueError("selected checkpoint name is not epoch-addressable")
    return int(match.group(1))


def ensure_selected_adapter_fingerprint(
    full_output_dir: Path,
    *,
    expected_adapter_path: str,
    expected_epoch: int = 2,
    created_at: str | None = None,
) -> tuple[Dict[str, Any], bool]:
    full = Path(full_output_dir)
    adapter = full / "selected_adapter"
    selection_path = full / "checkpoint_selection.json"
    gate_path = full / "format_gate.json"
    for path in (adapter, selection_path, gate_path):
        if not path.exists():
            raise FileNotFoundError(path)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    selected_epoch = selected_epoch_from_name(selection["selected_name"])
    if selected_epoch != expected_epoch:
        raise ValueError("frozen selected epoch is not %d" % expected_epoch)
    if Path(str(selection.get("selected_path", ""))).name != (
        selection["selected_name"]
    ):
        raise ValueError("checkpoint selection path/name mismatch")
    if selection.get("selected_format_gate_passed") is not True:
        raise ValueError("selected checkpoint did not pass the Format Gate")
    if selection.get("any_epoch_format_gate_passed") is not True:
        raise ValueError("checkpoint selection reports no passing Epoch")
    if selection.get("selection_split") != "dev":
        raise ValueError("checkpoint selection did not use Format Dev")
    if selection.get("test_metrics_used") is not False:
        raise PermissionError("checkpoint selection used Test metrics")
    if selection.get("policy_metrics_used") is not False:
        raise ValueError("checkpoint selection used policy metrics")
    if gate.get("passed") is not True:
        raise ValueError("frozen Format Gate has not passed")
    if gate.get("selected_checkpoint") != selection.get("selected_name"):
        raise ValueError("Format Gate and checkpoint selection disagree")
    current = {
        "selected_epoch": selected_epoch,
        "adapter_path": expected_adapter_path,
        "adapter_tree_sha256": sha256_directory(adapter),
        "checkpoint_selection_sha256": sha256_file(selection_path),
        "format_gate_sha256": sha256_file(gate_path),
    }
    fingerprint_path = full / "selected_adapter_fingerprint.json"
    if fingerprint_path.exists():
        frozen = json.loads(fingerprint_path.read_text(encoding="utf-8"))
        for key, value in current.items():
            if frozen.get(key) != value:
                raise RuntimeError(
                    "frozen Adapter fingerprint mismatch: %s" % key
                )
        return frozen, False
    value = {
        **current,
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
    }
    temporary = fingerprint_path.with_name(
        fingerprint_path.name + ".tmp-%d" % os.getpid()
    )
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, fingerprint_path)
    return value, True


def assert_raw_model_has_no_adapter(model: Any) -> None:
    peft_config = getattr(model, "peft_config", None)
    if peft_config or bool(
        getattr(model, "_hf_peft_config_loaded", False)
    ):
        raise RuntimeError("Raw model unexpectedly has a PEFT Adapter")
    active = getattr(model, "active_adapters", None)
    if callable(active):
        try:
            active = active()
        except (ImportError, ValueError):
            # Transformers exposes this method on plain base models and raises
            # "No adapter loaded" when queried.
            active = []
    if active:
        raise RuntimeError("Raw model unexpectedly reports active Adapters")


def assert_sft_adapter_loaded(model: Any) -> None:
    if not getattr(model, "peft_config", None):
        raise RuntimeError("SFT model did not load the selected Adapter")


def prompt_contract(
    renderer: Any,
    example: StateActionExample,
    generation_contract: Mapping[str, Any],
    image: Any | None = None,
) -> Dict[str, str]:
    rendered = renderer.render(example)
    prompt_hash = hashlib.sha256(
        rendered.prefix_text.encode("utf-8")
    ).hexdigest()
    image_hash = None
    if image is not None and hasattr(image, "tobytes"):
        image_digest = hashlib.sha256()
        image_digest.update(str(getattr(image, "mode", "")).encode("utf-8"))
        image_digest.update(b"\0")
        image_digest.update(str(getattr(image, "size", "")).encode("utf-8"))
        image_digest.update(b"\0")
        image_digest.update(image.tobytes())
        image_hash = image_digest.hexdigest()
    input_hash = _canonical_hash({
        "sample_id": example.example_id,
        "source_data_id": example.source_data_id,
        "trajectory_id": example.trajectory_id,
        "state_type": example.transition,
        "rendered_prompt_sha256": prompt_hash,
        "image_refs": example.image_refs,
        "image_content_sha256": image_hash,
        "renderer": renderer.manifest_metadata(),
        "generation": dict(generation_contract),
    })
    return {
        "rendered_prompt_sha256": prompt_hash,
        "input_contract_sha256": input_hash,
    }


def _detected_action_type(text: str) -> str:
    markers = {
        "image_search": text.count("<search>"),
        "text_search": text.count("<text_search>"),
        "answer": text.count("<answer>"),
    }
    present = [name for name, count in markers.items() if count > 0]
    return present[0] if len(present) == 1 else "invalid"


def _error_category(value: Mapping[str, Any]) -> str:
    if value["forged_information"]:
        return "forged_information"
    if value["multiple_actions"]:
        return "multiple_actions"
    if value["extra_text"]:
        return "extra_text"
    code = value.get("parse_error")
    if code == ProtocolError.EMPTY_REASON.value:
        return "empty_reason"
    if code in {
        ProtocolError.EMPTY_QUERY.value,
        ProtocolError.EMPTY_ANSWER.value,
    }:
        return "empty_payload"
    if not value["tag_closure_valid"]:
        return "tag_not_closed"
    if code == ProtocolError.UNKNOWN_ACTION.value:
        return "unknown_action"
    if value["parse_valid"]:
        return "valid"
    return "malformed_xml"


def build_prediction_record(
    example: StateActionExample,
    generated: GenerationRecord,
    *,
    split: str,
    model_variant: str,
    adapter_loaded: bool,
    hashes: Mapping[str, str],
) -> Dict[str, Any]:
    if generated.sample_id != example.example_id:
        raise ValueError("generation/example sample ID mismatch")
    parsed = parse_action(generated.generated_text)
    target = parse_action(example.target)
    if not target.valid or target.action_type is None:
        raise ValueError("evaluation target is invalid")
    exactly_one = has_exactly_one_action(generated.generated_text)
    closure = has_valid_tag_closure(generated.generated_text)
    forged = "<information" in generated.generated_text.casefold()
    action_type = (
        parsed.action_type.value
        if parsed.valid and parsed.action_type is not None else None
    )
    payload_requirement = (
        "not_applicable"
        if target.action_type == ActionType.IMAGE_SEARCH else "required"
    )
    payload_nonempty = (
        None
        if payload_requirement == "not_applicable"
        else bool(
            parsed.valid
            and parsed.action_type == target.action_type
            and (parsed.content or "").strip()
        )
    )
    answer_em = None
    if target.action_type == ActionType.ANSWER:
        answer_em = bool(
            parsed.valid
            and parsed.action_type == ActionType.ANSWER
            and normalize_answer(parsed.content or "")
            == normalize_answer(target.content or "")
        )
    target_copy = None
    query_has_xml = None
    query_length = None
    if target.action_type == ActionType.TEXT_SEARCH:
        query = (
            parsed.content or ""
            if parsed.valid and parsed.action_type == ActionType.TEXT_SEARCH
            else ""
        )
        target_copy = bool(
            query
            and normalize_answer(query)
            == normalize_answer(target.content or "")
        )
        query_has_xml = bool("<" in query or ">" in query)
        query_length = len(query.split())
    value: Dict[str, Any] = {
        "sample_id": example.example_id,
        "source_data_id": example.source_data_id,
        "trajectory_id": example.trajectory_id,
        "split": split,
        "route": example.route,
        "state_type": example.transition,
        "target_action_type": target.action_type.value,
        "target_text": example.target,
        "model_variant": model_variant,
        "adapter_loaded": adapter_loaded,
        **dict(hashes),
        "generated_text": generated.generated_text,
        "parse_valid": parsed.valid,
        "parse_error": parsed.error_code.value if parsed.error_code else None,
        "parsed_reason": parsed.reason,
        "parsed_action_type": action_type,
        "detected_action_type": _detected_action_type(
            generated.generated_text
        ),
        "parsed_action_payload": parsed.content,
        "exactly_one_action": exactly_one,
        "multiple_actions": (
            parsed.error_code == ProtocolError.MULTIPLE_ACTIONS
            or not exactly_one
            and sum(
                generated.generated_text.count(tag)
                for tag in ("<search>", "<text_search>", "<answer>")
            ) > 1
        ),
        "malformed": not parsed.valid,
        "extra_text": parsed.error_code == ProtocolError.EXTRA_TEXT,
        "forged_information": forged,
        "reason_nonempty": bool((parsed.reason or "").strip()),
        "action_payload_requirement": payload_requirement,
        "action_payload_nonempty": payload_nonempty,
        "tag_closure_valid": closure,
        "action_type_match": bool(
            parsed.valid and parsed.action_type == target.action_type
        ),
        "answer_exact_match": answer_em,
        "query_nonempty": (
            bool(
                parsed.valid
                and parsed.action_type == ActionType.TEXT_SEARCH
                and (parsed.content or "").strip()
            )
            if target.action_type == ActionType.TEXT_SEARCH else None
        ),
        "query_length": query_length,
        "query_copies_target": target_copy,
        "query_contains_xml": query_has_xml,
    }
    value["error_category"] = _error_category(value)
    return value


def _rate(numerator: int, denominator: int) -> float | str:
    return numerator / denominator if denominator else "not_applicable"


def _action_diagnostics(
    predictions: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    labels = ACTION_LABELS
    confusion = {
        target: {predicted: 0 for predicted in labels}
        for target in labels
    }
    predicted = Counter()
    target = Counter()
    correct = 0
    by_transition: Dict[str, list[int]] = defaultdict(list)
    for row in predictions:
        target_label = str(row["target_action_type"])
        predicted_label = str(row["parsed_action_type"] or "invalid")
        target[target_label] += 1
        predicted[predicted_label] += 1
        confusion[target_label][predicted_label] += 1
        match = int(row["action_type_match"])
        correct += match
        by_transition[str(row["state_type"])].append(match)
    precision: Dict[str, float | str] = {}
    recall: Dict[str, float | str] = {}
    f1_values = []
    for label in ("answer", "image_search", "text_search"):
        tp = confusion[label][label]
        p = _rate(tp, predicted[label])
        r = _rate(tp, target[label])
        precision[label] = p
        recall[label] = r
        pv = 0.0 if isinstance(p, str) else p
        rv = 0.0 if isinstance(r, str) else r
        f1_values.append(2 * pv * rv / (pv + rv) if pv + rv else 0.0)
    transition_recall = {
        name: (
            sum(by_transition[name]) / len(by_transition[name])
            if by_transition[name] else "not_applicable"
        )
        for name in ALL_TRANSITIONS
    }
    answer_values = [
        int(bool(row["answer_exact_match"]))
        for row in predictions
        if row["answer_exact_match"] is not None
    ]
    query_values = [
        int(bool(row["query_nonempty"]))
        for row in predictions
        if row["query_nonempty"] is not None
    ]
    return {
        "action_type_accuracy": correct / len(predictions),
        "action_confusion_matrix": confusion,
        "predicted_action_distribution": {
            label: predicted[label] for label in labels
        },
        "target_action_distribution": {
            label: target[label] for label in ("answer", "image_search", "text_search")
        },
        "action_precision_by_type": precision,
        "action_recall_by_type": recall,
        "macro_action_f1": sum(f1_values) / len(f1_values),
        "answer_exact_match": _rate(sum(answer_values), len(answer_values)),
        "text_query_nonempty_rate": _rate(
            sum(query_values), len(query_values)
        ),
        "transition_recall": transition_recall,
        **POLICY_BOUNDARY,
    }


def action_wise_metrics(
    predictions: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for action in ("answer", "image_search", "text_search"):
        rows = [
            row for row in predictions
            if row["target_action_type"] == action
        ]
        count = len(rows)
        applicable_payload = [
            row for row in rows
            if row["action_payload_nonempty"] is not None
        ]
        result[action] = {
            "target_count": count,
            "protocol_valid_count": sum(bool(row["parse_valid"]) for row in rows),
            "protocol_valid_rate": _rate(
                sum(bool(row["parse_valid"]) for row in rows), count
            ),
            "exactly_one_action_rate": _rate(
                sum(bool(row["exactly_one_action"]) for row in rows), count
            ),
            "tag_closure_rate": _rate(
                sum(bool(row["tag_closure_valid"]) for row in rows), count
            ),
            "nonempty_payload_rate": (
                "not_applicable"
                if action == "image_search"
                else _rate(
                    sum(
                        bool(row["action_payload_nonempty"])
                        for row in applicable_payload
                    ),
                    len(applicable_payload),
                )
            ),
            "query_nonempty_rate": (
                _rate(
                    sum(bool(row["query_nonempty"]) for row in rows),
                    count,
                )
                if action == "text_search" else "not_applicable"
            ),
            "multiple_action_rate": _rate(
                sum(bool(row["multiple_actions"]) for row in rows), count
            ),
            "malformed_rate": _rate(
                sum(bool(row["malformed"]) for row in rows), count
            ),
            "extra_text_rate": _rate(
                sum(bool(row["extra_text"]) for row in rows), count
            ),
            "forged_information_count": sum(
                bool(row["forged_information"]) for row in rows
            ),
        }
    return result


def evaluate_prediction_records(
    predictions: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    if not predictions:
        raise ValueError("prediction records cannot be empty")
    generation_records = [
        GenerationRecord(
            sample_id=str(row["sample_id"]),
            trajectory_id=str(row["trajectory_id"]),
            route=str(row["route"]),
            state_type=str(row["state_type"]),
            target_rendered=str(row["target_text"]),
            generated_text=str(row["generated_text"]),
        )
        for row in predictions
    ]
    base = evaluate_format_generation_records(generation_records)
    total = len(predictions)
    base.update({
        "forged_information_rate": sum(
            bool(row["forged_information"]) for row in predictions
        ) / total,
        "predicted_answer_format_valid_rate": _rate(
            sum(
                row["parse_valid"] and row["detected_action_type"] == "answer"
                for row in predictions
            ),
            sum(row["detected_action_type"] == "answer" for row in predictions),
        ),
        "predicted_image_search_format_valid_rate": _rate(
            sum(
                row["parse_valid"]
                and row["detected_action_type"] == "image_search"
                for row in predictions
            ),
            sum(
                row["detected_action_type"] == "image_search"
                for row in predictions
            ),
        ),
        "predicted_text_search_format_valid_rate": _rate(
            sum(
                row["parse_valid"]
                and row["detected_action_type"] == "text_search"
                for row in predictions
            ),
            sum(
                row["detected_action_type"] == "text_search"
                for row in predictions
            ),
        ),
        "answer_target_protocol_valid_rate": action_wise_metrics(
            predictions
        )["answer"]["protocol_valid_rate"],
        "image_search_target_protocol_valid_rate": action_wise_metrics(
            predictions
        )["image_search"]["protocol_valid_rate"],
        "text_search_target_protocol_valid_rate": action_wise_metrics(
            predictions
        )["text_search"]["protocol_valid_rate"],
        "action_wise": action_wise_metrics(predictions),
    })
    base.update(_action_diagnostics(predictions))
    base.pop("routing_collapse_computed", None)
    return base


def _comparison_row(
    raw: Mapping[str, Any],
    sft: Mapping[str, Any],
) -> Dict[str, Any]:
    for key in (
        "sample_id",
        "source_data_id",
        "trajectory_id",
        "split",
        "state_type",
        "target_action_type",
        "rendered_prompt_sha256",
        "input_contract_sha256",
    ):
        if raw.get(key) != sft.get(key):
            raise RuntimeError("PAIRED_INPUT_MISMATCH: %s" % key)
    raw_valid = raw["error_category"] == "valid"
    sft_valid = sft["error_category"] == "valid"
    if not raw_valid and sft_valid:
        category = "improvement"
    elif raw_valid and not sft_valid:
        category = "regression"
    elif raw_valid and sft_valid:
        category = "unchanged_valid"
    else:
        category = "unchanged_invalid"
    return {
        "sample_id": raw["sample_id"],
        "source_data_id": raw["source_data_id"],
        "trajectory_id": raw["trajectory_id"],
        "split": raw["split"],
        "state_type": raw["state_type"],
        "target_action_type": raw["target_action_type"],
        "target_text": raw["target_text"],
        "rendered_prompt_sha256": raw["rendered_prompt_sha256"],
        "input_contract_sha256": raw["input_contract_sha256"],
        "category": category,
        "raw_generated_text": raw["generated_text"],
        "sft_generated_text": sft["generated_text"],
        "raw_parse_valid": raw["parse_valid"],
        "sft_parse_valid": sft["parse_valid"],
        "raw_error_category": raw["error_category"],
        "sft_error_category": sft["error_category"],
        "raw_predicted_action": raw["parsed_action_type"] or "invalid",
        "sft_predicted_action": sft["parsed_action_type"] or "invalid",
        "raw_answer_exact_match": raw["answer_exact_match"],
        "sft_answer_exact_match": sft["answer_exact_match"],
        "raw_query_nonempty": raw["query_nonempty"],
        "sft_query_nonempty": sft["query_nonempty"],
        "raw_query": (
            raw["parsed_action_payload"]
            if raw["target_action_type"] == "text_search" else None
        ),
        "sft_query": (
            sft["parsed_action_payload"]
            if sft["target_action_type"] == "text_search" else None
        ),
    }


def pair_predictions(
    raw: Sequence[Mapping[str, Any]],
    sft: Sequence[Mapping[str, Any]],
) -> list[Dict[str, Any]]:
    if len(raw) != len(sft):
        raise ValueError("Raw/SFT prediction counts differ")
    raw_ids = [str(row["sample_id"]) for row in raw]
    sft_ids = [str(row["sample_id"]) for row in sft]
    if len(set(raw_ids)) != len(raw_ids) or len(set(sft_ids)) != len(sft_ids):
        raise ValueError("Raw/SFT predictions contain duplicate sample IDs")
    return [
        _comparison_row(raw_row, sft_row)
        for raw_row, sft_row in zip(raw, sft)
    ]


def metric_deltas(
    raw_metrics: Mapping[str, Any],
    sft_metrics: Mapping[str, Any],
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for metric in CORE_HIGHER_IS_BETTER:
        raw = float(raw_metrics[metric])
        sft = float(sft_metrics[metric])
        result[metric] = {
            "metric": metric,
            "raw": raw,
            "sft": sft,
            "absolute_delta": sft - raw,
            "relative_delta": (sft - raw) / raw if raw else None,
            "higher_is_better": True,
        }
    for metric in CORE_LOWER_IS_BETTER:
        raw = float(raw_metrics[metric])
        sft = float(sft_metrics[metric])
        result[metric] = {
            "metric": metric,
            "raw": raw,
            "sft": sft,
            "absolute_delta": sft - raw,
            "improvement_delta": raw - sft,
            "relative_delta": None,
            "higher_is_better": False,
        }
    return result


def transition_comparison(
    raw: Sequence[Mapping[str, Any]],
    sft: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for transition in ALL_TRANSITIONS:
        raw_rows = [row for row in raw if row["state_type"] == transition]
        sft_rows = [row for row in sft if row["state_type"] == transition]
        if not raw_rows and not sft_rows:
            result[transition] = "not_applicable"
            continue
        if len(raw_rows) != len(sft_rows):
            raise ValueError("transition pair count mismatch")
        count = len(raw_rows)
        raw_valid = sum(bool(row["parse_valid"]) for row in raw_rows) / count
        sft_valid = sum(bool(row["parse_valid"]) for row in sft_rows) / count
        result[transition] = {
            "sample_count": count,
            "raw_protocol_valid_rate": raw_valid,
            "sft_protocol_valid_rate": sft_valid,
            "delta_protocol_valid_rate": sft_valid - raw_valid,
            "raw_malformed_rate": sum(
                bool(row["malformed"]) for row in raw_rows
            ) / count,
            "sft_malformed_rate": sum(
                bool(row["malformed"]) for row in sft_rows
            ) / count,
            "raw_tag_closure_rate": sum(
                bool(row["tag_closure_valid"]) for row in raw_rows
            ) / count,
            "sft_tag_closure_rate": sum(
                bool(row["tag_closure_valid"]) for row in sft_rows
            ) / count,
            "raw_nonempty_payload_rate": sum(
                row["action_payload_nonempty"] is None
                or bool(row["action_payload_nonempty"])
                for row in raw_rows
            ) / count,
            "sft_nonempty_payload_rate": sum(
                row["action_payload_nonempty"] is None
                or bool(row["action_payload_nonempty"])
                for row in sft_rows
            ) / count,
        }
    return result


def action_comparison(
    raw: Sequence[Mapping[str, Any]],
    sft: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    return {
        action: {
            "raw": action_wise_metrics(raw)[action],
            "sft": action_wise_metrics(sft)[action],
        }
        for action in ("answer", "image_search", "text_search")
    }


def matrix(
    pairs: Sequence[Mapping[str, Any]],
    raw_key: str,
    sft_key: str,
    labels: Sequence[str] | None = None,
) -> Dict[str, Dict[str, int]]:
    matrix_labels = list(labels) if labels is not None else sorted(
        {
            str(row[raw_key]) for row in pairs
        } | {
            str(row[sft_key]) for row in pairs
        }
    )
    result = {
        source: {destination: 0 for destination in matrix_labels}
        for source in matrix_labels
    }
    for row in pairs:
        source = str(row[raw_key])
        destination = str(row[sft_key])
        if source not in result or destination not in result[source]:
            raise ValueError("matrix value is outside the declared labels")
        result[source][destination] += 1
    return result


def comparison_summary(
    pairs: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    counts = Counter(str(row["category"]) for row in pairs)
    total = len(pairs)
    return {
        "paired_prediction_count": total,
        "categories": {
            name: {
                "count": counts[name],
                "rate": counts[name] / total if total else 0.0,
            }
            for name in (
                "improvement",
                "regression",
                "unchanged_valid",
                "unchanged_invalid",
            )
        },
        **POLICY_BOUNDARY,
    }


def render_report(
    *,
    manifest: Mapping[str, Any],
    fingerprint: Mapping[str, Any],
    metrics: Mapping[str, Mapping[str, Mapping[str, Any]]],
    deltas: Mapping[str, Any],
    actions: Mapping[str, Any],
    transitions: Mapping[str, Any],
    summary: Mapping[str, Any],
    error_matrix: Mapping[str, Any],
    action_matrix: Mapping[str, Any],
    pairs: Sequence[Mapping[str, Any]],
) -> str:
    lines = [
        "# Protocol Format SFT v1: Raw vs Selected Adapter",
        "",
        "本报告主要衡量 Protocol Format SFT 的格式冷启动效果。",
        "",
        "格式指标用于判断模型是否学会协议语法和动作序列化。策略指标仅作为诊断，不能代表 GRPO 前模型已经具备最优路由、高质量搜索或正确答案能力。",
        "",
        "## 1. Executive Summary",
        "",
    ]
    combined_delta = deltas["combined"]["protocol_valid_rate"]
    lines += [
        "- Raw protocol valid rate: `%s`" % combined_delta["raw"],
        "- SFT protocol valid rate: `%s`" % combined_delta["sft"],
        "- Absolute improvement: `%s`" % combined_delta["absolute_delta"],
        "- Frozen selected epoch: `%s`" % fingerprint["selected_epoch"],
        "- Checkpoint reselection performed: `false`",
        "",
        "## 2. Evaluation Scope",
        "",
        "- Train 900: training-fit description, not generalization.",
        "- Format Dev 100: selection-set descriptive evaluation, not an independent benchmark.",
        "- Combined 1000: descriptive view of the full Format training dataset.",
        "- Test accessed: `false`.",
        "",
        "## 3. Dataset and Model Fingerprints",
        "",
        "```json",
        json.dumps({
            "dataset_schema": manifest["dataset_schema"],
            "adapter": fingerprint,
        }, ensure_ascii=False, indent=2, sort_keys=True),
        "```",
        "",
        "## 4. Raw vs SFT Format Metrics",
        "",
        "| Split | Variant | Protocol valid | One action | Malformed | Extra text | Forged count | Forged rate | Reason | Payload | Tag closure |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split in ("train", "dev", "combined"):
        for variant in ("raw", "sft"):
            value = metrics[variant][split]
            lines.append(
                "| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |"
                % (
                    split, variant,
                    value["protocol_valid_rate"],
                    value["exactly_one_action_rate"],
                    value["malformed_rate"],
                    value["extra_text_rate"],
                    value["forged_information_count"],
                    value["forged_information_rate"],
                    value["nonempty_reason_rate"],
                    value["nonempty_action_payload_rate"],
                    value["tag_closure_rate"],
                )
            )
    lines += [
        "",
        "## 5. Split-wise Results",
        "",
        "See `metric_deltas.json` for direction-explicit Train, Dev and Combined deltas.",
        "",
        "## 6. Action-wise Format Results",
        "",
        "```json",
        json.dumps(actions, ensure_ascii=False, indent=2, sort_keys=True),
        "```",
        "",
        "## 7. Transition-wise Format Results",
        "",
        "```json",
        json.dumps(transitions, ensure_ascii=False, indent=2, sort_keys=True),
        "```",
        "",
        "## 8. Paired Improvement and Regression",
        "",
        "```json",
        json.dumps(summary["categories"], ensure_ascii=False, indent=2, sort_keys=True),
        "```",
        "",
        "## 9. Error Transition Matrix",
        "",
        "```json",
        json.dumps(error_matrix, ensure_ascii=False, indent=2, sort_keys=True),
        "```",
        "",
        "## 10. Policy Diagnostics",
        "",
        "Action accuracy, confusion, precision/recall, macro F1, Answer EM, query non-emptiness and transition recall are diagnostic only. They were not used for selection, gating or training decisions.",
        "",
        "```json",
        json.dumps({
            "raw": {
                key: metrics["raw"]["combined"][key]
                for key in (
                    "action_type_accuracy",
                    "action_confusion_matrix",
                    "predicted_action_distribution",
                    "target_action_distribution",
                    "action_precision_by_type",
                    "action_recall_by_type",
                    "macro_action_f1",
                    "answer_exact_match",
                    "text_query_nonempty_rate",
                    "transition_recall",
                )
            },
            "sft": {
                key: metrics["sft"]["combined"][key]
                for key in (
                    "action_type_accuracy",
                    "action_confusion_matrix",
                    "predicted_action_distribution",
                    "target_action_distribution",
                    "action_precision_by_type",
                    "action_recall_by_type",
                    "macro_action_f1",
                    "answer_exact_match",
                    "text_query_nonempty_rate",
                    "transition_recall",
                )
            },
            "boundary": POLICY_BOUNDARY,
        }, ensure_ascii=False, indent=2, sort_keys=True),
        "```",
        "",
        "Raw → SFT Action matrix:",
        "",
        "```json",
        json.dumps(action_matrix, ensure_ascii=False, indent=2, sort_keys=True),
        "```",
        "",
        "## 11. Representative Cases",
        "",
    ]
    for category in ("improvement", "regression", "unchanged_valid"):
        lines += ["### %s" % category.replace("_", " ").title(), ""]
        examples = [row for row in pairs if row["category"] == category][:5]
        if not examples:
            lines += ["None.", ""]
            continue
        for row in examples:
            lines += [
                "#### `%s`" % row["sample_id"],
                "",
                "- Transition: `%s`" % row["state_type"],
                "- Target action: `%s`" % row["target_action_type"],
                "- Raw parse: `%s` (`%s`)" % (
                    row["raw_parse_valid"], row["raw_error_category"]
                ),
                "- SFT parse: `%s` (`%s`)" % (
                    row["sft_parse_valid"], row["sft_error_category"]
                ),
                "- Raw output: `%s`" % " ".join(
                    str(row["raw_generated_text"]).split()
                ),
                "- SFT output: `%s`" % " ".join(
                    str(row["sft_generated_text"]).split()
                ),
                "",
            ]
    lines += [
        "## 12. Limitations",
        "",
        "- Train is training data.",
        "- Format Dev was used for checkpoint selection.",
        "- No independent Test was read.",
        "- Query relevance and retrieval quality were not evaluated.",
        "- The report does not change the frozen Epoch 2 Adapter.",
        "",
        "## 13. Final Conclusion",
        "",
        "This is a descriptive Raw-vs-SFT format report with no performance gate. No training, checkpoint reselection, Test access or GRPO execution was performed.",
        "",
    ]
    return "\n".join(lines)
