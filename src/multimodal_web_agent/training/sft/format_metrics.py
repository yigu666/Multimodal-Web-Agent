from __future__ import annotations

from collections import Counter, defaultdict
import re
from typing import Any, Dict, Mapping, Sequence

from multimodal_web_agent.agent import ActionType, ProtocolError, parse_action
from multimodal_web_agent.data.protocol_sft.answer_normalizer import normalize_answer

from .dataset import FORMAT_TRANSITIONS
from .generation import GenerationRecord


ACTION_LABELS = ("answer", "image_search", "text_search", "invalid")
FORMAT_GATE_THRESHOLDS = {
    "protocol_valid_rate": (">=", 0.99),
    "exactly_one_action_rate": (">=", 0.99),
    "malformed_rate": ("<=", 0.01),
    "extra_text_rate": ("<=", 0.01),
    "forged_information_count": ("==", 0),
    "nonempty_reason_rate": (">=", 0.99),
    "nonempty_action_payload_rate": (">=", 0.99),
    "tag_closure_rate": (">=", 0.99),
    "target_truncation_count": ("==", 0),
}


def _record(value: GenerationRecord | Mapping[str, Any]) -> GenerationRecord:
    return value if isinstance(value, GenerationRecord) else GenerationRecord(**value)


def _has_one_action(text: str) -> bool:
    return sum(
        len(re.findall(re.escape(tag), text))
        for tag in ("<search>", "<text_search>", "<answer>")
    ) == 1


def _tags_closed(text: str) -> bool:
    if text.count("<reason>") != 1 or text.count("</reason>") != 1:
        return False
    variants = (
        text.count("<search>") == 1
        and text.count("<img>") == 1
        and text.count("</search>") == 1,
        text.count("<text_search>") == 1
        and text.count("</text_search>") == 1,
        text.count("<answer>") == 1 and text.count("</answer>") == 1,
    )
    return sum(variants) == 1


def has_exactly_one_action(text: str) -> bool:
    return _has_one_action(str(text))


def has_valid_tag_closure(text: str) -> bool:
    return _tags_closed(str(text))


def evaluate_format_generation_records(
    records: Sequence[GenerationRecord | Mapping[str, Any]],
    *,
    target_truncation_count: int = 0,
) -> Dict[str, Any]:
    values = [_record(item) for item in records]
    if not values:
        raise ValueError("Format Dev generation records cannot be empty")
    valid = exactly_one = extra_text = forged = nonempty_reason = 0
    nonempty_payload = closed = action_matches = 0
    answer_payload: list[int] = []
    text_payload: list[int] = []
    action_targets = Counter()
    valid_by_target = Counter()
    target_by_transition = Counter()
    correct_by_transition = Counter()
    confusion = {
        target: {prediction: 0 for prediction in ACTION_LABELS}
        for target in ACTION_LABELS
    }
    answer_em: list[int] = []
    for record in values:
        target = parse_action(record.target_rendered)
        if not target.valid or target.action_type is None:
            raise ValueError("Format Dev target is invalid: %s" % record.sample_id)
        target_label = target.action_type.value
        action_targets[target_label] += 1
        target_by_transition[record.state_type] += 1
        generated = record.generated_text
        if _has_one_action(generated):
            exactly_one += 1
        if _tags_closed(generated):
            closed += 1
        if "<information" in generated.casefold():
            forged += 1
        parsed = parse_action(generated)
        prediction = (
            parsed.action_type.value
            if parsed.valid and parsed.action_type is not None
            else "invalid"
        )
        confusion[target_label][prediction] += 1
        if parsed.valid:
            valid += 1
            valid_by_target[target_label] += 1
            nonempty_reason += int(bool((parsed.reason or "").strip()))
            # Image search has a fixed <img> structure and no free payload.
            nonempty_payload += int(
                parsed.action_type == ActionType.IMAGE_SEARCH
                or bool((parsed.content or "").strip())
            )
            matched = parsed.action_type == target.action_type
            action_matches += int(matched)
            correct_by_transition[record.state_type] += int(matched)
        elif parsed.error_code == ProtocolError.EXTRA_TEXT:
            extra_text += 1
        if target.action_type == ActionType.ANSWER:
            has_payload = bool(
                parsed.valid
                and parsed.action_type == ActionType.ANSWER
                and (parsed.content or "").strip()
            )
            answer_payload.append(int(has_payload))
            answer_em.append(
                int(
                    has_payload
                    and normalize_answer(parsed.content or "")
                    == normalize_answer(target.content or "")
                )
            )
        if target.action_type == ActionType.TEXT_SEARCH:
            text_payload.append(
                int(
                    parsed.valid
                    and parsed.action_type == ActionType.TEXT_SEARCH
                    and bool((parsed.content or "").strip())
                )
            )
    total = len(values)
    transition_recall: Dict[str, float | str] = {}
    for transition in (*FORMAT_TRANSITIONS, "initial_to_text_search"):
        denominator = target_by_transition[transition]
        transition_recall[transition] = (
            correct_by_transition[transition] / denominator
            if denominator
            else "not_applicable"
        )
    f1_values = []
    for action in ("answer", "image_search", "text_search"):
        tp = confusion[action][action]
        target_count = action_targets[action]
        predicted_count = sum(confusion[row][action] for row in ACTION_LABELS)
        precision = tp / predicted_count if predicted_count else 0.0
        recall = tp / target_count if target_count else 0.0
        f1_values.append(
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )
    result: Dict[str, Any] = {
        "examples_checked": total,
        "protocol_valid_rate": valid / total,
        "exactly_one_action_rate": exactly_one / total,
        "malformed_rate": (total - valid) / total,
        "extra_text_rate": extra_text / total,
        "forged_information_count": forged,
        "nonempty_reason_rate": nonempty_reason / total,
        "nonempty_action_payload_rate": nonempty_payload / total,
        "answer_payload_nonempty_rate": (
            sum(answer_payload) / len(answer_payload) if answer_payload else "not_applicable"
        ),
        "text_query_nonempty_rate": (
            sum(text_payload) / len(text_payload) if text_payload else "not_applicable"
        ),
        "tag_closure_rate": closed / total,
        "answer_format_valid_rate": (
            valid_by_target["answer"] / action_targets["answer"]
            if action_targets["answer"] else "not_applicable"
        ),
        "image_search_format_valid_rate": (
            valid_by_target["image_search"] / action_targets["image_search"]
            if action_targets["image_search"] else "not_applicable"
        ),
        "text_search_format_valid_rate": (
            valid_by_target["text_search"] / action_targets["text_search"]
            if action_targets["text_search"] else "not_applicable"
        ),
        "target_action_distribution": {
            action: action_targets[action]
            for action in ("answer", "image_search", "text_search")
        },
        "target_truncation_count": int(target_truncation_count),
        "action_type_accuracy": action_matches / total,
        "action_confusion_matrix": confusion,
        "transition_recall": transition_recall,
        "macro_action_f1": sum(f1_values) / len(f1_values),
        "answer_exact_match": (
            sum(answer_em) / len(answer_em) if answer_em else "not_applicable"
        ),
        "policy_diagnostic_only": True,
        "policy_metrics_used_for_checkpoint_selection": False,
        "policy_metrics_used_for_format_gate": False,
        "routing_collapse_computed": False,
    }
    return result


def check_format_gate(metrics: Mapping[str, Any]) -> Dict[str, bool]:
    targets = metrics.get("target_action_distribution", {})
    return {
        "protocol_valid_rate": float(metrics.get("protocol_valid_rate", 0.0)) >= 0.99,
        "exactly_one_action_rate": float(metrics.get("exactly_one_action_rate", 0.0)) >= 0.99,
        "malformed_rate": float(metrics.get("malformed_rate", 1.0)) <= 0.01,
        "extra_text_rate": float(metrics.get("extra_text_rate", 1.0)) <= 0.01,
        "forged_information_count": int(metrics.get("forged_information_count", 1)) == 0,
        "nonempty_reason_rate": float(metrics.get("nonempty_reason_rate", 0.0)) >= 0.99,
        "nonempty_action_payload_rate": float(metrics.get("nonempty_action_payload_rate", 0.0)) >= 0.99,
        "tag_closure_rate": float(metrics.get("tag_closure_rate", 0.0)) >= 0.99,
        "target_truncation_count": int(metrics.get("target_truncation_count", 1)) == 0,
        "answer_target_present": int(targets.get("answer", 0)) > 0,
        "image_search_target_present": int(targets.get("image_search", 0)) > 0,
        "text_search_target_present": int(targets.get("text_search", 0)) > 0,
    }


def format_gate_report(metrics: Mapping[str, Any]) -> Dict[str, Any]:
    checks = check_format_gate(metrics)
    return {
        "objective": "protocol_format_only",
        "passed": all(checks.values()),
        "checks": checks,
        "policy_metrics_used": False,
        "routing_collapse_used": False,
    }
