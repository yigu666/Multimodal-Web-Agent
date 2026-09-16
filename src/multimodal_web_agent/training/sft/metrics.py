from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, Mapping, Sequence

from multimodal_web_agent.agent import ActionType, ProtocolError, parse_action
from multimodal_web_agent.data.protocol_sft.answer_normalizer import normalize_answer

from .generation import GenerationRecord


ACTION_LABELS = ("answer", "image_search", "text_search", "invalid")
COLLAPSE_SENSITIVE_TRANSITIONS = (
    "initial_to_direct_answer",
    "initial_to_text_search",
    "image_information_to_text_search",
)
INITIAL_TRANSITIONS = (
    "initial_to_direct_answer",
    "initial_to_image_search",
    "initial_to_text_search",
)


def _target_action(record: GenerationRecord):
    parsed = parse_action(record.target_rendered)
    if not parsed.valid:
        raise ValueError("fixture target is invalid: %s" % record.sample_id)
    return parsed


def _generated_token_count(text: str, tokenizer: Any = None) -> int:
    if tokenizer is not None:
        try:
            return len(tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            pass
    return len(text.split())


def evaluate_generation_records(
    records: Sequence[GenerationRecord | Mapping[str, Any]],
    *,
    tokenizer: Any = None,
) -> Dict[str, Any]:
    normalized = [
        item if isinstance(item, GenerationRecord) else GenerationRecord(**item)
        for item in records
    ]
    if not normalized:
        return {
            "examples_checked": 0,
            "protocol_valid_rate": 0.0,
            "exactly_one_action_rate": 0.0,
            "malformed_rate": 0.0,
            "extra_text_rate": 0.0,
            "empty_reason_rate": 0.0,
            "forged_information_count": 0,
            "predicted_action_distribution": {
                label: 0 for label in ACTION_LABELS
            },
            "target_action_distribution": {
                label: 0 for label in ACTION_LABELS
            },
            "action_confusion_matrix": {
                target: {prediction: 0 for prediction in ACTION_LABELS}
                for target in ACTION_LABELS
            },
            "action_confusion_matrix_order": {
                "target_rows": list(ACTION_LABELS),
                "prediction_columns": list(ACTION_LABELS),
            },
            "action_confusion_matrix_values": [
                [0 for _ in ACTION_LABELS] for _ in ACTION_LABELS
            ],
            "action_recall_by_type": {
                label: None for label in ACTION_LABELS
            },
            "action_precision_by_type": {
                label: None for label in ACTION_LABELS
            },
            "macro_action_f1": 0.0,
            "action_type_accuracy": 0.0,
            "action_type_accuracy_by_transition": {},
            "zero_accuracy_transition_warnings": [],
            "initial_transition_recall": {
                transition: 0.0 for transition in INITIAL_TRANSITIONS
            },
            "minimum_initial_transition_recall": 0.0,
            "mean_initial_transition_recall": 0.0,
            "routing_collapsed": True,
            "routing_collapse_reasons": [
                *INITIAL_TRANSITIONS,
                "text_search_action_recall",
            ],
            "text_query_nonempty_rate": 0.0,
            "text_query_length_mean": 0.0,
            "text_query_length_max": 0,
            "answer_finish_rate": 0.0,
            "answer_exact_match": 0.0,
            "answer_exact_match_by_transition": {},
            "generation_token_mean": 0.0,
            "generation_token_max": 0,
        }
    valid = 0
    exactly_one = 0
    extra_text = 0
    empty_reason = 0
    forged = 0
    action_correct = 0
    action_total = 0
    action_by_transition: Dict[str, list[int]] = defaultdict(list)
    predicted_distribution = Counter()
    target_distribution = Counter()
    confusion = {
        target: {prediction: 0 for prediction in ACTION_LABELS}
        for target in ACTION_LABELS
    }
    text_queries: list[str] = []
    answer_targets = 0
    answer_finished = 0
    answer_em: list[int] = []
    answer_em_by_transition: Dict[str, list[int]] = defaultdict(list)
    generation_lengths = []
    for record in normalized:
        target = _target_action(record)
        generated = record.generated_text
        generation_lengths.append(_generated_token_count(generated, tokenizer))
        if "<information" in generated.casefold():
            forged += 1
        action_markers = sum(
            generated.count(marker)
            for marker in ("<search>", "<text_search>", "<answer>")
        )
        if action_markers == 1:
            exactly_one += 1
        parsed = parse_action(generated)
        target_label = target.action_type.value
        predicted_label = (
            parsed.action_type.value
            if parsed.valid and parsed.action_type is not None
            else "invalid"
        )
        target_distribution[target_label] += 1
        predicted_distribution[predicted_label] += 1
        confusion[target_label][predicted_label] += 1
        if parsed.valid:
            valid += 1
            if parsed.action_type == target.action_type:
                action_correct += 1
        else:
            if parsed.error_code == ProtocolError.EXTRA_TEXT:
                extra_text += 1
            if parsed.error_code == ProtocolError.EMPTY_REASON:
                empty_reason += 1
        action_total += 1
        action_by_transition[record.state_type].append(
            int(parsed.valid and parsed.action_type == target.action_type)
        )
        if target.action_type == ActionType.TEXT_SEARCH:
            text_queries.append(
                parsed.content or ""
                if parsed.valid and parsed.action_type == ActionType.TEXT_SEARCH
                else ""
            )
        if target.action_type == ActionType.ANSWER:
            answer_targets += 1
            if parsed.valid and parsed.action_type == ActionType.ANSWER:
                answer_finished += 1
                value = int(normalize_answer(parsed.content or "") == normalize_answer(target.content or ""))
                answer_em_by_transition[record.state_type].append(value)
                answer_em.append(value)
            else:
                answer_em_by_transition[record.state_type].append(0)
                answer_em.append(0)
    answer_finish_rate = answer_finished / answer_targets if answer_targets else 0.0
    transition_accuracy = {
        key: sum(values) / len(values)
        for key, values in sorted(action_by_transition.items())
    }
    initial_recall = {
        transition: float(transition_accuracy.get(transition, 0.0))
        for transition in INITIAL_TRANSITIONS
    }
    recalls: Dict[str, float | None] = {}
    precisions: Dict[str, float | None] = {}
    f1_values = []
    for label in ACTION_LABELS:
        true_positive = confusion[label][label]
        target_count = target_distribution[label]
        predicted_count = predicted_distribution[label]
        recall = true_positive / target_count if target_count else None
        precision = true_positive / predicted_count if predicted_count else None
        recalls[label] = recall
        precisions[label] = precision
        if target_count:
            precision_value = precision or 0.0
            recall_value = recall or 0.0
            denominator = precision_value + recall_value
            f1_values.append(
                2 * precision_value * recall_value / denominator
                if denominator
                else 0.0
            )
    text_search_recall = recalls.get("text_search")
    collapse_reasons = [
        transition
        for transition, recall in initial_recall.items()
        if recall == 0.0
    ]
    if text_search_recall in (None, 0.0):
        collapse_reasons.append("text_search_action_recall")
    return {
        "examples_checked": len(normalized),
        "protocol_valid_rate": valid / len(normalized),
        "exactly_one_action_rate": exactly_one / len(normalized),
        "malformed_rate": 1.0 - valid / len(normalized),
        "extra_text_rate": extra_text / len(normalized),
        "empty_reason_rate": empty_reason / len(normalized),
        "forged_information_count": forged,
        "action_type_accuracy": action_correct / action_total if action_total else 0.0,
        "action_type_accuracy_by_transition": transition_accuracy,
        "zero_accuracy_transition_warnings": [
            transition
            for transition in COLLAPSE_SENSITIVE_TRANSITIONS
            if transition_accuracy.get(transition) == 0.0
        ],
        "initial_transition_recall": initial_recall,
        "minimum_initial_transition_recall": min(
            initial_recall.values(), default=0.0
        ),
        "mean_initial_transition_recall": (
            sum(initial_recall.values()) / len(initial_recall)
            if initial_recall else 0.0
        ),
        "routing_collapsed": bool(collapse_reasons),
        "routing_collapse_reasons": collapse_reasons,
        "predicted_action_distribution": {
            label: predicted_distribution[label] for label in ACTION_LABELS
        },
        "target_action_distribution": {
            label: target_distribution[label] for label in ACTION_LABELS
        },
        "action_confusion_matrix": confusion,
        "action_confusion_matrix_order": {
            "target_rows": list(ACTION_LABELS),
            "prediction_columns": list(ACTION_LABELS),
        },
        "action_confusion_matrix_values": [
            [confusion[target][prediction] for prediction in ACTION_LABELS]
            for target in ACTION_LABELS
        ],
        "action_recall_by_type": recalls,
        "action_precision_by_type": precisions,
        "macro_action_f1": (
            sum(f1_values) / len(f1_values) if f1_values else 0.0
        ),
        "text_query_nonempty_rate": (
            sum(bool(query.strip()) for query in text_queries) / len(text_queries)
            if text_queries else 0.0
        ),
        "text_query_length_mean": (
            sum(len(query.split()) for query in text_queries) / len(text_queries)
            if text_queries else 0.0
        ),
        "text_query_length_max": max((len(query.split()) for query in text_queries), default=0),
        "answer_finish_rate": answer_finish_rate,
        "answer_exact_match": sum(answer_em) / len(answer_em) if answer_em else 0.0,
        "answer_exact_match_by_transition": {
            key: sum(values) / len(values) for key, values in sorted(answer_em_by_transition.items())
        },
        "generation_token_mean": sum(generation_lengths) / len(generation_lengths),
        "generation_token_max": max(generation_lengths),
    }


def check_format_gates(metrics: Mapping[str, Any]) -> Dict[str, bool]:
    return {
        "protocol_valid_rate": float(metrics.get("protocol_valid_rate", 0.0)) >= 0.97,
        "exactly_one_action_rate": float(metrics.get("exactly_one_action_rate", 0.0)) >= 0.98,
        "malformed_rate": float(metrics.get("malformed_rate", 1.0)) <= 0.03,
        "forged_information_count": int(metrics.get("forged_information_count", 1)) == 0,
        "text_query_nonempty_rate": float(metrics.get("text_query_nonempty_rate", 0.0)) >= 0.95,
        "answer_finish_rate": float(metrics.get("answer_finish_rate", 0.0)) >= 0.90,
        "target_truncation_count": int(metrics.get("target_truncation_count", 1)) == 0,
    }
