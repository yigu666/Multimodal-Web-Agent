from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

from multimodal_web_agent.data.protocol_sft.schema import StateActionExample

from .masking import (
    EmptyTargetError,
    PrefixMismatchError,
    SequenceOverflowError,
    tokenize_current_turn,
)
from .renderer import ProtocolRenderer


def audit_mask_examples(
    processor: Any,
    renderer: ProtocolRenderer,
    examples: Sequence[StateActionExample],
    images: Sequence[Any],
    *,
    max_seq_len: int,
) -> Dict[str, Any]:
    if len(examples) != len(images):
        raise ValueError("examples and images must have equal lengths")
    report: Dict[str, Any] = {
        "examples_checked": len(examples),
        "mask_failure_count": 0,
        "target_truncation_count": 0,
        "sequence_overflow_count": 0,
        "empty_target_count": 0,
        "prefix_mismatch_count": 0,
        "history_active_label_count": 0,
        "information_active_label_count": 0,
        "image_prefix_active_label_count": 0,
        "failures": [],
    }
    for example, image in zip(examples, images):
        try:
            tokenized = tokenize_current_turn(
                processor, renderer, example, image, max_seq_len=max_seq_len
            )
        except SequenceOverflowError as exc:
            report["sequence_overflow_count"] += 1
            report["mask_failure_count"] += 1
            report["failures"].append({"sample_id": example.example_id, "error": str(exc)})
            continue
        except PrefixMismatchError as exc:
            report["prefix_mismatch_count"] += 1
            report["mask_failure_count"] += 1
            report["failures"].append({"sample_id": example.example_id, "error": str(exc)})
            continue
        except EmptyTargetError as exc:
            report["empty_target_count"] += 1
            report["mask_failure_count"] += 1
            report["failures"].append({"sample_id": example.example_id, "error": str(exc)})
            continue
        except ValueError as exc:
            report["mask_failure_count"] += 1
            report["failures"].append({"sample_id": example.example_id, "error": str(exc)})
            continue
        labels = tokenized.labels
        context_active = int((labels[: tokenized.metadata.target_start] != -100).sum().item())
        if tokenized.metadata.history_assistant_turn_count:
            context_labels = labels[: tokenized.metadata.target_start]
            report["history_active_label_count"] += int(
                (context_labels != -100).sum().item()
            )
        if tokenized.metadata.information_turn_count:
            context_labels = labels[: tokenized.metadata.target_start]
            report["information_active_label_count"] += int(
                (context_labels != -100).sum().item()
            )
        report["image_prefix_active_label_count"] += context_active
        if context_active:
            report["mask_failure_count"] += 1
            report["failures"].append({
                "sample_id": example.example_id,
                "error": "context contains active labels",
            })
    report["target_truncation_count"] = int(report["target_truncation_count"])
    return report


def combine_mask_reports(reports: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    keys = (
        "examples_checked", "mask_failure_count", "target_truncation_count",
        "sequence_overflow_count", "empty_target_count", "prefix_mismatch_count",
        "history_active_label_count", "information_active_label_count",
        "image_prefix_active_label_count",
    )
    combined = {key: sum(int(report.get(key, 0)) for report in reports.values()) for key in keys}
    combined["splits"] = {name: dict(report) for name, report in reports.items()}
    return combined


def audit_target_weight_examples(
    processor: Any,
    renderer: ProtocolRenderer,
    examples: Sequence[StateActionExample],
    images: Sequence[Any],
    *,
    max_seq_len: int,
    loss_weights: Any,
) -> Dict[str, Any]:
    if len(examples) != len(images):
        raise ValueError("examples and images must have equal lengths")
    report: Dict[str, Any] = {
        "examples_checked": len(examples),
        "target_weight_alignment_failure_count": 0,
        "alignment_method_counts": {},
        "failures": [],
    }
    methods: Dict[str, int] = {}
    for example, image in zip(examples, images):
        try:
            tokenized = tokenize_current_turn(
                processor,
                renderer,
                example,
                image,
                max_seq_len=max_seq_len,
                loss_weights=loss_weights,
            )
            method = str(
                (tokenized.weight_alignment or {}).get(
                    "alignment_method", "unknown"
                )
            )
            methods[method] = methods.get(method, 0) + 1
        except Exception as exc:
            report["target_weight_alignment_failure_count"] += 1
            report["failures"].append({
                "sample_id": example.example_id,
                "error": repr(exc),
            })
    report["alignment_method_counts"] = methods
    return report
