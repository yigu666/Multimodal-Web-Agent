from __future__ import annotations

from typing import Sequence


def set_valued_action_correct(
    prediction: str,
    valid_action_set: tuple[str, ...],
) -> bool:
    return prediction in valid_action_set


def set_valued_action_validity(
    predictions: Sequence[str],
    valid_action_sets: Sequence[tuple[str, ...]],
) -> float:
    if len(predictions) != len(valid_action_sets):
        raise ValueError("prediction and target lengths differ")
    if not predictions:
        return 0.0
    return sum(
        set_valued_action_correct(prediction, targets)
        for prediction, targets in zip(predictions, valid_action_sets)
    ) / len(predictions)
