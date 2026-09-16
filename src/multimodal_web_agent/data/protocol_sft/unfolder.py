from __future__ import annotations

from typing import Dict, List, Mapping, Sequence

from multimodal_web_agent.agent import parse_action

from .schema import StateActionExample, Trajectory


def unfold_trajectories(
    trajectories: Sequence[Trajectory],
    split_assignments: Mapping[str, str],
) -> List[StateActionExample]:
    examples = []
    for trajectory in trajectories:
        if trajectory.trajectory_id not in split_assignments:
            raise KeyError("trajectory has no split assignment: %s" % trajectory.trajectory_id)
        split = split_assignments[trajectory.trajectory_id]
        for step_index, step in enumerate(trajectory.steps):
            parsed = parse_action(step.target)
            history_turn_count = sum(
                message.role == "assistant" for message in step.state
            )
            example = StateActionExample(
                example_id="%s:step:%d" % (trajectory.trajectory_id, step_index),
                trajectory_id=trajectory.trajectory_id,
                source_data_id=trajectory.source_data_id,
                split=split,
                route=trajectory.route,
                transition=step.transition,
                state=step.state,
                target=step.target,
                image_refs=step.image_refs,
                canonical_answer=trajectory.canonical_answer,
                accepted_answers=trajectory.accepted_answers,
                source=trajectory.source,
                information_provenance=step.information_provenance,
                schema_version=trajectory.schema_version,
                target_turn_index=step_index,
                history_turn_count=history_turn_count,
                target_action_type=(
                    parsed.action_type.value if parsed.action_type else ""
                ),
            )
            example.validate()
            examples.append(example)
    return examples


def group_examples_by_split(
    examples: Sequence[StateActionExample],
) -> Dict[str, List[StateActionExample]]:
    grouped: Dict[str, List[StateActionExample]] = {"train": [], "dev": [], "test": []}
    for example in examples:
        grouped[example.split].append(example)
    for values in grouped.values():
        values.sort(key=lambda item: item.example_id)
    return grouped
