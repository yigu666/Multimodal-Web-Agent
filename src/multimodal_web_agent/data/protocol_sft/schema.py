from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from html import escape
from typing import Any, Dict, List, Mapping, Optional

from multimodal_web_agent.agent import ActionType, parse_action


SCHEMA_VERSION = "protocol-sft-v0"
SUPPORTED_SCHEMA_VERSIONS = {
    SCHEMA_VERSION,
    "protocol-sft-v0.1",
    "protocol-sft-v0.2",
    "protocol-sft-v0.3",
    "protocol-sft-v0.4",
    "protocol-sft-v0.5",
    "protocol-format-sft-v1",
}


class RouteType(str, Enum):
    DIRECT_ANSWER = "direct_answer"
    # Legacy v0/v0.1 route names remain readable and auditable.
    IMAGE_SEARCH = "image_search"
    TEXT_SEARCH = "text_search"
    IMAGE_SEARCH_ANSWER = "image_search_answer"
    TEXT_SEARCH_ANSWER = "text_search_answer"
    IMAGE_TEXT_SEARCH_ANSWER = "image_text_search_answer"


class TransitionType(str, Enum):
    INITIAL_TO_DIRECT_ANSWER = "initial_to_direct_answer"
    INITIAL_TO_IMAGE_SEARCH = "initial_to_image_search"
    IMAGE_INFORMATION_TO_ANSWER = "image_information_to_answer"
    INITIAL_TO_TEXT_SEARCH = "initial_to_text_search"
    IMAGE_INFORMATION_TO_TEXT_SEARCH = "image_information_to_text_search"
    TEXT_INFORMATION_TO_ANSWER = "text_information_to_answer"


@dataclass(frozen=True)
class Message:
    role: str
    content: str
    trainable: bool = False

    def validate(self) -> None:
        if self.role not in {"system", "user", "assistant", "tool"}:
            raise ValueError("unsupported message role: %s" % self.role)
        if not isinstance(self.content, str) or not self.content.strip():
            raise ValueError("message content must be non-empty")
        if self.trainable:
            raise ValueError("state/history messages must never be trainable")


@dataclass(frozen=True)
class TrajectoryStep:
    transition: str
    state: List[Message]
    target: str
    image_refs: List[Dict[str, Any]] = field(default_factory=list)
    information_provenance: Optional[Dict[str, Any]] = None

    def validate(self) -> None:
        TransitionType(self.transition)
        if not self.state:
            raise ValueError("trajectory step state must be non-empty")
        for message in self.state:
            message.validate()
        parsed = parse_action(self.target)
        if not parsed.valid:
            raise ValueError("invalid target action: %s" % parsed.error_code)


@dataclass(frozen=True)
class Trajectory:
    trajectory_id: str
    source_data_id: str
    route: str
    question: str
    canonical_answer: str
    accepted_answers: List[str]
    steps: List[TrajectoryStep]
    source: Dict[str, Any]
    schema_version: str = SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema_version not in SUPPORTED_SCHEMA_VERSIONS:
            raise ValueError("unsupported schema version")
        route = RouteType(self.route)
        if not self.trajectory_id or not self.source_data_id:
            raise ValueError("trajectory identifiers must be non-empty")
        if not self.question.strip() or not self.canonical_answer.strip():
            raise ValueError("question and canonical answer must be non-empty")
        if not self.accepted_answers:
            raise ValueError("accepted_answers must be non-empty")
        if self.canonical_answer not in self.accepted_answers:
            raise ValueError("canonical answer must be included in accepted_answers")
        expected_steps = {
            RouteType.DIRECT_ANSWER: 1,
            RouteType.IMAGE_SEARCH: 2,
            RouteType.TEXT_SEARCH: 2,
            RouteType.IMAGE_SEARCH_ANSWER: 2,
            RouteType.TEXT_SEARCH_ANSWER: 2,
            RouteType.IMAGE_TEXT_SEARCH_ANSWER: 3,
        }[route]
        if len(self.steps) != expected_steps:
            raise ValueError("route has wrong number of steps")
        for step in self.steps:
            step.validate()
        _validate_route_steps(route, self.steps)
        expected_answer = escape(" ".join(self.canonical_answer.split()), quote=False)
        for step in self.steps:
            parsed = parse_action(step.target)
            if parsed.action_type == ActionType.ANSWER and parsed.content != expected_answer:
                raise ValueError("answer target does not match canonical answer")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Trajectory":
        steps = []
        for raw_step in value["steps"]:
            state = [Message(**message) for message in raw_step["state"]]
            steps.append(
                TrajectoryStep(
                    transition=raw_step["transition"],
                    state=state,
                    target=raw_step["target"],
                    image_refs=list(raw_step.get("image_refs", [])),
                    information_provenance=raw_step.get("information_provenance"),
                )
            )
        return cls(
            trajectory_id=value["trajectory_id"],
            source_data_id=value["source_data_id"],
            route=value["route"],
            question=value["question"],
            canonical_answer=value["canonical_answer"],
            accepted_answers=list(value["accepted_answers"]),
            steps=steps,
            source=dict(value["source"]),
            schema_version=value.get("schema_version", SCHEMA_VERSION),
        )


@dataclass(frozen=True)
class StateActionExample:
    example_id: str
    trajectory_id: str
    source_data_id: str
    split: str
    route: str
    transition: str
    state: List[Message]
    target: str
    image_refs: List[Dict[str, Any]]
    canonical_answer: str
    accepted_answers: List[str]
    source: Dict[str, Any]
    information_provenance: Optional[Dict[str, Any]] = None
    schema_version: str = SCHEMA_VERSION
    target_turn_index: int = 0
    history_turn_count: int = 0
    target_action_type: str = ""

    def validate(self) -> None:
        if self.schema_version not in SUPPORTED_SCHEMA_VERSIONS:
            raise ValueError("unsupported schema version")
        if self.split not in {"train", "dev", "test"}:
            raise ValueError("invalid split")
        RouteType(self.route)
        transition = TransitionType(self.transition)
        if not self.example_id or not self.trajectory_id or not self.source_data_id:
            raise ValueError("example identifiers must be non-empty")
        if not self.canonical_answer or self.canonical_answer not in self.accepted_answers:
            raise ValueError("canonical answer must be present in accepted_answers")
        for message in self.state:
            message.validate()
        parsed = parse_action(self.target)
        if not parsed.valid:
            raise ValueError("invalid target action: %s" % parsed.error_code)
        expected_action = {
            TransitionType.INITIAL_TO_DIRECT_ANSWER: ActionType.ANSWER,
            TransitionType.INITIAL_TO_IMAGE_SEARCH: ActionType.IMAGE_SEARCH,
            TransitionType.IMAGE_INFORMATION_TO_ANSWER: ActionType.ANSWER,
            TransitionType.INITIAL_TO_TEXT_SEARCH: ActionType.TEXT_SEARCH,
            TransitionType.IMAGE_INFORMATION_TO_TEXT_SEARCH: ActionType.TEXT_SEARCH,
            TransitionType.TEXT_INFORMATION_TO_ANSWER: ActionType.ANSWER,
        }[transition]
        if parsed.action_type != expected_action:
            raise ValueError("target action does not match transition")
        if self.schema_version in {
            "protocol-sft-v0.2",
            "protocol-sft-v0.3",
            "protocol-sft-v0.4",
            "protocol-sft-v0.5",
            "protocol-format-sft-v1",
        }:
            if self.target_turn_index < 0:
                raise ValueError("target_turn_index cannot be negative")
            actual_history_turns = sum(
                message.role == "assistant" for message in self.state
            )
            if self.history_turn_count != actual_history_turns:
                raise ValueError("history_turn_count does not match assistant history")
            if self.target_turn_index != actual_history_turns:
                raise ValueError("target_turn_index does not match current turn")
            if self.target_action_type != parsed.action_type.value:
                raise ValueError("target_action_type does not match parsed target")
        if expected_action == ActionType.ANSWER:
            expected_answer = escape(" ".join(self.canonical_answer.split()), quote=False)
            if parsed.content != expected_answer:
                raise ValueError("answer target does not match canonical answer")
        history_actions = [
            parse_action(message.content)
            for message in self.state
            if message.role == "assistant"
        ]
        if any(not action.valid for action in history_actions):
            raise ValueError("assistant history contains an invalid action")
        if any(message.trainable for message in self.state):
            raise ValueError("history action is accidentally trainable")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StateActionExample":
        state = [Message(**message) for message in value["state"]]
        parsed = parse_action(value["target"])
        history_turn_count = sum(message.role == "assistant" for message in state)
        return cls(
            example_id=value["example_id"],
            trajectory_id=value["trajectory_id"],
            source_data_id=value["source_data_id"],
            split=value["split"],
            route=value["route"],
            transition=value["transition"],
            state=state,
            target=value["target"],
            image_refs=list(value.get("image_refs", [])),
            canonical_answer=value["canonical_answer"],
            accepted_answers=list(value["accepted_answers"]),
            source=dict(value["source"]),
            information_provenance=value.get("information_provenance"),
            schema_version=value.get("schema_version", SCHEMA_VERSION),
            target_turn_index=int(
                value.get("target_turn_index", history_turn_count)
            ),
            history_turn_count=int(
                value.get("history_turn_count", history_turn_count)
            ),
            target_action_type=str(
                value.get(
                    "target_action_type",
                    parsed.action_type.value if parsed.action_type else "",
                )
            ),
        )


def _validate_route_steps(route: RouteType, steps: List[TrajectoryStep]) -> None:
    expected = {
        RouteType.DIRECT_ANSWER: [
            (TransitionType.INITIAL_TO_DIRECT_ANSWER, ActionType.ANSWER)
        ],
        RouteType.IMAGE_SEARCH: [
            (TransitionType.INITIAL_TO_IMAGE_SEARCH, ActionType.IMAGE_SEARCH),
            (TransitionType.IMAGE_INFORMATION_TO_ANSWER, ActionType.ANSWER),
        ],
        RouteType.TEXT_SEARCH: [
            (TransitionType.INITIAL_TO_TEXT_SEARCH, ActionType.TEXT_SEARCH),
            (TransitionType.TEXT_INFORMATION_TO_ANSWER, ActionType.ANSWER),
        ],
        RouteType.IMAGE_SEARCH_ANSWER: [
            (TransitionType.INITIAL_TO_IMAGE_SEARCH, ActionType.IMAGE_SEARCH),
            (TransitionType.IMAGE_INFORMATION_TO_ANSWER, ActionType.ANSWER),
        ],
        RouteType.TEXT_SEARCH_ANSWER: [
            (TransitionType.INITIAL_TO_TEXT_SEARCH, ActionType.TEXT_SEARCH),
            (TransitionType.TEXT_INFORMATION_TO_ANSWER, ActionType.ANSWER),
        ],
        RouteType.IMAGE_TEXT_SEARCH_ANSWER: [
            (TransitionType.INITIAL_TO_IMAGE_SEARCH, ActionType.IMAGE_SEARCH),
            (
                TransitionType.IMAGE_INFORMATION_TO_TEXT_SEARCH,
                ActionType.TEXT_SEARCH,
            ),
            (TransitionType.TEXT_INFORMATION_TO_ANSWER, ActionType.ANSWER),
        ],
    }[route]
    actual = [
        (TransitionType(step.transition), parse_action(step.target).action_type)
        for step in steps
    ]
    if actual != expected:
        raise ValueError("trajectory steps do not match route")


def validate_trajectory_dict(value: Mapping[str, Any]) -> Trajectory:
    trajectory = Trajectory.from_dict(value)
    trajectory.validate()
    return trajectory


def validate_example_dict(value: Mapping[str, Any]) -> StateActionExample:
    example = StateActionExample.from_dict(value)
    example.validate()
    return example
