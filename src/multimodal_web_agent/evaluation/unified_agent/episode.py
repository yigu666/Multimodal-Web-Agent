from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class EpisodeTurn:
    turn_index: int
    generated_text: str
    protocol_valid: bool
    action_type: str | None
    tool_executed: bool
    output_token_count: int
    prompt_sha256: str
    parse_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EpisodeResult:
    eval_id: str
    model_id: str
    task_type: str
    search_required: bool
    turns: tuple[EpisodeTurn, ...]
    final_answer: str | None
    normalized_em: int
    token_f1: float
    tool_call_count: int
    image_search_call_count: int
    text_search_call_count: int
    agent_turn_count: int
    episode_protocol_valid: bool
    within_budget: bool
    agent_success_at_budget: bool
    tool_execution_failure: bool
    max_turn_exhausted: bool
    wall_clock_seconds: float
    initial_prompt_sha256: str

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["turns"] = [turn.to_dict() for turn in self.turns]
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EpisodeResult":
        return cls(
            eval_id=str(value["eval_id"]),
            model_id=str(value["model_id"]),
            task_type=str(value["task_type"]),
            search_required=bool(value["search_required"]),
            turns=tuple(EpisodeTurn(**turn) for turn in value["turns"]),
            final_answer=value.get("final_answer"),
            normalized_em=int(value["normalized_em"]),
            token_f1=float(value["token_f1"]),
            tool_call_count=int(value["tool_call_count"]),
            image_search_call_count=int(value["image_search_call_count"]),
            text_search_call_count=int(value["text_search_call_count"]),
            agent_turn_count=int(value["agent_turn_count"]),
            episode_protocol_valid=bool(value["episode_protocol_valid"]),
            within_budget=bool(value["within_budget"]),
            agent_success_at_budget=bool(value["agent_success_at_budget"]),
            tool_execution_failure=bool(value["tool_execution_failure"]),
            max_turn_exhausted=bool(value["max_turn_exhausted"]),
            wall_clock_seconds=float(value["wall_clock_seconds"]),
            initial_prompt_sha256=str(value["initial_prompt_sha256"]),
        )
