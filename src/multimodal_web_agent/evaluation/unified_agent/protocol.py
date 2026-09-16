from __future__ import annotations

import hashlib
from typing import Sequence

from multimodal_web_agent.data.protocol_sft.schema import (
    Message,
    StateActionExample,
)
from multimodal_web_agent.training.sft.renderer import ProtocolRenderer

from .schema import UnifiedEvalExample


UNIFIED_AGENT_SYSTEM_PROMPT = (
    "You are a multimodal research agent. Return exactly one protocol action "
    "and no other text. Valid actions are: "
    "<reason>...</reason><search><img></search>, "
    "<reason>...</reason><text_search>...</text_search>, or "
    "<reason>...</reason><answer>...</answer>. "
    "Tool observations are provided only as <information>...</information>."
)
DUMMY_TARGET = "<reason>placeholder</reason>\n<answer>placeholder</answer>"


def state_example(
    example: UnifiedEvalExample,
    history: Sequence[Message],
) -> StateActionExample:
    state = [
        Message("system", UNIFIED_AGENT_SYSTEM_PROMPT),
        Message("user", "<image>\nQuestion: %s" % example.question),
        *history,
    ]
    return StateActionExample(
        example_id=example.eval_id,
        trajectory_id="unified:" + example.eval_id,
        source_data_id=example.source_data_id,
        split="dev",
        route="direct_answer",
        transition="initial_to_direct_answer",
        state=state,
        target=DUMMY_TARGET,
        image_refs=[{
            "kind": "unified_eval_image",
            "path": example.image_path,
            "sha256": example.image_sha256,
        }],
        canonical_answer="placeholder",
        accepted_answers=["placeholder"],
        source={"source_dataset": example.source_dataset},
        schema_version="protocol-format-sft-v1",
        target_turn_index=sum(item.role == "assistant" for item in state),
        history_turn_count=sum(item.role == "assistant" for item in state),
        target_action_type="answer",
    )


def render_prompt(
    renderer: ProtocolRenderer,
    example: UnifiedEvalExample,
    history: Sequence[Message],
) -> tuple[StateActionExample, str]:
    state = state_example(example, history)
    rendered = renderer.render(state)
    digest = hashlib.sha256(
        rendered.prefix_text.encode("utf-8")
    ).hexdigest()
    return state, digest
