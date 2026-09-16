from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from multimodal_web_agent.data.protocol_sft.schema import Message, StateActionExample


@dataclass(frozen=True)
class RenderedPrompt:
    prefix_messages: List[Dict[str, Any]]
    full_messages: List[Dict[str, Any]]
    prefix_text: str
    full_text: str
    tool_role_supported: bool
    tool_role_rendering: str
    chat_template_sha256: str


class ProtocolRenderer:
    """Single renderer shared by tokenization, generation and mask audits."""

    def __init__(self, processor: Any, tool_role_supported: bool | None = None):
        self.processor = processor
        self.tokenizer = getattr(processor, "tokenizer", processor)
        self._tool_role_supported = tool_role_supported

    @property
    def chat_template(self) -> str:
        return str(getattr(self.tokenizer, "chat_template", ""))

    @property
    def chat_template_sha256(self) -> str:
        return hashlib.sha256(self.chat_template.encode("utf-8")).hexdigest()

    @property
    def tool_role_supported(self) -> bool:
        if self._tool_role_supported is not None:
            return self._tool_role_supported
        marker = "__PROTOCOL_SFT_TOOL_PROBE_8f4c__"
        messages = [
            {"role": "system", "content": "system-probe"},
            {"role": "user", "content": "question-probe"},
            {"role": "assistant", "content": "<reason>x</reason>\n<search><img></search>"},
            {"role": "tool", "content": "<information>%s</information>" % marker},
        ]
        try:
            tool_rendering = str(self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            ))
            user_messages = list(messages)
            user_messages[-1] = {
                "role": "user",
                "content": messages[-1]["content"],
            }
            user_rendering = str(self.processor.apply_chat_template(
                user_messages, tokenize=False, add_generation_prompt=True
            ))
            # Some templates silently ignore or alias unknown roles.  Treat the
            # role as supported only when the tool content survives and its
            # rendering is observably distinct from the documented user
            # fallback.
            supported = marker in tool_rendering and tool_rendering != user_rendering
            if not supported:
                self._tool_role_supported = False
            else:
                self._tool_role_supported = True
        except Exception:
            self._tool_role_supported = False
        return bool(self._tool_role_supported)

    @staticmethod
    def _validate_information_message(message: Message) -> None:
        content = message.content.strip()
        if not (
            content.startswith("<information>")
            and content.endswith("</information>")
        ):
            raise ValueError("environment messages must be exactly wrapped in <information>")
        forbidden = (
            "<|image_pad|>",
            "<|vision_start|>",
            "<|vision_end|>",
            "<image>",
        )
        if any(token in content for token in forbidden):
            raise ValueError("cached information must remain text-only")

    @staticmethod
    def _image_count(messages: List[Dict[str, Any]]) -> int:
        return sum(
            sum(item.get("type") == "image" for item in message.get("content", []))
            if isinstance(message.get("content"), list)
            else 0
            for message in messages
        )

    @staticmethod
    def _target_count(messages: List[Dict[str, Any]], target: str) -> int:
        return sum(
            message.get("role") == "assistant"
            and isinstance(message.get("content"), str)
            and message["content"] == target
            for message in messages
        )

    @staticmethod
    def assert_compatible_metadata(
        actual: Dict[str, Any],
        expected: Dict[str, Any],
    ) -> None:
        keys = (
            "tool_role_supported",
            "tool_role_rendering",
            "chat_template_sha256",
        )
        mismatched = [
            key for key in keys if actual.get(key) != expected.get(key)
        ]
        if mismatched:
            raise RuntimeError(
                "renderer metadata differs from training: %s"
                % ", ".join(mismatched)
            )

    @property
    def tool_role_rendering(self) -> str:
        return "tool" if self.tool_role_supported else "user_information_fallback"

    def _message(self, message: Message) -> Dict[str, Any]:
        role = message.role
        if role == "tool" or message.content.lstrip().startswith("<information>"):
            self._validate_information_message(message)
        if role == "tool" and not self.tool_role_supported:
            role = "user"
        content = message.content
        if role == "user" and "<image>" in content:
            text = content.replace("<image>", "", 1).lstrip("\n ")
            value: Any = [{"type": "image"}, {"type": "text", "text": text}]
        else:
            value = content
        return {"role": role, "content": value}

    def messages_for_example(self, example: StateActionExample) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        prefix = [self._message(message) for message in example.state]
        full = list(prefix) + [{"role": "assistant", "content": example.target}]
        if self._image_count(prefix) != 1 or self._image_count(full) != 1:
            raise ValueError("each Protocol-SFT example must contain exactly one original image")
        if self._target_count(prefix, example.target) != 0:
            raise ValueError("current target already appears as assistant history")
        if self._target_count(full, example.target) != 1:
            raise ValueError("current target must appear exactly once")
        return prefix, full

    def render(self, example: StateActionExample) -> RenderedPrompt:
        prefix_messages, full_messages = self.messages_for_example(example)
        prefix_text = self.processor.apply_chat_template(
            prefix_messages, tokenize=False, add_generation_prompt=True
        )
        full_text = self.processor.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False
        )
        return RenderedPrompt(
            prefix_messages=prefix_messages,
            full_messages=full_messages,
            prefix_text=str(prefix_text),
            full_text=str(full_text),
            tool_role_supported=self.tool_role_supported,
            tool_role_rendering=self.tool_role_rendering,
            chat_template_sha256=self.chat_template_sha256,
        )

    def manifest_metadata(self) -> Dict[str, Any]:
        return {
            "tool_role_supported": self.tool_role_supported,
            "tool_role_rendering": self.tool_role_rendering,
            "chat_template_sha256": self.chat_template_sha256,
        }
