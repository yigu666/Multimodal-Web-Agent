from __future__ import annotations

from typing import Any


FORBIDDEN_ACTION_OPENINGS = ("<search>", "<text_search>")


class ToolAbsentEnvironment:
    """Fail-closed environment for the tool-absent evaluation condition."""

    def __init__(self) -> None:
        self.begin_episode_calls = 0
        self.visual_search_invocations = 0
        self.text_search_invocations = 0

    def begin_episode(self, example: Any, image: Any) -> None:
        self.begin_episode_calls += 1

    def image_search(self, image_sha256: str) -> str:
        self.visual_search_invocations += 1
        raise AssertionError("NO_SEARCH_BACKEND_INVOCATION: visual_search")

    def text_search(self, query: str) -> str:
        self.text_search_invocations += 1
        raise AssertionError("NO_SEARCH_BACKEND_INVOCATION: text_search")

    def episode_log(self) -> list[dict[str, Any]]:
        return []

    def audit(self) -> dict[str, int | bool]:
        return {
            "search_subsystem_disabled": True,
            "begin_episode_calls": self.begin_episode_calls,
            "executed_visual_search_calls": self.visual_search_invocations,
            "executed_text_search_calls": self.text_search_invocations,
            "retrieval_backend_calls": self.visual_search_invocations + self.text_search_invocations,
            "information_injection_count": 0,
        }


__all__ = ["FORBIDDEN_ACTION_OPENINGS", "ToolAbsentEnvironment"]
