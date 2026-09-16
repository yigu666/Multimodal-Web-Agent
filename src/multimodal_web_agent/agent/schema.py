from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .protocol_errors import ProtocolError


class ActionType(str, Enum):
    IMAGE_SEARCH = "image_search"
    TEXT_SEARCH = "text_search"
    ANSWER = "answer"


@dataclass(frozen=True)
class ParsedAction:
    valid: bool
    action_type: Optional[ActionType]
    reason: Optional[str]
    content: Optional[str]
    error_code: Optional[ProtocolError]
    raw_text: str
