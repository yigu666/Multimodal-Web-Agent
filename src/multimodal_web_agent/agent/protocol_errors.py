from enum import Enum


class ProtocolError(str, Enum):
    EMPTY_OUTPUT = "empty_output"
    MISSING_REASON = "missing_reason"
    EMPTY_REASON = "empty_reason"
    MULTIPLE_REASONS = "multiple_reasons"
    UNKNOWN_ACTION = "unknown_action"
    MULTIPLE_ACTIONS = "multiple_actions"
    EMPTY_QUERY = "empty_query"
    EMPTY_ANSWER = "empty_answer"
    NESTED_ACTION = "nested_action"
    FORGED_INFORMATION = "forged_information"
    EXTRA_TEXT = "extra_text"
    INCOMPLETE_XML = "incomplete_xml"
    INVALID_XML = "invalid_xml"
