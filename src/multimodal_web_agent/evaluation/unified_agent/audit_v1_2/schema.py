from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


EVIDENCE_AUDIT_SCHEMA = "unified-agent-eval-evidence-audit-v1"
ANSWER_EVALUATOR_SCHEMA = "unified-answer-evaluator-v2"
AUDIT_RUN_SCHEMA = "unified-agent-eval-v1-2-audit-run-v1"


@dataclass(frozen=True)
class EvidenceAuditRecord:
    episode_id: str
    model: str
    task_type: str
    search_required: bool
    source_dataset: str
    source_split: str
    question: str
    accepted_answers: list[str]
    final_answer: str | None
    protocol_valid: bool
    finished_with_answer: bool
    em_v1: int
    token_f1_v1: float
    em_v2_strict: int
    token_f1_v2: float
    semantic_equivalence_v2: bool
    numeric_equivalence_v2: bool
    unit_equivalence_v2: bool
    alias_equivalence_v2: bool
    search_attempt_count: int
    successful_search_count: int
    image_search_count: int
    text_search_count: int
    tool_budget_exceeded: bool
    tool_execution_failure: bool
    retrieved_information: list[dict[str, Any]]
    information_total_chars: int
    information_truncated: bool
    information_truncation_unknown: bool
    exact_answer_string_hit: bool
    normalized_answer_hit: bool
    numeric_answer_hit: bool
    unit_equivalent_hit: bool
    alias_answer_hit: bool
    evidence_hit_any: bool
    evidence_hit_rank: int | None
    evidence_hit_tool: str | None
    evidence_hit_turn: int | None
    evidence_quadrant_v1: str
    evidence_available_final_correct: bool
    evidence_available_final_wrong: bool
    evidence_available_no_answer: bool
    evidence_absent_final_correct: bool
    evidence_absent_final_wrong: bool
    evidence_available_final_correct_v2: bool
    evidence_available_final_wrong_v2: bool
    evidence_absent_final_correct_v2: bool
    evidence_absent_final_wrong_v2: bool
    entity_copy_failure: bool
    relation_extraction_failure: bool
    answer_span_copy_failure: bool
    termination_failure: bool
    repeat_search_failure: bool
    query_quality_failure: bool
    first_action: str
    action_sequence: list[str]
    final_answer_normalized: str
    diagnostic_rule_types: dict[str, str]
    diagnostic_notes: list[str] = field(default_factory=list)
    schema_version: str = EVIDENCE_AUDIT_SCHEMA

    def validate(self) -> None:
        if self.schema_version != EVIDENCE_AUDIT_SCHEMA:
            raise ValueError("Evidence Audit schema mismatch")
        if self.model not in {"raw", "sft", "reward_v21", "stage2"}:
            raise ValueError("unsupported audited model")
        if self.search_attempt_count < self.successful_search_count:
            raise ValueError("successful searches exceed attempts")
        if self.successful_search_count != (
            self.image_search_count + self.text_search_count
        ):
            raise ValueError("successful tool counts are inconsistent")
        layers = (
            self.exact_answer_string_hit,
            self.normalized_answer_hit,
            self.numeric_answer_hit,
            self.unit_equivalent_hit,
            self.alias_answer_hit,
        )
        if self.evidence_hit_any != any(layers):
            raise ValueError("evidence_hit_any differs from layer union")
        if self.evidence_hit_any != (self.evidence_hit_rank is not None):
            raise ValueError("evidence hit rank presence is inconsistent")
        allowed_quadrants = {
            "not_applicable_no_successful_search",
            "A_evidence_hit_final_correct",
            "B_evidence_hit_final_wrong",
            "C_no_evidence_hit_final_correct",
            "D_no_evidence_hit_final_wrong",
            "E_evidence_hit_no_final_answer",
            "H_no_evidence_hit_no_final_answer",
        }
        if self.evidence_quadrant_v1 not in allowed_quadrants:
            raise ValueError("invalid evidence utilization quadrant")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvidenceAuditRecord":
        record = cls(**dict(value))
        record.validate()
        return record
