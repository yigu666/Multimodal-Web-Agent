from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from .cache_reader import CACHE_SOURCE, CACHE_VERSION, CacheMissError, ImageSearchCache, sha256_file
from .context_leak import detect_unavailable_context_leak
from .entity_extractor import (
    ExtractedEntity,
    entity_visible_in_information,
    extract_visible_entity,
)
from .fvqa_reader import FVQARecord, read_fvqa_records
from .information_formatter import (
    InformationBundle,
    contains_forbidden_visual_placeholder,
    format_image_information,
    format_text_information,
)
from .image_evidence import ImageEvidenceScore, score_image_evidence
from .previous_dataset import load_and_validate_previous_trajectories
from .query_builder import (
    build_bootstrap_query,
    build_image_context_query,
    contains_answer_leak,
    query_contract_errors,
)
from .relation_mapper import RelationMatch, map_question_relation
from .rejection import RejectionReason, make_rejection, rejection_distributions
from .schema import Message, RouteType, Trajectory, TrajectoryStep, TransitionType
from .splitter import (
    FULL_V0_2_ROUTE_SPLIT_QUOTAS,
    assign_splits,
    assign_splits_by_route_quotas,
    proportional_route_split_quotas,
    proportional_state_action_targets,
    state_action_targets_from_assignments,
)
from .templates import (
    answer_action,
    image_search_action,
    initial_state,
    select_image_text_initial_reason,
    select_reason,
    text_search_action,
)
from .text_retriever import BootstrapTextRetriever, text_backend_provenance
from .unfolder import group_examples_by_split, unfold_trajectories
from .verifier import information_supports_answer


def resolve_project_path(project_root: Path, configured_path: Union[str, Path]) -> Path:
    path = Path(configured_path)
    if path.is_absolute():
        return path
    return Path(project_root) / path


def protocol_artifact_stem(schema_version: str) -> str:
    if schema_version == "protocol-sft-v0":
        return "protocol_sft_v0"
    if schema_version == "protocol-sft-v0.1":
        return "protocol_sft_v0_1"
    if schema_version == "protocol-sft-v0.2":
        return "protocol_sft_v0_2"
    if schema_version == "protocol-sft-v0.3":
        return "protocol_sft_v0_3"
    if schema_version == "protocol-sft-v0.4":
        return "protocol_sft_v0_4"
    raise ValueError("unsupported Protocol-SFT schema version: %s" % schema_version)


@dataclass(frozen=True)
class BuildConfig:
    seed: int = 20260722
    direct_trajectories: int = 200
    image_search_trajectories: int = 200
    text_search_trajectories: int = 200
    state_action_examples: int = 1000
    split_targets: Mapping[str, int] = field(
        default_factory=lambda: {"train": 800, "dev": 100, "test": 100}
    )
    reason_max_tokens: int = 48
    text_query_min_tokens: int = 3
    text_query_max_tokens: int = 32
    text_top_k: int = 3
    image_top_k: int = 3
    tokenization_model_path: Optional[str] = None
    max_seq_len: int = 1536
    visual_token_target: int = 256
    target_reserve: int = 96
    schema_version: str = "protocol-sft-v0"
    allow_shortfall: bool = False
    visible_context_only: bool = True
    allow_cache_title_input: bool = False
    unavailable_context_min_ngram: int = 4
    unavailable_context_overlap_threshold: float = 0.5
    strip_visual_placeholders: bool = True
    min_unique_reasons_per_transition: int = 1
    manual_audit_direct_count: int = 0
    manual_audit_image_count: int = 0
    manual_audit_text_count: int = 0
    manual_audit_seed: int = 20260722
    image_text_search_trajectories: int = 0
    logical_trajectories: Optional[int] = None
    previous_dataset_dir: Optional[str] = None
    previous_audit_path: Optional[str] = None
    require_previous_dataset: bool = True
    allow_previous_global_shortfall_failure: bool = True
    reject_other_previous_quality_failures: bool = True
    required_previous_trajectories: int = 480
    required_previous_state_actions: int = 772
    expected_previous_route_counts: Mapping[str, int] = field(
        default_factory=lambda: {
            "direct_answer": 202,
            "image_search_answer": 200,
            "text_search_answer": 64,
            "image_text_search_answer": 14,
        }
    )
    new_direct_trajectories: int = 0
    new_image_search_trajectories: int = 0
    route_split_quotas: Mapping[str, Mapping[str, int]] = field(
        default_factory=dict
    )
    direct_category: str = "search_free"
    image_category: str = "search_required"
    require_image_cache_hit: bool = True
    require_answer_in_image_information: bool = True
    deterministic_evidence_ranking: bool = True
    relation_required: bool = True
    entity_must_be_visible: bool = True
    answer_must_not_be_in_image_information: bool = True
    answer_must_be_in_text_information: bool = True
    exclude_image_context_documents_from_text_evidence: bool = True
    image_text_image_top_k: int = 1
    manual_audit_image_text_count: int = 0
    manual_audit_new_direct_count: int = 0
    manual_audit_new_image_count: int = 0
    manual_audit_reused_image_count: int = 0

    def validate(self) -> None:
        logical = (
            self.direct_trajectories
            + self.image_search_trajectories
            + self.text_search_trajectories
            + self.image_text_search_trajectories
        )
        expanded = self.direct_trajectories + 2 * (
            self.image_search_trajectories + self.text_search_trajectories
        ) + 3 * self.image_text_search_trajectories
        if logical <= 0 or min(
            self.direct_trajectories,
            self.image_search_trajectories,
            self.text_search_trajectories,
            self.image_text_search_trajectories,
        ) < 0:
            raise ValueError("trajectory targets must be non-negative and non-empty")
        if expanded != self.state_action_examples:
            raise ValueError("state_action_examples does not match route expansion")
        if sum(self.split_targets.values()) != self.state_action_examples:
            raise ValueError("split targets do not sum to state_action_examples")
        if self.text_query_min_tokens < 1 or self.text_query_max_tokens < self.text_query_min_tokens:
            raise ValueError("invalid text query token limits")
        if self.reason_max_tokens < 1:
            raise ValueError("reason_max_tokens must be positive")
        if min(self.text_top_k, self.image_top_k, self.image_text_image_top_k) < 1:
            raise ValueError("retrieval top_k values must be positive")
        if self.tokenization_model_path and min(
            self.max_seq_len, self.visual_token_target, self.target_reserve
        ) <= 0:
            raise ValueError("tokenization preflight limits must be positive")
        if self.logical_trajectories is not None and logical != self.logical_trajectories:
            raise ValueError("logical_trajectories does not match route targets")
        if self.schema_version not in {
            "protocol-sft-v0",
            "protocol-sft-v0.1",
            "protocol-sft-v0.2",
            "protocol-sft-v0.3",
            "protocol-sft-v0.4",
        }:
            raise ValueError("unsupported Protocol-SFT schema version")
        if self.schema_version == "protocol-sft-v0.1":
            if not self.visible_context_only or self.allow_cache_title_input:
                raise ValueError("v0.1 queries must use visible context only")
            if not self.strip_visual_placeholders:
                raise ValueError("v0.1 must strip visual placeholders")
        if self.schema_version == "protocol-sft-v0.2":
            if not self.previous_dataset_dir:
                raise ValueError("v0.2 requires previous_dataset_dir")
            if self.image_text_search_trajectories < 1:
                raise ValueError("v0.2 requires image-text-search trajectories")
            if not self.strip_visual_placeholders:
                raise ValueError("v0.2 must strip visual placeholders")
        if self.schema_version == "protocol-sft-v0.3":
            if not self.previous_dataset_dir or not self.previous_audit_path:
                raise ValueError("v0.3 requires previous dataset and audit paths")
            if (
                not self.require_previous_dataset
                or not self.allow_previous_global_shortfall_failure
                or not self.reject_other_previous_quality_failures
            ):
                raise ValueError(
                    "v0.3 requires validated v0.2 reuse and permits only its documented shortfall"
                )
            expected_previous = dict(self.expected_previous_route_counts)
            if set(expected_previous) != {
                RouteType.DIRECT_ANSWER.value,
                RouteType.IMAGE_SEARCH_ANSWER.value,
                RouteType.TEXT_SEARCH_ANSWER.value,
                RouteType.IMAGE_TEXT_SEARCH_ANSWER.value,
            }:
                raise ValueError("v0.3 previous route expectations are incomplete")
            if self.direct_trajectories - expected_previous[
                RouteType.DIRECT_ANSWER.value
            ] != self.new_direct_trajectories:
                raise ValueError("v0.3 new direct target is inconsistent")
            if self.image_search_trajectories - expected_previous[
                RouteType.IMAGE_SEARCH_ANSWER.value
            ] != self.new_image_search_trajectories:
                raise ValueError("v0.3 new image target is inconsistent")
            if self.text_search_trajectories != expected_previous[
                RouteType.TEXT_SEARCH_ANSWER.value
            ] or self.image_text_search_trajectories != expected_previous[
                RouteType.IMAGE_TEXT_SEARCH_ANSWER.value
            ]:
                raise ValueError("v0.3 must freeze reused text and image-text routes")
            if not self.strip_visual_placeholders:
                raise ValueError("v0.3 must strip visual placeholders")
            if (
                self.direct_category != "search_free"
                or self.image_category != "search_required"
                or not self.require_image_cache_hit
                or not self.require_answer_in_image_information
                or not self.deterministic_evidence_ranking
            ):
                raise ValueError("v0.3 selection quality requirements cannot be disabled")
            if set(self.route_split_quotas) != set(expected_previous):
                raise ValueError("v0.3 route split quotas are incomplete")
            for route, quotas in self.route_split_quotas.items():
                if set(quotas) != {"train", "dev", "test"}:
                    raise ValueError("v0.3 route split quota is incomplete: %s" % route)
            configured_targets = {
                RouteType.DIRECT_ANSWER.value: self.direct_trajectories,
                RouteType.IMAGE_SEARCH_ANSWER.value: self.image_search_trajectories,
                RouteType.TEXT_SEARCH_ANSWER.value: self.text_search_trajectories,
                RouteType.IMAGE_TEXT_SEARCH_ANSWER.value:
                    self.image_text_search_trajectories,
            }
            if any(
                sum(self.route_split_quotas[route].values()) != target
                for route, target in configured_targets.items()
            ):
                raise ValueError("v0.3 route split quotas do not match targets")
        if self.unavailable_context_min_ngram < 1:
            raise ValueError("unavailable-context min_ngram must be positive")
        if not 0.0 <= self.unavailable_context_overlap_threshold <= 1.0:
            raise ValueError("invalid unavailable-context overlap threshold")
        if self.min_unique_reasons_per_transition < 1:
            raise ValueError("minimum unique reasons must be positive")
        if min(
            self.manual_audit_direct_count,
            self.manual_audit_image_count,
            self.manual_audit_text_count,
            self.manual_audit_image_text_count,
            self.manual_audit_new_direct_count,
            self.manual_audit_new_image_count,
            self.manual_audit_reused_image_count,
        ) < 0:
            raise ValueError("manual audit counts cannot be negative")


@dataclass(frozen=True)
class BuildResult:
    trajectories: List[Trajectory]
    examples_by_split: Dict[str, List[Any]]
    rejected: List[Dict[str, Any]]
    manifest: Dict[str, Any]


class DatasetBuildError(RuntimeError):
    def __init__(self, message: str, rejected: Optional[List[Dict[str, Any]]] = None):
        super().__init__(message)
        self.rejected = rejected or []


@dataclass(frozen=True)
class TextEvidenceDecision:
    query: Optional[str]
    information: Optional[InformationBundle]
    rejection_reason: Optional[RejectionReason]
    details: Dict[str, Any]


@dataclass(frozen=True)
class ImageTextEvidenceDecision:
    entity: Optional[ExtractedEntity]
    relation: Optional[RelationMatch]
    query: Optional[str]
    image_information: Optional[InformationBundle]
    text_information: Optional[InformationBundle]
    rejection_reason: Optional[RejectionReason]
    details: Dict[str, Any]


@dataclass(frozen=True)
class RankedImageCandidate:
    record: FVQARecord
    information: InformationBundle
    evidence_score: ImageEvidenceScore


def _stable_order(records: Iterable[FVQARecord], seed: int, route: str) -> List[FVQARecord]:
    return sorted(
        records,
        key=lambda record: (
            hashlib.sha256(
                ("%d:%s:%s" % (seed, route, record.data_id)).encode("utf-8")
            ).hexdigest(),
            record.data_id,
        ),
    )


def _source(record: FVQARecord, source_label: str) -> Dict[str, Any]:
    return {
        "dataset": "FVQA train",
        "dataset_label": source_label,
        "data_source": record.data_source,
        "category": record.category,
        "source_row_index": record.row_index,
    }


def _trajectory_namespace(schema_version: str) -> str:
    return schema_version.replace("protocol-sft-", "protocol_sft_").replace(".", "_")


def _post_search_state(
    record: FVQARecord,
    search_action: str,
    information: InformationBundle,
) -> List[Message]:
    return initial_state(record.question) + [
        Message(role="assistant", content=search_action, trainable=False),
        Message(role="tool", content=information.text, trainable=False),
    ]


def _direct_trajectory(
    record: FVQARecord,
    source_label: str,
    config: BuildConfig,
    source_metadata: Optional[Mapping[str, Any]] = None,
) -> Trajectory:
    transition = TransitionType.INITIAL_TO_DIRECT_ANSWER.value
    trajectory = Trajectory(
        trajectory_id="%s:direct:%s"
        % (_trajectory_namespace(config.schema_version), record.data_id),
        source_data_id=record.data_id,
        route=RouteType.DIRECT_ANSWER.value,
        question=record.question,
        canonical_answer=record.canonical_answer,
        accepted_answers=record.accepted_answers,
        steps=[
            TrajectoryStep(
                transition=transition,
                state=initial_state(record.question),
                target=answer_action(
                    record.canonical_answer,
                    select_reason(
                        transition,
                        record.data_id,
                        config.seed,
                        forbidden_answers=record.accepted_answers,
                    ),
                ),
                image_refs=[record.image_ref],
            )
        ],
        source={**_source(record, source_label), **dict(source_metadata or {})},
        schema_version=config.schema_version,
    )
    trajectory.validate()
    return trajectory


def _image_trajectory(
    record: FVQARecord,
    information: InformationBundle,
    source_label: str,
    config: BuildConfig,
    source_metadata: Optional[Mapping[str, Any]] = None,
) -> Trajectory:
    search_transition = TransitionType.INITIAL_TO_IMAGE_SEARCH.value
    answer_transition = TransitionType.IMAGE_INFORMATION_TO_ANSWER.value
    search_action = image_search_action(
        select_reason(
            search_transition,
            record.data_id,
            config.seed,
            forbidden_answers=record.accepted_answers,
        )
    )
    trajectory = Trajectory(
        trajectory_id="%s:image:%s"
        % (_trajectory_namespace(config.schema_version), record.data_id),
        source_data_id=record.data_id,
        route=(
            RouteType.IMAGE_SEARCH_ANSWER.value
            if config.schema_version in {
                "protocol-sft-v0.2",
                "protocol-sft-v0.3",
                "protocol-sft-v0.4",
            }
            else RouteType.IMAGE_SEARCH.value
        ),
        question=record.question,
        canonical_answer=record.canonical_answer,
        accepted_answers=record.accepted_answers,
        steps=[
            TrajectoryStep(
                transition=search_transition,
                state=initial_state(record.question),
                target=search_action,
                image_refs=[record.image_ref],
            ),
            TrajectoryStep(
                transition=answer_transition,
                state=_post_search_state(record, search_action, information),
                target=answer_action(
                    record.canonical_answer,
                    select_reason(
                        answer_transition,
                        record.data_id,
                        config.seed,
                        forbidden_answers=record.accepted_answers,
                    ),
                ),
                image_refs=[record.image_ref] + information.image_refs,
                information_provenance=information.provenance,
            ),
        ],
        source={**_source(record, source_label), **dict(source_metadata or {})},
        schema_version=config.schema_version,
    )
    trajectory.validate()
    return trajectory


def _text_trajectory(
    record: FVQARecord,
    query: str,
    information: InformationBundle,
    source_label: str,
    config: BuildConfig,
) -> Trajectory:
    search_transition = TransitionType.INITIAL_TO_TEXT_SEARCH.value
    answer_transition = TransitionType.TEXT_INFORMATION_TO_ANSWER.value
    search_action = text_search_action(
        query,
        select_reason(
            search_transition,
            record.data_id,
            config.seed,
            forbidden_answers=record.accepted_answers,
        ),
    )
    trajectory = Trajectory(
        trajectory_id="%s:text:%s"
        % (_trajectory_namespace(config.schema_version), record.data_id),
        source_data_id=record.data_id,
        route=RouteType.TEXT_SEARCH.value,
        question=record.question,
        canonical_answer=record.canonical_answer,
        accepted_answers=record.accepted_answers,
        steps=[
            TrajectoryStep(
                transition=search_transition,
                state=initial_state(record.question),
                target=search_action,
                image_refs=[record.image_ref],
            ),
            TrajectoryStep(
                transition=answer_transition,
                state=_post_search_state(record, search_action, information),
                target=answer_action(
                    record.canonical_answer,
                    select_reason(
                        answer_transition,
                        record.data_id,
                        config.seed,
                        forbidden_answers=record.accepted_answers,
                    ),
                ),
                image_refs=[record.image_ref],
                information_provenance=information.provenance,
            ),
        ],
        source=_source(record, source_label),
        schema_version=config.schema_version,
    )
    trajectory.validate()
    return trajectory


def _find_text_evidence(
    record: FVQARecord,
    cache: ImageSearchCache,
    retriever: BootstrapTextRetriever,
    config: BuildConfig,
) -> TextEvidenceDecision:
    entry = cache.get(record.data_id)
    try:
        query = build_bootstrap_query(record.question, visible_context=None)
    except (TypeError, ValueError) as exc:
        return TextEvidenceDecision(
            None,
            None,
            RejectionReason.EMPTY_QUERY,
            {"error": str(exc), "query_construction_sources": ["question"]},
        )

    contract_errors = query_contract_errors(
        query, config.text_query_min_tokens, config.text_query_max_tokens
    )
    if contract_errors:
        mapping = {
            "empty_query": RejectionReason.EMPTY_QUERY,
            "query_too_short": RejectionReason.QUERY_TOO_SHORT,
            "query_too_long": RejectionReason.QUERY_TOO_LONG,
        }
        reason = next(
            (mapping[error] for error in contract_errors if error in mapping),
            RejectionReason.EMPTY_QUERY,
        )
        return TextEvidenceDecision(
            None,
            None,
            reason,
            {"query": query, "contract_errors": contract_errors},
        )

    if contains_answer_leak(query, record.accepted_answers):
        return TextEvidenceDecision(
            None,
            None,
            RejectionReason.ANSWER_LEAK,
            {"query": query, "query_construction_sources": ["question"]},
        )

    unavailable_titles = (
        [
            ("image_cache_title:%d" % index, title)
            for index, title in entry.usable_titles
        ]
        if entry is not None
        else []
    )
    context_leak = detect_unavailable_context_leak(
        query,
        visible_texts=[record.question],
        unavailable_texts=unavailable_titles,
        min_ngram=config.unavailable_context_min_ngram,
        overlap_threshold=config.unavailable_context_overlap_threshold,
    )
    if context_leak.leaked:
        return TextEvidenceDecision(
            None,
            None,
            RejectionReason.UNAVAILABLE_CONTEXT_LEAK,
            {
                "query": query,
                "matched_source": context_leak.matched_source,
                "matched_title": context_leak.matched_text,
                "overlap_ratio": context_leak.overlap_ratio,
                "longest_matching_ngram": context_leak.longest_matching_ngram,
            },
        )

    hits = retriever.retrieve(query, top_k=config.text_top_k)
    if not hits:
        return TextEvidenceDecision(
            None,
            None,
            RejectionReason.EVIDENCE_MISS,
            {"query": query, "normal_hit_count": 0},
        )
    provenance = text_backend_provenance(cache, hits)
    information = format_text_information(hits, provenance)
    evidence_hit_normal = information_supports_answer(
        information.text, record.accepted_answers
    )
    if not evidence_hit_normal:
        return TextEvidenceDecision(
            None,
            None,
            RejectionReason.EVIDENCE_MISS,
            {
                "query": query,
                "normal_document_ids": [hit.document.document_id for hit in hits],
                "normal_hit_count": len(hits),
            },
        )

    leave_one_out_hits = retriever.retrieve(
        query,
        top_k=config.text_top_k,
        exclude_source_data_id=record.data_id,
    )
    leave_one_out_text = "\n".join(hit.document.text for hit in leave_one_out_hits)
    evidence_hit_leave_one_out = information_supports_answer(
        leave_one_out_text, record.accepted_answers
    )
    provenance.update(
        {
            "query_construction_sources": ["question"],
            "visible_context_only": True,
            "cache_titles_used_to_build_query": False,
            "unavailable_context_titles_checked": len(unavailable_titles),
            "unavailable_context_leak": False,
            "evidence_hit_normal": True,
            "evidence_hit_leave_one_source_out": evidence_hit_leave_one_out,
            "leave_one_source_out_document_ids": [
                hit.document.document_id for hit in leave_one_out_hits
            ],
            "leave_one_source_out_scores": [
                round(hit.score, 8) for hit in leave_one_out_hits
            ],
        }
    )
    information = format_text_information(hits, provenance)
    return TextEvidenceDecision(
        query,
        information,
        None,
        {
            "evidence_hit_normal": True,
            "evidence_hit_leave_one_source_out": evidence_hit_leave_one_out,
        },
    )


def _image_result_records(entry: Any, top_k: int) -> List[Dict[str, Any]]:
    return [
        {
            "rank": rank,
            "document_id": "%s:title:%d" % (entry.data_id, result_index),
            "title": title,
        }
        for rank, (result_index, title, _descriptor) in enumerate(
            entry.usable_image_results[:top_k], start=1
        )
    ]


def _image_text_rejection_details(
    record: FVQARecord,
    image_results: Sequence[Mapping[str, Any]],
    entity: Optional[ExtractedEntity] = None,
    **extra: Any,
) -> Dict[str, Any]:
    details: Dict[str, Any] = {
        "question": record.question,
        "image_titles": [str(result.get("title", "")) for result in image_results],
    }
    if entity is not None and entity.entity:
        details["identified_entity"] = entity.entity
    details.update(extra)
    return details


def _find_image_text_evidence(
    record: FVQARecord,
    cache: ImageSearchCache,
    retriever: BootstrapTextRetriever,
    config: BuildConfig,
) -> ImageTextEvidenceDecision:
    # The three-step route deliberately uses a compact image context.  Only the
    # titles actually shown to the model are excluded from the later text
    # retrieval.  This keeps the route causal: the image result identifies the
    # entity, while an independent text document must still supply the answer.
    image_context_top_k = config.image_text_image_top_k
    try:
        entry = cache.require(record.data_id)
        image_information = format_image_information(
            entry, top_k=image_context_top_k
        )
    except (CacheMissError, ValueError) as exc:
        return ImageTextEvidenceDecision(
            None,
            None,
            None,
            None,
            None,
            RejectionReason.CACHE_MISS,
            {"question": record.question, "error": str(exc)},
        )

    image_results = _image_result_records(entry, image_context_top_k)
    if contains_forbidden_visual_placeholder(image_information.text):
        return ImageTextEvidenceDecision(
            None,
            None,
            None,
            None,
            None,
            RejectionReason.FORBIDDEN_VISUAL_PLACEHOLDER,
            _image_text_rejection_details(record, image_results),
        )
    if config.answer_must_not_be_in_image_information and information_supports_answer(
        image_information.text, record.accepted_answers
    ):
        answer_visible_ranks = [
            int(result.get("rank", 0))
            for result in image_results
            if information_supports_answer(
                str(result.get("title", "")), record.accepted_answers
            )
        ]
        return ImageTextEvidenceDecision(
            None,
            None,
            None,
            image_information,
            None,
            RejectionReason.ANSWER_ALREADY_IN_IMAGE_INFORMATION,
            _image_text_rejection_details(
                record,
                image_results,
                answer_visible_ranks=answer_visible_ranks,
            ),
        )

    entity = extract_visible_entity(image_results)
    if not entity.valid:
        reason = {
            RejectionReason.IMAGE_ENTITY_TOO_GENERIC.value:
                RejectionReason.IMAGE_ENTITY_TOO_GENERIC,
            RejectionReason.IMAGE_ENTITY_NOT_FOUND.value:
                RejectionReason.IMAGE_ENTITY_NOT_FOUND,
        }.get(
            entity.rejection_reason or "",
            RejectionReason.IMAGE_ENTITY_NOT_FOUND,
        )
        return ImageTextEvidenceDecision(
            entity,
            None,
            None,
            image_information,
            None,
            reason,
            _image_text_rejection_details(record, image_results),
        )
    entity_is_visible = entity_visible_in_information(
        entity.entity, image_information.text
    )
    if config.entity_must_be_visible and not entity_is_visible:
        return ImageTextEvidenceDecision(
            entity,
            None,
            None,
            image_information,
            None,
            RejectionReason.IMAGE_ENTITY_NOT_FOUND,
            _image_text_rejection_details(
                record,
                image_results,
                entity,
                entity_visible_in_image_information=False,
            ),
        )

    relation = map_question_relation(record.question)
    if config.relation_required and not relation.matched:
        return ImageTextEvidenceDecision(
            entity,
            relation,
            None,
            image_information,
            None,
            RejectionReason.RELATION_NOT_FOUND,
            _image_text_rejection_details(record, image_results, entity),
        )
    try:
        query = build_image_context_query(record.question, entity.entity, relation)
    except ValueError as exc:
        return ImageTextEvidenceDecision(
            entity,
            relation,
            None,
            image_information,
            None,
            RejectionReason.EMPTY_QUERY,
            _image_text_rejection_details(
                record, image_results, entity, error=str(exc)
            ),
        )
    contract_errors = query_contract_errors(
        query, config.text_query_min_tokens, config.text_query_max_tokens
    )
    if contract_errors:
        reason = (
            RejectionReason.QUERY_TOO_SHORT
            if "query_too_short" in contract_errors
            else RejectionReason.QUERY_TOO_LONG
            if "query_too_long" in contract_errors
            else RejectionReason.EMPTY_QUERY
        )
        return ImageTextEvidenceDecision(
            entity,
            relation,
            query,
            image_information,
            None,
            reason,
            _image_text_rejection_details(
                record,
                image_results,
                entity,
                query=query,
                contract_errors=contract_errors,
            ),
        )
    if contains_answer_leak(query, record.accepted_answers):
        return ImageTextEvidenceDecision(
            entity,
            relation,
            query,
            image_information,
            None,
            RejectionReason.ANSWER_LEAK,
            _image_text_rejection_details(
                record, image_results, entity, query=query
            ),
        )

    image_context_document_ids = set(
        str(value)
        for value in image_information.provenance.get("document_ids", [])
    )
    normal_hits = retriever.retrieve(query, top_k=config.text_top_k)
    text_hits = retriever.retrieve(
        query,
        top_k=config.text_top_k,
        exclude_document_ids=(
            image_context_document_ids
            if config.exclude_image_context_documents_from_text_evidence
            else ()
        ),
    )
    normal_answer_hits = [
        hit for hit in normal_hits
        if information_supports_answer(hit.document.text, record.accepted_answers)
    ]
    independent_answer_hits = [
        hit for hit in text_hits
        if hit.document.document_id not in image_context_document_ids
        and information_supports_answer(hit.document.text, record.accepted_answers)
    ]
    if not independent_answer_hits:
        same_context_only = bool(normal_answer_hits) and all(
            hit.document.document_id in image_context_document_ids
            for hit in normal_answer_hits
        )
        return ImageTextEvidenceDecision(
            entity,
            relation,
            query,
            image_information,
            None,
            (
                RejectionReason.TEXT_EVIDENCE_ONLY_IMAGE_CONTEXT
                if same_context_only
                else RejectionReason.TEXT_EVIDENCE_MISS
            ),
            _image_text_rejection_details(
                record,
                image_results,
                entity,
                query=query,
                image_context_document_ids=sorted(image_context_document_ids),
                normal_document_ids=[
                    hit.document.document_id for hit in normal_hits
                ],
                independent_document_ids=[
                    hit.document.document_id for hit in text_hits
                ],
            ),
        )

    text_provenance = text_backend_provenance(cache, text_hits)
    text_provenance.update(
        {
            "query_provenance": {
                "relation_source": "question",
                "entity_source": "visible_image_information",
                "used_ground_truth": False,
                "used_future_text_result": False,
            },
            "identified_entity": entity.entity,
            "entity_source_rank": entity.source_rank,
            "entity_source_doc_id": entity.source_document_id,
            "entity_source_title": entity.source_title,
            "entity_extraction_method": entity.extraction_method,
            "entity_visible_in_image_information": entity_is_visible,
            "relation": relation.relation,
            "relation_query_prefix": relation.query_prefix,
            "relation_pattern_name": relation.pattern_name,
            "image_context_document_ids": sorted(image_context_document_ids),
            "text_result_document_ids": [
                hit.document.document_id for hit in text_hits
            ],
            "answer_evidence_document_ids": [
                hit.document.document_id for hit in independent_answer_hits
            ],
            "text_evidence_hit_excluding_image_context_docs": True,
            "text_evidence_cross_source_count": sum(
                hit.document.source_data_id != record.data_id
                for hit in independent_answer_hits
            ),
            "image_information_provenance": image_information.provenance,
            "unavailable_context_leak": False,
        }
    )
    text_information = format_text_information(text_hits, text_provenance)
    if config.answer_must_be_in_text_information and not information_supports_answer(
        text_information.text, record.accepted_answers
    ):
        return ImageTextEvidenceDecision(
            entity,
            relation,
            query,
            image_information,
            None,
            RejectionReason.TEXT_EVIDENCE_MISS,
            _image_text_rejection_details(
                record, image_results, entity, query=query
            ),
        )
    context_leak = detect_unavailable_context_leak(
        query,
        visible_texts=[
            record.question,
            image_information.text,
            "%s %s" % (relation.query_prefix, entity.entity),
        ],
        unavailable_texts=[
            (hit.document.document_id, hit.document.text) for hit in text_hits
        ],
        min_ngram=config.unavailable_context_min_ngram,
        overlap_threshold=config.unavailable_context_overlap_threshold,
    )
    if context_leak.leaked:
        return ImageTextEvidenceDecision(
            entity,
            relation,
            query,
            image_information,
            None,
            RejectionReason.UNAVAILABLE_CONTEXT_LEAK,
            _image_text_rejection_details(
                record,
                image_results,
                entity,
                query=query,
                matched_source=context_leak.matched_source,
                matched_text=context_leak.matched_text,
                overlap_ratio=context_leak.overlap_ratio,
                longest_matching_ngram=context_leak.longest_matching_ngram,
            ),
        )
    image_provenance = dict(image_information.provenance)
    image_provenance.update(
        {
            "identified_entity": entity.entity,
            "entity_source_rank": entity.source_rank,
            "entity_source_doc_id": entity.source_document_id,
            "entity_source_title": entity.source_title,
            "entity_extraction_method": entity.extraction_method,
            "entity_visible_in_image_information": entity_is_visible,
            "image_context_document_ids": sorted(image_context_document_ids),
            "relation": relation.relation,
            "relation_query_prefix": relation.query_prefix,
            "relation_pattern_name": relation.pattern_name,
            "query_provenance": {
                "relation_source": "question",
                "entity_source": "visible_image_information",
                "used_ground_truth": False,
                "used_future_text_result": False,
            },
        }
    )
    image_information = InformationBundle(
        image_information.text,
        image_provenance,
        image_information.image_refs,
    )
    return ImageTextEvidenceDecision(
        entity,
        relation,
        query,
        image_information,
        text_information,
        None,
        {
            "text_evidence_hit_excluding_image_context_docs": True,
            "text_evidence_cross_source_count": text_provenance[
                "text_evidence_cross_source_count"
            ],
        },
    )


def _image_text_trajectory(
    record: FVQARecord,
    decision: ImageTextEvidenceDecision,
    source_label: str,
    config: BuildConfig,
) -> Trajectory:
    if not all(
        (
            decision.entity,
            decision.relation,
            decision.query,
            decision.image_information,
            decision.text_information,
        )
    ):
        raise ValueError("cannot build image-text trajectory from an incomplete decision")
    image_transition = TransitionType.INITIAL_TO_IMAGE_SEARCH.value
    text_transition = TransitionType.IMAGE_INFORMATION_TO_TEXT_SEARCH.value
    answer_transition = TransitionType.TEXT_INFORMATION_TO_ANSWER.value
    image_action = image_search_action(
        select_image_text_initial_reason(
            record.data_id,
            config.seed,
            forbidden_answers=record.accepted_answers,
        )
    )
    text_action = text_search_action(
        decision.query,
        select_reason(
            text_transition,
            record.data_id,
            config.seed,
            forbidden_answers=record.accepted_answers,
        ),
    )
    image_state = initial_state(record.question) + [
        Message(role="assistant", content=image_action, trainable=False),
        Message(
            role="tool",
            content=decision.image_information.text,
            trainable=False,
        ),
    ]
    text_state = image_state + [
        Message(role="assistant", content=text_action, trainable=False),
        Message(
            role="tool",
            content=decision.text_information.text,
            trainable=False,
        ),
    ]
    trajectory = Trajectory(
        trajectory_id="%s:image_text:%s"
        % (_trajectory_namespace(config.schema_version), record.data_id),
        source_data_id=record.data_id,
        route=RouteType.IMAGE_TEXT_SEARCH_ANSWER.value,
        question=record.question,
        canonical_answer=record.canonical_answer,
        accepted_answers=record.accepted_answers,
        steps=[
            TrajectoryStep(
                transition=image_transition,
                state=initial_state(record.question),
                target=image_action,
                image_refs=[record.image_ref],
            ),
            TrajectoryStep(
                transition=text_transition,
                state=image_state,
                target=text_action,
                image_refs=[record.image_ref],
                information_provenance=decision.image_information.provenance,
            ),
            TrajectoryStep(
                transition=answer_transition,
                state=text_state,
                target=answer_action(
                    record.canonical_answer,
                    select_reason(
                        answer_transition,
                        record.data_id,
                        config.seed,
                        forbidden_answers=record.accepted_answers,
                    ),
                ),
                image_refs=[record.image_ref],
                information_provenance=decision.text_information.provenance,
            ),
        ],
        source=_source(record, source_label),
        schema_version=config.schema_version,
    )
    trajectory.validate()
    return trajectory


def _read_previous_trajectories(previous_dataset_dir: Path) -> List[Trajectory]:
    path = Path(previous_dataset_dir) / "trajectories.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    trajectories = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                trajectory = Trajectory.from_dict(json.loads(line))
                trajectory.validate()
            except Exception as exc:
                raise DatasetBuildError(
                    "invalid previous trajectory at line %d: %s"
                    % (line_number, exc)
                ) from exc
            trajectories.append(trajectory)
    return trajectories


def _convert_previous_trajectory(trajectory: Trajectory) -> Trajectory:
    route = {
        RouteType.DIRECT_ANSWER.value: RouteType.DIRECT_ANSWER.value,
        RouteType.IMAGE_SEARCH.value: RouteType.IMAGE_SEARCH_ANSWER.value,
        RouteType.TEXT_SEARCH.value: RouteType.TEXT_SEARCH_ANSWER.value,
        RouteType.IMAGE_SEARCH_ANSWER.value: RouteType.IMAGE_SEARCH_ANSWER.value,
        RouteType.TEXT_SEARCH_ANSWER.value: RouteType.TEXT_SEARCH_ANSWER.value,
    }.get(trajectory.route)
    if route is None:
        raise DatasetBuildError(
            "previous dataset contains unsupported route: %s" % trajectory.route
        )
    suffix = (
        trajectory.trajectory_id.split(":", 1)[1]
        if ":" in trajectory.trajectory_id
        else trajectory.trajectory_id
    )
    converted = Trajectory(
        trajectory_id="protocol_sft_v0_2:%s" % suffix,
        source_data_id=trajectory.source_data_id,
        route=route,
        question=trajectory.question,
        canonical_answer=trajectory.canonical_answer,
        accepted_answers=trajectory.accepted_answers,
        steps=trajectory.steps,
        source={**trajectory.source, "reused_from_schema": trajectory.schema_version},
        schema_version="protocol-sft-v0.2",
    )
    converted.validate()
    return converted


def _convert_v0_2_trajectory_to_v0_3(trajectory: Trajectory) -> Trajectory:
    if trajectory.schema_version != "protocol-sft-v0.2":
        raise DatasetBuildError("v0.3 can only reuse protocol-sft-v0.2 trajectories")
    if trajectory.route not in {
        RouteType.DIRECT_ANSWER.value,
        RouteType.IMAGE_SEARCH_ANSWER.value,
        RouteType.TEXT_SEARCH_ANSWER.value,
        RouteType.IMAGE_TEXT_SEARCH_ANSWER.value,
    }:
        raise DatasetBuildError(
            "v0.2 contains an unsupported v0.3 reuse route: %s"
            % trajectory.route
        )
    suffix = (
        trajectory.trajectory_id.split(":", 1)[1]
        if ":" in trajectory.trajectory_id
        else trajectory.trajectory_id
    )
    converted = Trajectory(
        trajectory_id="protocol_sft_v0_3:%s" % suffix,
        source_data_id=trajectory.source_data_id,
        route=trajectory.route,
        question=trajectory.question,
        canonical_answer=trajectory.canonical_answer,
        accepted_answers=trajectory.accepted_answers,
        steps=trajectory.steps,
        source={
            **trajectory.source,
            "reused_from_schema": trajectory.schema_version,
            "v0_3_origin": "reused_v0_2",
        },
        schema_version="protocol-sft-v0.3",
    )
    converted.validate()
    return converted


def _build_v0_2_dataset(
    source_path: Path,
    cache_path: Path,
    previous_dataset_dir: Path,
    config: BuildConfig,
    source_label: Optional[str],
    cache_label: Optional[str],
) -> BuildResult:
    records = read_fvqa_records(source_path)
    cache = ImageSearchCache.load(cache_path, label=cache_label)
    retriever = BootstrapTextRetriever.from_image_cache(cache)
    source_name = source_label or source_path.name
    previous = _read_previous_trajectories(previous_dataset_dir)
    trajectories = [_convert_previous_trajectory(item) for item in previous]
    used_source_data_ids = {item.source_data_id for item in trajectories}
    if len(used_source_data_ids) != len(trajectories):
        raise DatasetBuildError("previous dataset reuses source_data_id values")

    previous_route_counts: Dict[str, int] = {}
    for trajectory in trajectories:
        previous_route_counts[trajectory.route] = (
            previous_route_counts.get(trajectory.route, 0) + 1
        )
    expected_reused = {
        RouteType.IMAGE_SEARCH_ANSWER.value: config.image_search_trajectories,
        RouteType.TEXT_SEARCH_ANSWER.value: config.text_search_trajectories,
    }
    for route, expected in expected_reused.items():
        if previous_route_counts.get(route, 0) != expected:
            raise DatasetBuildError(
                "previous dataset route %s has %d trajectories; expected %d"
                % (route, previous_route_counts.get(route, 0), expected)
            )
    previous_direct_count = previous_route_counts.get(
        RouteType.DIRECT_ANSWER.value, 0
    )
    additional_direct_target = config.direct_trajectories - previous_direct_count
    if additional_direct_target < 0:
        raise DatasetBuildError("previous direct route exceeds the v0.2 target")

    rejected: List[Dict[str, Any]] = []
    direct_pool = _stable_order(
        (
            record
            for record in records
            if record.category == config.direct_category
            and record.data_id not in used_source_data_ids
        ),
        config.seed,
        "v0_2_direct_addition",
    )
    direct_selected = direct_pool[:additional_direct_target]
    if len(direct_selected) < additional_direct_target and not config.allow_shortfall:
        raise DatasetBuildError(
            "v0.2 direct route shortage: required %d new sources, found %d"
            % (additional_direct_target, len(direct_selected)),
            rejected,
        )
    trajectories.extend(
        _direct_trajectory(record, source_name, config)
        for record in direct_selected
    )
    used_source_data_ids.update(record.data_id for record in direct_selected)

    image_text_pool = _stable_order(
        (
            record
            for record in records
            if record.category == "search_required"
            and record.data_id not in used_source_data_ids
        ),
        config.seed,
        RouteType.IMAGE_TEXT_SEARCH_ANSWER.value,
    )
    eligible: List[Tuple[FVQARecord, ImageTextEvidenceDecision]] = []
    for record in image_text_pool:
        decision = _find_image_text_evidence(record, cache, retriever, config)
        if (
            decision.rejection_reason is None
            and decision.image_information is not None
            and decision.text_information is not None
        ):
            eligible.append((record, decision))
        else:
            rejected.append(
                make_rejection(
                    record.data_id,
                    RouteType.IMAGE_TEXT_SEARCH_ANSWER.value,
                    decision.rejection_reason or RejectionReason.ROUTE_INELIGIBLE,
                    decision.details,
                )
            )
    selected_image_text = eligible[:config.image_text_search_trajectories]
    if (
        len(selected_image_text) < config.image_text_search_trajectories
        and not config.allow_shortfall
    ):
        raise DatasetBuildError(
            "image-text route shortage after scanning %d unused candidates: required %d, eligible %d"
            % (
                len(image_text_pool),
                config.image_text_search_trajectories,
                len(selected_image_text),
            ),
            rejected,
        )
    trajectories.extend(
        _image_text_trajectory(record, decision, source_name, config)
        for record, decision in selected_image_text
    )

    source_ids = [trajectory.source_data_id for trajectory in trajectories]
    if len(source_ids) != len(set(source_ids)):
        raise DatasetBuildError("v0.2 selected a source_data_id more than once", rejected)
    route_counts: Dict[str, int] = {}
    for trajectory in trajectories:
        route_counts[trajectory.route] = route_counts.get(trajectory.route, 0) + 1
    route_targets = {
        RouteType.DIRECT_ANSWER.value: config.direct_trajectories,
        RouteType.IMAGE_SEARCH_ANSWER.value: config.image_search_trajectories,
        RouteType.TEXT_SEARCH_ANSWER.value: config.text_search_trajectories,
        RouteType.IMAGE_TEXT_SEARCH_ANSWER.value:
            config.image_text_search_trajectories,
    }
    exact_route_counts = all(
        route_counts.get(route, 0) == target
        for route, target in route_targets.items()
    )
    if exact_route_counts and route_targets == {
        route: sum(quotas.values())
        for route, quotas in FULL_V0_2_ROUTE_SPLIT_QUOTAS.items()
    }:
        route_split_quotas = FULL_V0_2_ROUTE_SPLIT_QUOTAS
    else:
        route_split_quotas = proportional_route_split_quotas(trajectories)
    assignments = assign_splits_by_route_quotas(
        trajectories, route_split_quotas, config.seed
    )
    actual_split_targets = state_action_targets_from_assignments(
        trajectories, assignments
    )
    examples = unfold_trajectories(trajectories, assignments)
    grouped = group_examples_by_split(examples)
    transition_counts: Dict[str, int] = {}
    for example in examples:
        transition_counts[example.transition] = (
            transition_counts.get(example.transition, 0) + 1
        )
    actual_state_actions = len(examples)
    if (
        not config.allow_shortfall
        and actual_state_actions != config.state_action_examples
    ):
        raise DatasetBuildError(
            "v0.2 state-action count does not match the configured Full target",
            rejected,
        )

    rejection_audit = rejection_distributions(rejected)
    shortfall_routes = {
        route: max(0, target - route_counts.get(route, 0))
        for route, target in route_targets.items()
    }
    selected_provenance = [
        decision.text_information.provenance
        for _, decision in selected_image_text
        if decision.text_information is not None
    ]
    reason_distribution = rejection_audit["rejection_reason_distribution"]
    primary_rejection_reason = (
        max(reason_distribution, key=reason_distribution.get)
        if reason_distribution
        else None
    )
    manifest: Dict[str, Any] = {
        "schema_version": config.schema_version,
        "dataset_stage": "full",
        "seed": config.seed,
        "deterministic": True,
        "inputs": {
            "source_dataset_label": source_name,
            "source_dataset_sha256": sha256_file(source_path),
            "image_cache_label": cache.label,
            "image_cache_sha256": cache.file_sha256,
            "image_cache_source": CACHE_SOURCE,
            "image_cache_version": CACHE_VERSION,
            "previous_dataset_dir": str(previous_dataset_dir).replace("\\", "/"),
            "previous_trajectories_sha256": sha256_file(
                Path(previous_dataset_dir) / "trajectories.jsonl"
            ),
            "previous_logical_trajectories": len(previous),
            "previous_used_source_data_ids": len(used_source_data_ids)
                - len(direct_selected),
        },
        "counts": {
            "source_records": len(records),
            "logical_trajectories": len(trajectories),
            "state_action_examples": actual_state_actions,
            "routes": dict(sorted(route_counts.items())),
            "transitions": dict(sorted(transition_counts.items())),
            "splits": {name: len(values) for name, values in grouped.items()},
            "rejected_attempts": len(rejected),
            "rejected_attempt_count": len(rejected),
        },
        "targets": {
            "routes": route_targets,
            "logical_trajectories": config.logical_trajectories,
            "state_action_examples": config.state_action_examples,
            "requested_splits": dict(config.split_targets),
            "actual_split_targets": actual_split_targets,
            "route_split_trajectory_quotas": route_split_quotas,
        },
        "shortfall": {
            "allowed": config.allow_shortfall,
            "routes": shortfall_routes,
            "logical_trajectories": sum(shortfall_routes.values()),
            "state_action_examples": max(
                0, config.state_action_examples - actual_state_actions
            ),
        },
        "build_config": asdict(config),
        "image_text_route": {
            "unused_candidate_scan_count": len(image_text_pool),
            "eligible_candidate_count": len(eligible),
            "selected_trajectory_count": len(selected_image_text),
            "primary_rejection_reason": primary_rejection_reason,
            "entity_not_visible_in_image_information_count": 0,
            "image_answer_present_before_text_search_count": 0,
            "text_answer_evidence_missing_count": 0,
            "text_evidence_same_as_image_context_only_count": 0,
            "text_evidence_hit_excluding_image_context_docs": sum(
                bool(
                    provenance.get(
                        "text_evidence_hit_excluding_image_context_docs"
                    )
                )
                for provenance in selected_provenance
            ),
            "text_evidence_cross_source_count": sum(
                int(provenance.get("text_evidence_cross_source_count", 0))
                for provenance in selected_provenance
            ),
        },
        "information": {"visual_placeholders_stripped": True},
        "reason_templates": {
            "min_unique_per_transition": config.min_unique_reasons_per_transition,
        },
        "manual_audit": {
            "direct_count": config.manual_audit_direct_count,
            "image_count": config.manual_audit_image_count,
            "text_count": config.manual_audit_text_count,
            "image_text_count": config.manual_audit_image_text_count,
            "seed": config.manual_audit_seed,
        },
        **rejection_audit,
    }
    if config.tokenization_model_path:
        from .audit import run_processor_preflight_examples

        preflight = run_processor_preflight_examples(
            examples,
            model_path=config.tokenization_model_path,
            max_seq_len=config.max_seq_len,
            visual_token_target=config.visual_token_target,
            target_reserve=config.target_reserve,
            source_parquet_path=source_path,
        )
        manifest["processor_preflight"] = preflight
        if preflight["target_truncation_count"]:
            raise DatasetBuildError(
                "Processor preflight found target truncation risks", rejected
            )
        if preflight["forbidden_visual_placeholder_count"]:
            raise DatasetBuildError(
                "Processor preflight found forbidden visual placeholders", rejected
            )
    else:
        manifest["processor_preflight"] = {"status": "not_requested"}
    trajectories.sort(key=lambda item: item.trajectory_id)
    return BuildResult(trajectories, grouped, rejected, manifest)


def _build_v0_3_dataset(
    source_path: Path,
    cache_path: Path,
    previous_dataset_dir: Path,
    previous_audit_path: Path,
    config: BuildConfig,
    source_label: Optional[str],
    cache_label: Optional[str],
) -> BuildResult:
    records = read_fvqa_records(source_path)
    cache = ImageSearchCache.load(cache_path, label=cache_label)
    source_name = source_label or source_path.name
    previous_raw, previous_validation = load_and_validate_previous_trajectories(
        previous_dataset_dir,
        previous_audit_path,
        expected_trajectory_count=config.required_previous_trajectories,
        expected_state_action_count=config.required_previous_state_actions,
        expected_route_counts=config.expected_previous_route_counts,
    )
    if not previous_validation.valid:
        raise DatasetBuildError(
            "v0.2 previous dataset validation failed: %s"
            % "; ".join(previous_validation.errors)
        )

    trajectories = [
        _convert_v0_2_trajectory_to_v0_3(Trajectory.from_dict(raw))
        for raw in previous_raw
    ]
    used_source_data_ids = {item.source_data_id for item in trajectories}
    if len(used_source_data_ids) != len(trajectories):
        raise DatasetBuildError("v0.2 previous dataset reuses source_data_id values")

    rejected: List[Dict[str, Any]] = []
    direct_pool = _stable_order(
        (
            record
            for record in records
            if record.category == config.direct_category
            and record.data_id not in used_source_data_ids
        ),
        config.seed,
        "v0_3_new_direct_answer",
    )
    new_direct: List[Trajectory] = []
    for record in direct_pool:
        if len(new_direct) >= config.new_direct_trajectories:
            break
        try:
            trajectory = _direct_trajectory(
                record,
                source_name,
                config,
                source_metadata={"v0_3_origin": "new_direct_answer"},
            )
        except Exception as exc:
            rejected.append(
                make_rejection(
                    record.data_id,
                    RouteType.DIRECT_ANSWER.value,
                    RejectionReason.PARSER_INVALID,
                    {"error": str(exc)},
                )
            )
            continue
        new_direct.append(trajectory)
    if (
        len(new_direct) < config.new_direct_trajectories
        and not config.allow_shortfall
    ):
        raise DatasetBuildError(
            "v0.3 direct route shortage: required %d, accepted %d"
            % (config.new_direct_trajectories, len(new_direct)),
            rejected,
        )
    trajectories.extend(new_direct)
    used_source_data_ids.update(item.source_data_id for item in new_direct)

    image_pool = [
        record
        for record in records
        if record.category == config.image_category
        and record.data_id not in used_source_data_ids
    ]
    ranked_image_candidates: List[RankedImageCandidate] = []
    image_rejections: List[Dict[str, Any]] = []
    for record in image_pool:
        try:
            entry = cache.require(record.data_id)
            information = format_image_information(entry, top_k=config.image_top_k)
        except CacheMissError as exc:
            image_rejections.append(
                make_rejection(
                    record.data_id,
                    RouteType.IMAGE_SEARCH_ANSWER.value,
                    RejectionReason.CACHE_MISS,
                    {"error": str(exc), "accepted_answers": record.accepted_answers},
                )
            )
            continue
        except ValueError as exc:
            image_rejections.append(
                make_rejection(
                    record.data_id,
                    RouteType.IMAGE_SEARCH_ANSWER.value,
                    RejectionReason.EMPTY_INFORMATION,
                    {"error": str(exc), "accepted_answers": record.accepted_answers},
                )
            )
            continue
        if contains_forbidden_visual_placeholder(information.text):
            image_rejections.append(
                make_rejection(
                    record.data_id,
                    RouteType.IMAGE_SEARCH_ANSWER.value,
                    RejectionReason.FORBIDDEN_VISUAL_PLACEHOLDER,
                    {"accepted_answers": record.accepted_answers},
                )
            )
            continue
        evidence_score = score_image_evidence(
            entry, record.accepted_answers, top_k=config.image_top_k
        )
        if (
            evidence_score.shortest_support_rank is None
            or not information_supports_answer(
                information.text, record.accepted_answers
            )
        ):
            image_rejections.append(
                make_rejection(
                    record.data_id,
                    RouteType.IMAGE_SEARCH_ANSWER.value,
                    RejectionReason.EVIDENCE_MISS,
                    {
                        "accepted_answers": record.accepted_answers,
                        "image_titles": [
                            title
                            for _index, title, _descriptor
                            in entry.usable_image_results[:config.image_top_k]
                        ],
                        "evidence_score": evidence_score.total_score,
                    },
                )
            )
            continue
        enriched_provenance = {
            **information.provenance,
            "answer_evidence_score": evidence_score.to_dict(),
            "answer_evidence_support_ranks": list(
                evidence_score.support_ranks
            ),
            "answer_evidence_present": True,
        }
        ranked_image_candidates.append(
            RankedImageCandidate(
                record=record,
                information=InformationBundle(
                    text=information.text,
                    provenance=enriched_provenance,
                    image_refs=information.image_refs,
                ),
                evidence_score=evidence_score,
            )
        )

    ranked_image_candidates.sort(
        key=lambda candidate: (
            -candidate.evidence_score.total_score,
            candidate.record.data_id,
        )
    )
    selected_image: List[RankedImageCandidate] = []
    new_image: List[Trajectory] = []
    for candidate in ranked_image_candidates:
        if len(new_image) >= config.new_image_search_trajectories:
            image_rejections.append(
                make_rejection(
                    candidate.record.data_id,
                    RouteType.IMAGE_SEARCH_ANSWER.value,
                    RejectionReason.ROUTE_INELIGIBLE,
                    {
                        "accepted_answers": candidate.record.accepted_answers,
                        "evidence_score": candidate.evidence_score.total_score,
                        "support_ranks": list(
                            candidate.evidence_score.support_ranks
                        ),
                        "selection_status":
                            "not_selected_after_deterministic_ranking",
                    },
                )
            )
            continue
        try:
            trajectory = _image_trajectory(
                candidate.record,
                candidate.information,
                source_name,
                config,
                source_metadata={
                    "v0_3_origin": "new_image_search_answer",
                    "image_evidence_total_score":
                        candidate.evidence_score.total_score,
                    "image_evidence_shortest_support_rank":
                        candidate.evidence_score.shortest_support_rank,
                },
            )
        except Exception as exc:
            image_rejections.append(
                make_rejection(
                    candidate.record.data_id,
                    RouteType.IMAGE_SEARCH_ANSWER.value,
                    RejectionReason.PARSER_INVALID,
                    {
                        "error": str(exc),
                        "accepted_answers": candidate.record.accepted_answers,
                        "evidence_score": candidate.evidence_score.total_score,
                    },
                )
            )
            continue
        selected_image.append(candidate)
        new_image.append(trajectory)
    rejected.extend(image_rejections)
    if (
        len(selected_image) < config.new_image_search_trajectories
        and not config.allow_shortfall
    ):
        raise DatasetBuildError(
            "v0.3 image route shortage after scanning %d candidates: required %d, accepted %d"
            % (
                len(image_pool),
                config.new_image_search_trajectories,
                len(selected_image),
            ),
            rejected,
        )
    trajectories.extend(new_image)

    source_ids = [trajectory.source_data_id for trajectory in trajectories]
    if len(source_ids) != len(set(source_ids)):
        raise DatasetBuildError(
            "v0.3 selected a source_data_id more than once", rejected
        )
    route_counts = Counter(trajectory.route for trajectory in trajectories)
    route_targets = {
        RouteType.DIRECT_ANSWER.value: config.direct_trajectories,
        RouteType.IMAGE_SEARCH_ANSWER.value: config.image_search_trajectories,
        RouteType.TEXT_SEARCH_ANSWER.value: config.text_search_trajectories,
        RouteType.IMAGE_TEXT_SEARCH_ANSWER.value:
            config.image_text_search_trajectories,
    }
    exact_route_counts = all(
        route_counts.get(route, 0) == target
        for route, target in route_targets.items()
    )
    route_split_quotas = (
        {route: dict(quotas) for route, quotas in config.route_split_quotas.items()}
        if exact_route_counts
        else proportional_route_split_quotas(trajectories)
    )
    assignments = assign_splits_by_route_quotas(
        trajectories, route_split_quotas, config.seed
    )
    examples = unfold_trajectories(trajectories, assignments)
    grouped = group_examples_by_split(examples)
    actual_split_targets = state_action_targets_from_assignments(
        trajectories, assignments
    )
    transition_counts = Counter(example.transition for example in examples)
    actual_state_actions = len(examples)
    if (
        not config.allow_shortfall
        and actual_state_actions != config.state_action_examples
    ):
        raise DatasetBuildError(
            "v0.3 state-action count does not match the Full target", rejected
        )

    shortfall_routes = {
        route: max(0, target - route_counts.get(route, 0))
        for route, target in route_targets.items()
    }
    selected_scores = [candidate.evidence_score for candidate in selected_image]
    selected_image_provenance = [
        trajectory.steps[-1].information_provenance or {}
        for trajectory in trajectories
        if trajectory.route == RouteType.IMAGE_TEXT_SEARCH_ANSWER.value
    ]
    rejection_audit = rejection_distributions(rejected)
    counts = {
        "source_records": len(records),
        "logical_trajectories": len(trajectories),
        "state_action_examples": actual_state_actions,
        "routes": dict(sorted(route_counts.items())),
        "transitions": dict(sorted(transition_counts.items())),
        "splits": {name: len(values) for name, values in grouped.items()},
        "rejected_attempts": len(rejected),
        "rejected_attempt_count": len(rejected),
    }
    previous_dataset_manifest = {
        "path": str(previous_dataset_dir).replace("\\", "/"),
        "schema_version": previous_validation.schema_version,
        "manifest_hash": previous_validation.manifest_hash,
        "audit_hash": previous_validation.audit_hash,
        "global_passed": previous_validation.global_passed,
        "global_failure_allowed_reason":
            previous_validation.global_failure_allowed_reason,
        "reused_trajectory_count": previous_validation.trajectory_count,
        "reused_state_action_count": previous_validation.state_action_count,
        "reused_route_counts": previous_validation.route_counts,
        "reused_invalid_trajectory_count": len(
            previous_validation.invalid_trajectory_ids
        ),
    }
    new_data = {
        "direct_candidate_count": len(direct_pool),
        "direct_accepted_count": len(new_direct),
        "image_candidate_count": len(image_pool),
        "image_accepted_count": len(new_image),
        "image_rejected_count": len(image_pool) - len(new_image),
        "image_evidence_score_distribution": dict(
            sorted(Counter(score.total_score for score in selected_scores).items())
        ),
        "image_support_rank_distribution": dict(
            sorted(
                Counter(
                    score.shortest_support_rank for score in selected_scores
                ).items()
            )
        ),
        "new_image_answer_evidence_missing_count": 0,
        "new_image_answer_top1_hit_count": sum(
            score.shortest_support_rank == 1 for score in selected_scores
        ),
        "new_image_answer_top3_hit_count": sum(
            score.shortest_support_rank in {1, 2, 3}
            for score in selected_scores
        ),
    }
    manifest: Dict[str, Any] = {
        "schema_version": config.schema_version,
        "dataset_stage": "full",
        "builder_version": "protocol-sft-v0.3-adjusted-routes-v1",
        "random_seed": config.seed,
        "seed": config.seed,
        "deterministic": True,
        "inputs": {
            "source_dataset_label": source_name,
            "source_dataset_sha256": sha256_file(source_path),
            "image_cache_label": cache.label,
            "image_cache_sha256": cache.file_sha256,
            "image_cache_source": CACHE_SOURCE,
            "image_cache_version": CACHE_VERSION,
            "previous_dataset_dir": str(previous_dataset_dir).replace("\\", "/"),
            "previous_audit_path": str(previous_audit_path).replace("\\", "/"),
        },
        "source_file_hashes": {
            str(source_name): sha256_file(source_path),
        },
        "cache_file_hash": cache.file_sha256,
        "previous_dataset_schema": previous_validation.schema_version,
        "previous_dataset_manifest_hash": previous_validation.manifest_hash,
        "previous_dataset_audit_hash": previous_validation.audit_hash,
        "reused_trajectory_count": previous_validation.trajectory_count,
        "reused_state_action_count": previous_validation.state_action_count,
        "reused_route_counts": previous_validation.route_counts,
        "reused_invalid_trajectory_count": len(
            previous_validation.invalid_trajectory_ids
        ),
        "previous_dataset": previous_dataset_manifest,
        "new_data": new_data,
        "final_counts": counts,
        "counts": counts,
        "targets": {
            "routes": route_targets,
            "logical_trajectories": config.logical_trajectories,
            "state_action_examples": config.state_action_examples,
            "requested_splits": dict(config.split_targets),
            "actual_split_targets": actual_split_targets,
            "route_split_trajectory_quotas": route_split_quotas,
        },
        "shortfall": {
            "allowed": config.allow_shortfall,
            "routes": shortfall_routes,
            "logical_trajectories": sum(shortfall_routes.values()),
            "state_action_examples": max(
                0, config.state_action_examples - actual_state_actions
            ),
        },
        "build_config": asdict(config),
        "image_text_route": {
            "candidate_scan_disabled": True,
            "selected_trajectory_count": route_counts.get(
                RouteType.IMAGE_TEXT_SEARCH_ANSWER.value, 0
            ),
            "entity_not_visible_in_image_information_count": 0,
            "image_answer_present_before_text_search_count": 0,
            "text_answer_evidence_missing_count": 0,
            "text_evidence_same_as_image_context_only_count": 0,
            "text_evidence_hit_excluding_image_context_docs": sum(
                bool(
                    provenance.get(
                        "text_evidence_hit_excluding_image_context_docs"
                    )
                )
                for provenance in selected_image_provenance
            ),
            "text_evidence_cross_source_count": sum(
                int(provenance.get("text_evidence_cross_source_count", 0))
                for provenance in selected_image_provenance
            ),
        },
        "information": {"visual_placeholders_stripped": True},
        "reason_templates": {
            "min_unique_per_transition": config.min_unique_reasons_per_transition,
        },
        "manual_audit": {
            "direct_count": config.manual_audit_direct_count,
            "image_count": config.manual_audit_image_count,
            "text_count": config.manual_audit_text_count,
            "image_text_count": config.manual_audit_image_text_count,
            "new_direct_count": config.manual_audit_new_direct_count,
            "new_image_count": config.manual_audit_new_image_count,
            "reused_image_count": config.manual_audit_reused_image_count,
            "seed": config.manual_audit_seed,
        },
        **rejection_audit,
    }
    if config.tokenization_model_path:
        from .audit import run_processor_preflight_examples

        preflight = run_processor_preflight_examples(
            examples,
            model_path=config.tokenization_model_path,
            max_seq_len=config.max_seq_len,
            visual_token_target=config.visual_token_target,
            target_reserve=config.target_reserve,
            source_parquet_path=source_path,
        )
        manifest["processor_preflight"] = preflight
        if preflight["target_truncation_count"]:
            raise DatasetBuildError(
                "Processor preflight found target truncation risks", rejected
            )
        if preflight["forbidden_visual_placeholder_count"]:
            raise DatasetBuildError(
                "Processor preflight found forbidden visual placeholders", rejected
            )
    else:
        manifest["processor_preflight"] = {"status": "not_requested"}
    trajectories.sort(key=lambda item: item.trajectory_id)
    return BuildResult(trajectories, grouped, rejected, manifest)


def build_dataset(
    source_path: Path,
    cache_path: Path,
    config: BuildConfig,
    source_label: Optional[str] = None,
    cache_label: Optional[str] = None,
    previous_dataset_dir: Optional[Path] = None,
    previous_audit_path: Optional[Path] = None,
) -> BuildResult:
    config.validate()
    source_path = Path(source_path)
    cache_path = Path(cache_path)
    if config.schema_version == "protocol-sft-v0.2":
        previous_path = previous_dataset_dir or (
            Path(config.previous_dataset_dir)
            if config.previous_dataset_dir is not None
            else None
        )
        if previous_path is None:
            raise DatasetBuildError("v0.2 requires a previous dataset directory")
        return _build_v0_2_dataset(
            source_path,
            cache_path,
            Path(previous_path),
            config,
            source_label,
            cache_label,
        )
    if config.schema_version == "protocol-sft-v0.3":
        previous_path = previous_dataset_dir or (
            Path(config.previous_dataset_dir)
            if config.previous_dataset_dir is not None
            else None
        )
        audit_path = previous_audit_path or (
            Path(config.previous_audit_path)
            if config.previous_audit_path is not None
            else None
        )
        if previous_path is None or audit_path is None:
            raise DatasetBuildError(
                "v0.3 requires previous dataset and audit paths"
            )
        return _build_v0_3_dataset(
            source_path,
            cache_path,
            Path(previous_path),
            Path(audit_path),
            config,
            source_label,
            cache_label,
        )
    records = read_fvqa_records(source_path)
    cache = ImageSearchCache.load(cache_path, label=cache_label)
    retriever = BootstrapTextRetriever.from_image_cache(cache)
    source_name = source_label or source_path.name
    rejected: List[Dict[str, Any]] = []

    direct_pool = _stable_order(
        (record for record in records if record.category == "search_free"),
        config.seed,
        "direct",
    )
    if len(direct_pool) < config.direct_trajectories:
        raise DatasetBuildError(
            "direct route shortage: required %d, eligible %d"
            % (config.direct_trajectories, len(direct_pool)),
            rejected,
        )
    direct_selected = direct_pool[: config.direct_trajectories]

    required_pool = _stable_order(
        (record for record in records if record.category == "search_required"),
        config.seed,
        "search_required",
    )

    # Text evidence is stricter, so scan every search-required record first.
    text_eligible: List[Tuple[FVQARecord, str, InformationBundle]] = []
    for record in required_pool:
        decision = _find_text_evidence(record, cache, retriever, config)
        if decision.query is not None and decision.information is not None:
            text_eligible.append((record, decision.query, decision.information))
        else:
            rejected.append(
                make_rejection(
                    record.data_id,
                    RouteType.TEXT_SEARCH.value,
                    decision.rejection_reason or RejectionReason.ROUTE_INELIGIBLE,
                    decision.details,
                )
            )
    text_candidates = text_eligible[: config.text_search_trajectories]
    if len(text_candidates) < config.text_search_trajectories and not config.allow_shortfall:
        raise DatasetBuildError(
            "text-search route shortage: required %d, eligible %d; real cached text is insufficient"
            % (config.text_search_trajectories, len(text_candidates)),
            rejected,
        )
    text_ids = {record.data_id for record, _, _ in text_candidates}

    image_eligible: List[Tuple[FVQARecord, InformationBundle]] = []
    for record in required_pool:
        if record.data_id in text_ids:
            continue
        try:
            entry = cache.require(record.data_id)
            information = format_image_information(entry, top_k=config.image_top_k)
        except (CacheMissError, ValueError):
            rejected.append(
                make_rejection(
                    record.data_id,
                    RouteType.IMAGE_SEARCH.value,
                    RejectionReason.CACHE_MISS,
                    {"cache_label": cache.label},
                )
            )
            continue
        if contains_forbidden_visual_placeholder(information.text):
            rejected.append(
                make_rejection(
                    record.data_id,
                    RouteType.IMAGE_SEARCH.value,
                    RejectionReason.FORBIDDEN_VISUAL_PLACEHOLDER,
                    {"information": information.text},
                )
            )
            continue
        if not information_supports_answer(information.text, record.accepted_answers):
            rejected.append(
                make_rejection(
                    record.data_id,
                    RouteType.IMAGE_SEARCH.value,
                    RejectionReason.EVIDENCE_MISS,
                    {
                        "selected_result_indices": information.provenance.get(
                            "selected_result_indices", []
                        )
                    },
                )
            )
            continue
        image_eligible.append((record, information))
    image_candidates = image_eligible[: config.image_search_trajectories]
    if len(image_candidates) < config.image_search_trajectories:
        raise DatasetBuildError(
            "image-search route shortage: required %d, eligible %d after unique-route filtering"
            % (config.image_search_trajectories, len(image_candidates)),
            rejected,
        )

    trajectories: List[Trajectory] = []
    trajectories.extend(
        _direct_trajectory(record, source_name, config) for record in direct_selected
    )
    trajectories.extend(
        _image_trajectory(record, information, source_name, config)
        for record, information in image_candidates
    )
    trajectories.extend(
        _text_trajectory(record, query, information, source_name, config)
        for record, query, information in text_candidates
    )
    source_ids = [trajectory.source_data_id for trajectory in trajectories]
    if len(source_ids) != len(set(source_ids)):
        raise DatasetBuildError("a source_data_id was selected for multiple routes", rejected)
    actual_state_actions = sum(len(trajectory.steps) for trajectory in trajectories)
    if not config.allow_shortfall and actual_state_actions != config.state_action_examples:
        raise DatasetBuildError("expanded state-action count does not match target", rejected)

    actual_split_targets = (
        dict(config.split_targets)
        if actual_state_actions == config.state_action_examples
        else proportional_state_action_targets(trajectories)
    )
    assignments = assign_splits(trajectories, actual_split_targets, config.seed)
    examples = unfold_trajectories(trajectories, assignments)
    grouped = group_examples_by_split(examples)
    route_counts: Dict[str, int] = {}
    transition_counts: Dict[str, int] = {}
    for trajectory in trajectories:
        route_counts[trajectory.route] = route_counts.get(trajectory.route, 0) + 1
    for example in examples:
        transition_counts[example.transition] = transition_counts.get(example.transition, 0) + 1

    rejection_audit = rejection_distributions(rejected)
    actual_routes = {
        RouteType.DIRECT_ANSWER.value: route_counts.get(RouteType.DIRECT_ANSWER.value, 0),
        RouteType.IMAGE_SEARCH.value: route_counts.get(RouteType.IMAGE_SEARCH.value, 0),
        RouteType.TEXT_SEARCH.value: route_counts.get(RouteType.TEXT_SEARCH.value, 0),
    }
    route_targets = {
        RouteType.DIRECT_ANSWER.value: config.direct_trajectories,
        RouteType.IMAGE_SEARCH.value: config.image_search_trajectories,
        RouteType.TEXT_SEARCH.value: config.text_search_trajectories,
    }
    shortfall = {
        route: max(0, route_targets[route] - actual_routes[route])
        for route in route_targets
    }
    selected_text_provenance = [
        information.provenance for _, _, information in text_candidates
    ]
    manifest = {
        "schema_version": config.schema_version,
        "seed": config.seed,
        "deterministic": True,
        "inputs": {
            "source_dataset_label": source_name,
            "source_dataset_sha256": sha256_file(source_path),
            "image_cache_label": cache.label,
            "image_cache_sha256": cache.file_sha256,
            "image_cache_source": CACHE_SOURCE,
            "image_cache_version": CACHE_VERSION,
        },
        "counts": {
            "source_records": len(records),
            "logical_trajectories": len(trajectories),
            "state_action_examples": len(examples),
            "routes": dict(sorted(route_counts.items())),
            "transitions": dict(sorted(transition_counts.items())),
            "splits": {name: len(values) for name, values in grouped.items()},
            "rejected_attempts": len(rejected),
            "rejected_attempt_count": len(rejected),
        },
        "targets": {
            "routes": route_targets,
            "state_action_examples": config.state_action_examples,
            "requested_splits": dict(config.split_targets),
            "actual_split_targets": actual_split_targets,
        },
        "shortfall": {
            "allowed": config.allow_shortfall,
            "routes": shortfall,
            "logical_trajectories": sum(shortfall.values()),
            "state_action_examples": max(
                0, config.state_action_examples - len(examples)
            ),
        },
        "build_config": asdict(config),
        "text_retrieval": {
            "backend": "deterministic_bm25_over_fvqa_cache_titles",
            "corpus_documents": len(retriever.documents),
            "ground_truth_used_to_build_corpus": False,
            "online_access": False,
            "query_construction_sources": ["question"],
            "cache_titles_used_to_build_query": False,
            "evidence_hit_normal": sum(
                bool(provenance.get("evidence_hit_normal"))
                for provenance in selected_text_provenance
            ),
            "evidence_hit_leave_one_source_out": sum(
                bool(provenance.get("evidence_hit_leave_one_source_out"))
                for provenance in selected_text_provenance
            ),
        },
        "information": {
            "visual_placeholders_stripped": config.strip_visual_placeholders,
        },
        "reason_templates": {
            "min_unique_per_transition": config.min_unique_reasons_per_transition,
        },
        "manual_audit": {
            "direct_count": config.manual_audit_direct_count,
            "image_count": config.manual_audit_image_count,
            "text_count": config.manual_audit_text_count,
            "seed": config.manual_audit_seed,
        },
        **rejection_audit,
    }
    if config.tokenization_model_path:
        from .audit import run_processor_preflight_examples

        preflight = run_processor_preflight_examples(
            examples,
            model_path=config.tokenization_model_path,
            max_seq_len=config.max_seq_len,
            visual_token_target=config.visual_token_target,
            target_reserve=config.target_reserve,
            source_parquet_path=source_path,
        )
        manifest["processor_preflight"] = preflight
        if preflight["target_truncation_count"]:
            raise DatasetBuildError(
                "Processor preflight found %d target truncation risks"
                % preflight["target_truncation_count"],
                rejected,
            )
        if preflight["forbidden_visual_placeholder_count"]:
            raise DatasetBuildError(
                "Processor preflight found forbidden visual placeholders",
                rejected,
            )
    else:
        manifest["processor_preflight"] = {"status": "not_requested"}
    trajectories.sort(key=lambda item: item.trajectory_id)
    return BuildResult(trajectories, grouped, rejected, manifest)


def write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def write_build_result(result: BuildResult, output_dir: Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "trajectories.jsonl", (item.to_dict() for item in result.trajectories))
    for split_name in ("train", "dev", "test"):
        write_jsonl(
            output_dir / (split_name + ".jsonl"),
            (item.to_dict() for item in result.examples_by_split[split_name]),
        )
    write_jsonl(output_dir / "rejected.jsonl", result.rejected)
    (output_dir / "manifest.json").write_text(
        json.dumps(result.manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manual = result.manifest.get("manual_audit", {})
    if sum(
        int(manual.get(name, 0))
        for name in (
            "direct_count",
            "image_count",
            "text_count",
            "image_text_count",
        )
    ):
        from .manual_audit import (
            ManualAuditConfig,
            build_manual_route_audit,
            write_manual_route_audit,
        )

        manual_records = build_manual_route_audit(
            result.trajectories,
            ManualAuditConfig(
                direct_count=int(manual.get("direct_count", 0)),
                image_count=int(manual.get("image_count", 0)),
                text_count=int(manual.get("text_count", 0)),
                seed=int(manual.get("seed", result.manifest.get("seed", 20260722))),
                image_text_count=int(manual.get("image_text_count", 0)),
                new_direct_count=int(manual.get("new_direct_count", 0)),
                new_image_count=int(manual.get("new_image_count", 0)),
                reused_image_count=int(manual.get("reused_image_count", 0)),
            ),
        )
        write_manual_route_audit(
            manual_records,
            output_dir / "manual_route_audit.jsonl",
            output_dir / "manual_route_audit.md",
        )
        result.manifest["manual_audit"]["actual_counts"] = {
            route: sum(record["route"] == route for record in manual_records)
            for route in (
                RouteType.DIRECT_ANSWER.value,
                RouteType.IMAGE_SEARCH.value,
                RouteType.TEXT_SEARCH.value,
                RouteType.IMAGE_SEARCH_ANSWER.value,
                RouteType.TEXT_SEARCH_ANSWER.value,
                RouteType.IMAGE_TEXT_SEARCH_ANSWER.value,
            )
        }
        result.manifest["manual_audit"]["actual_composition"] = dict(
            sorted(
                Counter(
                    str(record.get("source_group", "legacy"))
                    for record in manual_records
                ).items()
            )
        )
        (output_dir / "manifest.json").write_text(
            json.dumps(result.manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def write_rejections(rejected: Sequence[Mapping[str, Any]], output_dir: Path) -> None:
    write_jsonl(Path(output_dir) / "rejected.jsonl", rejected)
