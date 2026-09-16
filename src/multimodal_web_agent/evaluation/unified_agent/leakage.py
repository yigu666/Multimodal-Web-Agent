from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import re
import unicodedata
from typing import Any, Iterable, Mapping, Sequence

from .answer_metrics import normalize_answer
from .source_adapters.base import SourceCandidate, image_bytes


@dataclass(frozen=True)
class LeakageReference:
    source_label: str
    source_data_id: str
    question: str
    answer_aliases: tuple[str, ...]
    image_sha256: str | None
    image_dhash: int | None


def normalized_question(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    return " ".join(re.findall(r"\w+", text, re.UNICODE))


def image_dhash(raw: bytes) -> int:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("image dHash requires Pillow") from exc
    with Image.open(io.BytesIO(raw)) as image:
        gray = image.convert("L").resize((9, 8))
        pixels = list(gray.getdata())
    bits = 0
    for row in range(8):
        for column in range(8):
            bits = (bits << 1) | int(
                pixels[row * 9 + column] > pixels[row * 9 + column + 1]
            )
    return bits


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def _question_similarity(left: str, right: str) -> float:
    a = set(normalized_question(left).split())
    b = set(normalized_question(right).split())
    return len(a & b) / len(a | b) if a or b else 1.0


def references_from_fvqa_parquet(
    path: Path,
    source_label: str,
) -> list[LeakageReference]:
    result = []
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("FVQA leakage scan requires pyarrow") from exc
    parquet = pq.ParquetFile(path)
    available = set(parquet.schema_arrow.names)
    columns = [
        name for name in (
            "data_id", "prompt", "images", "reward_model", "question"
        ) if name in available
    ]
    for batch in parquet.iter_batches(batch_size=64, columns=columns):
        for row in batch.to_pylist():
            prompt = row.get("prompt") or row.get("question") or ""
            if isinstance(prompt, list):
                values = [
                    str(item.get("content", ""))
                    for item in prompt if isinstance(item, Mapping)
                ]
                prompt = values[-1] if values else ""
            images = row.get("images") or []
            raw = image_bytes(images[0])[0] if len(images) == 1 else None
            reward = row.get("reward_model") or row
            answer = str(reward.get("ground_truth", "")).strip()
            result.append(LeakageReference(
                source_label=source_label,
                source_data_id=str(row.get("data_id", "")),
                question=str(prompt),
                answer_aliases=(answer,) if answer else (),
                image_sha256=hashlib.sha256(raw).hexdigest() if raw else None,
                image_dhash=image_dhash(raw) if raw else None,
            ))
    return result


def references_from_jsonl(
    path: Path,
    source_label: str,
) -> list[LeakageReference]:
    result = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            result.append(LeakageReference(
                source_label=source_label,
                source_data_id=str(
                    row.get("source_data_id") or row.get("data_id") or ""
                ),
                question=str(row.get("question") or row.get("prompt") or ""),
                answer_aliases=tuple(
                    row.get("answer_aliases")
                    or row.get("accepted_answers")
                    or ([row["canonical_answer"]]
                        if row.get("canonical_answer") else [])
                ),
                image_sha256=row.get("image_sha256"),
                image_dhash=(
                    int(row["image_dhash"], 16)
                    if isinstance(row.get("image_dhash"), str)
                    else row.get("image_dhash")
                ),
            ))
    return result


def audit_leakage(
    candidates: Sequence[SourceCandidate],
    references: Sequence[LeakageReference],
    *,
    near_image_hamming: int = 5,
    high_question_similarity: float = 0.9,
) -> dict[str, Any]:
    source_ids = {row.source_data_id for row in references if row.source_data_id}
    image_hashes = {row.image_sha256 for row in references if row.image_sha256}
    questions = {
        normalized_question(row.question) for row in references if row.question
    }
    qa = {
        (normalized_question(row.question), normalize_answer(alias))
        for row in references
        for alias in row.answer_aliases if row.question and alias
    }
    dhashes = [
        (row.source_label, row.source_data_id, row.image_dhash)
        for row in references if row.image_dhash is not None
    ]
    reference_questions = [
        (row.source_label, row.source_data_id, row.question)
        for row in references if row.question
    ]
    counts = {
        "exact_source_overlap_count": 0,
        "exact_image_overlap_count": 0,
        "near_duplicate_image_overlap_count": 0,
        "exact_question_overlap_count": 0,
        "question_answer_pair_overlap_count": 0,
        "high_similarity_question_count": 0,
    }
    hard_reject = {}
    high_similarity = []
    for candidate in candidates:
        reasons = []
        if candidate.source_data_id in source_ids:
            counts["exact_source_overlap_count"] += 1
            reasons.append("exact_source_overlap")
        if candidate.image_sha256 in image_hashes:
            counts["exact_image_overlap_count"] += 1
            reasons.append("exact_image_overlap")
        question = normalized_question(candidate.question)
        if question in questions:
            counts["exact_question_overlap_count"] += 1
            reasons.append("exact_question_overlap")
        if any(
            (question, normalize_answer(alias)) in qa
            for alias in candidate.answer_aliases
        ):
            counts["question_answer_pair_overlap_count"] += 1
            reasons.append("question_answer_pair_overlap")
        candidate_dhash = image_dhash(candidate.image_bytes)
        if any(
            hamming_distance(candidate_dhash, value) <= near_image_hamming
            for _, _, value in dhashes
        ):
            counts["near_duplicate_image_overlap_count"] += 1
            reasons.append("near_duplicate_image_overlap")
        matches = [
            {
                "source_label": label,
                "source_data_id": source_id,
                "similarity": score,
            }
            for label, source_id, other in reference_questions
            if question != normalized_question(other)
            for score in [_question_similarity(candidate.question, other)]
            if score >= high_question_similarity
        ]
        if matches:
            counts["high_similarity_question_count"] += 1
            high_similarity.append({
                "candidate_key": candidate.candidate_key,
                "matches": sorted(
                    matches,
                    key=lambda item: (
                        -item["similarity"], item["source_data_id"]
                    ),
                )[:20],
            })
        if reasons:
            hard_reject[candidate.candidate_key] = sorted(set(reasons))
    hard_keys = (
        "exact_source_overlap_count",
        "exact_image_overlap_count",
        "near_duplicate_image_overlap_count",
        "exact_question_overlap_count",
        "question_answer_pair_overlap_count",
    )
    return {
        **counts,
        "hard_reject_candidates": hard_reject,
        "high_similarity_manual_review": high_similarity,
        "hard_gate_passed": all(counts[key] == 0 for key in hard_keys),
        "entity_name_overlap_is_not_a_rejection_rule": True,
    }


def audit_internal_duplicates(
    candidates: Sequence[SourceCandidate],
    *,
    near_image_hamming: int = 5,
    pre_rejected_keys: set[str] | None = None,
) -> dict[str, Any]:
    pre_rejected = set(pre_rejected_keys or ())
    fields = {
        "source_data_id": {},
        "image_sha256": {},
        "normalized_question": {},
        "question_answer_pair": {},
    }
    dhashes = []
    for candidate in candidates:
        values = {
            "source_data_id": candidate.source_data_id,
            "image_sha256": candidate.image_sha256,
            "normalized_question": normalized_question(candidate.question),
            "question_answer_pair": (
                normalized_question(candidate.question),
                tuple(sorted(
                    normalize_answer(alias)
                    for alias in candidate.answer_aliases
                )),
            ),
        }
        for name, value in values.items():
            fields[name].setdefault(value, []).append(candidate.candidate_key)
        dhashes.append((candidate.candidate_key, image_dhash(candidate.image_bytes)))
    groups = {
        name: [
            sorted(keys) for keys in mapping.values() if len(keys) > 1
        ]
        for name, mapping in fields.items()
    }
    near_groups = []
    for index, (left_key, left_hash) in enumerate(dhashes):
        for right_key, right_hash in dhashes[index + 1:]:
            distance = hamming_distance(left_hash, right_hash)
            if distance <= near_image_hamming:
                near_groups.append({
                    "left": left_key,
                    "right": right_key,
                    "hamming_distance": distance,
                })

    # Duplicate constraints form a graph because one candidate may share a
    # question with one row and a near-duplicate image with another. Keep one
    # deterministic representative per connected component and reject only
    # the remaining members. Prefer a candidate that has not already failed
    # an external training-leakage gate.
    parent = {candidate.candidate_key: candidate.candidate_key for candidate in candidates}

    def find(key: str) -> str:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    for values in groups.values():
        for group in values:
            for key in group[1:]:
                union(group[0], key)
    for pair in near_groups:
        union(pair["left"], pair["right"])
    components: dict[str, list[str]] = {}
    for key in parent:
        components.setdefault(find(key), []).append(key)
    duplicate_components = [
        sorted(component)
        for component in components.values()
        if len(component) > 1
    ]
    representatives = []
    reject = set()
    for component in sorted(duplicate_components):
        representative = min(
            component,
            key=lambda key: (key in pre_rejected, key),
        )
        representatives.append(representative)
        reject.update(key for key in component if key != representative)
    return {
        "exact_source_duplicate_group_count": len(groups["source_data_id"]),
        "exact_image_duplicate_group_count": len(groups["image_sha256"]),
        "exact_question_duplicate_group_count": len(
            groups["normalized_question"]
        ),
        "question_answer_duplicate_group_count": len(
            groups["question_answer_pair"]
        ),
        "near_duplicate_image_pair_count": len(near_groups),
        "groups": groups,
        "near_duplicate_image_pairs": near_groups,
        "duplicate_component_count": len(duplicate_components),
        "duplicate_components": duplicate_components,
        "representative_candidates": representatives,
        "hard_reject_candidates": sorted(reject),
        "deduplication_policy": (
            "one_deterministic_representative_per_connected_component;"
            "prefer_not_pre_rejected"
        ),
        "passed": not duplicate_components,
    }
