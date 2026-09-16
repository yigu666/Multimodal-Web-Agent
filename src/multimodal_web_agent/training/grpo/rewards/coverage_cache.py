from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

from multimodal_web_agent.data.protocol_sft.text_retriever import BootstrapTextRetriever
from multimodal_web_agent.evaluation.unified_agent.answer_metrics import normalize_answer
from multimodal_web_agent.evaluation.unified_agent.audit_v1_2.answer_equivalence import (
    question_requires_numeric_answer,
)

from .evidence_support import (
    MATCHER_VERSION,
    deterministic_quantity_signatures,
    normalized_evidence_tokens,
)
from .text_query_utility import rank_sensitive_utility


COVERAGE_SCHEMA = "grpo-reward-v2-text-corpus-coverage-v1"
BASELINE_SCHEMA = "grpo-reward-v2-question-baseline-v1"


def accepted_answers(prompt: Mapping[str, Any]) -> tuple[str, ...]:
    values = (
        prompt.get("candidate_answers")
        or prompt.get("accepted_answers")
        or prompt.get("answer_aliases")
        or ()
    )
    result = tuple(str(value) for value in values if str(value).strip())
    ground_truth = str(prompt.get("ground_truth", "") or "").strip()
    if ground_truth and ground_truth not in result:
        result = (ground_truth, *result)
    return result


def prompt_id(prompt: Mapping[str, Any]) -> str:
    return str(prompt.get("prompt_uid") or prompt.get("eval_id") or prompt.get("prompt_id") or "")


def answers_sha256(values: Iterable[str]) -> str:
    canonical = json.dumps(list(values), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def canonical_corpus_sha256(documents: Sequence[Any]) -> str:
    digest = hashlib.sha256()
    for document in sorted(documents, key=lambda item: str(item.document_id)):
        row = {"document_id": str(document.document_id), "text": str(document.text)}
        digest.update(json.dumps(row, ensure_ascii=False, sort_keys=True,
                                 separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def build_coverage_rows(
    prompts: Sequence[Mapping[str, Any]],
    documents: Sequence[Any],
    *,
    text_corpus_sha256: str,
) -> list[dict[str, Any]]:
    # Precompute canonical token strings once. Most strict matches then become
    # fast boundary-safe substring checks in C rather than Python document scans.
    quantity_postings: dict[tuple[str, str], set[int]] = {}
    document_token_strings: list[str] = []
    document_singular_strings: list[str] = []
    for index, document in enumerate(documents):
        ordered_tokens = list(normalized_evidence_tokens(str(document.text)))
        document_token_strings.append(" " + " ".join(ordered_tokens) + " ")
        singular_tokens = []
        for token in ordered_tokens:
            if token.endswith("ies") and len(token) > 4:
                singular_tokens.append(token[:-3] + "y")
            elif token.endswith("s") and len(token) > 3 and not token.endswith(("ss", "us", "is")):
                singular_tokens.append(token[:-1])
            else:
                singular_tokens.append(token)
        document_singular_strings.append(" " + " ".join(singular_tokens) + " ")
        if re.search(r"\d", str(document.text)):
            for signature in deterministic_quantity_signatures(str(document.text)):
                quantity_postings.setdefault(signature, set()).add(index)
    rows = []
    for prompt in prompts:
        aliases = accepted_answers(prompt)
        best_document = None
        best_match = None
        for alias in aliases:
            tokens = normalize_answer(alias).split()
            if not tokens:
                continue
            phrase = " " + " ".join(tokens) + " "
            exact = next((index for index, text in enumerate(document_token_strings) if phrase in text), None)
            if exact is not None:
                best_document = str(documents[exact].document_id)
                best_match = "normalized_exact"
                break
            singular = []
            for token in tokens:
                if token.endswith("ies") and len(token) > 4:
                    singular.append(token[:-3] + "y")
                elif token.endswith("s") and len(token) > 3 and not token.endswith(("ss", "us", "is")):
                    singular.append(token[:-1])
                else:
                    singular.append(token)
            singular_phrase = " " + " ".join(singular) + " "
            alias_index = next((index for index, text in enumerate(document_singular_strings) if singular_phrase in text), None)
            if alias_index is not None:
                best_document = str(documents[alias_index].document_id)
                best_match = "alias_equivalent"
                break
            numeric_like = bool(re.search(
                r"\d|\b(?:one|two|three|four|five|six|seven|eight|nine|ten|"
                r"hundred|thousand|million)\b", alias.casefold()
            ))
            title_like = " of " in normalize_answer(alias) or normalize_answer(alias).startswith((
                "grand duke ", "grand duchess ", "duke ", "duchess ", "king ",
                "queen ", "prince ", "princess ", "emperor ", "empress ",
                "president ", "saint ", "sir ", "dame ", "dr ", "doctor ", "professor ",
            ))
            if numeric_like:
                for signature in deterministic_quantity_signatures(alias):
                    matching = quantity_postings.get(signature, ())
                    if not matching:
                        continue
                    dimension = signature[0]
                    if dimension != "number" or question_requires_numeric_answer(str(prompt.get("question", ""))):
                        quantity_index = min(matching, key=lambda value: str(documents[value].document_id))
                        best_document = str(documents[quantity_index].document_id)
                        best_match = "unit_equivalent" if dimension != "number" else "numeric_equivalent"
                        break
                if best_document is not None:
                    break
            if title_like:
                person_core = normalize_answer(alias)
                for prefix in (
                    "grand duke", "grand duchess", "duke", "duchess", "king", "queen",
                    "prince", "princess", "emperor", "empress", "president", "saint",
                    "sir", "dame", "dr", "doctor", "professor",
                ):
                    if person_core.startswith(prefix + " "):
                        person_core = person_core[len(prefix):].strip()
                        break
                person_core = re.sub(r"\s+of\s+(?:the\s+)?[a-z][a-z\s-]+$", "", person_core)
                core_tokens = person_core.split()
                if len(core_tokens) >= 2:
                    core_phrase = " " + " ".join(core_tokens) + " "
                    core_index = next((index for index, text in enumerate(document_token_strings) if core_phrase in text), None)
                    if core_index is not None:
                        best_document = str(documents[core_index].document_id)
                        best_match = "alias_equivalent"
            if best_document is not None:
                break
        rows.append({
            "prompt_id": prompt_id(prompt),
            "accepted_answers_hash": answers_sha256(aliases),
            "text_corpus_sha256": text_corpus_sha256,
            "matcher_version": MATCHER_VERSION,
            "corpus_has_answer": best_document is not None,
            "coverage_mask": 1.0 if best_document is not None else 0.0,
            "best_document_id": best_document,
            "best_match_type": best_match,
        })
    return rows


def build_question_baseline_rows(
    prompts: Sequence[Mapping[str, Any]],
    retriever: BootstrapTextRetriever,
    *,
    text_corpus_sha256: str,
    top_k: int,
) -> list[dict[str, Any]]:
    rows = []
    for prompt in prompts:
        question = str(prompt.get("question", ""))
        source_data_id = str(prompt.get("source_data_id", ""))
        hits = retriever.retrieve(question, top_k=top_k,
                                  exclude_source_data_id=source_data_id)
        rank = rank_sensitive_utility(
            accepted_answers(prompt),
            [hit.document.text for hit in hits],
            question=question,
        )
        rows.append({
            "prompt_id": prompt_id(prompt),
            "accepted_answers_hash": answers_sha256(accepted_answers(prompt)),
            "text_corpus_sha256": text_corpus_sha256,
            "matcher_version": MATCHER_VERSION,
            "retriever_version": "deterministic-bm25-v1",
            "tokenizer_version": "protocol-query-tokens-v1",
            "top_k": int(top_k),
            "tie_breaking": "score_desc_document_id_asc",
            "question_sha256": hashlib.sha256(question.encode("utf-8")).hexdigest(),
            "question_baseline_rank_utility": rank.value,
            "best_rank": rank.best_rank,
            "best_match_type": rank.best_support.match_type,
            "document_ids": [hit.document.document_id for hit in hits],
        })
    return rows


def rows_by_prompt(rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {}
    for row in rows:
        key = str(row["prompt_id"])
        if key in result:
            raise ValueError(f"duplicate Reward v2 cache prompt_id={key}")
        result[key] = dict(row)
    return result


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def validate_cache_pair(
    coverage_rows: Sequence[Mapping[str, Any]],
    baseline_rows: Sequence[Mapping[str, Any]],
) -> None:
    coverage = rows_by_prompt(coverage_rows)
    baseline = rows_by_prompt(baseline_rows)
    if set(coverage) != set(baseline):
        raise ValueError("coverage and baseline cache prompt IDs differ")
    for key in coverage:
        left, right = coverage[key], baseline[key]
        for field in ("accepted_answers_hash", "text_corpus_sha256", "matcher_version"):
            if left[field] != right[field]:
                raise ValueError(f"Reward v2 cache mismatch for {key}: {field}")


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
