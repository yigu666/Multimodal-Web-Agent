from __future__ import annotations

from dataclasses import dataclass
from html import escape
from typing import Any, Dict, List, Sequence

from .cache_reader import CacheEntry
from .text_retriever import TextSearchHit


FORBIDDEN_VISUAL_PLACEHOLDERS = (
    "<|vision_start|>",
    "<|image_pad|>",
    "<|vision_end|>",
)


def contains_forbidden_visual_placeholder(text: str) -> bool:
    return any(token in text for token in FORBIDDEN_VISUAL_PLACEHOLDERS)


@dataclass(frozen=True)
class InformationBundle:
    text: str
    provenance: Dict[str, Any]
    image_refs: List[Dict[str, Any]]


def _safe_text(value: str) -> str:
    return escape(" ".join(value.split()), quote=False)


def format_image_information(entry: CacheEntry, top_k: int = 3) -> InformationBundle:
    selected = entry.usable_image_results[:top_k]
    if not selected:
        raise ValueError("cannot format an empty cache entry")
    lines = [
        "<information>",
        "[Image Search Results] Cached results ranked in original cache order:",
    ]
    indices = []
    for rank, (result_index, title, _descriptor) in enumerate(selected, start=1):
        indices.append(result_index)
        lines.append("%d. title: %s" % (rank, _safe_text(title)))
    lines.append("</information>")
    provenance = entry.provenance(indices)
    provenance["document_ids"] = [
        "%s:title:%d" % (entry.data_id, result_index)
        for result_index in indices
    ]
    return InformationBundle(
        text="\n".join(lines),
        provenance=provenance,
        # Cached thumbnail pixels are not loaded by this dataset builder. Keeping
        # only titles prevents fake multi-image tokens and phantom image inputs.
        image_refs=[],
    )


def format_text_information(
    hits: List[TextSearchHit],
    provenance: Dict[str, Any],
) -> InformationBundle:
    if not hits:
        raise ValueError("cannot format empty text-search hits")
    lines = [
        "<information>",
        "[Text Search Results] Cached webpage titles ranked by deterministic BM25:",
    ]
    for rank, hit in enumerate(hits, start=1):
        lines.append(
            "%d. %s [cache_document=%s]"
            % (rank, _safe_text(hit.document.text), hit.document.document_id)
        )
    lines.append("</information>")
    return InformationBundle(text="\n".join(lines), provenance=provenance, image_refs=[])


def format_frozen_information(
    kind: str,
    records: Sequence[str],
) -> str:
    """Format generic offline Unified-Eval records with the frozen XML wrapper."""
    if kind not in {"Image Search", "Text Search"}:
        raise ValueError("unsupported frozen information kind")
    if not records:
        raise ValueError("cannot format empty frozen information")
    lines = ["<information>", "[%s Results]" % kind]
    lines.extend(
        "%d. %s" % (index, _safe_text(record))
        for index, record in enumerate(records, 1)
    )
    lines.append("</information>")
    return "\n".join(lines)
