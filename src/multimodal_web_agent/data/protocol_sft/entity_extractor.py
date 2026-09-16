from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from html import unescape
from typing import List, Mapping, Optional, Sequence, Tuple


_TOKEN_RE = re.compile(r"[^\W_]+(?:['’-][^\W_]+)?", re.UNICODE)
_XML_RE = re.compile(r"<[^>]*>")
_FILE_EXTENSION_RE = re.compile(
    r"\.(?:jpe?g|png|gif|webp|svg|tiff?|bmp)$", re.IGNORECASE
)
_NOISE_PREFIX_RE = re.compile(
    r"^(?:file\s*:\s*|premium\s+photo\s*[|:\-]\s*|"
    r"editorial\s+image\s*[|:\-]\s*)",
    re.IGNORECASE,
)
_SITE_SUFFIX_RE = re.compile(
    r"\s*(?:[-|–—]\s*)?(?:wikipedia|wikimedia\s+commons|britannica|"
    r"youtube|official\s+site|amazon|ebay|pinterest|instagram|tiktok)\s*$",
    re.IGNORECASE,
)
_GENERIC_TITLES = {
    "premium photo",
    "editorial image",
    "home",
    "homepage",
    "official site",
    "official website",
    "image",
    "image result",
    "search result",
    "untitled",
    "photo",
    "picture",
    "gallery",
    "wikipedia",
    "wikimedia commons",
    "britannica",
    "youtube",
    "amazon",
    "ebay",
    "pinterest",
    "instagram",
    "tiktok",
}
_GENERIC_TOKENS = {
    "home",
    "official",
    "website",
    "image",
    "result",
    "photo",
    "picture",
    "gallery",
    "untitled",
    "premium",
    "editorial",
}


@dataclass(frozen=True)
class ExtractedEntity:
    entity: str
    source_rank: int
    source_document_id: Optional[str]
    source_title: str
    extraction_method: str
    valid: bool
    rejection_reason: Optional[str]


def _tokens(text: str) -> List[str]:
    return _TOKEN_RE.findall(unicodedata.normalize("NFKC", text))


def clean_search_title(title: str) -> str:
    """Remove a small, deterministic set of search-title wrappers and sites."""

    normalized = " ".join(unescape(unicodedata.normalize("NFKC", str(title))).split())
    if not normalized or _XML_RE.search(normalized):
        return ""
    cleaned = _NOISE_PREFIX_RE.sub("", normalized).strip(" -|:;,.")
    previous = None
    while cleaned and cleaned != previous:
        previous = cleaned
        cleaned = _SITE_SUFFIX_RE.sub("", cleaned).strip(" -|:;,.")
    cleaned = _FILE_EXTENSION_RE.sub("", cleaned).strip(" -|:;,.")
    return " ".join(cleaned.split())


def _entity_rejection_reason(candidate: str) -> Optional[str]:
    tokens = _tokens(candidate)
    folded = " ".join(token.casefold() for token in tokens)
    if not candidate or not tokens or len(tokens) > 12 or _XML_RE.search(candidate):
        return "image_entity_not_found"
    if folded in _GENERIC_TITLES or all(
        token.casefold() in _GENERIC_TOKENS for token in tokens
    ):
        return "image_entity_too_generic"
    proper_tokens = sum(
        token[:1].isupper()
        or token.isupper()
        or any(character.isupper() for character in token[1:])
        or any(character.isdigit() for character in token)
        or any(ord(character) > 127 for character in token)
        for token in tokens
    )
    if proper_tokens < (1 if len(tokens) == 1 else 2):
        return "image_entity_too_generic"
    return None


def _contains_token_sequence(container: str, candidate: str) -> bool:
    container_tokens = [token.casefold() for token in _tokens(container)]
    candidate_tokens = [token.casefold() for token in _tokens(candidate)]
    width = len(candidate_tokens)
    return bool(width) and any(
        container_tokens[index:index + width] == candidate_tokens
        for index in range(len(container_tokens) - width + 1)
    )


def extract_visible_entity(
    image_results: Sequence[Mapping[str, object]],
) -> ExtractedEntity:
    """Extract an entity using image-result titles and no answer-side inputs."""

    candidates: List[Tuple[int, Optional[str], str, str]] = []
    saw_generic = False
    for position, result in enumerate(image_results, start=1):
        rank = int(result.get("rank", position))
        title = str(result.get("title", ""))
        cleaned = clean_search_title(title)
        rejection = _entity_rejection_reason(cleaned)
        if rejection:
            saw_generic = saw_generic or rejection == "image_entity_too_generic"
            continue
        document_id_value = result.get("document_id")
        document_id = (
            str(document_id_value) if document_id_value is not None else None
        )
        candidates.append((rank, document_id, title, cleaned))

    # Prefer a concise entity phrase that is independently visible in another
    # ranked title. This is deterministic and never consults an answer field.
    for rank, document_id, title, cleaned in sorted(
        candidates, key=lambda item: (len(_tokens(item[3])), item[0], item[3].casefold())
    ):
        if any(
            other_title != title and _contains_token_sequence(other_cleaned, cleaned)
            for _, _, other_title, other_cleaned in candidates
        ):
            return ExtractedEntity(
                cleaned,
                rank,
                document_id,
                title,
                "shared_visible_title_phrase",
                True,
                None,
            )

    if candidates:
        rank, document_id, title, cleaned = min(candidates, key=lambda item: item[0])
        return ExtractedEntity(
            cleaned,
            rank,
            document_id,
            title,
            "cleaned_top%d_title" % rank,
            True,
            None,
        )
    return ExtractedEntity(
        "",
        0,
        None,
        "",
        "none",
        False,
        "image_entity_too_generic" if saw_generic else "image_entity_not_found",
    )


def entity_visible_in_information(entity: str, information: str) -> bool:
    return _contains_token_sequence(unescape(information), entity)
