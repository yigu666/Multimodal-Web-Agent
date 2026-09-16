from __future__ import annotations

import hashlib
from html import escape
from typing import Any, Dict, List, Sequence, Tuple

from .query_builder import contains_answer_leak
from .schema import Message, TransitionType


SYSTEM_PROMPT = (
    "You are a visual question-answering agent. Return exactly one <reason> element "
    "followed by exactly one action: <search><img></search>, "
    "<text_search>query</text_search>, or <answer>answer</answer>. Do not emit "
    "<information>; it is supplied only by the environment. Do not add text outside tags."
)


REASON_TEMPLATES: Dict[str, Tuple[str, ...]] = {
    TransitionType.INITIAL_TO_DIRECT_ANSWER.value: (
        "The available visual evidence is sufficient to answer the question directly.",
        "The question can be answered from the image without external search.",
        "No external information is required because the image provides enough evidence.",
        "The image already contains enough information to provide the requested answer.",
        "The visible content supports a direct answer without using retrieval tools.",
        "A direct response is appropriate because the relevant evidence is visible.",
        "The requested detail can be determined reliably from the supplied image.",
        "The visual input alone provides sufficient evidence for a confident answer.",
        "Everything needed to answer is already available in the current image.",
        "External retrieval is unnecessary because the visual evidence resolves the question.",
    ),
    TransitionType.INITIAL_TO_IMAGE_SEARCH.value: (
        "The visual subject cannot be identified confidently from the image alone.",
        "A reverse image search is needed to identify the visual entity reliably.",
        "The image contains an unfamiliar entity that requires additional visual retrieval.",
        "Additional image-search evidence is required before the question can be answered.",
        "The pictured subject needs visual retrieval for a reliable identification.",
        "The available image does not identify the subject with enough confidence.",
        "A cached image search can provide the missing identity evidence needed here.",
        "Visual matching is necessary to determine what the depicted entity represents.",
        "The subject requires external visual evidence before producing a final answer.",
        "Image-based retrieval is the appropriate next step for identifying this subject.",
    ),
    TransitionType.INITIAL_TO_TEXT_SEARCH.value: (
        "The requested fact is not directly available from the image alone.",
        "A focused text search is needed to obtain the requested factual information.",
        "The question requires external textual evidence before it can be answered.",
        "Relevant text-search results are needed to resolve the requested factual relation.",
        "The visible input does not provide the external fact requested by the question.",
        "A concise textual query is required to retrieve the missing information.",
        "The answer depends on factual context that should be retrieved through text search.",
        "External written evidence is necessary to answer this knowledge-focused question.",
        "The requested relationship requires a targeted search of cached textual information.",
        "Textual retrieval is the appropriate next action for finding the requested detail.",
    ),
    TransitionType.IMAGE_INFORMATION_TO_TEXT_SEARCH.value: (
        "The image results identify the subject, but the requested attribute still needs verification.",
        "The visual search resolves the entity, while the requested fact requires textual evidence.",
        "The subject is now identifiable, so a focused factual search is required.",
        "The image-search titles establish the entity but do not provide the requested detail.",
        "The retrieved visual evidence identifies the object, but an external fact remains missing.",
        "The visible search results reveal the subject while leaving its requested property unresolved.",
        "The entity is established from image evidence, so its external attribute needs retrieval.",
        "Visual identification is complete, but textual verification is required for the remaining fact.",
        "The image context supplies a reliable entity for a targeted factual query.",
        "The subject can now anchor a text search for the requested relationship.",
    ),
    TransitionType.IMAGE_INFORMATION_TO_ANSWER.value: (
        "The image-search results provide sufficient evidence to answer the question.",
        "The retrieved visual information identifies the subject clearly enough to answer.",
        "The cached image-result titles contain the information needed for a final response.",
        "The visual retrieval evidence now supports a confident answer to the question.",
        "The ranked image-search titles resolve the identity required by the question.",
        "The retrieved image evidence is adequate to determine the requested answer.",
        "The cached visual-search information provides a reliable basis for answering now.",
        "The subject is sufficiently identified by the available image-search results.",
        "The returned visual evidence supplies the missing information needed to answer.",
        "The image-search context is now sufficient for a grounded final response.",
    ),
    TransitionType.TEXT_INFORMATION_TO_ANSWER.value: (
        "The retrieved textual evidence provides the fact requested by the question.",
        "The text-search results are sufficient to support a grounded final answer.",
        "The available search information contains the requested factual detail for answering.",
        "The retrieved webpage titles provide enough evidence to answer the question.",
        "The textual results now supply the missing information needed for a response.",
        "The ranked text evidence supports a confident answer to the requested fact.",
        "The cached textual information resolves the question and permits a final answer.",
        "The search results contain adequate factual evidence for answering at this point.",
        "The returned textual context establishes the detail required by the question.",
        "The retrieved text is sufficient to produce an evidence-grounded final response.",
    ),
}


IMAGE_TEXT_INITIAL_REASON_TEMPLATES: Tuple[str, ...] = (
    "The visual subject must be identified before the requested fact can be retrieved.",
    "Image retrieval is needed first to establish the entity for factual research.",
    "The depicted entity must be identified before searching for its external attribute.",
    "A visual search should establish the subject before a targeted text query.",
    "The image alone does not name the entity needed for subsequent factual retrieval.",
    "Visual identification is the necessary first step toward answering this factual question.",
    "The subject must be resolved through image search before investigating its requested property.",
    "A cached visual search can identify the entity that anchors the later fact search.",
    "The requested relationship depends on identifying the pictured subject through visual retrieval first.",
    "Image-search evidence is required to expose the entity before textual verification.",
)


DIRECT_REASON = REASON_TEMPLATES[TransitionType.INITIAL_TO_DIRECT_ANSWER.value][0]
IMAGE_SEARCH_REASON = REASON_TEMPLATES[TransitionType.INITIAL_TO_IMAGE_SEARCH.value][0]
TEXT_SEARCH_REASON = REASON_TEMPLATES[TransitionType.INITIAL_TO_TEXT_SEARCH.value][0]
IMAGE_ANSWER_REASON = REASON_TEMPLATES[TransitionType.IMAGE_INFORMATION_TO_ANSWER.value][0]
TEXT_ANSWER_REASON = REASON_TEMPLATES[TransitionType.TEXT_INFORMATION_TO_ANSWER.value][0]


def select_reason(
    transition: str,
    source_data_id: str,
    seed: int,
    *,
    forbidden_answers: Sequence[str] = (),
) -> str:
    templates = REASON_TEMPLATES[TransitionType(transition).value]
    return _select_reason_template(
        templates,
        "%d:%s:%s" % (seed, transition, source_data_id),
        forbidden_answers,
    )


def _select_reason_template(
    templates: Sequence[str],
    deterministic_key: str,
    forbidden_answers: Sequence[str],
) -> str:
    digest = hashlib.sha256(
        deterministic_key.encode("utf-8")
    ).digest()
    start = int.from_bytes(digest[:8], "big") % len(templates)
    for offset in range(len(templates)):
        candidate = templates[(start + offset) % len(templates)]
        if not contains_answer_leak(candidate, forbidden_answers):
            return candidate
    raise ValueError("no reason template is safe for the accepted answers")


def select_image_text_initial_reason(
    source_data_id: str,
    seed: int,
    *,
    forbidden_answers: Sequence[str] = (),
) -> str:
    return _select_reason_template(
        IMAGE_TEXT_INITIAL_REASON_TEMPLATES,
        "%d:image_text_initial:%s" % (seed, source_data_id),
        forbidden_answers,
    )


def protocol_text(value: str) -> str:
    return escape(" ".join(str(value).split()), quote=False)


def answer_action(answer: str, reason: str) -> str:
    return "<reason>%s</reason>\n<answer>%s</answer>" % (
        protocol_text(reason),
        protocol_text(answer),
    )


def image_search_action(reason: str = IMAGE_SEARCH_REASON) -> str:
    return "<reason>%s</reason>\n<search><img></search>" % protocol_text(reason)


def text_search_action(query: str, reason: str = TEXT_SEARCH_REASON) -> str:
    return "<reason>%s</reason>\n<text_search>%s</text_search>" % (
        protocol_text(reason),
        protocol_text(query),
    )


def initial_state(question: str) -> List[Message]:
    return [
        Message(role="system", content=SYSTEM_PROMPT),
        Message(role="user", content="<image>\nQuestion: %s" % " ".join(question.split())),
    ]


def source_image_ref(data_id: str, row_index: int) -> Dict[str, Any]:
    return {
        "kind": "fvqa_parquet_row",
        "data_id": data_id,
        "row_index": row_index,
        "image_index": 0,
    }
