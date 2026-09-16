from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Sequence

from multimodal_web_agent.data.protocol_sft.text_retriever import (
    BootstrapTextRetriever,
    TextDocument,
)


@dataclass(frozen=True)
class PageChunk:
    chunk_id: str
    url: str
    title: str
    text: str
    page_rank: int
    chunk_index: int


def clean_text(value: str) -> str:
    text = str(value).replace("\x00", " ")
    return re.sub(r"\s+", " ", text).strip()


def chunk_page(
    text: str,
    *,
    url: str,
    title: str,
    page_rank: int,
    chunk_words: int = 320,
    overlap_words: int = 48,
) -> list[PageChunk]:
    if chunk_words <= overlap_words or overlap_words < 0:
        raise ValueError("invalid chunk size/overlap")
    words = clean_text(text).split()
    chunks = []
    step = chunk_words - overlap_words
    for start in range(0, len(words), step):
        selected = words[start:start + chunk_words]
        if not selected:
            break
        chunk_index = len(chunks)
        chunks.append(PageChunk(
            chunk_id="page-%04d-chunk-%04d" % (page_rank, chunk_index),
            url=url,
            title=title,
            text=" ".join(selected),
            page_rank=page_rank,
            chunk_index=chunk_index,
        ))
        if start + chunk_words >= len(words):
            break
    return chunks


def select_chunks(query: str, chunks: Sequence[PageChunk], *, top_k: int) -> list[PageChunk]:
    documents = [
        TextDocument(
            document_id=chunk.chunk_id,
            text=chunk.text,
            source_data_id=chunk.url,
            cache_result_index=chunk.chunk_index,
            raw_result_hash="",
        )
        for chunk in chunks
    ]
    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    hits = BootstrapTextRetriever(documents).retrieve(query, top_k=top_k)
    selected = [by_id[hit.document.document_id] for hit in hits]
    if selected:
        return selected
    return sorted(chunks, key=lambda item: (item.page_rank, item.chunk_index))[:top_k]
