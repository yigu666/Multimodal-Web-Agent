from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
from typing import Any, Iterable, Mapping, Sequence

from ..answer_metrics import normalize_answer
from .errors import SourceInvalidError
from .provenance import sha256_file


def _open_text(path: Path):
    if path.name.casefold().endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def iter_wiki_records(path: Path) -> Iterable[dict[str, Any]]:
    with _open_text(Path(path)) as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError(
                    "Wiki6M line %d is not an object" % line_number
                )
            yield dict(value)


def _first(row: Mapping[str, Any], fields: Sequence[str]) -> str:
    return next((
        str(row[field]).strip()
        for field in fields if row.get(field) not in (None, "")
    ), "")


def _body(row: Mapping[str, Any]) -> str:
    value: Any = next((
        row[field] for field in (
            "wikipedia_content", "text", "content", "document",
            "paragraphs", "page"
        ) if row.get(field) not in (None, "")
    ), "")
    if isinstance(value, list):
        return "\n\n".join(
            str(item.get("text") if isinstance(item, Mapping) else item)
            for item in value
        )
    if isinstance(value, Mapping):
        return str(value.get("text") or value.get("content") or "")
    return str(value)


def build_wikipedia_index(
    source_path: Path,
    index_path: Path,
    *,
    include_body: bool,
    source_sha256: str | None = None,
    required_entity_ids: Sequence[str] | None = None,
    required_titles: Sequence[str] | None = None,
) -> dict[str, Any]:
    source_path = Path(source_path)
    index_path = Path(index_path)
    digest = source_sha256 or sha256_file(source_path)
    selected_entity_ids = {
        str(value).strip()
        for value in (required_entity_ids or ())
        if str(value).strip()
    }
    selected_titles = {
        str(value).strip().casefold()
        for value in (required_titles or ())
        if str(value).strip()
    }
    selection_enabled = (
        required_entity_ids is not None or required_titles is not None
    )
    selection_sha256 = hashlib.sha256(json.dumps(
        {
            "entity_ids": sorted(selected_entity_ids),
            "titles": sorted(selected_titles),
            "selection_enabled": selection_enabled,
        },
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")).hexdigest()
    if index_path.is_file():
        with sqlite3.connect(index_path) as connection:
            meta = dict(connection.execute(
                "SELECT key, value FROM metadata"
            ).fetchall())
        if (
            meta.get("source_sha256") == digest
            and meta.get("include_body") == str(bool(include_body))
            and meta.get("selection_sha256") == selection_sha256
        ):
            return {
                **meta,
                "record_count": int(meta["record_count"]),
                "status": "SKIPPED_ALREADY_VERIFIED",
                "index_path": str(index_path),
            }
        raise ValueError(
            "SOURCE_CACHE_CONFLICT: Wikipedia index provenance changed"
        )
    temporary = index_path.with_name(index_path.name + ".partial")
    if temporary.exists():
        temporary.unlink()
    temporary.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    connection = sqlite3.connect(temporary)
    try:
        connection.executescript("""
            PRAGMA journal_mode=OFF;
            PRAGMA synchronous=OFF;
            PRAGMA locking_mode=EXCLUSIVE;
            PRAGMA temp_store=FILE;
            PRAGMA cache_size=-65536;
            CREATE TABLE metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE documents (
                record_id TEXT PRIMARY KEY,
                entity_id TEXT,
                title TEXT NOT NULL,
                body TEXT NOT NULL,
                document_sha256 TEXT NOT NULL
            );
            CREATE VIRTUAL TABLE documents_fts USING fts5(
                record_id UNINDEXED, title, body
            );
        """)
        for index, row in enumerate(iter_wiki_records(source_path)):
            entity_id = _first(row, (
                "wikidata_id", "entity_id", "wikipedia_id",
                "page_id", "id",
            ))
            title = _first(row, (
                "wikipedia_title", "title", "page_title", "name",
            ))
            if selection_enabled and (
                entity_id not in selected_entity_ids
                and title.casefold() not in selected_titles
            ):
                continue
            body = _body(row) if include_body else ""
            if not title:
                continue
            record_id = _first(row, ("record_id", "id")) or (
                entity_id or "wiki-record-%08d" % index
            )
            document_hash = hashlib.sha256(
                body.encode("utf-8")
            ).hexdigest()
            try:
                connection.execute(
                    "INSERT INTO documents VALUES (?, ?, ?, ?, ?)",
                    (record_id, entity_id, title, body, document_hash),
                )
                connection.execute(
                    "INSERT INTO documents_fts VALUES (?, ?, ?)",
                    (record_id, title, body),
                )
            except sqlite3.IntegrityError:
                continue
            count += 1
            if count % 10_000 == 0:
                connection.commit()
        connection.executescript("""
            CREATE INDEX documents_entity_id ON documents(entity_id);
            CREATE INDEX documents_title ON documents(title COLLATE NOCASE);
        """)
        metadata = {
            "source_path": str(source_path),
            "source_sha256": digest,
            "include_body": str(bool(include_body)),
            "record_count": str(count),
            "wiki_dump_version": "Wiki6M_ver_1_0",
            "selection_sha256": selection_sha256,
            "selection_enabled": str(selection_enabled),
            "required_entity_id_count": str(len(selected_entity_ids)),
            "required_title_count": str(len(selected_titles)),
        }
        connection.executemany(
            "INSERT INTO metadata VALUES (?, ?)", metadata.items()
        )
        connection.commit()
    except sqlite3.OperationalError as exc:
        free_bytes = shutil.disk_usage(temporary.parent).free
        raise SourceInvalidError(
            "WIKIPEDIA_INDEX_IO_ERROR: %s; index=%s; free_bytes=%d; "
            "the verified Wiki6M download remains reusable"
            % (exc, temporary, free_bytes)
        ) from exc
    finally:
        connection.close()
    os.replace(temporary, index_path)
    return {
        **metadata,
        "record_count": count,
        "status": "INDEX_BUILT",
        "index_path": str(index_path),
    }


def _paragraphs(body: str, maximum_chars: int) -> Iterable[tuple[int, int, int, str]]:
    cursor = 0
    for paragraph_index, part in enumerate(
        re.split(r"\n\s*\n+", body)
    ):
        paragraph = part.strip()
        if not paragraph:
            cursor += len(part)
            continue
        start = body.find(paragraph, cursor)
        end = start + len(paragraph)
        cursor = end
        yield (
            paragraph_index,
            start,
            min(end, start + maximum_chars),
            paragraph[:maximum_chars],
        )


def connect_official_evidence(
    index_path: Path,
    *,
    candidate_id: str,
    entity_id: str | None,
    wikipedia_title: str | None,
    answer_aliases: Sequence[str],
    maximum_passage_chars: int = 1800,
) -> dict[str, Any] | None:
    """Resolve only by official entity/title mapping, never by answer."""
    if not entity_id and not wikipedia_title:
        return None
    with sqlite3.connect(index_path) as connection:
        connection.row_factory = sqlite3.Row
        row = None
        if entity_id:
            row = connection.execute(
                "SELECT * FROM documents WHERE entity_id = ? "
                "ORDER BY record_id LIMIT 1",
                (entity_id,),
            ).fetchone()
        if row is None and wikipedia_title:
            row = connection.execute(
                "SELECT * FROM documents WHERE title = ? COLLATE NOCASE "
                "ORDER BY record_id LIMIT 1",
                (wikipedia_title,),
            ).fetchone()
    if row is None or not row["body"]:
        return None
    aliases = {
        normalize_answer(alias) for alias in answer_aliases if alias
    }
    for paragraph_index, start, end, passage in _paragraphs(
        row["body"], maximum_passage_chars
    ):
        normalized = normalize_answer(passage)
        if not any(alias and alias in normalized for alias in aliases):
            continue
        passage_hash = hashlib.sha256(
            passage.encode("utf-8")
        ).hexdigest()
        return {
            "candidate_id": candidate_id,
            "entity_id": entity_id or row["entity_id"],
            "wikipedia_title": row["title"],
            "evidence_record_id": row["record_id"],
            "evidence_text": passage,
            "evidence_sha256": passage_hash,
            "answer_reachable": True,
            "evidence_connection_method": "official_entity_mapping",
            "document_id": row["record_id"],
            "paragraph_index": paragraph_index,
            "character_start": start,
            "character_end": end,
            "document_sha256": row["document_sha256"],
            "passage_sha256": passage_hash,
            "wiki_dump_version": "Wiki6M_ver_1_0",
        }
    return None
