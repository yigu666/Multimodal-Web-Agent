from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image

from ..answer_metrics import answer_reachable
from ..schema import TASK_TYPES
from .base import (
    SourceCandidate,
    SourceScan,
    image_bytes,
    input_file_hashes,
    read_tabular,
    unique_aliases,
)


def _configured_paths(
    root_dir: Path,
    values: Sequence[str] | str | None,
) -> list[Path]:
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    result = []
    for value in values:
        path = Path(str(value))
        result.append(path if path.is_absolute() else root_dir / path)
    return result


def _field(row: Mapping[str, Any], specification: Any) -> Any:
    if specification is None:
        return None
    choices = (
        specification
        if isinstance(specification, (list, tuple))
        else [specification]
    )
    for choice in choices:
        value: Any = row
        for part in str(choice).split("."):
            if not isinstance(value, Mapping) or part not in value:
                value = None
                break
            value = value[part]
        if value is not None and value != "":
            return value
    return None


def _records(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, Mapping):
        values = [value]
    elif isinstance(value, Sequence):
        values = list(value)
    else:
        values = [value]
    result = []
    for item in values:
        if isinstance(item, Mapping):
            text = " | ".join(
                str(item.get(key, "")).strip()
                for key in ("title", "snippet", "text", "content")
                if str(item.get(key, "")).strip()
            )
        else:
            text = str(item).strip()
        if text:
            result.append(text)
    return tuple(result)


def _aliases(value: Any) -> tuple[str, ...]:
    if value is None:
        raise ValueError("answer is missing")
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            decoded = value
        value = decoded
    values = value if isinstance(value, (list, tuple)) else [value]
    return unique_aliases(values)


def _answer_values(row: Mapping[str, Any], specification: Any) -> list[Any]:
    choices = (
        specification
        if isinstance(specification, (list, tuple))
        else [specification]
    )
    result = []
    for choice in choices:
        value = _field(row, choice)
        if value is None:
            continue
        if isinstance(value, str):
            try:
                decoded = json.loads(value)
            except json.JSONDecodeError:
                decoded = value
            value = decoded
        if isinstance(value, (list, tuple)):
            result.extend(value)
        else:
            result.append(value)
    return result


def _evidence_from_roots(
    source_id: str,
    roots: Sequence[Path],
) -> tuple[str, ...]:
    records = []
    for root in roots:
        if not root.is_dir():
            continue
        for extension in (".json", ".jsonl", ".txt"):
            path = root / (source_id + extension)
            if not path.is_file():
                continue
            if extension == ".txt":
                records.extend(_records(path.read_text(encoding="utf-8")))
            elif extension == ".json":
                value = json.loads(path.read_text(encoding="utf-8"))
                records.extend(_records(value))
            else:
                with path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        if line.strip():
                            records.extend(_records(json.loads(line)))
    return tuple(records)


def _resolve_query_image(
    value: Any,
    *,
    root_dir: Path,
    image_roots: Sequence[Path],
) -> tuple[bytes, str, str]:
    if isinstance(value, Mapping):
        raw_value = value.get("bytes")
        if isinstance(raw_value, (bytes, bytearray)) and raw_value:
            raw, extension = image_bytes(value)
            return raw, extension, "embedded_bytes"
        path_value = value.get("path")
        if path_value:
            return _resolve_query_image(
                str(path_value),
                root_dir=root_dir,
                image_roots=image_roots,
            )
        raise ValueError("missing_query_image")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("missing_query_image")
    raw_path = Path(value)
    candidates = (
        [raw_path] if raw_path.is_absolute()
        else [root_dir / raw_path, *[root / raw_path for root in image_roots]]
    )
    for path in candidates:
        if path.is_file():
            return path.read_bytes(), path.suffix or ".jpg", path.as_posix()
    raise FileNotFoundError(value)


def _verify_image(raw: bytes) -> None:
    from io import BytesIO

    with Image.open(BytesIO(raw)) as image:
        image.verify()


class GenericHeldoutSourceAdapter:
    source_name = "generic_heldout"

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        project_root: Path,
        source_name: str | None = None,
    ):
        self.config = dict(config)
        self.project_root = Path(project_root)
        if source_name:
            self.source_name = source_name

    def _root(self) -> Path | None:
        value = self.config.get("root_dir")
        if value is None:
            return None
        path = Path(str(value))
        return path if path.is_absolute() else self.project_root / path

    def scan(self) -> SourceScan:
        configured = self.config.get("enabled") is True
        root_dir = self._root()
        if not configured:
            return SourceScan(
                self.source_name,
                False,
                0,
                "%s source is disabled" % self.source_name,
                inventory={
                    "configured": False,
                    "status": "unavailable",
                },
            )
        if root_dir is None or not root_dir.is_dir():
            return SourceScan(
                self.source_name,
                False,
                0,
                "%s root is unavailable" % self.source_name,
                inventory={
                    "configured": True,
                    "root_exists": False,
                    "status": "unavailable",
                    "missing_root": (
                        str(root_dir) if root_dir is not None else None
                    ),
                },
            )
        annotation_values = (
            self.config.get("annotations")
            or self.config.get("annotation_file")
        )
        annotations = _configured_paths(root_dir, annotation_values)
        annotation_files = [path for path in annotations if path.is_file()]
        image_roots = _configured_paths(
            root_dir, self.config.get("image_roots")
        )
        evidence_roots = _configured_paths(
            root_dir, self.config.get("evidence_roots")
        )
        inventory = {
            "configured": True,
            "root_exists": True,
            "annotation_files_found": len(annotation_files),
            "annotation_files_missing": [
                path.as_posix() for path in annotations if not path.is_file()
            ],
            "image_files_found": sum(
                1 for root in image_roots if root.is_dir()
                for path in root.rglob("*") if path.is_file()
            ),
            "evidence_files_found": sum(
                1 for root in evidence_roots if root.is_dir()
                for path in root.rglob("*") if path.is_file()
            ),
        }
        if not annotation_files:
            inventory["status"] = "unavailable"
            return SourceScan(
                self.source_name,
                False,
                0,
                "%s annotations are unavailable" % self.source_name,
                inventory=inventory,
            )
        field_map = dict(self.config.get("field_map", {}))
        if "answer_aliases" in field_map and "answers" not in field_map:
            field_map["answers"] = field_map["answer_aliases"]
        if (
            self.source_name == "generic_heldout"
            and not all(
                key in field_map
                for key in (
                    "source_id",
                    "question",
                    "answer_aliases",
                    "query_image",
                    "evidence",
                )
            )
        ):
            inventory["status"] = "invalid"
            inventory["blocking_reasons"] = [
                "generic_adapter_requires_explicit_field_map"
            ]
            return SourceScan(
                self.source_name,
                False,
                0,
                "generic Held-out field_map is incomplete",
                inventory=inventory,
            )
        rows = []
        for annotation in annotation_files:
            for row in read_tabular(annotation):
                rows.append((annotation, row))
        candidates = []
        rejected = []
        rejection_counts: dict[str, int] = {}
        image_file_count = 0
        evidence_record_count = 0
        evidence_mode = str(self.config.get("evidence_mode", "text"))
        configured_license = str(
            self.config.get("license") or ""
        ).strip()
        license_file_value = self.config.get("license_file")
        if not configured_license and license_file_value:
            license_file = Path(str(license_file_value))
            if not license_file.is_absolute():
                license_file = root_dir / license_file
            if license_file.is_file():
                configured_license = license_file.read_text(
                    encoding="utf-8"
                ).strip()
        for index, (annotation, row) in enumerate(rows):
            source_id = str(_field(
                row,
                field_map.get(
                    "source_id", ("source_data_id", "data_id", "id")
                ),
            ) or "").strip()
            reasons = []
            question = str(_field(
                row, field_map.get("question", ("question", "query"))
            ) or "").strip()
            if not source_id:
                reasons.append("missing_source_data_id")
            if not question:
                reasons.append("empty_question")
            try:
                aliases = _aliases(_answer_values(
                    row,
                    field_map.get(
                        "answers",
                        (
                            "answer_aliases",
                            "answers",
                            "answer",
                            "original_answer",
                        ),
                    ),
                ))
            except ValueError:
                aliases = ()
                reasons.append("empty_answer")
            image_value = _field(
                row,
                field_map.get(
                    "query_image",
                    ("query_image", "image_path", "image"),
                ),
            )
            try:
                raw_image, extension, image_path = _resolve_query_image(
                    image_value,
                    root_dir=root_dir,
                    image_roots=image_roots,
                )
                _verify_image(raw_image)
                image_file_count += 1
            except (FileNotFoundError, OSError, TypeError, ValueError):
                raw_image = b""
                extension = ".jpg"
                image_path = ""
                reasons.append(
                    "missing_query_image"
                    if image_value is None else "unreadable_query_image"
                )
            image_records = _records(_field(
                row, field_map.get("image_evidence")
            ))
            text_records = _records(_field(
                row, field_map.get("text_evidence")
            ))
            generic_records = _records(_field(
                row,
                field_map.get(
                    "evidence",
                    ("offline_evidence_records", "evidence"),
                ),
            ))
            generic_records += _evidence_from_roots(
                source_id, evidence_roots
            )
            if generic_records:
                if evidence_mode in {"image", "both"}:
                    image_records += generic_records
                if evidence_mode in {"text", "both"}:
                    text_records += generic_records
            evidence_record_count += len(image_records) + len(text_records)
            if not image_records and not text_records:
                reasons.append("offline_evidence_missing")
            elif aliases and not answer_reachable(
                aliases, list(image_records + text_records)
            ):
                reasons.append("offline_evidence_answer_unreachable")
            row_license = str(
                _field(row, field_map.get("license", "license"))
                or configured_license
                or ""
            ).strip()
            if not row_license:
                reasons.append("missing_license_metadata")
            raw_eligible = _field(
                row, field_map.get("eligible_task_types")
            )
            eligible_types = (
                [str(value) for value in raw_eligible]
                if isinstance(raw_eligible, (list, tuple)) else []
            )
            if any(value not in TASK_TYPES for value in eligible_types):
                reasons.append("invalid_eligible_task_type")
            suggested = str(_field(
                row, field_map.get("task_type")
            ) or "").strip() or None
            if suggested is not None and suggested not in TASK_TYPES:
                reasons.append("invalid_suggested_task_type")
            if reasons:
                normalized_reasons = sorted(set(reasons))
                rejected.append({
                    "source_dataset": self.source_name,
                    "source_data_id": source_id or "row-%d" % index,
                    "source_annotation": annotation.as_posix(),
                    "source_row_index": index,
                    "rejection_reasons": normalized_reasons,
                })
                for reason in normalized_reasons:
                    rejection_counts[reason] = (
                        rejection_counts.get(reason, 0) + 1
                    )
                continue
            metadata_fields = (
                "entity",
                "entity_id",
                "wikipedia_title",
                "question_type",
                "knowledge_source",
                "original_answer",
                "source_url",
                "split",
                "license",
            )
            metadata = {
                key: _field(row, field_map.get(key, key))
                for key in metadata_fields
                if _field(row, field_map.get(key, key)) is not None
            }
            metadata["license"] = row_license
            if (
                "source_url" not in metadata
                and self.config.get("source_url")
            ):
                metadata["source_url"] = self.config["source_url"]
            metadata.update({
                "source_row_index": index,
                "source_annotation": annotation.as_posix(),
                "query_image_path": image_path,
                "query_image_role_verified": True,
                "query_image_role_source": "configured_dataset_annotation",
                "retrieval_result_images_excluded_from_input": True,
                "declared_eligible_task_types": eligible_types,
                "online_access": False,
                "source_split": (
                    _field(row, field_map.get("source_split", "split"))
                    or self.config.get("source_split")
                ),
                "source_manifest": self.config.get("source_manifest"),
            })
            candidate = SourceCandidate(
                source_dataset=self.source_name,
                source_data_id=source_id,
                question=question,
                image_bytes=raw_image,
                image_extension=extension,
                answer_aliases=aliases,
                suggested_task_type=suggested,
                image_search_records=image_records,
                text_corpus_records=text_records,
                source_metadata=metadata,
            )
            candidate.validate()
            candidates.append(candidate)
        status = (
            "available" if candidates and not rejected
            else "partial" if candidates
            else "invalid"
        )
        inventory.update({
            "status": status,
            "query_images_resolved": image_file_count,
            "offline_evidence_record_count": evidence_record_count,
            "acquisition": self.config.get("acquisition"),
        })
        return SourceScan(
            source_name=self.source_name,
            available=bool(candidates),
            candidate_count=len(candidates),
            message="%s held-out source scanned" % self.source_name,
            candidates=tuple(candidates),
            input_files_sha256=input_file_hashes({
                path.relative_to(root_dir).as_posix(): path
                for path in annotation_files
            }),
            skipped_candidate_counts=rejection_counts,
            raw_record_count=len(rows),
            hard_rejected_rows=tuple(rejected),
            inventory=inventory,
        )
