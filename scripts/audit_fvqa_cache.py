#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import io
import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pyarrow.parquet as pq
from PIL import Image


CACHE_VERSION = (
    "lmms-lab/FVQA@bb4a4ff4c9c3fd0382d11f5d7fccd66d0b8428b5"
)
CACHE_SOURCE = "lmms-lab/FVQA official image search cache"
CLEANER_VERSION = "raw-audit-v1-no-cleaning"

TITLE_FIELD = "tool_returned_web_title_list"
IMAGE_FIELD = "tool_returned_images_urls"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def load_data_ids(parquet_path: Path) -> list[Any]:
    table = pq.read_table(parquet_path, columns=["data_id"])
    return table.column("data_id").to_pylist()


def normalize_sequence(
    record: dict[str, Any],
    field: str,
) -> tuple[list[Any], bool]:
    """返回字段内容和字段结构是否非法。"""

    if field not in record:
        return [], True

    value = record[field]

    if value is None:
        return [], False

    if isinstance(value, (list, tuple)):
        return list(value), False

    return [], True


def validate_thumbnail(value: Any) -> tuple[str, bool]:
    """
    返回：
      type_name
      broken

    注意：URL 只做格式检查，不进行联网可达性检查。
    """

    if isinstance(value, str):
        value = value.strip()

        if not value:
            return "empty_string", True

        parsed = urlparse(value)

        if parsed.scheme in {"http", "https"} and parsed.netloc:
            return "remote_url", False

        return "invalid_url_string", True

    if isinstance(value, Image.Image):
        try:
            copied = value.copy()
            copied.load()
            return "embedded_pil_image", False
        except Exception:
            return "embedded_pil_decode_error", True

    if isinstance(value, (bytes, bytearray)):
        try:
            with Image.open(io.BytesIO(value)) as image:
                image.verify()
            return "embedded_image_bytes", False
        except Exception:
            return "embedded_bytes_decode_error", True

    if value is None:
        return "none", True

    return f"unsupported:{type(value).__name__}", True


def audit_split(
    root: Path,
    output_dir: Path,
    split: str,
) -> dict[str, Any]:
    parquet_path = root / f"fvqa_{split}.parquet"
    cache_path = (
        root
        / f"fvqa_{split}_image_search_results_cache.pkl"
    )

    if not parquet_path.is_file():
        raise FileNotFoundError(parquet_path)

    if not cache_path.is_file():
        raise FileNotFoundError(cache_path)

    data_ids = load_data_ids(parquet_path)
    data_id_counter = Counter(data_ids)
    unique_data_ids = set(data_ids)

    with cache_path.open("rb") as file:
        cache = pickle.load(file)

    if not isinstance(cache, dict):
        raise TypeError(
            f"{cache_path} 顶层对象不是 dict，而是 "
            f"{type(cache).__name__}"
        )

    cache_keys = set(cache.keys())

    missing_ids = sorted(
        unique_data_ids - cache_keys,
        key=str,
    )
    extra_keys = sorted(
        cache_keys - unique_data_ids,
        key=str,
    )

    duplicate_ids = {
        str(data_id): count
        for data_id, count in data_id_counter.items()
        if count > 1
    }

    metrics: dict[str, Any] = {
        "split": split,
        "candidate_count": len(data_ids),
        "unique_candidate_count": len(unique_data_ids),
        "duplicate_candidate_id_count": sum(
            count - 1
            for count in data_id_counter.values()
            if count > 1
        ),
        "duplicate_candidate_ids": duplicate_ids,
        "cache_key_count": len(cache),
        "cache_hit_count": sum(
            data_id in cache
            for data_id in data_ids
        ),
        "unique_cache_hit_count": len(
            unique_data_ids & cache_keys
        ),
        "cache_hit_ratio": (
            sum(data_id in cache for data_id in data_ids)
            / len(data_ids)
            if data_ids
            else 0.0
        ),
        "missing_unique_count": len(missing_ids),
        "missing_data_ids": [str(item) for item in missing_ids],
        "extra_cache_key_count": len(extra_keys),
        "extra_cache_keys": [str(item) for item in extra_keys],
        "empty_result_count": 0,
        "invalid_result_count": 0,
        "invalid_title_count": 0,
        "broken_thumbnail_count": 0,
        "non_serializable_object_count": 0,
        "json_non_serializable_record_count": 0,
        "length_mismatch_record_count": 0,
        "records_with_no_titles": 0,
        "records_with_no_thumbnails": 0,
        "total_title_count": 0,
        "total_thumbnail_count": 0,
        "valid_title_count": 0,
        "valid_thumbnail_count": 0,
        "thumbnail_type_counts": {},
        "cache_file": str(cache_path.resolve()),
        "cache_file_sha256": sha256_file(cache_path),
        "cache_version": CACHE_VERSION,
        "source": CACHE_SOURCE,
        "cleaner_version": CLEANER_VERSION,
        "remote_thumbnail_reachability_checked": False,
    }

    thumbnail_type_counter: Counter[str] = Counter()
    record_manifest_path = (
        output_dir / f"fvqa_cache_records_{split}.jsonl"
    )

    with record_manifest_path.open(
        "w",
        encoding="utf-8",
    ) as manifest_file:
        for data_id in sorted(unique_data_ids, key=str):
            if data_id not in cache:
                manifest_record = {
                    "data_id": str(data_id),
                    "present": False,
                    "cache_version": CACHE_VERSION,
                    "source": CACHE_SOURCE,
                    "raw_result_hash": None,
                    "cleaner_version": CLEANER_VERSION,
                }

                manifest_file.write(
                    json.dumps(
                        manifest_record,
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                continue

            record = cache[data_id]
            record_invalid = False

            if not isinstance(record, dict):
                record_invalid = True
                titles: list[Any] = []
                thumbnails: list[Any] = []
            else:
                titles, title_field_invalid = normalize_sequence(
                    record,
                    TITLE_FIELD,
                )
                thumbnails, image_field_invalid = normalize_sequence(
                    record,
                    IMAGE_FIELD,
                )

                record_invalid = (
                    title_field_invalid
                    or image_field_invalid
                )

            valid_title_count = 0
            invalid_title_count = 0

            for title in titles:
                if isinstance(title, str) and title.strip():
                    valid_title_count += 1
                else:
                    invalid_title_count += 1

            valid_thumbnail_count = 0
            broken_thumbnail_count = 0

            for thumbnail in thumbnails:
                thumbnail_type, broken = validate_thumbnail(
                    thumbnail
                )
                thumbnail_type_counter[thumbnail_type] += 1

                if broken:
                    broken_thumbnail_count += 1
                else:
                    valid_thumbnail_count += 1

            if len(titles) != len(thumbnails):
                metrics["length_mismatch_record_count"] += 1

            if valid_title_count == 0:
                metrics["records_with_no_titles"] += 1

            if valid_thumbnail_count == 0:
                metrics["records_with_no_thumbnails"] += 1

            if (
                valid_title_count == 0
                and valid_thumbnail_count == 0
            ):
                metrics["empty_result_count"] += 1

            if invalid_title_count:
                record_invalid = True

            if record_invalid:
                metrics["invalid_result_count"] += 1

            metrics["invalid_title_count"] += invalid_title_count
            metrics["broken_thumbnail_count"] += (
                broken_thumbnail_count
            )
            metrics["total_title_count"] += len(titles)
            metrics["total_thumbnail_count"] += len(thumbnails)
            metrics["valid_title_count"] += valid_title_count
            metrics["valid_thumbnail_count"] += (
                valid_thumbnail_count
            )

            raw_result_hash = None

            try:
                serialized = pickle.dumps(
                    record,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
                raw_result_hash = hashlib.sha256(
                    serialized
                ).hexdigest()
            except Exception:
                metrics["non_serializable_object_count"] += 1

            try:
                json.dumps(record)
            except (TypeError, ValueError):
                metrics[
                    "json_non_serializable_record_count"
                ] += 1

            manifest_record = {
                "data_id": str(data_id),
                "present": True,
                "title_count": len(titles),
                "valid_title_count": valid_title_count,
                "thumbnail_count": len(thumbnails),
                "valid_thumbnail_count": valid_thumbnail_count,
                "broken_thumbnail_count": broken_thumbnail_count,
                "invalid_record": record_invalid,
                "cache_version": CACHE_VERSION,
                "source": CACHE_SOURCE,
                "raw_result_hash": raw_result_hash,
                "cleaner_version": CLEANER_VERSION,
            }

            manifest_file.write(
                json.dumps(
                    manifest_record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    metrics["thumbnail_type_counts"] = dict(
        sorted(thumbnail_type_counter.items())
    )
    metrics["record_manifest"] = str(
        record_manifest_path.resolve()
    )

    report_path = (
        output_dir / f"fvqa_cache_audit_{split}.json"
    )
    report_path.write_text(
        json.dumps(
            metrics,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=" * 72)
    print(f"split: {split}")
    print(f"candidate_count: {metrics['candidate_count']}")
    print(
        "unique_candidate_count:",
        metrics["unique_candidate_count"],
    )
    print(f"cache_key_count: {metrics['cache_key_count']}")
    print(f"cache_hit_count: {metrics['cache_hit_count']}")
    print(
        "cache_hit_ratio:",
        f"{metrics['cache_hit_ratio']:.6f}",
    )
    print(
        "missing_unique_count:",
        metrics["missing_unique_count"],
    )
    print(
        "extra_cache_key_count:",
        metrics["extra_cache_key_count"],
    )
    print(
        "empty_result_count:",
        metrics["empty_result_count"],
    )
    print(
        "invalid_result_count:",
        metrics["invalid_result_count"],
    )
    print(
        "broken_thumbnail_count:",
        metrics["broken_thumbnail_count"],
    )
    print(
        "non_serializable_object_count:",
        metrics["non_serializable_object_count"],
    )
    print(f"report: {report_path.resolve()}")
    print(f"records: {record_manifest_path.resolve()}")

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/raw/fvqa"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/manifests"),
    )
    parser.add_argument(
        "--splits", nargs="+", choices=("train", "test"),
        default=("train", "test"),
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    combined: dict[str, Any] = {}

    for split in args.splits:
        combined[split] = audit_split(
            root=args.root,
            output_dir=args.output_dir,
            split=split,
        )

    combined_path = (
        args.output_dir / "fvqa_cache_audit_summary.json"
    )
    combined_path.write_text(
        json.dumps(
            combined,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=" * 72)
    print("combined report:", combined_path.resolve())


if __name__ == "__main__":
    main()
