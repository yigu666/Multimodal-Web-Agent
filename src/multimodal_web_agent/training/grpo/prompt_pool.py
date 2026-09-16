from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from multimodal_web_agent.data.protocol_sft.answer_normalizer import accepted_answer_list
from multimodal_web_agent.data.protocol_sft.cache_reader import ImageSearchCache, sha256_file
from multimodal_web_agent.data.quality.duplicate_grouper import entity_group_id, image_dhash, image_sha256, near_duplicate_group_id, stable_group_id
from multimodal_web_agent.data.quality.pool_manifest import write_json, write_jsonl, write_sha256_manifest

from .schema import PromptPoolItem


POOL_SCHEMA = "grpo-prompt-pool-v1"
DEFAULT_COUNTS = {
    "train": {"search_free": 768, "search_required": 1280},
    "reward_audit": {"search_free": 128, "search_required": 128},
    "dev": {"search_free": 128, "search_required": 128},
}


def _row_image_bytes(row: Mapping[str, Any]) -> bytes | None:
    images = row.get("images") or []
    if isinstance(images, Mapping):
        images = [images]
    if images and isinstance(images[0], Mapping):
        value = images[0].get("bytes")
        if isinstance(value, (bytes, bytearray)):
            return bytes(value)
    if isinstance(images, (bytes, bytearray)):
        return bytes(images)
    return None


def _question(row: Mapping[str, Any]) -> str:
    prompt = row.get("prompt", row.get("question", ""))
    if isinstance(prompt, str):
        return prompt.strip()
    if isinstance(prompt, Sequence):
        contents = []
        for message in prompt:
            if isinstance(message, Mapping):
                content = str(message.get("content", "")).strip()
                if content:
                    contents.append(content)
        return contents[-1] if contents else ""
    return str(prompt).strip()


def _reward(row: Mapping[str, Any]) -> Mapping[str, Any]:
    value = row.get("reward_model", row)
    return value if isinstance(value, Mapping) else {}


def _read_rows(path: Path) -> Iterable[dict[str, Any]]:
    if path.suffix.casefold() in {".parquet", ".pq"}:
        import pyarrow.parquet as pq
        parquet = pq.ParquetFile(path)
        columns = [name for name in ("prompt", "images", "reward_model", "data_id", "category", "data_source") if name in parquet.schema_arrow.names]
        for batch in parquet.iter_batches(batch_size=128, columns=columns):
            yield from batch.to_pylist()
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield json.loads(line)


def _read_id_file(path: Path) -> set[str]:
    result: set[str] = set()
    if not path.is_file():
        return result
    for line in path.read_text(
        encoding="utf-8", errors="ignore"
    ).splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, Mapping):
            for key in ("data_id", "source_data_id", "source_id"):
                if value.get(key) is not None:
                    result.add(str(value[key]))
    return result


def _historical_split_ids(
    paths: Iterable[Path],
) -> dict[str, set[str]]:
    splits = {"train": set(), "dev": set(), "test": set()}
    for path in paths:
        if not path.exists():
            continue
        if path.is_file():
            split = path.stem if path.stem in splits else "train"
            splits[split].update(_read_id_file(path))
        else:
            # Only split payloads define historical SFT membership.  Do not
            # scan predictions, diagnostics, manifests, or arbitrary nested
            # JSONL files: those can contain the complete candidate inventory.
            for split in splits:
                splits[split].update(
                    _read_id_file(path / f"{split}.jsonl")
                )
    return splits


def _historical_ids(paths: Iterable[Path]) -> set[str]:
    by_split = _historical_split_ids(paths)
    return set().union(*by_split.values())


def _candidate(row: Mapping[str, Any], row_index: int, cache: ImageSearchCache) -> PromptPoolItem | None:
    data_id = str(row.get("data_id", "")).strip()
    question = _question(row)
    reward = _reward(row)
    ground_truth = str(reward.get("ground_truth", "")).strip()
    candidates = accepted_answer_list(ground_truth, reward.get("candidate_answers"))
    category = str(row.get("category", "")).casefold().replace("-", "_")
    raw_image = _row_image_bytes(row)
    entry = cache.get(data_id)
    if not data_id or not question or not ground_truth or not candidates or category not in {"search_free", "search_required"} or raw_image is None:
        return None
    image_hash = image_sha256(raw_image)
    if category == "search_required" and (entry is None or not entry.usable_image_results):
        return None
    titles = [title for _, title in (entry.usable_titles if entry else [])]
    entity = entity_group_id(titles=titles, question=question)
    near = near_duplicate_group_id(source_hash=image_hash, dhash=image_dhash(raw_image), source_data_id=data_id)
    uid = hashlib.sha256(f"{data_id}|{image_hash}|{question}".encode()).hexdigest()
    valid_actions = ["answer", "image_search"] if category == "search_free" else ["image_search"]
    return PromptPoolItem(
        prompt_uid=f"grpo:v1:{uid}", source_data_id=data_id, data_id=data_id, source_split="FVQA Train",
        image_ref={"kind": "fvqa_parquet_row", "path": str(row.get("image_ref", "")), "row_index": row_index, "image_index": 0, "data_id": data_id},
        image_sha256=image_hash, question=question, ground_truth=ground_truth, candidate_answers=candidates,
        category=category, image_cache_key=data_id, valid_action_set=valid_actions,
        entity_group_id=entity, near_duplicate_group_id=near, source_group_id=stable_group_id("source", data_id),
        metadata={"row_index": row_index, "data_source": str(row.get("data_source", "mmsearch_r1/fvqa_train")), "cache_file_sha256": cache.file_sha256},
    )


def _overlap_report(pools: Mapping[str, Sequence[PromptPoolItem]]) -> dict[str, Any]:
    report: dict[str, Any] = {}
    fields = ("source_data_id", "image_sha256", "entity_group_id", "near_duplicate_group_id")
    for left, right in (("train", "reward_audit"), ("train", "dev"), ("reward_audit", "dev")):
        report[f"{left}_vs_{right}"] = {field: len({getattr(item, field) for item in pools[left]} & {getattr(item, field) for item in pools[right]}) for field in fields}
    return report


def build_prompt_pool_v1(
    *, source_path: Path, cache_path: Path, output_dir: Path, manifest_dir: Path,
    counts: Mapping[str, Mapping[str, int]] | None = None, seed: int = 20260730,
    historical_paths: Sequence[Path] = (), strict: bool = True,
) -> dict[str, Any]:
    counts = counts or DEFAULT_COUNTS
    cache = ImageSearchCache.load(cache_path, label="fvqa_train_official_cache")
    historical_by_split = _historical_split_ids(historical_paths)
    rejection_counts: Counter[str] = Counter()
    candidates: list[PromptPoolItem] = []
    for index, row in enumerate(_read_rows(Path(source_path))):
        item = _candidate(row, index, cache)
        if item is not None:
            candidates.append(item)
            continue
        data_id = str(row.get("data_id", "")).strip()
        reward = _reward(row)
        category = str(row.get("category", "")).casefold().replace("-", "_")
        if not data_id:
            rejection_counts["missing_data_id"] += 1
        elif not _question(row):
            rejection_counts["missing_question"] += 1
        elif not str(reward.get("ground_truth", "")).strip() or not accepted_answer_list(reward.get("ground_truth", ""), reward.get("candidate_answers")):
            rejection_counts["missing_or_unparseable_answer"] += 1
        elif category not in {"search_free", "search_required"}:
            rejection_counts["unsupported_category"] += 1
        elif _row_image_bytes(row) is None:
            rejection_counts["unreadable_image"] += 1
        elif category == "search_required" and (cache.get(data_id) is None or not cache.get(data_id).usable_image_results):
            rejection_counts["image_cache_miss"] += 1
        else:
            rejection_counts["other_executability_rejection"] += 1
    candidates.sort(key=lambda item: item.prompt_uid)
    unique: dict[str, PromptPoolItem] = {item.prompt_uid: item for item in candidates}
    candidates = list(unique.values())
    historical_groups: dict[str, dict[str, set[str]]] = {
        split: {
            field: set()
            for field in (
                "source_data_id",
                "image_sha256",
                "entity_group_id",
                "near_duplicate_group_id",
            )
        }
        for split in ("train", "dev", "test")
    }
    for split, ids in historical_by_split.items():
        for item in candidates:
            if item.source_data_id not in ids:
                continue
            for field in historical_groups[split]:
                historical_groups[split][field].add(
                    str(getattr(item, field))
                )

    def in_historical_split(item: PromptPoolItem, split: str) -> bool:
        return any(
            str(getattr(item, field)) in values
            for field, values in historical_groups[split].items()
        )

    def in_any_historical_split(item: PromptPoolItem) -> bool:
        return any(
            in_historical_split(item, split)
            for split in ("train", "dev", "test")
        )

    unused = [
        item for item in candidates if not in_any_historical_split(item)
    ]
    reusable_sft_train = [
        item
        for item in candidates
        if in_historical_split(item, "train")
        and not in_historical_split(item, "dev")
        and not in_historical_split(item, "test")
    ]
    # Stable, seeded ordering; historical SFT ids are deprioritized, not silently
    # promoted into Dev or Audit.
    def order(items: Sequence[PromptPoolItem]) -> list[PromptPoolItem]:
        return sorted(items, key=lambda item: hashlib.sha256(f"{seed}|{item.prompt_uid}".encode()).hexdigest())
    available = order(unused) + order(reusable_sft_train)
    pools: dict[str, list[PromptPoolItem]] = {"train": [], "reward_audit": [], "dev": []}
    used: set[str] = set()
    isolation_fields = (
        "source_data_id",
        "image_sha256",
        "entity_group_id",
        "near_duplicate_group_id",
    )
    group_owner: dict[str, dict[str, str]] = {
        field: {} for field in isolation_fields
    }

    def can_assign(item: PromptPoolItem, pool_name: str) -> bool:
        return all(
            group_owner[field].get(str(getattr(item, field))) in {None, pool_name}
            for field in isolation_fields
        )

    def assign(item: PromptPoolItem, pool_name: str) -> None:
        used.add(item.prompt_uid)
        pools[pool_name].append(item)
        for field in isolation_fields:
            group_owner[field][str(getattr(item, field))] = pool_name

    # Dev and audit are allocated first so they cannot accidentally be consumed by train.
    for pool_name in ("dev", "reward_audit", "train"):
        for category in ("search_free", "search_required"):
            target = int(counts[pool_name][category])
            choices = [
                item
                for item in available
                if item.prompt_uid not in used
                and item.category == category
                and (
                    pool_name == "train"
                    or not in_any_historical_split(item)
                )
                and can_assign(item, pool_name)
            ]
            if len(choices) < target:
                if strict:
                    raise RuntimeError(
                        f"GRPO Prompt Pool v1 build failed: pool={pool_name} category={category} "
                        f"required={target} available={len(choices)} candidate_count={len(candidates)} "
                        f"rejection_reasons={dict(sorted(rejection_counts.items()))}"
                    )
                target = len(choices)
            for item in choices[:target]:
                assign(item, pool_name)
    pools["train"].sort(key=lambda item: item.prompt_uid)
    pools["reward_audit"].sort(key=lambda item: item.prompt_uid)
    pools["dev"].sort(key=lambda item: item.prompt_uid)
    overlaps = _overlap_report(pools)
    size_report = {name: {category: sum(item.category == category for item in values) for category in ("search_free", "search_required")} for name, values in pools.items()}
    manifest = {
        "schema_version": POOL_SCHEMA, "seed": seed, "source_path": str(source_path), "cache_path": str(cache_path),
        "source_sha256": sha256_file(Path(source_path)), "cache_sha256": cache.file_sha256, "counts": size_report,
        "requested_counts": counts, "total_unique_prompts": len({item.prompt_uid for values in pools.values() for item in values}),
        "reused_sft_train_group_count": len({item.source_group_id for item in pools["train"] if in_historical_split(item, "train")}),
        "excluded_sft_dev_group_count": len({item.source_group_id for item in candidates if in_historical_split(item, "dev")}),
        "excluded_test_group_count": len({item.source_group_id for item in candidates if in_historical_split(item, "test")}),
        "historical_metadata_available": bool(historical_paths), "group_isolation": overlaps,
        "pool_pass": {name: int(all(size_report[name][category] == int(counts[name][category]) for category in ("search_free", "search_required"))) for name in pools}, "strict": strict,
    }
    exact_counts = all(
        size_report[name][category] == int(counts[name][category])
        for name in pools for category in ("search_free", "search_required")
    )
    audit = {"schema_version": POOL_SCHEMA, "candidate_count": len(candidates), "selected_count": sum(map(len, pools.values())), "rejection_summary": dict(sorted(rejection_counts.items())), "group_isolation": overlaps, "exact_counts": exact_counts, "passed": exact_counts and all(all(value == 0 for value in row.values()) for row in overlaps.values())}
    if strict and not audit["passed"]:
        raise RuntimeError(
            "GRPO Prompt Pool v1 audit failed before publish: "
            + json.dumps(audit, ensure_ascii=False, sort_keys=True)
        )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "train.jsonl", [item.to_dict() for item in pools["train"]])
    write_jsonl(output_dir / "reward_audit.jsonl", [item.to_dict() for item in pools["reward_audit"]])
    write_jsonl(output_dir / "dev.jsonl", [item.to_dict() for item in pools["dev"]])
    write_json(output_dir / "manifest.json", manifest)
    write_json(output_dir / "audit.json", audit)
    marker = "\n\nGRPO_PROMPT_POOL_V1_READY\n" if audit["passed"] else "\n"
    (output_dir / "audit_report.md").write_text("# GRPO Prompt Pool v1 Audit\n\n" + json.dumps(audit, ensure_ascii=False, indent=2) + marker, encoding="utf-8")
    preview = ["# Sample Preview", ""]
    for name in ("train", "reward_audit", "dev"):
        preview.append(f"## {name}")
        preview.extend(f"- `{item.prompt_uid}` [{item.category}] {item.question}" for item in pools[name][:5])
    (output_dir / "sample_preview.md").write_text("\n".join(preview) + "\n", encoding="utf-8")
    manifest_dir = Path(manifest_dir)
    write_json(manifest_dir / "grpo_prompt_pool_v1_manifest.json", manifest)
    write_json(manifest_dir / "grpo_prompt_pool_v1_audit.json", audit)
    write_sha256_manifest(
        [output_dir / name for name in ("train.jsonl", "reward_audit.jsonl", "dev.jsonl", "manifest.json", "audit.json")],
        project_root=manifest_dir.parent,
        output_path=manifest_dir / "grpo_prompt_pool_v1_files.sha256",
    )
    return manifest


class PromptPoolBuilder:
    """Object-oriented facade used by server orchestration and tests."""

    def __init__(self, *, source_path: Path, cache_path: Path, output_dir: Path, manifest_dir: Path, seed: int = 20260730, historical_paths: Sequence[Path] = ()):
        self.source_path = Path(source_path)
        self.cache_path = Path(cache_path)
        self.output_dir = Path(output_dir)
        self.manifest_dir = Path(manifest_dir)
        self.seed = seed
        self.historical_paths = tuple(Path(path) for path in historical_paths)

    def build(self, *, strict: bool = True) -> dict[str, Any]:
        return build_prompt_pool_v1(
            source_path=self.source_path, cache_path=self.cache_path,
            output_dir=self.output_dir, manifest_dir=self.manifest_dir,
            seed=self.seed, historical_paths=self.historical_paths, strict=strict,
        )
