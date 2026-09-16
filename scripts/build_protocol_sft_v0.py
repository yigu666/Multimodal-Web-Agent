#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from multimodal_web_agent.data.protocol_sft.builder import (  # noqa: E402
    BuildConfig,
    DatasetBuildError,
    build_dataset,
    protocol_artifact_stem,
    resolve_project_path,
    write_build_result,
    write_rejections,
)


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read the server build config") from exc
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("config root must be a mapping")
    return value


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build versioned Protocol-SFT data from FVQA Train and its image cache")
    parser.add_argument("--project-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--config", default="configs/protocol_sft/data_v0_server.yaml")
    parser.add_argument("--source-parquet")
    parser.add_argument("--image-search-cache")
    parser.add_argument("--output-dir")
    parser.add_argument("--previous-dataset-dir")
    parser.add_argument("--previous-audit-path")
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    project_root = args.project_root.resolve()
    config_path = resolve_project_path(project_root, args.config)
    raw = _load_yaml(config_path)
    dataset = raw.get("dataset", {})
    targets = raw.get("targets", {})
    split = raw.get("split", {})
    protocol = raw.get("protocol", {})
    retrieval = raw.get("retrieval", {})
    tokenization = raw.get("tokenization", {})
    query = raw.get("query", {})
    information = raw.get("information", {})
    reason_templates = raw.get("reason_templates", {})
    manual_audit = raw.get("manual_audit", {})
    image_text_route = raw.get("image_text_route", {})
    reuse = raw.get("reuse", {})
    selection = raw.get("selection", {})

    source_value = args.source_parquet or dataset["source_parquet"]
    cache_value = args.image_search_cache or dataset["image_search_cache"]
    output_value = args.output_dir or dataset["output_dir"]
    source_path = resolve_project_path(project_root, source_value)
    cache_path = resolve_project_path(project_root, cache_value)
    output_dir = resolve_project_path(project_root, output_value)
    schema_version = str(dataset.get("schema_version", "protocol-sft-v0"))
    previous_value = args.previous_dataset_dir or dataset.get("previous_dataset_dir")
    previous_dataset_dir = (
        resolve_project_path(project_root, previous_value)
        if previous_value
        else None
    )
    previous_audit_value = (
        args.previous_audit_path or dataset.get("previous_audit_path")
    )
    previous_audit_path = (
        resolve_project_path(project_root, previous_audit_value)
        if previous_audit_value
        else None
    )
    if (
        schema_version in {"protocol-sft-v0.2", "protocol-sft-v0.3"}
        and previous_dataset_dir is not None
        and output_dir.resolve() == previous_dataset_dir.resolve()
    ):
        raise ValueError("output_dir must not overwrite the previous dataset")

    route_names = (
        "direct_answer",
        "image_search_answer",
        "text_search_answer",
        "image_text_search_answer",
    )
    route_split_quotas = {
        route: {
            "train": int(split[route]["train"]),
            "dev": int(split[route]["dev"]),
            "test": int(split[route]["test"]),
        }
        for route in route_names
        if isinstance(split.get(route), dict)
    }

    build_config = BuildConfig(
        seed=int(dataset.get("seed", 20260722)),
        direct_trajectories=int(
            targets.get(
                "direct_answer_trajectories",
                targets.get("direct_trajectories", 200),
            )
        ),
        image_search_trajectories=int(
            targets.get(
                "image_search_answer_trajectories",
                targets.get("image_search_trajectories", 200),
            )
        ),
        text_search_trajectories=int(
            targets.get(
                "text_search_answer_trajectories",
                targets.get("text_search_trajectories", 200),
            )
        ),
        image_text_search_trajectories=int(
            targets.get("image_text_search_answer_trajectories", 0)
        ),
        logical_trajectories=(
            int(targets["logical_trajectories"])
            if "logical_trajectories" in targets
            else None
        ),
        state_action_examples=int(targets.get("state_action_examples", 1000)),
        split_targets={
            "train": int(split.get("train_examples", split.get("train_state_actions", 800))),
            "dev": int(split.get("dev_examples", split.get("dev_state_actions", 100))),
            "test": int(split.get("test_examples", split.get("test_state_actions", 100))),
        },
        reason_max_tokens=int(protocol.get("reason_max_tokens", 48)),
        text_query_min_tokens=int(protocol.get("text_query_min_tokens", 3)),
        text_query_max_tokens=int(protocol.get("text_query_max_tokens", 32)),
        text_top_k=int(retrieval.get("text_top_k", 3)),
        image_top_k=int(retrieval.get("image_top_k", 3)),
        image_text_image_top_k=int(
            image_text_route.get(
                "image_top_k", retrieval.get("image_top_k", 3)
            )
        ),
        tokenization_model_path=tokenization.get("model_path"),
        max_seq_len=int(tokenization.get("max_seq_len", 1536)),
        visual_token_target=int(tokenization.get("visual_token_target", 256)),
        target_reserve=int(tokenization.get("target_reserve", 96)),
        schema_version=schema_version,
        allow_shortfall=bool(
            targets.get(
                "allow_shortfall",
                schema_version in {"protocol-sft-v0.2", "protocol-sft-v0.3"},
            )
        ),
        previous_dataset_dir=(
            str(previous_value).replace("\\", "/") if previous_value else None
        ),
        previous_audit_path=(
            str(previous_audit_value).replace("\\", "/")
            if previous_audit_value
            else None
        ),
        require_previous_dataset=bool(
            reuse.get("require_previous_dataset", True)
        ),
        allow_previous_global_shortfall_failure=bool(
            reuse.get("allow_previous_global_shortfall_failure", True)
        ),
        reject_other_previous_quality_failures=bool(
            reuse.get("reject_other_previous_quality_failures", True)
        ),
        required_previous_trajectories=int(
            reuse.get("require_previous_trajectories", 480)
        ),
        required_previous_state_actions=int(
            reuse.get("require_previous_state_actions", 772)
        ),
        expected_previous_route_counts={
            str(route): int(count)
            for route, count in reuse.get(
                "expected_previous_routes",
                {
                    "direct_answer": 202,
                    "image_search_answer": 200,
                    "text_search_answer": 64,
                    "image_text_search_answer": 14,
                },
            ).items()
        },
        new_direct_trajectories=int(
            targets.get("new_direct_answer_trajectories", 0)
        ),
        new_image_search_trajectories=int(
            targets.get("new_image_search_answer_trajectories", 0)
        ),
        route_split_quotas=route_split_quotas,
        direct_category=str(selection.get("direct_category", "search_free")),
        image_category=str(selection.get("image_category", "search_required")),
        require_image_cache_hit=bool(
            selection.get("require_image_cache_hit", True)
        ),
        require_answer_in_image_information=bool(
            selection.get("require_answer_in_image_information", True)
        ),
        deterministic_evidence_ranking=bool(
            selection.get("deterministic_evidence_ranking", True)
        ),
        visible_context_only=bool(query.get("visible_context_only", True)),
        allow_cache_title_input=bool(query.get("allow_cache_title_input", False)),
        unavailable_context_min_ngram=int(
            query.get("unavailable_context_min_ngram", 4)
        ),
        unavailable_context_overlap_threshold=float(
            query.get("unavailable_context_overlap_threshold", 0.5)
        ),
        strip_visual_placeholders=bool(
            information.get("strip_visual_placeholders", True)
        ),
        min_unique_reasons_per_transition=int(
            reason_templates.get("min_unique_per_transition", 1)
        ),
        manual_audit_direct_count=int(manual_audit.get("direct_count", 0)),
        manual_audit_image_count=int(manual_audit.get("image_count", 0)),
        manual_audit_text_count=int(manual_audit.get("text_count", 0)),
        manual_audit_image_text_count=int(
            manual_audit.get("image_text_count", 0)
        ),
        manual_audit_new_direct_count=int(
            manual_audit.get("new_direct_count", 0)
        ),
        manual_audit_new_image_count=int(
            manual_audit.get("new_image_count", 0)
        ),
        manual_audit_reused_image_count=int(
            manual_audit.get("reused_image_count", 0)
        ),
        manual_audit_seed=int(manual_audit.get("seed", 20260722)),
        relation_required=bool(image_text_route.get("relation_required", True)),
        entity_must_be_visible=bool(
            image_text_route.get("entity_must_be_visible", True)
        ),
        answer_must_not_be_in_image_information=bool(
            image_text_route.get(
                "answer_must_not_be_in_image_information", True
            )
        ),
        answer_must_be_in_text_information=bool(
            image_text_route.get("answer_must_be_in_text_information", True)
        ),
        exclude_image_context_documents_from_text_evidence=bool(
            image_text_route.get(
                "exclude_image_context_documents_from_text_evidence", True
            )
        ),
    )
    try:
        result = build_dataset(
            source_path,
            cache_path,
            build_config,
            source_label=str(source_value).replace("\\", "/"),
            cache_label=str(cache_value).replace("\\", "/"),
            previous_dataset_dir=previous_dataset_dir,
            previous_audit_path=previous_audit_path,
        )
    except DatasetBuildError as exc:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_rejections(exc.rejected, output_dir)
        print("ERROR: %s" % exc, file=sys.stderr)
        print("Rejected attempts were written to %s" % (output_dir / "rejected.jsonl"), file=sys.stderr)
        return 2

    write_build_result(result, output_dir)
    manifest_dir = project_root / "data" / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(
        output_dir / "manifest.json",
        manifest_dir / (
            protocol_artifact_stem(build_config.schema_version) + "_manifest.json"
        ),
    )
    print(
        json.dumps(
            {
                "counts": result.manifest["counts"],
                "shortfall": result.manifest["shortfall"],
                "rejection_reason_distribution": result.manifest[
                    "rejection_reason_distribution"
                ],
                "rejection_reason_distribution_by_route": result.manifest[
                    "rejection_reason_distribution_by_route"
                ],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    print("Output: %s" % output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
