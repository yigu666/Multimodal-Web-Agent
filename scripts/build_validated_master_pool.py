#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from multimodal_web_agent.data.protocol_sft.builder import resolve_project_path
from multimodal_web_agent.data.quality.pool_manifest import sha256_file
from multimodal_web_agent.data.quality.pool_builder import (
    MasterPoolBuildConfig,
    build_validated_master_pool,
    write_master_pool,
)


def _yaml(path: Path) -> dict[str, Any]:
    import yaml
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("config root must be a mapping")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument(
        "--config",
        default="configs/data_quality/validated_master_pool_v0_1_server.yaml",
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--manifest-output")
    parser.add_argument("--no-manifest-copy", action="store_true")
    args = parser.parse_args()
    root = args.project_root.resolve()
    raw = _yaml(resolve_project_path(root, args.config))
    expected_schema = str(raw["schema_version"])
    gate_raw = raw["gate"]
    gate_schema = str(gate_raw["schema_version"])
    gate_config = resolve_project_path(root, gate_raw["config_file"])
    gate_config_raw = _yaml(gate_config)
    if gate_config_raw.get("schema_version") != gate_schema:
        raise ValueError("configured Gate schema does not match Gate config")
    source_key = "parquet_file" if "parquet_file" in raw["source"] else "parquet"
    source = resolve_project_path(root, raw["source"][source_key])
    cache = resolve_project_path(root, raw["source"]["image_search_cache"])
    output = resolve_project_path(
        root, args.output_dir or raw["output"]["directory"]
    )
    if output.exists():
        raise FileExistsError("refusing to overwrite output: %s" % output)
    previous = resolve_project_path(
        root, raw["source"].get("previous_trajectory_path", "")
    )
    policy = raw.get("policy", {})
    gate = raw.get("gate", {})
    config = MasterPoolBuildConfig(
        source_label=str(raw["source"][source_key]).replace("\\", "/"),
        cache_label=str(raw["source"]["image_search_cache"]).replace("\\", "/"),
        seed=int(raw.get("seed", 20260727)),
        image_top_k=int(raw.get("image_search", {}).get("top_k", 3)),
        previous_trajectory_path=str(previous),
        maximum_question_copy_ratio_without_entity=float(
            gate.get("maximum_question_copy_ratio_without_entity", 0.85)
        ),
        require_evidence_reachability=bool(
            gate.get("require_evidence_reachability", True)
        ),
        repair_mode=str(policy.get("repair_mode", "reject_only")),
        schema_version=expected_schema,
        gate_schema_version=gate_schema,
        gate_config_sha256=sha256_file(gate_config),
        automatic_route_conversion=bool(
            policy.get("automatic_route_conversion", False)
        ),
        automatic_query_rewrite=bool(
            policy.get("automatic_query_rewrite", False)
        ),
        protocol_sft_schema=raw.get("chain", {}).get(
            "protocol_sft_schema"
        ),
    )
    result = build_validated_master_pool(
        source_path=source,
        cache_path=cache,
        config=config,
    )
    write_master_pool(result, output)
    manifest_target = args.manifest_output or raw["output"].get(
        "manifest_file", raw["output"].get("manifest")
    )
    if manifest_target and not args.no_manifest_copy:
        manifest_copy = resolve_project_path(root, manifest_target)
        manifest_copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(output / "manifest.json", manifest_copy)
    print(json.dumps(result.manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result.manifest["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
