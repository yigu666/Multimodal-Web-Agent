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
from multimodal_web_agent.data.protocol_sft.v0_5_manifest import (
    GATE_SCHEMA,
    POOL_SCHEMA,
    PROTOCOL_SCHEMA,
    build_protocol_sft_v0_5,
    write_protocol_sft_v0_5,
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
        "--config", default="configs/protocol_sft/data_v0_5_server.yaml"
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--manifest-output")
    parser.add_argument("--no-manifest-copy", action="store_true")
    args = parser.parse_args()
    root = args.project_root.resolve()
    raw = _yaml(resolve_project_path(root, args.config))
    if raw.get("schema_version") != PROTOCOL_SCHEMA:
        raise ValueError("Protocol-SFT schema mismatch")
    source = raw["source_pool"]
    if source.get("schema_version") != POOL_SCHEMA:
        raise ValueError("source pool schema mismatch")
    if source.get("gate_schema_version") != GATE_SCHEMA:
        raise ValueError("Gate schema mismatch")
    output = resolve_project_path(
        root, args.output_dir or raw["output"]["directory"]
    )
    historical = {
        split: [
            resolve_project_path(root, path)
            for path in raw.get("historical_exposure", {}).get(split, ())
        ]
        for split in ("train", "dev", "test")
    }
    reserved = [
        resolve_project_path(root, path)
        for path in raw.get("historical_reserved", {}).get("test", ())
    ]
    historical_manifests = [
        resolve_project_path(root, path)
        for path in raw.get("historical_manifests", ())
    ]
    missing_history = [
        path
        for path in (
            *[item for values in historical.values() for item in values],
            *reserved,
            *historical_manifests,
        )
        if not path.is_file()
    ]
    if missing_history:
        raise FileNotFoundError(
            "missing historical exposure inputs: %s"
            % ", ".join(str(path) for path in missing_history)
        )
    result = build_protocol_sft_v0_5(
        master_pool_dir=resolve_project_path(root, source["directory"]),
        master_pool_audit_path=resolve_project_path(
            root, source["audit_file"]
        ),
        historical_paths=historical,
        reserved_paths=reserved,
        historical_manifest_paths=historical_manifests,
        seed=int(raw.get("seed", 20260727)),
        route_targets=raw.get("route_targets", {}),
    )
    write_protocol_sft_v0_5(result, output)
    target = args.manifest_output or raw["output"].get("manifest_file")
    if target and not args.no_manifest_copy:
        manifest_output = resolve_project_path(root, target)
        manifest_output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(output / "manifest.json", manifest_output)
    print(json.dumps(result.manifest, ensure_ascii=False, indent=2, sort_keys=True))
    print("Test content is embargoed; no Test preview was generated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
