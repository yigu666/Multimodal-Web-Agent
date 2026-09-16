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

from multimodal_web_agent.data.protocol_format_sft import (
    ROUTE_TARGETS,
    SCHEMA,
    build_protocol_format_sft_v1,
    write_protocol_format_sft_v1,
)
from multimodal_web_agent.data.protocol_sft.builder import resolve_project_path


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
        "--config", default="configs/protocol_sft/data_format_v1.yaml"
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--no-manifest-copy", action="store_true")
    args = parser.parse_args()
    root = args.project_root.resolve()
    raw = _yaml(resolve_project_path(root, args.config))
    if raw.get("schema_version") != SCHEMA:
        raise ValueError("format view schema mismatch")
    if raw.get("selection") != ROUTE_TARGETS:
        raise ValueError("format view route/split plan is not frozen")
    history = raw["historical_provenance"]
    paths = lambda key: [
        resolve_project_path(root, value) for value in history.get(key, ())
    ]
    protocol_v0_5_dir = resolve_project_path(
        root, raw["sources"]["protocol_v0_5_dir"]
    )
    master_pool_dir = resolve_project_path(
        root, raw["sources"]["master_pool_dir"]
    )
    required = [
        protocol_v0_5_dir,
        master_pool_dir,
        protocol_v0_5_dir / "train.jsonl",
        protocol_v0_5_dir / "dev.jsonl",
        *paths("train"),
        *paths("dev"),
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "missing format view input: %s"
            % ", ".join(str(path) for path in missing)
        )
    result = build_protocol_format_sft_v1(
        protocol_v0_5_dir=protocol_v0_5_dir,
        master_pool_dir=master_pool_dir,
        historical_train_paths=paths("train"),
        historical_dev_paths=paths("dev"),
        seed=int(raw.get("seed", 20260728)),
    )
    output = resolve_project_path(
        root, args.output_dir or raw["output"]["directory"]
    )
    write_protocol_format_sft_v1(result, output)
    if not args.no_manifest_copy:
        destination = resolve_project_path(
            root, raw["output"]["manifest_file"]
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(output / "manifest.json", destination)
    print(json.dumps(result.manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
