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
from multimodal_web_agent.data.protocol_sft.v0_4_manifest import (
    build_protocol_sft_v0_4,
    write_protocol_sft_v0_4,
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
        "--config", default="configs/protocol_sft/data_v0_4_server.yaml"
    )
    args = parser.parse_args()
    root = args.project_root.resolve()
    raw = _yaml(resolve_project_path(root, args.config))
    pool = resolve_project_path(root, raw["source"]["master_pool_dir"])
    historical = resolve_project_path(
        root, raw["source"]["historical_test_path"]
    )
    output = resolve_project_path(root, raw["output"]["directory"])
    if output.resolve() == resolve_project_path(
        root, "data/processed/protocol_sft_v0_3"
    ).resolve():
        raise ValueError("v0.4 output must not overwrite v0.3")
    result = build_protocol_sft_v0_4(
        master_pool_dir=pool,
        historical_test_path=historical,
        seed=int(raw.get("seed", 20260727)),
        route_targets=raw.get("route_targets", {}),
    )
    write_protocol_sft_v0_4(result, output)
    manifest_copy = resolve_project_path(root, raw["output"]["manifest"])
    manifest_copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(output / "manifest.json", manifest_copy)
    print(json.dumps(result.manifest, ensure_ascii=False, indent=2, sort_keys=True))
    print("Test content is embargoed; no Test preview was generated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
