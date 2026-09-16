#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from multimodal_web_agent.data.protocol_sft.audit import (  # noqa: E402
    audit_data_dir,
    write_audit_markdown,
    write_sha256_manifest,
)
from multimodal_web_agent.data.protocol_sft.builder import (  # noqa: E402
    protocol_artifact_stem,
    resolve_project_path,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit an already-built versioned Protocol-SFT dataset")
    parser.add_argument("--project-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--data-dir", default="data/processed/protocol_sft_v0")
    parser.add_argument("--output", default="data/manifests/protocol_sft_v0_audit.json")
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    data_dir = resolve_project_path(project_root, args.data_dir)
    output = resolve_project_path(project_root, args.output)
    report = audit_data_dir(data_dir)
    stem = protocol_artifact_stem(str(report["schema_version"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path = data_dir / "audit_report.md"
    write_audit_markdown(report, markdown_path)

    hash_inputs = [
        data_dir / name
        for name in (
            "trajectories.jsonl",
            "train.jsonl",
            "dev.jsonl",
            "test.jsonl",
            "rejected.jsonl",
            "manifest.json",
            "sample_preview.md",
            "manual_route_audit.jsonl",
            "manual_route_audit.md",
            "audit_report.md",
        )
    ]
    hash_inputs.append(output)
    hash_inputs.append(
        project_root / "data" / "manifests" / (stem + "_manifest.json")
    )
    hash_inputs = [path for path in hash_inputs if path.is_file()]
    write_sha256_manifest(
        hash_inputs,
        root=project_root,
        output=project_root / "data" / "manifests" / (stem + "_files.sha256"),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
