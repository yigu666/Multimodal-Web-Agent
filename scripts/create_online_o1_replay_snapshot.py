#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cache_rows(cache_root: Path) -> list[dict]:
    rows = []
    for path in sorted(cache_root.rglob("*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        payload = value.get("payload", {}) if isinstance(value, dict) else {}
        metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
        rows.append({
            "path": path.relative_to(cache_root).as_posix(),
            "namespace": path.parent.relative_to(cache_root).as_posix(),
            "sha256": _sha(path),
            "bytes": path.stat().st_size,
            "mtime_utc": datetime.fromtimestamp(
                path.stat().st_mtime, timezone.utc
            ).isoformat(),
            "provider": payload.get("backend") or metadata.get("reader_provider") or metadata.get("provider"),
            "backend_version": metadata.get("backend_version") or metadata.get("reader_version") or metadata.get("reader_pipeline_version"),
            "tool_type": payload.get("tool_type"),
        })
    return rows


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--stage", choices=("before", "after"), required=True)
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("outputs/online_web_agent_v1_o1_100"),
    )
    args = parser.parse_args()
    root = args.project_root.resolve()
    output = args.output_root if args.output_root.is_absolute() else root / args.output_root
    cache_root = root / "outputs/online_web_cache"
    snapshot_root = output / "replay_snapshot"
    rows = _cache_rows(cache_root)
    if args.stage == "before":
        target = snapshot_root / "pre_live_cache_manifest.json"
        if target.exists():
            raise FileExistsError(target)
        _write(target, {
            "schema_version": "online-web-agent-o1-cache-baseline-v1",
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "cache_root": str(cache_root),
            "file_count": len(rows),
            "total_bytes": sum(row["bytes"] for row in rows),
            "files": rows,
            "network_accessed": False,
        })
        print("ONLINE_O1_CACHE_BASELINE_COMPLETE")
        return 0

    baseline_path = snapshot_root / "pre_live_cache_manifest.json"
    if not baseline_path.is_file():
        raise FileNotFoundError(baseline_path)
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    before = {row["path"]: row["sha256"] for row in baseline["files"]}
    after = {row["path"]: row["sha256"] for row in rows}
    new_paths = sorted(path for path in after if path not in before)
    changed_paths = sorted(path for path in after if path in before and after[path] != before[path])

    timestamps = []
    provenance_root = output / "live_provenance"
    provenance_files = sorted(provenance_root.rglob("*.json")) if provenance_root.is_dir() else []
    for path in provenance_files:
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("timestamp"):
            timestamps.append(str(value["timestamp"]))
    manifest = {
        "schema_version": "online-web-agent-o1-replay-snapshot-v1",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "cache_root": str(cache_root),
        "cache_file_count": len(rows),
        "cache_total_bytes": sum(row["bytes"] for row in rows),
        "baseline_cache_file_count": len(before),
        "new_cache_file_count": len(new_paths),
        "changed_cache_file_count": len(changed_paths),
        "new_cache_paths": new_paths,
        "changed_cache_paths": changed_paths,
        "provenance_file_count": len(provenance_files),
        "provider_timestamp_min": min(timestamps) if timestamps else None,
        "provider_timestamp_max": max(timestamps) if timestamps else None,
        "providers": ["serper", "serpapi_google_lens", "local", "jina"],
        "visual_auto_crop": False,
        "files": rows,
        "network_accessed": False,
    }
    _write(snapshot_root / "manifest.json", manifest)
    print("ONLINE_O1_REPLAY_SNAPSHOT_COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
