#!/usr/bin/env python3
"""Extract the already downloaded official KB without loading JSON into RAM."""
from __future__ import annotations
import subprocess
import os
from pathlib import Path

ROOT = Path(os.environ.get("MWA_ROOT", Path(__file__).resolve().parents[2])).resolve()
KB = ROOT / "data/external_benchmarks/encyclopedic_vqa/knowledge_base"
ARCHIVE = KB / "download_tmp/encyclopedic_kb_wiki.zip"
OUT = ROOT / "outputs/final_evqa_enriched_visual_agent_r2k/p1_kb_download"

def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    target = KB / "encyclopedic_kb_wiki.json"
    if target.exists() and target.stat().st_size > 16_000_000_000:
        print(f"json_already_extracted size={target.stat().st_size}")
        return 0
    if not ARCHIVE.exists() or ARCHIVE.stat().st_size != 5_295_637_128:
        print("archive_incomplete", ARCHIVE.stat().st_size if ARCHIVE.exists() else 0)
        return 2
    log = (OUT / "extract.log").open("ab", buffering=0)
    proc = subprocess.Popen(["unzip", "-o", "-q", str(ARCHIVE), "-d", str(KB)], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    (OUT / "extract.pid").write_text(str(proc.pid) + "\n", encoding="utf-8")
    print(f"extract_started pid={proc.pid} target={target}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
