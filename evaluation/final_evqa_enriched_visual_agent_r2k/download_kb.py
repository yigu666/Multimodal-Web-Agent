#!/usr/bin/env python3
"""Start/resume the official E-VQA controlled-KB download on Titan."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(os.environ.get("MWA_ROOT", Path(__file__).resolve().parents[2])).resolve()
KB = ROOT / "data/external_benchmarks/encyclopedic_vqa/knowledge_base"
TMP = KB / "download_tmp"
OUT = ROOT / "outputs/final_evqa_enriched_visual_agent_r2k/p1_kb_download"
URL = "https://storage.googleapis.com/encyclopedic-vqa/encyclopedic_kb_wiki.zip"
DEST = TMP / "encyclopedic_kb_wiki.zip"
LOG = OUT / "download.log"


def main() -> int:
    TMP.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    if DEST.exists() and DEST.stat().st_size >= 5_295_637_128:
        print(f"archive_already_complete size={DEST.stat().st_size}")
        return 0
    with LOG.open("ab", buffering=0) as log:
        cmd = [
            "curl", "-L", "-C", "-", "--retry", "8", "--retry-delay", "15",
            "--fail", "--output", str(DEST), URL,
        ]
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    (OUT / "download.pid").write_text(str(proc.pid) + "\n", encoding="utf-8")
    print(f"download_started pid={proc.pid} destination={DEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
