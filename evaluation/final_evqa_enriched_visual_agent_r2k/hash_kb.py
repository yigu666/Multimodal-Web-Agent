#!/usr/bin/env python3
from pathlib import Path
import os
import subprocess

ROOT = Path(os.environ.get("MWA_ROOT", Path(__file__).resolve().parents[2])).resolve()
JSON = ROOT / "data/external_benchmarks/encyclopedic_vqa/knowledge_base/encyclopedic_kb_wiki.json"
OUT = ROOT / "outputs/final_evqa_enriched_visual_agent_r2k/p2_kb_integrity"

def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    log = (OUT / "json.sha256.log").open("ab", buffering=0)
    p = subprocess.Popen(["sha256sum", str(JSON)], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    (OUT / "hash.pid").write_text(str(p.pid) + "\n", encoding="utf-8")
    print("hash_started", p.pid)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
