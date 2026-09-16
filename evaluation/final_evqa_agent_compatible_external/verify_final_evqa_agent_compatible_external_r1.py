#!/usr/bin/env python3
"""Pure-Python contract verifier for FINAL-EVQA-AGENT-COMPATIBLE-EXTERNAL-R1."""
from __future__ import annotations
import hashlib, json, os, sys
from pathlib import Path

ROOT = Path(os.environ.get("MWA_ROOT", Path(__file__).resolve().parents[2])).resolve()
DATA = ROOT / "data/external_benchmarks/encyclopedic_vqa"
OUT = ROOT / "outputs/final_evqa_agent_compatible_external_r1"
EXPECTED_TEST = "dbf3cf7336b7904cb0f996d2cea1762f0ae5186cd42f0c9f6a74c1c16d1d9bb5"
EXPECTED_LENS = "348c7043c51184e327337538e889c26832081c6dc16f0f349d903f884793dd68"

def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1024*1024), b""): h.update(b)
    return h.hexdigest()

def rows(p: Path):
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]

def main() -> int:
    errors = []
    contract_p = OUT / "contracts/final_contract.json"
    manifest_p = DATA / "processed/agent_compatible_r1/final_manifest.jsonl"
    if not contract_p.exists(): errors.append("missing final_contract.json")
    if not manifest_p.exists(): errors.append("missing final_manifest.jsonl")
    if not (DATA / "raw/test.csv").exists() or sha(DATA / "raw/test.csv") != EXPECTED_TEST: errors.append("official test.csv hash mismatch")
    if not (DATA / "raw/lens_entities.csv").exists() or sha(DATA / "raw/lens_entities.csv") != EXPECTED_LENS: errors.append("official lens_entities.csv hash mismatch")
    if manifest_p.exists():
        mr = rows(manifest_p); ids = [r.get("sample_id") for r in mr]
        if len(mr) < 160: errors.append("final N below 160")
        if len(ids) != len(set(ids)): errors.append("duplicate sample_id")
        for r in mr:
            for k in ("sample_id","dataset_name","dataset_image_id","image_path","image_sha256","question","answer_refs","wikipedia_url_hidden","lens_wiki_urls","visual_anchored"): 
                if k not in r: errors.append("manifest missing " + k); break
            p = Path(str(r.get("image_path", "")))
            if not p.exists(): errors.append("missing image " + str(p)); continue
            if sha(p) != str(r.get("image_sha256")): errors.append("image hash mismatch " + str(r.get("sample_id")))
    if contract_p.exists():
        c = json.loads(contract_p.read_text(encoding="utf-8"))
        required = {"FINAL_EVQA_AGENT_COMPATIBLE_EXTERNAL_R1_COMPLETE", "OFFICIAL_EVQA_SOURCE_PINNED", "FINAL_MANIFEST_SHA256", "FRESH_LENS_CALLS", "SERPAPI_LENS_CALLS", "VISUAL_SEARCH_PROVIDER", "AUTO_CONTINUE"}
        for k in required:
            if k not in c: errors.append("contract missing " + k)
        if c.get("FRESH_LENS_CALLS") != 0 or c.get("SERPAPI_LENS_CALLS") != 0: errors.append("fresh Lens calls nonzero")
        if c.get("VISUAL_SEARCH_PROVIDER") != "EVQA_OFFICIAL_FROZEN_GOOGLE_LENS_REPLAY": errors.append("visual provider mismatch")
        if c.get("AUTO_CONTINUE") is not False: errors.append("AUTO_CONTINUE must be false")
        if manifest_p.exists() and c.get("FINAL_MANIFEST_SHA256") != sha(manifest_p): errors.append("manifest contract hash mismatch")
    for model in ("protocol_sft", "reward_v21"):
        for cond, sub in (("notool", "p8_notool"), ("agent", "p9_agent")):
            p = OUT / sub / model / "episodes.jsonl"
            if not p.exists(): errors.append(f"missing {sub}/{model}/episodes.jsonl")
            elif manifest_p.exists() and len(rows(p)) != len(rows(manifest_p)): errors.append(f"incomplete {sub}/{model}")
    if errors:
        print("FINAL_EVQA_AGENT_COMPATIBLE_EXTERNAL_R1_VERIFY_FAIL")
        for e in errors: print("- " + e)
        return 1
    print("FINAL_EVQA_AGENT_COMPATIBLE_EXTERNAL_R1_VERIFY_PASS")
    return 0

if __name__ == "__main__": sys.exit(main())
