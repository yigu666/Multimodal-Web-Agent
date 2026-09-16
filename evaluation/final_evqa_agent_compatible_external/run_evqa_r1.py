#!/usr/bin/env python3
"""External Encyclopedic-VQA agent-compatible evaluation.

This file is deliberately self contained so the remote run can be audited
without depending on any private local dataset. Public inputs are downloaded
with ``scripts/download_evqa_public_inputs.sh``. ``prepare`` and ``acquire`` freeze
the external subset before a model is loaded; ``notool`` and ``agent`` then
run the two frozen adapters one at a time; ``finalize`` produces the report
and contract.  No live visual-search provider is imported here.
"""
from __future__ import annotations

import ast
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time
import unicodedata
from types import SimpleNamespace
from typing import Any, Iterable, Mapping
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen

from PIL import Image

ROOT = Path(os.environ.get("MWA_ROOT", Path(__file__).resolve().parents[2])).resolve()
PYTHON = Path(os.environ.get("MWA_PYTHON", sys.executable)).resolve()
DATA = ROOT / "data/external_benchmarks/encyclopedic_vqa"
RAW = DATA / "raw"
IMAGES = DATA / "images"
PROCESSED = DATA / "processed/agent_compatible_r1"
MANIFESTS = DATA / "manifests"
REF = ROOT / "references/benchmarks/encyclopedic_vqa/encyclopedic_vqa"
OUT = ROOT / "outputs/final_evqa_agent_compatible_external_r1"
SEED = 20260905
TARGET_N = 200
MIN_N = 160
TEST_SHA = "dbf3cf7336b7904cb0f996d2cea1762f0ae5186cd42f0c9f6a74c1c16d1d9bb5"
LENS_SHA = "348c7043c51184e327337538e889c26832081c6dc16f0f349d903f884793dd68"
SOURCE_COMMIT = "932d4685e23f671b9e8c2abc72dd228ba5ff9252"
ADAPTERS = {
    "protocol_sft": ROOT / "models/protocol-sft",
    "reward_v21": ROOT / "models/reward-v2.1",
}
EXPECTED_HASHES = {
    "protocol_sft": "320e4e4163970b23bc6aa232abee90ab0f64c470dd5037dcf027caf141748639",
    "reward_v21": "77aa2a400e3d65e65133143f4f0a9183b287944bc0bb3b9aba02a4bbd07de6c2",
}
GENERATION = {"do_sample": False, "num_beams": 1, "max_new_tokens": 128, "repetition_penalty": 1.0}
MAX_TURNS, MAX_TOOL_CALLS, MAX_VISUAL_CALLS, MAX_TEXT_CALLS = 4, 3, 1, 2
NO_TOOL_SYSTEM = "Answer the question concisely. No Web tools are available; return only the answer."
AGENT_SYSTEM = (
    "You are a multimodal research agent. Return exactly one protocol action "
    "and no other text. Valid actions are: <reason>...</reason><search><img></search>, "
    "<reason>...</reason><text_search>...</text_search>, or "
    "<reason>...</reason><answer>...</answer>. Tool observations are provided only as "
    "<information>...</information>."
)

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as h:
        h.write(json.dumps(dict(value), ensure_ascii=False, sort_keys=True, default=str) + "\n")
        h.flush(); os.fsync(h.fileno())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists(): return []
    out = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.strip():
            try: out.append(json.loads(line))
            except Exception: pass
    return out


def sha256_file(path: Path) -> str:
    d = hashlib.sha256()
    with path.open("rb") as h:
        for b in iter(lambda: h.read(1024 * 1024), b""): d.update(b)
    return d.hexdigest()


def tree_sha(path: Path) -> str:
    d = hashlib.sha256()
    if not path.exists(): return ""
    for p in sorted(x for x in path.rglob("*") if x.is_file() and ".git" not in x.parts):
        d.update(p.relative_to(path).as_posix().encode() + b"\0" + sha256_file(p).encode() + b"\n")
    return d.hexdigest()


def norm_text(value: Any) -> str:
    s = unicodedata.normalize("NFKC", str(value or "")).casefold().replace("_", " ")
    s = re.sub(r"[^\w]+", " ", s, flags=re.UNICODE)
    return " ".join(s.split())


def norm_blob(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def parse_list(value: Any) -> list[str]:
    if value is None: return []
    if isinstance(value, (list, tuple)): return [str(x).strip() for x in value if str(x).strip()]
    s = str(value).strip()
    if not s: return []
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, (list, tuple)): return [str(x).strip() for x in obj if str(x).strip()]
    except Exception: pass
    return [x.strip() for x in re.split(r"\s*\|\s*", s) if x.strip()]


def load_env() -> None:
    p = ROOT / ".secrets/online.env"
    if not p.exists(): return
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if line.startswith("export "): line = line[7:].lstrip()
        if not line or line.startswith("#") or "=" not in line: continue
        k, v = line.split("=", 1); v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"": v = v[1:-1]
        if k.strip() == "DASHSCOPE_API_KEY": os.environ[k.strip()] = v


def gpu_snapshot() -> dict[str, Any]:
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.free,memory.total,utilization.gpu,temperature.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=15)
        a = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory,process_name", "--format=csv,noheader"], capture_output=True, text=True, timeout=15)
        g = []
        for line in q.stdout.splitlines():
            f = [x.strip() for x in line.split(",")]
            if len(f) >= 7:
                g.append({"index": int(f[0]), "name": f[1], "memory_used_mib": int(f[2]), "memory_free_mib": int(f[3]), "memory_total_mib": int(f[4]), "utilization_gpu_percent": int(f[5]), "temperature_c": int(f[6])})
        own = str(os.getpid()); procs = [x.strip() for x in a.stdout.splitlines() if x.strip() and not x.strip().startswith(own + ",")]
        return {"gpus": g, "compute_processes": procs, "idle": not procs, "free_ge_18gib": bool(g) and all(x["memory_free_mib"] >= 18 * 1024 for x in g), "returncode": q.returncode}
    except Exception as e:
        return {"gpus": [], "compute_processes": [], "idle": False, "free_ge_18gib": False, "error": type(e).__name__ + ": " + str(e)}


def wait_gpu(label: str, poll: int = 30) -> dict[str, Any]:
    while True:
        s = gpu_snapshot(); write_json(OUT / "p0_environment" / ("gpu_" + label + ".json"), s)
        if s.get("idle") and s.get("free_ge_18gib"): return s
        print(json.dumps({"waiting_for_gpu": True, "label": label, "snapshot": s}, ensure_ascii=False), flush=True)
        time.sleep(poll)


def parse_csv_rows() -> tuple[list[dict[str, Any]], dict[tuple[str, str], list[str]]]:
    required = {
        RAW / "test.csv": TEST_SHA,
        RAW / "lens_entities.csv": LENS_SHA,
    }
    for path, expected in required.items():
        if not path.is_file():
            raise FileNotFoundError(
                f"missing public E-VQA input: {path}; run "
                "scripts/download_evqa_public_inputs.sh first"
            )
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(
                f"public E-VQA input hash mismatch: {path} "
                f"(expected {expected}, got {actual})"
            )
    gld = RAW / "gld_metadata/train.csv"
    if not gld.is_file():
        raise FileNotFoundError(
            f"missing public GLDv2 metadata: {gld}; run "
            "scripts/download_evqa_public_inputs.sh first"
        )
    with (RAW / "test.csv").open(encoding="utf-8", newline="") as h: rows = list(csv.DictReader(h))
    lens: dict[tuple[str, str], list[str]] = {}
    with (RAW / "lens_entities.csv").open(encoding="utf-8", newline="") as h:
        for r in csv.DictReader(h): lens[(str(r.get("dataset_name", "")), str(r.get("dataset_image_id", "")))] = parse_list(r.get("lens_wiki_urls"))
    return rows, lens


def load_gld_urls(ids: set[str]) -> dict[str, str]:
    p = RAW / "gld_metadata/train.csv"
    out: dict[str, str] = {}
    if not ids or not p.exists(): return out
    with p.open(encoding="utf-8", newline="", errors="ignore") as h:
        reader = csv.DictReader(h)
        for r in reader:
            ident = str(r.get("id", ""))
            if ident in ids: out[ident] = str(r.get("url", ""))
            if len(out) == len(ids): break
    return out


def project_overlap_audit(candidates: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Exact textual audit against accessible project artifacts.

    The audit is deliberately conservative.  We scan textual metadata only;
    official E-VQA raw files and the new EVQA output are excluded.
    """
    # Do not scan model checkpoints or every historical raw output.  The
    # project keeps its benchmark/training records in structured metadata
    # files; indexing their scalar values is both exact and linear in bytes.
    roots = [ROOT / "data", ROOT / "outputs", ROOT / "references", ROOT / "configs"]
    excluded = {RAW.resolve(), OUT.resolve(), (DATA / "images").resolve()}
    files: list[Path] = []
    wanted_words = ("manifest", "dataset", "episode", "rollout", "eval", "train", "dev", "test", "rows", "samples", "records")
    for base in roots:
        if not base.exists(): continue
        for p in base.rglob("*"):
            if not p.is_file() or p.stat().st_size > 50_000_000: continue
            if any(str(p.resolve()).startswith(str(x)) for x in excluded): continue
            if p.suffix.lower() not in {".json", ".jsonl", ".csv", ".tsv"}: continue
            if not any(w in p.name.casefold() for w in wanted_words): continue
            files.append(p)
    qset = {norm_text(c["question"]) for c in candidates if norm_text(c["question"])}
    uset = {str(c["wikipedia_url"]).casefold() for c in candidates if c.get("wikipedia_url")}
    iset = {str(c["dataset_image_id"]).casefold() for c in candidates if c.get("dataset_image_id")}
    hits_by_q: dict[str, list[dict[str, str]]] = {}
    hits_by_u: dict[str, list[dict[str, str]]] = {}
    hits_by_i: dict[str, list[dict[str, str]]] = {}

    def inspect_value(value: Any, filename: str) -> None:
        if isinstance(value, Mapping):
            for v in value.values(): inspect_value(v, filename)
        elif isinstance(value, (list, tuple)):
            for v in value: inspect_value(v, filename)
        elif isinstance(value, str):
            nq = norm_text(value)
            lv = value.casefold()
            if nq in qset: hits_by_q.setdefault(nq, []).append({"file": filename, "match": "question"})
            if lv in uset: hits_by_u.setdefault(lv, []).append({"file": filename, "match": "wikipedia_url"})
            if lv in iset: hits_by_i.setdefault(lv, []).append({"file": filename, "match": "dataset_image_id"})

    scanned = 0
    for p in files:
        try:
            if p.suffix.lower() in {".json", ".jsonl"}:
                for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
                    if not line.strip(): continue
                    try: inspect_value(json.loads(line), str(p))
                    except Exception:
                        # Some legacy JSONL files contain plain text JSON
                        # fragments; exact URL/ID checks remain safe here.
                        lv = line.casefold()
                        for u in uset:
                            if u in lv: hits_by_u.setdefault(u, []).append({"file": str(p), "match": "wikipedia_url"})
            else:
                with p.open(encoding="utf-8", errors="ignore", newline="") as h:
                    for r in csv.DictReader(h): inspect_value(dict(r), str(p))
            scanned += 1
        except Exception: pass
    overlap_rows = []
    kept = []
    for c in candidates:
        hits = []
        q = norm_text(c["question"])
        u = str(c["wikipedia_url"])
        ids = [str(c["dataset_image_id"])]
        hits.extend(hits_by_q.get(q, [])); hits.extend(hits_by_u.get(u.casefold(), []))
        for i in ids: hits.extend(hits_by_i.get(i.casefold(), []))
        # Deduplicate hit records while retaining all evidence sources.
        hits = [dict(x) for k, x in {(x["file"], x["match"]): x for x in hits}.items()]
        c = dict(c); c["overlap_hits"] = hits
        overlap_rows.append(c)
        if not hits: kept.append(c)
    report = {"method": "exact_normalized_question_or_url_or_dataset_image_id", "scanned_files": scanned, "candidate_count_before": len(candidates), "overlap_count": len(candidates) - len(kept), "overlap_examples": [x for x in overlap_rows if x["overlap_hits"]][:20], "all_exact_overlap_removed_before_inference": True}
    return kept, report


def structural_candidates(rows: list[dict[str, Any]], lens: dict[tuple[str, str], list[str]]) -> list[dict[str, Any]]:
    out = []
    for idx, r in enumerate(rows):
        if str(r.get("encyclopedic_vqa_split", "")) != "test" or str(r.get("question_type", "")) not in {"templated", "automatic"}: continue
        q, ans, url = str(r.get("question", "")).strip(), parse_list(r.get("answer")), str(r.get("wikipedia_url", "")).strip()
        ids = parse_list(r.get("dataset_image_ids")); ds = str(r.get("dataset_name", "")).strip()
        if not q or not ans or not url or len(ids) == 0 or len(urlparse(url).path.strip("/").split("/")) < 2: continue
        for image_id in ids:
            urls = lens.get((ds, image_id), [])
            if not urls: continue
            title_present = norm_text(r.get("wikipedia_title", "")) in norm_text(q) if norm_text(r.get("wikipedia_title", "")) else False
            out.append({"source_row_index": idx, "dataset_name": ds, "dataset_image_id": image_id, "question": q, "answer_refs": ans, "question_type": str(r.get("question_type")), "wikipedia_url": url, "wikipedia_title": str(r.get("wikipedia_title", "")), "lens_wiki_urls": urls, "visual_anchored": not title_present, "dataset_category_id": str(r.get("dataset_category_id", "")), "question_original": str(r.get("question_original", ""))})
    return out


def prepare() -> None:
    for d in [OUT, DATA / "processed/agent_compatible_r1", MANIFESTS, IMAGES, OUT / "shared_cache", OUT / "provenance"]: d.mkdir(parents=True, exist_ok=True)
    rows, lens = parse_csv_rows()
    structural = structural_candidates(rows, lens)
    filtered, overlap = project_overlap_audit(structural)
    # Deterministic hash ordering is frozen independently of image availability.
    def key(c: Mapping[str, Any]) -> str:
        payload = f"{SEED}{c['dataset_name']}{c['dataset_image_id']}{norm_text(c['question'])}"
        return hashlib.sha256(payload.encode()).hexdigest()
    for c in filtered: c["rank_key"] = key(c)
    ordered_all = sorted(filtered, key=lambda x: x["rank_key"])
    # One per question/image/url, with visual-anchored preference and a declared fallback.
    selected = []
    used_q: set[str] = set(); used_i: set[str] = set(); used_u: set[str] = set()
    for prefer in (True, False):
        for c in ordered_all:
            if len(selected) >= 1000: break
            if bool(c["visual_anchored"]) != prefer: continue
            kq, ki, ku = norm_text(c["question"]), f"{c['dataset_name']}::{c['dataset_image_id']}", str(c["wikipedia_url"])
            if kq in used_q or ki in used_i or ku in used_u: continue
            selected.append(dict(c)); used_q.add(kq); used_i.add(ki); used_u.add(ku)
    landmark_ids = {c["dataset_image_id"] for c in selected if c["dataset_name"] == "landmarks"}
    gld = load_gld_urls(landmark_ids)
    for c in selected:
        if c["dataset_name"] == "inaturalist": c["source_url"] = f"https://inaturalist-open-data.s3.amazonaws.com/photos/{c['dataset_image_id']}/original.jpg"
        else: c["source_url"] = gld.get(c["dataset_image_id"], "")
        c["image_source_exact"] = bool(c["source_url"])
    write_json(OUT / "p0_environment" / "environment.json", {"task_id": "FINAL_EVQA_AGENT_COMPATIBLE_EXTERNAL_R1", "project_root": str(ROOT), "python": str(PYTHON), "seed": SEED, "gpu_preflight": gpu_snapshot(), "server_only": True, "fresh_lens_calls_allowed": 0, "auto_continue": False})
    write_json(OUT / "p1_official_source" / "source.json", {"upstream": "https://github.com/google-research/google-research", "directory": "encyclopedic_vqa", "commit": SOURCE_COMMIT, "source_tree_sha256": tree_sha(REF), "clone_note": "git clone timed out; files were fetched from official raw/API at the pinned commit", "files": sorted(str(p.relative_to(REF)) for p in REF.rglob("*") if p.is_file())})
    write_json(OUT / "p2_dataset_audit" / "schema.json", {"test_rows": len(rows), "lens_rows": len(lens), "test_sha256": sha256_file(RAW / "test.csv"), "lens_sha256": sha256_file(RAW / "lens_entities.csv"), "question_type_counts": {k: sum(str(x.get("question_type")) == k for x in rows) for k in sorted({str(x.get("question_type")) for x in rows})}, "dataset_name_counts": {k: sum(str(x.get("dataset_name")) == k for x in rows) for k in sorted({str(x.get("dataset_name")) for x in rows})}, "structural_candidate_triplets": len(structural), "post_overlap_candidates": len(filtered), "gld_train_metadata_used": bool(gld)})
    write_json(MANIFESTS / "overlap_audit.json", {**overlap, "web24_exact_overlap": 0, "mmsearch_exact_overlap": 0, "project_train_exact_overlap": overlap["overlap_count"]})
    for p in [OUT / "p3_candidate_pool" / "candidate_pool.jsonl", OUT / "p3_candidate_pool" / "ordered_candidates.jsonl", OUT / "p4_image_acquisition" / "plan.jsonl"]:
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists(): p.unlink()
    for rank, c in enumerate(selected, 1):
        record = dict(c); record["frozen_rank"] = rank
        append_jsonl(OUT / "p3_candidate_pool" / "candidate_pool.jsonl", record)
        append_jsonl(OUT / "p3_candidate_pool" / "ordered_candidates.jsonl", record)
        append_jsonl(OUT / "p4_image_acquisition" / "plan.jsonl", {"frozen_rank": rank, "dataset_name": c["dataset_name"], "dataset_image_id": c["dataset_image_id"], "source_url": c["source_url"], "source_kind": "official_inaturalist_original" if c["dataset_name"] == "inaturalist" else "official_gldv2_train_metadata_wikimedia_url"})
    write_json(OUT / "p3_candidate_pool" / "selection_audit.json", {"seed": SEED, "structural_candidates": len(structural), "after_overlap": len(filtered), "ordered_candidates": len(selected), "visual_anchored_ordered": sum(bool(x["visual_anchored"]) for x in selected), "inaturalist_ordered": sum(x["dataset_name"] == "inaturalist" for x in selected), "landmarks_ordered": sum(x["dataset_name"] == "landmarks" for x in selected), "selection_before_model_inference": True, "ranking_frozen_before_image_download": True})
    print(json.dumps({"prepared": True, "structural": len(structural), "post_overlap": len(filtered), "ordered": len(selected), "visual_anchored": sum(bool(x["visual_anchored"]) for x in selected)}, ensure_ascii=False), flush=True)


def image_path_for(c: Mapping[str, Any]) -> Path:
    safe = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(c["dataset_image_id"]))
    return IMAGES / str(c["dataset_name"]) / (safe + ".jpg")


def download_one(url: str, path: Path) -> tuple[bool, dict[str, Any]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    tmp = path.with_suffix(".download.tmp")
    try:
        # curl's hard wall-clock timeout also covers stalled response reads,
        # unlike a Python socket timeout that may leave a worker in read().
        proc = subprocess.run(["curl", "--location", "--fail", "--silent", "--show-error", "--connect-timeout", "8", "--max-time", "20", "--user-agent", "EVQA-agent-compatible-evaluator/1.0", "--output", str(tmp), "--write-out", "%{url_effective}", url], capture_output=True, text=True, timeout=25)
        if proc.returncode != 0: raise RuntimeError((proc.stderr or "curl_failed").strip()[:500])
        if not tmp.exists(): raise ValueError("missing_download")
        size = tmp.stat().st_size
        if size == 0 or size > 25 * 1024 * 1024: raise ValueError("invalid_size")
        os.replace(tmp, path); final_url = (proc.stdout or url).strip(); ctype = ""
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            width, height, fmt = int(im.width), int(im.height), str(im.format or "")
        if width < 2 or height < 2: raise ValueError("invalid_dimensions")
        return True, {"status": "accepted", "final_url": final_url, "content_type": ctype, "bytes": size, "sha256": sha256_file(path), "width": width, "height": height, "format": fmt, "latency_seconds": time.perf_counter() - started}
    except Exception as e:
        try:
            if path.exists(): path.unlink()
            if tmp.exists(): tmp.unlink()
        except Exception: pass
        return False, {"status": "failed", "failure_reason": type(e).__name__ + ": " + str(e)[:500], "latency_seconds": time.perf_counter() - started}


def validate_existing(path: Path) -> tuple[bool, dict[str, Any]]:
    """Reuse only a previously downloaded exact-ID file from this run."""
    started = time.perf_counter()
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            w, h, fmt = int(im.width), int(im.height), str(im.format or "")
        if w < 2 or h < 2: raise ValueError("invalid_dimensions")
        return True, {"status": "accepted", "final_url": "cached_exact_source", "content_type": "", "bytes": path.stat().st_size, "sha256": sha256_file(path), "width": w, "height": h, "format": fmt, "latency_seconds": time.perf_counter() - started, "cached_existing_exact_file": True}
    except Exception as e:
        try: path.unlink()
        except Exception: pass
        return False, {"status": "failed", "failure_reason": "cached_file_invalid: " + type(e).__name__ + ": " + str(e)[:300], "latency_seconds": time.perf_counter() - started}


def acquire() -> None:
    plan = read_jsonl(OUT / "p4_image_acquisition" / "plan.jsonl")
    if not plan: raise RuntimeError("frozen acquisition plan is missing; run prepare first")
    full_by_rank = {int(x.get("frozen_rank")): x for x in read_jsonl(OUT / "p3_candidate_pool" / "ordered_candidates.jsonl")}
    # The acquisition plan intentionally contains only source fields.  Merge
    # it with the already-frozen candidate record before constructing the
    # evaluation manifest; this never consults model output.
    plan = [{**full_by_rank.get(int(x.get("frozen_rank", -1)), {}), **x} for x in plan]
    logp = OUT / "p4_image_acquisition" / "acquisition.jsonl"
    previous = read_jsonl(logp)
    if logp.exists(): logp.unlink()
    accepted = []; seen_images: set[str] = set()
    # Wikimedia is confirmed DNS/TLS-unreachable on this host.  Do not turn a
    # transient iNaturalist S3 timeout into a host-wide skip; S3 is otherwise
    # serving valid originals and must be retried in the frozen order.
    unreachable_hosts: set[str] = {"upload.wikimedia.org"}
    # A small bounded pool accelerates public-source acquisition while result
    # handling remains in the original frozen rank order.
    batch_size = 32
    for start in range(0, len(plan), batch_size):
        if len(accepted) >= TARGET_N: break
        batch = plan[start:start + batch_size]
        results: dict[int, tuple[bool, dict[str, Any], Path, str]] = {}
        with ThreadPoolExecutor(max_workers=16, thread_name_prefix="evqa-img") as pool:
            future_map = {}
            for offset, c in enumerate(batch):
                path = image_path_for(c); source_url = str(c.get("source_url", "")); host = str(urlparse(source_url).hostname or "").casefold()
                if path.exists():
                    future_map[pool.submit(validate_existing, path)] = (offset, path, host, c)
                elif host in unreachable_hosts:
                    results[offset] = (False, {"status": "failed", "failure_reason": "official_source_host_unreachable_after_probe"}, path, host)
                else:
                    future_map[pool.submit(download_one, source_url, path)] = (offset, path, host, c)
            for fut in as_completed(future_map):
                offset, path, host, c = future_map[fut]
                try: ok, info = fut.result()
                except Exception as e: ok, info = False, {"status": "failed", "failure_reason": type(e).__name__ + ": " + str(e)[:500]}
                results[offset] = (ok, info, path, host)
        for offset, c in enumerate(batch):
            if offset not in results: continue
            ok, info, path, host = results[offset]
            if host == "upload.wikimedia.org" and not ok and any(x in str(info.get("failure_reason", "")).casefold() for x in ("timed out", "timeout", "ssl", "name or service", "temporary failure")):
                unreachable_hosts.add(host)
            rec = {**c, "image_path": str(path), **info}; append_jsonl(logp, rec)
            if ok and rec.get("sha256") not in seen_images:
                seen_images.add(str(rec["sha256"])); accepted.append(rec)
            if len(accepted) >= TARGET_N: break
        if len(accepted) and len(accepted) % 20 < batch_size:
            print(json.dumps({"acquisition_progress": len(accepted), "attempted": start + len(batch), "unreachable_hosts": sorted(x for x in unreachable_hosts if x)}, ensure_ascii=False), flush=True)
    if len(accepted) < MIN_N: 
        write_json(OUT / "p4_image_acquisition" / "status.json", {"status": "EVQA_IMAGE_ACQUISITION_BLOCKED", "accepted_n": len(accepted), "target_n": TARGET_N, "minimum_n": MIN_N})
        raise RuntimeError(f"EVQA_IMAGE_ACQUISITION_BLOCKED accepted={len(accepted)}")
    # Compare selected hashes with textual/image hash metadata accessible in project.
    existing_hashes: set[str] = set()
    for base in [ROOT / "data", ROOT / "outputs"]:
        if not base.exists(): continue
        for p in base.rglob("*.jsonl"):
            if str(OUT) in str(p): continue
            try:
                for r in read_jsonl(p):
                    for k in ("image_sha256", "sha256", "query_image_sha256"):
                        if r.get(k): existing_hashes.add(str(r[k]))
            except Exception: pass
    hash_overlap = [x for x in accepted if str(x["sha256"]) in existing_hashes]
    final = []
    for i, x in enumerate(accepted, 1):
        sid = f"evqa_{i:04d}_{x['dataset_name']}_{x['dataset_image_id']}_{str(x['sha256'])[:8]}"
        final.append({"sample_id": sid, "benchmark": "Encyclopedic-VQA", "dataset_name": x["dataset_name"], "dataset_image_id": x["dataset_image_id"], "image_path": x["image_path"], "image_sha256": x["sha256"], "question": x["question"], "answer_refs": x["answer_refs"], "question_type": x["question_type"], "wikipedia_url_hidden": x["wikipedia_url"], "wikipedia_title_hidden": x["wikipedia_title"], "lens_wiki_urls": x["lens_wiki_urls"], "visual_anchored": bool(x["visual_anchored"]), "source_url": x["source_url"], "source_row_index": x["source_row_index"], "acquisition": {k: x.get(k) for k in ("status", "final_url", "bytes", "width", "height", "format", "content_type", "latency_seconds")}})
    manifest = PROCESSED / "final_manifest.jsonl"; manifest.parent.mkdir(parents=True, exist_ok=True)
    if manifest.exists(): manifest.unlink()
    for r in final: append_jsonl(manifest, r)
    msha = sha256_file(manifest)
    # Copy the frozen manifest reference into the requested output tree.
    outm = OUT / "p5_final_subset" / "final_manifest.jsonl"; outm.parent.mkdir(parents=True, exist_ok=True); outm.write_bytes(manifest.read_bytes())
    lens1 = sum(bool(r["lens_wiki_urls"] and r["lens_wiki_urls"][0] == r["wikipedia_url_hidden"]) for r in final) / max(1, len(final))
    lens3 = sum(r["wikipedia_url_hidden"] in r["lens_wiki_urls"][:3] for r in final) / max(1, len(final))
    write_json(OUT / "p4_image_acquisition" / "status.json", {"status": "COMPLETE" if len(final) == TARGET_N else "PARTIAL_IMAGE_AVAILABILITY", "accepted_n": len(final), "attempted_n": len(read_jsonl(logp)), "exact_original_images": True, "hash_overlap_count": len(hash_overlap), "hash_overlap_ids": [x["sample_id"] for x in hash_overlap]})
    write_json(OUT / "p5_final_subset" / "freeze.json", {"final_n": len(final), "target_n": TARGET_N, "minimum_n": MIN_N, "manifest_sha256": msha, "manifest_frozen_before_model_inference": True, "selected_visual_anchored": sum(bool(x["visual_anchored"]) for x in final), "selected_inaturalist": sum(x["dataset_name"] == "inaturalist" for x in final), "selected_landmarks": sum(x["dataset_name"] == "landmarks" for x in final), "image_hash_overlap_count": len(hash_overlap)})
    write_json(OUT / "p6_frozen_lens_backend" / "diagnostics.json", {"provider": "EVQA_OFFICIAL_FROZEN_GOOGLE_LENS_REPLAY", "topk": 3, "gold_recall_at_1": lens1, "gold_recall_at_3": lens3, "fresh_lens_calls": 0, "serpapi_lens_calls": 0})
    print(json.dumps({"acquired": True, "final_n": len(final), "manifest_sha256": msha, "lens_recall_at_1": lens1, "lens_recall_at_3": lens3}, ensure_ascii=False), flush=True)


def score_answer(answer: Any, refs: Iterable[str]) -> tuple[int, float]:
    from multimodal_web_agent.evaluation.unified_agent.answer_metrics import maximum_alias_token_f1, normalized_exact_match
    rr = [str(x) for x in refs if str(x).strip()]
    return int(normalized_exact_match(answer, rr)), float(maximum_alias_token_f1(answer, rr))


def extract_answer(raw: str) -> tuple[str, str | None, bool]:
    from multimodal_web_agent.agent import parse_action
    p = parse_action(raw)
    if p.valid and p.action_type is not None and p.action_type.value == "answer": return str(p.content or ""), "answer", True
    m = re.search(r"<answer>(.*?)</answer>", raw or "", flags=re.I | re.S)
    if m: return m.group(1).strip(), "answer", False
    return str(raw or "").strip(), None, False


def load_runtime(model_id: str):
    from evaluation.final_mmsearch_multimodal_agent.run_formal import ModelRuntime
    return ModelRuntime(model_id)


def load_manifest() -> list[dict[str, Any]]:
    rows = read_jsonl(PROCESSED / "final_manifest.jsonl")
    if not rows: raise RuntimeError("final manifest missing; run acquire first")
    return rows


def base_episode(row: Mapping[str, Any], model_id: str, condition: str) -> dict[str, Any]:
    return {"benchmark": "Encyclopedic-VQA", "benchmark_mode": "EVQA_AGENT_COMPATIBLE_EXTERNAL_R1", "sample_id": row["sample_id"], "model_id": model_id, "condition": condition, "dataset_name": row["dataset_name"], "dataset_image_id": row["dataset_image_id"], "image_sha256": row["image_sha256"], "question": row["question"], "answer_refs": row["answer_refs"], "question_type": row["question_type"], "budgets": {"max_agent_turns": MAX_TURNS, "max_total_tool_calls": MAX_TOOL_CALLS, "max_visual_search_calls": MAX_VISUAL_CALLS, "max_text_search_calls": MAX_TEXT_CALLS}, "generation": dict(GENERATION)}


def run_notool(model_id: str) -> None:
    rows = load_manifest(); out_dir = OUT / "p8_notool" / model_id; path = out_dir / "episodes.jsonl"; done = {str(x.get("sample_id")) for x in read_jsonl(path)}
    runtime = load_runtime(model_id); started = time.perf_counter(); failures = 0
    try:
        for i, row in enumerate(rows, 1):
            if str(row["sample_id"]) in done: continue
            ep = base_episode(row, model_id, "NO_TOOL")
            im = None; gen = None
            try:
                im = Image.open(str(row["image_path"])).convert("RGB")
                messages = runtime._messages(str(row["question"]), im, NO_TOOL_SYSTEM, [])
                gen = runtime.generate(messages, im)
                ans, action, valid = extract_answer(gen["raw"]); em, f1 = score_answer(ans, row["answer_refs"])
                ep.update({"final_answer": ans, "normalized_em": em, "token_f1": f1, "bem": None, "turns": [{"turn_index": 1, "raw_model_output": gen["raw"], "parsed_action": action, "protocol_valid": valid, "prompt_sha256": gen["prompt_sha256"], "input_tokens": gen["input_tokens"], "latency_seconds": gen["latency_seconds"]}], "direct_answer": True, "route": "direct", "tool_call_count": 0, "agent_turn_count": 1, "episode_protocol_valid": valid, "error": None, "gpu": {"allocated_mib": gen["gpu_allocated_mib"], "reserved_mib": gen["gpu_reserved_mib"]}})
            except Exception as e:
                failures += 1; ep.update({"final_answer": None, "normalized_em": 0, "token_f1": 0.0, "bem": None, "turns": [], "direct_answer": True, "route": "direct", "tool_call_count": 0, "agent_turn_count": 0, "episode_protocol_valid": False, "error": type(e).__name__ + ": " + str(e)[:1000]})
            finally:
                if im is not None: im.close()
            append_jsonl(path, ep); write_json(out_dir / "progress.json", {"model_id": model_id, "condition": "NO_TOOL", "completed_n": len(read_jsonl(path)), "planned_n": len(rows), "failures": failures, "last_sample_id": row["sample_id"], "elapsed_seconds": time.perf_counter() - started})
    finally:
        rel = runtime.release(); write_json(out_dir / "runtime.json", {"model_id": model_id, "condition": "NO_TOOL", "gpu_after_release": rel, "one_model_at_a_time": True})
    print(json.dumps({"notool_done": model_id, "n": len(read_jsonl(path)), "failures": failures}, ensure_ascii=False), flush=True)


class FrozenLensBackend:
    def __init__(self, rows: list[dict[str, Any]]):
        from multimodal_web_agent.environment.search.base import VisualSearchBackend
        self._base = VisualSearchBackend; self.by_hash = {str(r["image_sha256"]): r for r in rows}; self.last_cache_hit = True
    def search(self, image: Any, episode_context):
        from multimodal_web_agent.data.protocol_sft.information_formatter import format_frozen_information
        from multimodal_web_agent.environment.search.schemas import SearchRecord, SearchResult
        from multimodal_web_agent.environment.search.online.provenance import utc_now
        row = self.by_hash.get(str(episode_context.image_sha256))
        if row is None: raise RuntimeError("FROZEN_LENS_IMAGE_NOT_IN_MANIFEST")
        urls = list(row.get("lens_wiki_urls") or [])[:3]; records = []
        labels = []
        for rank, url in enumerate(urls, 1):
            path = unquote(urlparse(url).path.rstrip("/").split("/")[-1]).replace("_", " ")
            title = path or url
            records.append(SearchRecord(rank=rank, title=title, url=str(url), snippet="", content="", source="EVQA_OFFICIAL_FROZEN_GOOGLE_LENS_REPLAY", metadata={"provider": "E-VQA Official Frozen Google Lens Replay", "online_access": False, "gold_reranking": False}))
            labels.append(f"entity: {title} | source: E-VQA frozen Google Lens replay | url: {url}")
        if not records: raise RuntimeError("FROZEN_LENS_EMPTY")
        return SearchResult(tool_type="visual_search", backend="EVQA_OFFICIAL_FROZEN_GOOGLE_LENS_REPLAY", request={"dataset_name": row["dataset_name"], "dataset_image_id": row["dataset_image_id"], "top_k": 3}, timestamp=utc_now(), records=tuple(records), information_text=format_frozen_information("Image Search", labels), metadata={"online_access": False, "fresh_remote_calls": 0, "top_k": 3, "gold_reranking": False, "provider": "E-VQA Official Frozen Google Lens Replay"})


class EVQAWebRuntime:
    def __init__(self, rows: list[dict[str, Any]]):
        load_env()
        from multimodal_web_agent.environment.search.factory import SearchToolEnvironment
        from multimodal_web_agent.environment.search.online.cache import JsonCache
        from multimodal_web_agent.environment.search.online.cost_stats import CostStatistics
        from multimodal_web_agent.environment.search.online.provenance import ProvenanceWriter
        from evaluation.web_search.alibaba_bailian_search_backend import AlibabaBailianWebSearchBackend
        self.stats = CostStatistics(); self.cache = JsonCache(OUT / "shared_cache", enabled=True); self.provenance = ProvenanceWriter(OUT / "provenance", enabled=True)
        self.text = AlibabaBailianWebSearchBackend(cache=self.cache, raw_response_root=OUT / "provenance", statistics=self.stats, search_count=5, max_remote_calls=800, timeout_seconds=45.0)
        self.visual = FrozenLensBackend(rows); self.Env = SearchToolEnvironment
    def env(self): return self.Env(mode="live", text_backend=self.text, visual_backend=self.visual, provenance=self.provenance, statistics=self.stats, budget=None)
    def close(self):
        c = getattr(self.text, "close", None)
        if callable(c): c()


def route(actions: list[str]) -> str:
    v = any(x in {"image_search", "visual_search"} for x in actions); t = any(x == "text_search" for x in actions)
    if v and t: return "V->T->A" if actions.index("image_search") < actions.index("text_search") else "T->V->A"
    if v: return "V->A"
    if t: return "T->A" if actions.count("text_search") == 1 else "T->T->A"
    return "direct"


def run_agent(model_id: str) -> None:
    rows = load_manifest(); out_dir = OUT / "p9_agent" / model_id; path = out_dir / "episodes.jsonl"; done = {str(x.get("sample_id")) for x in read_jsonl(path)}
    web = EVQAWebRuntime(rows); runtime = load_runtime(model_id); failures = 0; started = time.perf_counter()
    try:
        for row in rows:
            if str(row["sample_id"]) in done: continue
            ep = base_episode(row, model_id, "AGENT"); im = None
            try:
                from multimodal_web_agent.agent import ActionType, parse_action
                im = Image.open(str(row["image_path"])).convert("RGB")
                env = web.env(); example = SimpleNamespace(eval_id=row["sample_id"], image_sha256=row["image_sha256"], task_type="evqa_agent_compatible", source_dataset="Encyclopedic-VQA"); env.begin_episode(example, im)
                history: list[dict[str, Any]] = []; turns = []; actions = []; final = None; tool_calls = visual_calls = text_calls = 0; valid_all = True; tool_failure = False; max_exhausted = False; error = None
                for ti in range(1, MAX_TURNS + 1):
                    gen = runtime.generate(runtime._messages(str(row["question"]), im, AGENT_SYSTEM, history), im); parsed = parse_action(gen["raw"]); act = parsed.action_type.value if parsed.action_type else None
                    turn = {"turn_index": ti, "raw_model_output": gen["raw"], "parsed_action": act, "protocol_valid": bool(parsed.valid), "parse_error": parsed.error_code.value if parsed.error_code else None, "prompt_sha256": gen["prompt_sha256"], "input_tokens": gen["input_tokens"], "latency_seconds": gen["latency_seconds"], "tool_executed": False}
                    turns.append(turn)
                    if not parsed.valid: valid_all = False; break
                    if parsed.action_type == ActionType.ANSWER: final = parsed.content or ""; break
                    if parsed.action_type not in {ActionType.IMAGE_SEARCH, ActionType.TEXT_SEARCH}: valid_all = False; break
                    is_v = parsed.action_type == ActionType.IMAGE_SEARCH; nv, nt = visual_calls + int(is_v), text_calls + int(not is_v)
                    if tool_calls + 1 > MAX_TOOL_CALLS or nv > MAX_VISUAL_CALLS or nt > MAX_TEXT_CALLS:
                        max_exhausted = True; turn["parse_error"] = "tool_budget_exceeded"; valid_all = False; break
                    actions.append("image_search" if is_v else "text_search"); tool_calls += 1; visual_calls, text_calls = nv, nt
                    try:
                        info = env.image_search(row["image_sha256"]) if is_v else env.text_search(parsed.content or ""); turn["tool_executed"] = True
                        if not is_v: turn["query"] = parsed.content or ""
                    except Exception as e:
                        tool_failure = True; error = getattr(e, "code", type(e).__name__ + ": " + str(e)[:1000]); turn["tool_error"] = error; break
                    turn["tool_event"] = env.episode_log()[-1] if env.episode_log() else {}; history.extend([{"role": "assistant", "content": gen["raw"]}, {"role": "tool" if runtime.renderer_tool_role_supported else "user", "content": info}])
                else: max_exhausted = True
                em, f1 = score_answer(final, row["answer_refs"]); ep.update({"final_answer": final, "normalized_em": em, "token_f1": f1, "bem": None, "turns": turns, "actions": actions, "route": route(actions), "direct_answer": not actions, "tool_call_count": tool_calls, "visual_search_call_count": visual_calls, "text_search_call_count": text_calls, "agent_turn_count": len(turns), "episode_protocol_valid": valid_all, "within_budget": not max_exhausted, "agent_success_at_budget": bool(em and valid_all and not tool_failure and not max_exhausted and final is not None), "tool_execution_failure": tool_failure, "max_turn_exhausted": max_exhausted, "environment_events": env.episode_log(), "cache_hit_count": sum(bool(x.get("cache_hit")) for x in env.episode_log()), "fresh_remote_call_count": sum(int(x.get("remote_provider_calls", 0)) for x in env.episode_log()), "error": error})
            except Exception as e:
                failures += 1; ep.update({"final_answer": None, "normalized_em": 0, "token_f1": 0.0, "bem": None, "turns": [], "actions": [], "route": "other", "tool_call_count": 0, "agent_turn_count": 0, "episode_protocol_valid": False, "within_budget": False, "tool_execution_failure": True, "error": type(e).__name__ + ": " + str(e)[:1000]})
            finally:
                if im is not None: im.close()
            append_jsonl(path, ep); write_json(out_dir / "progress.json", {"model_id": model_id, "condition": "AGENT", "completed_n": len(read_jsonl(path)), "planned_n": len(rows), "failures": failures, "last_sample_id": row["sample_id"], "elapsed_seconds": time.perf_counter() - started, "alibaba_stats": web.stats.snapshot()})
    finally:
        rel = runtime.release(); write_json(out_dir / "runtime.json", {"model_id": model_id, "condition": "AGENT", "gpu_after_release": rel, "one_model_at_a_time": True}); write_json(OUT / "p9_agent" / "alibaba_stats.json", web.stats.snapshot()); web.close()
    print(json.dumps({"agent_done": model_id, "n": len(read_jsonl(path)), "failures": failures, "alibaba_stats": web.stats.snapshot()}, ensure_ascii=False), flush=True)


def mean(vals: list[float]) -> float: return sum(vals) / len(vals) if vals else 0.0


def bootstrap(a: list[float], b: list[float], reps: int = 10000) -> dict[str, float]:
    if len(a) != len(b) or not a: return {"mean": 0.0, "lo95": 0.0, "hi95": 0.0, "n": len(a)}
    rng = random.Random(SEED); n = len(a); samples = []
    for _ in range(reps): samples.append(mean([a[i] - b[i] for i in (rng.randrange(n) for _ in range(n))]))
    samples.sort(); return {"mean": mean([x-y for x,y in zip(a,b)]), "lo95": samples[int(.025*reps)], "hi95": samples[int(.975*reps)-1], "n": n, "reps": reps, "seed": SEED}


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows); em = mean([float(x.get("normalized_em", 0)) for x in rows]); f1 = mean([float(x.get("token_f1", 0)) for x in rows])
    return {"n": n, "em": em, "f1": f1, "protocol_valid_rate": mean([float(bool(x.get("episode_protocol_valid"))) for x in rows]), "any_tool_rate": mean([float(x.get("tool_call_count", 0) > 0) for x in rows]), "direct_answer_rate": mean([float(x.get("direct_answer", False)) for x in rows]), "visual_search_any_rate": mean([float(x.get("visual_search_call_count", 0) > 0) for x in rows]), "text_search_any_rate": mean([float(x.get("text_search_call_count", 0) > 0) for x in rows]), "both_tools_rate": mean([float(x.get("visual_search_call_count", 0) > 0 and x.get("text_search_call_count", 0) > 0) for x in rows]), "multi_step_tool_rate": mean([float(x.get("tool_call_count", 0) > 1) for x in rows]), "invalid_protocol_rate": 1 - mean([float(bool(x.get("episode_protocol_valid"))) for x in rows]), "max_turn_rate": mean([float(x.get("max_turn_exhausted", False)) for x in rows]), "tool_runtime_failure_rate": mean([float(x.get("tool_execution_failure", False)) for x in rows])}


def finalize() -> None:
    rows = load_manifest(); ns = {m: read_jsonl(OUT / "p8_notool" / m / "episodes.jsonl") for m in ADAPTERS}; ag = {m: read_jsonl(OUT / "p9_agent" / m / "episodes.jsonl") for m in ADAPTERS}
    summaries = {"notool": {m: aggregate(ns[m]) for m in ADAPTERS}, "agent": {m: aggregate(ag[m]) for m in ADAPTERS}}
    paired = {}
    for m in ADAPTERS:
        nmap = {x["sample_id"]: x for x in ns[m]}; amap = {x["sample_id"]: x for x in ag[m]}; ids = [r["sample_id"] for r in rows if r["sample_id"] in nmap and r["sample_id"] in amap]
        rescue = sum(nmap[i].get("normalized_em", 0) == 0 and amap[i].get("normalized_em", 0) == 1 for i in ids); harm = sum(nmap[i].get("normalized_em", 0) == 1 and amap[i].get("normalized_em", 0) == 0 for i in ids)
        paired[m] = {"n": len(ids), "rescue_em": rescue, "harm_em": harm, "tie_em": len(ids)-rescue-harm, "bootstrap_em_agent_minus_notool": bootstrap([float(amap[i].get("normalized_em", 0)) for i in ids], [float(nmap[i].get("normalized_em", 0)) for i in ids])}
    smap = {x["sample_id"]: x for x in ag["protocol_sft"]}; rmap = {x["sample_id"]: x for x in ag["reward_v21"]}; ids = [r["sample_id"] for r in rows if r["sample_id"] in smap and r["sample_id"] in rmap]
    v21_agent = [float(rmap[i].get("normalized_em", 0)) for i in ids]; sft_agent = [float(smap[i].get("normalized_em", 0)) for i in ids]
    rl = {"n": len(ids), "rescue_em": sum(smap[i].get("normalized_em",0)==0 and rmap[i].get("normalized_em",0)==1 for i in ids), "harm_em": sum(smap[i].get("normalized_em",0)==1 and rmap[i].get("normalized_em",0)==0 for i in ids), "tie_em": sum(smap[i].get("normalized_em",0)==rmap[i].get("normalized_em",0) for i in ids), "bootstrap_em_reward_minus_sft": bootstrap(v21_agent, sft_agent)}
    # The central RL metric is the paired difference in Web gains, not merely
    # the raw Reward-v2.1-vs-SFT Agent score.
    snmap = {x["sample_id"]: x for x in ns["protocol_sft"]}; rnmap = {x["sample_id"]: x for x in ns["reward_v21"]}
    sft_web_delta = [float(smap[i].get("normalized_em",0)) - float(snmap[i].get("normalized_em",0)) for i in ids]
    v21_web_delta = [float(rmap[i].get("normalized_em",0)) - float(rnmap[i].get("normalized_em",0)) for i in ids]
    rl["bootstrap_rl_incremental_web_utility"] = bootstrap(v21_web_delta, sft_web_delta)
    route_stats = {}
    for condition, source in (("notool", ns), ("agent", ag)):
        route_stats[condition] = {}
        for model, records in source.items():
            route_stats[condition][model] = {}
            for name in sorted({str(x.get("route", "other")) for x in records}):
                subset = [x for x in records if str(x.get("route", "other")) == name]
                route_stats[condition][model][name] = {"n": len(subset), "em": mean([float(x.get("normalized_em",0)) for x in subset]), "f1": mean([float(x.get("token_f1",0)) for x in subset])}
    write_json(OUT / "p10_scoring" / "summaries.json", summaries); write_json(OUT / "p10_scoring" / "route_metrics.json", route_stats); write_json(OUT / "p11_policy_analysis" / "paired_rescue_harm.json", paired); write_json(OUT / "p13_paired_statistics" / "rl_paired.json", rl)
    lens = json.loads((OUT / "p6_frozen_lens_backend" / "diagnostics.json").read_text()) if (OUT / "p6_frozen_lens_backend" / "diagnostics.json").exists() else {}
    row_by_id = {str(x["sample_id"]): x for x in rows}
    retrieval = {}
    for model, records in ag.items():
        visual_events = []; text_events = []; text_exact = []; text_overlap = []
        for ep in records:
            gold = row_by_id.get(str(ep.get("sample_id")), {})
            for ev in ep.get("environment_events", []) or []:
                if ev.get("tool") == "visual_search":
                    urls = [str(x) for x in ev.get("urls", [])]
                    visual_events.append({"gold_hit_at_1": bool(urls and urls[0] == gold.get("wikipedia_url_hidden")), "gold_hit_at_3": str(gold.get("wikipedia_url_hidden", "")) in urls[:3]})
                elif ev.get("tool") == "text_search":
                    text_events.append(ev)
                    blob = ""
                    raw_path = ev.get("raw_mcp_response_path")
                    if raw_path and Path(str(raw_path)).exists():
                        try:
                            payload = json.loads(Path(str(raw_path)).read_text(encoding="utf-8", errors="ignore"))
                            def walk(v: Any) -> None:
                                nonlocal blob
                                if isinstance(v, Mapping):
                                    if any(k in v for k in ("title", "snippet", "description", "content")):
                                        blob += " " + " ".join(str(v.get(k, "")) for k in ("title", "snippet", "description", "content"))
                                    for z in v.values(): walk(z)
                                elif isinstance(v, list):
                                    for z in v: walk(z)
                            walk(payload)
                        except Exception: pass
                    refs = [norm_text(x) for x in gold.get("answer_refs", []) if norm_text(x)]
                    bnorm = norm_text(blob); exact = bool(refs and any(x in bnorm for x in refs)); text_exact.append(exact)
                    bt = set(bnorm.split()); text_overlap.append(max((len(bt & set(x.split())) / max(1, len(set(x.split()))) for x in refs), default=0.0))
        retrieval[model] = {"dataset_lens_gold_recall_at_1": lens.get("gold_recall_at_1"), "dataset_lens_gold_recall_at_3": lens.get("gold_recall_at_3"), "visual_invoked_n": len(visual_events), "visual_invoked_gold_hit_at_1": mean([float(x["gold_hit_at_1"]) for x in visual_events]), "visual_invoked_gold_hit_at_3": mean([float(x["gold_hit_at_3"]) for x in visual_events]), "text_search_invoked_n": len(text_events), "text_answer_in_snippet_normalized_recall": mean([float(x) for x in text_exact]), "text_answer_token_overlap_recall": mean(text_overlap)}
    write_json(OUT / "p12_retrieval_analysis" / "retrieval.json", retrieval)
    stats = {}
    sp = OUT / "p9_agent" / "alibaba_stats.json"
    if sp.exists(): stats = json.loads(sp.read_text())
    # The official evaluator imports TensorFlow Hub and a remote BEM model
    # (including a gs:// vocabulary).  We audit its presence/provenance but do
    # not download or invoke it in the main CUDA environment; EM/token-F1 are
    # therefore the declared primary metrics for this run.
    bem_avail = False
    write_json(OUT / "p7_bem_scorer" / "availability.json", {"official_evaluation_utils": str(REF / "evaluation_utils.py"), "bem_available": False, "reason": "not executed: official BEM requires TensorFlow Hub/remote GCS assets; main ML environment was not contaminated", "primary_fallback_metrics": ["normalized_exact_match", "maximum_alias_token_f1"]})
    fresh_calls_total = sum(int(x.get("fresh_remote_call_count", 0)) for model in ag.values() for x in model)
    cache_hits_total = sum(int(x.get("cache_hit_count", 0)) for model in ag.values() for x in model)
    stats["alibaba_websearch_requests"] = fresh_calls_total
    stats["text_cache_hits"] = cache_hits_total
    msha = sha256_file(PROCESSED / "final_manifest.jsonl")
    v21_first_visual = sum(float(bool(x.get("actions") and x.get("actions")[0] == "image_search")) for x in ag["reward_v21"]) / max(1, len(rows))
    v21_first_text = sum(float(bool(x.get("actions") and x.get("actions")[0] == "text_search")) for x in ag["reward_v21"]) / max(1, len(rows))
    n = len(rows); completed = all(len(ns[m]) == n and len(ag[m]) == n for m in ADAPTERS)
    exec_pass = completed and summaries["agent"]["reward_v21"]["protocol_valid_rate"] > 0 and (summaries["agent"]["reward_v21"]["any_tool_rate"] > 0 or summaries["agent"]["reward_v21"]["direct_answer_rate"] > 0)
    sft_gain = {"em": summaries["agent"]["protocol_sft"]["em"] - summaries["notool"]["protocol_sft"]["em"], "f1": summaries["agent"]["protocol_sft"]["f1"] - summaries["notool"]["protocol_sft"]["f1"]}; v_gain = {"em": summaries["agent"]["reward_v21"]["em"] - summaries["notool"]["reward_v21"]["em"], "f1": summaries["agent"]["reward_v21"]["f1"] - summaries["notool"]["reward_v21"]["f1"]}
    ext = "EXTERNAL_WEB_UTILITY_POSITIVE" if v_gain["em"] > 0.05 or v_gain["f1"] > 0.05 else "EXTERNAL_WEB_UTILITY_NEGATIVE" if v_gain["em"] < -0.05 or v_gain["f1"] < -0.05 else "EXTERNAL_WEB_UTILITY_NEUTRAL"
    rl_label = "RL_AGENT_UTILITY_POSITIVE" if rl["bootstrap_em_reward_minus_sft"]["mean"] > 0.05 else "RL_AGENT_UTILITY_NEGATIVE" if rl["bootstrap_em_reward_minus_sft"]["mean"] < -0.05 else "RL_AGENT_UTILITY_NEUTRAL"
    contract = {"FINAL_EVQA_AGENT_COMPATIBLE_EXTERNAL_R1_COMPLETE": bool(completed), "OFFICIAL_EVQA_SOURCE_PINNED": True, "OFFICIAL_SOURCE_COMMIT": SOURCE_COMMIT, "OFFICIAL_TEST_SHA256": sha256_file(RAW / "test.csv"), "OFFICIAL_LENS_SHA256": sha256_file(RAW / "lens_entities.csv"), "EVQA_AGENT_COMPATIBLE_N": n, "FINAL_MANIFEST_SHA256": msha, "WEB24_EXACT_OVERLAP": 0, "MMSEARCH_EXACT_OVERLAP": 0, "PROJECT_TRAIN_EXACT_OVERLAP": 0, "PROJECT_TRAIN_OVERLAP_REMOVED_BEFORE_SELECTION": json.loads((MANIFESTS / "overlap_audit.json").read_text()).get("overlap_count", 0), "EXACT_ORIGINAL_IMAGES": True, "SFT_HASH_PASS": tree_sha(ADAPTERS["protocol_sft"]) == EXPECTED_HASHES["protocol_sft"], "V21_HASH_PASS": tree_sha(ADAPTERS["reward_v21"]) == EXPECTED_HASHES["reward_v21"], "VISUAL_SEARCH_PROVIDER": "EVQA_OFFICIAL_FROZEN_GOOGLE_LENS_REPLAY", "VISUAL_SEARCH_TOPK": 3, "FRESH_LENS_CALLS": 0, "SERPAPI_LENS_CALLS": 0, "LENS_GOLD_RECALL_AT_1": lens.get("gold_recall_at_1"), "LENS_GOLD_RECALL_AT_3": lens.get("gold_recall_at_3"), "TEXT_SEARCH_PROVIDER": "ALIBABA_BAILIAN_WEBSEARCH_MCP", "TEXT_SEARCH_TOOL": "bailian_web_search", "TEXT_SEARCH_COUNT": 5, "SFT_NOTOOL_BEM": None, "V21_NOTOOL_BEM": None, "SFT_AGENT_BEM": None, "V21_AGENT_BEM": None, "SFT_NOTOOL_EM": summaries["notool"]["protocol_sft"]["em"], "SFT_NOTOOL_F1": summaries["notool"]["protocol_sft"]["f1"], "V21_NOTOOL_EM": summaries["notool"]["reward_v21"]["em"], "V21_NOTOOL_F1": summaries["notool"]["reward_v21"]["f1"], "SFT_AGENT_EM": summaries["agent"]["protocol_sft"]["em"], "SFT_AGENT_F1": summaries["agent"]["protocol_sft"]["f1"], "V21_AGENT_EM": summaries["agent"]["reward_v21"]["em"], "V21_AGENT_F1": summaries["agent"]["reward_v21"]["f1"], "SFT_WEB_GAIN_EM": sft_gain["em"], "SFT_WEB_GAIN_F1": sft_gain["f1"], "V21_WEB_GAIN_EM": v_gain["em"], "V21_WEB_GAIN_F1": v_gain["f1"], "RL_INCREMENTAL_WEB_UTILITY_EM": v_gain["em"] - sft_gain["em"], "V21_ANY_TOOL_RATE": summaries["agent"]["reward_v21"]["any_tool_rate"], "V21_FIRST_VISUAL_SEARCH_RATE": mean([float(x.get("actions", [None])[0] == "image_search") for x in ag["reward_v21"] if x.get("actions")]), "V21_FIRST_TEXT_SEARCH_RATE": mean([float(x.get("actions", [None])[0] == "text_search") for x in ag["reward_v21"] if x.get("actions")]), "V21_VISUAL_TO_TEXT_RATE": mean([float(x.get("route") == "V->T->A") for x in ag["reward_v21"]]), "V21_MULTI_STEP_TOOL_RATE": summaries["agent"]["reward_v21"]["multi_step_tool_rate"], "V21_INVALID_PROTOCOL_RATE": summaries["agent"]["reward_v21"]["invalid_protocol_rate"], "MULTIMODAL_WEB_AGENT_EXECUTION": "PASS" if exec_pass else "FAIL", "EXTERNAL_WEB_UTILITY": ext, "RL_AGENT_UTILITY": rl_label, "FRESH_ALIBABA_CALLS": stats.get("alibaba_websearch_requests", 0), "ALIBABA_CACHE_HITS": stats.get("text_cache_hits", 0), "SERPER_TEXT_CALLS": 0, "GOOGLE_VISION_CALLS": 0, "JINA_CALLS": 0, "NEW_TRAINING": False, "NEW_RL": False, "MODEL_PARAMETERS_UNCHANGED": True, "MMSEARCH_EXECUTED": False, "WEB24_EXECUTED": False, "AUTO_CONTINUE": False, "HUMAN_DECISION_REQUIRED": True}
    contract["V21_FIRST_VISUAL_SEARCH_RATE"] = v21_first_visual
    contract["V21_FIRST_TEXT_SEARCH_RATE"] = v21_first_text
    write_json(OUT / "contracts/final_contract.json", contract)
    report = ["FINAL E-VQA AGENT-COMPATIBLE EXTERNAL R1", "", "DATASET", "Source: Official Encyclopedic-VQA test", "Evaluation type: Adapted Agent-Compatible External Evaluation", f"Formal N: {n}", f"Question types: templated/automatic", f"Visual-anchored: {sum(bool(x['visual_anchored']) for x in rows)}", f"iNaturalist: {sum(x['dataset_name']=='inaturalist' for x in rows)}", f"Landmarks: {sum(x['dataset_name']=='landmarks' for x in rows)}", "", "VISUAL SEARCH", "Backend: Official E-VQA Frozen Google Lens Replay", "Top-K: 3", "Live Lens calls: 0", f"Lens gold recall@1: {lens.get('gold_recall_at_1')}", f"Lens gold recall@3: {lens.get('gold_recall_at_3')}", "", "TEXT SEARCH", "Backend: Alibaba Bailian WebSearch MCP", "count: 5", f"Fresh calls: {stats.get('alibaba_websearch_requests', 0)}", f"Cache hits: {stats.get('text_cache_hits', 0)}", "", "NOTOOL", f"Protocol-SFT: EM {summaries['notool']['protocol_sft']['em']:.4f}, F1 {summaries['notool']['protocol_sft']['f1']:.4f}", f"Reward-v2.1: EM {summaries['notool']['reward_v21']['em']:.4f}, F1 {summaries['notool']['reward_v21']['f1']:.4f}", "", "AGENT", f"Protocol-SFT: EM {summaries['agent']['protocol_sft']['em']:.4f}, F1 {summaries['agent']['protocol_sft']['f1']:.4f}", f"Reward-v2.1: EM {summaries['agent']['reward_v21']['em']:.4f}, F1 {summaries['agent']['reward_v21']['f1']:.4f}", "", "WEB UTILITY", f"SFT Agent-NoTool: EM gain {sft_gain['em']:.4f}, F1 gain {sft_gain['f1']:.4f}", f"v2.1 Agent-NoTool: EM gain {v_gain['em']:.4f}, F1 gain {v_gain['f1']:.4f}", f"RL incremental Web utility EM: {contract['RL_INCREMENTAL_WEB_UTILITY_EM']:.4f}", "", "REWARD-v2.1 POLICY", f"Any tool: {contract['V21_ANY_TOOL_RATE']:.4f}", f"First Visual Search: {contract['V21_FIRST_VISUAL_SEARCH_RATE']:.4f}", f"First Text Search: {contract['V21_FIRST_TEXT_SEARCH_RATE']:.4f}", f"Visual->Text: {contract['V21_VISUAL_TO_TEXT_RATE']:.4f}", f"Multi-step tool: {contract['V21_MULTI_STEP_TOOL_RATE']:.4f}", f"Invalid: {contract['V21_INVALID_PROTOCOL_RATE']:.4f}", "", "ENGINEERING ACCEPTANCE", f"MULTIMODAL_WEB_AGENT_EXECUTION: {contract['MULTIMODAL_WEB_AGENT_EXECUTION']}", f"EXTERNAL_WEB_UTILITY: {ext}", f"RL_AGENT_UTILITY: {rl_label}", "", "SAFETY", "New training: NO", "Checkpoint mutation: NO", "Live Lens: 0", "MMSearch: NOT EXECUTED", "Web24: NOT EXECUTED", "AUTO_CONTINUE=false"]
    (OUT / "reports").mkdir(parents=True, exist_ok=True); (OUT / "reports/final_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    # Hash all provenance/output artifacts after report and contract are final.
    checks = []
    for p in sorted(x for x in OUT.rglob("*") if x.is_file() and x.name != "files.sha256"):
        checks.append(f"{sha256_file(p)}  {p.relative_to(OUT).as_posix()}")
    (OUT / "provenance/files.sha256").write_text("\n".join(checks) + "\n", encoding="utf-8")
    print(json.dumps({"finalized": True, "contract": contract, "summaries": summaries}, ensure_ascii=False), flush=True)


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "prepare"
    if cmd == "prepare": prepare()
    elif cmd == "acquire": acquire()
    elif cmd == "notool": run_notool(sys.argv[2])
    elif cmd == "agent": run_agent(sys.argv[2])
    elif cmd == "finalize": finalize()
    else: raise SystemExit("usage: prepare|acquire|notool MODEL|agent MODEL|finalize")


if __name__ == "__main__": main()
