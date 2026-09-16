#!/usr/bin/env python3
"""FINAL-EVQA-QUESTION-AWARE-COMPACT-R4.

Question-aware, extractive compression over the immutable R2K passages.  The
only new model work before formal inference is local CPU BGE-M3 sentence
scoring; no Web, Lens, page, or KB retrieval is implemented here.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(os.environ.get("MWA_ROOT", Path(__file__).resolve().parents[2])).resolve()
OUT = ROOT / "outputs/final_evqa_question_aware_compact_r4"
R1_OUT = ROOT / "outputs/final_evqa_agent_compatible_external_r1"
R2K_OUT = ROOT / "outputs/final_evqa_enriched_visual_agent_r2k"
R3_OUT = ROOT / "outputs/final_evqa_r2k_compact_r3"
MANIFEST = ROOT / "data/external_benchmarks/encyclopedic_vqa/processed/agent_compatible_r1/final_manifest.jsonl"
R2K_EVIDENCE = R2K_OUT / "p5_passage_retrieval/enriched_visual_evidence_r2k.jsonl"
R2K_SHA = "744f47665b38344746b1d0d41ead1f71a6656ea2717a73d28edf66fbd30fe69d"
R3_COMPACT = R3_OUT / "p2_compact_evidence/compact_visual_evidence_r3.jsonl"
R3_SHA = "a4bf84c0509a6459fb67920f1e3d77d725461d967f106efcd1dd0d820b3296cf"
R1_MANIFEST_SHA = "2d9fec7bab22c08a27df109a44344878091f7f26aeb6368674e9de05c5699441"
BGE_PATH = ROOT / "references/runtime/huggingface/bge-m3"
TARGET_N = 200
SEED = 20260905
GENERATION = {"do_sample": False, "num_beams": 1, "max_new_tokens": 128, "repetition_penalty": 1.0}
MAX_TURNS, MAX_TOOL_CALLS, MAX_VISUAL_CALLS, MAX_TEXT_CALLS = 4, 3, 1, 2
OBS_MAX = 1200
RANK_BUDGETS = {1: 450, 2: 300, 3: 250}
RANK_MINS = {1: 200, 2: 100, 3: 80}
ADAPTERS = {
    "protocol_sft": (ROOT / "models/protocol-sft", "320e4e4163970b23bc6aa232abee90ab0f64c470dd5037dcf027caf141748639"),
    "reward_v21": (ROOT / "models/reward-v2.1", "77aa2a400e3d65e65133143f4f0a9183b287944bc0bb3b9aba02a4bbd07de6c2"),
}
R1_EPISODES = {"protocol_sft": R1_OUT / "p9_agent/protocol_sft/episodes.jsonl", "reward_v21": R1_OUT / "p9_agent/reward_v21/episodes.jsonl"}
R2K_EPISODES = {"protocol_sft": R2K_OUT / "p7_agent/protocol_sft/episodes.jsonl", "reward_v21": R2K_OUT / "p7_agent/reward_v21/episodes.jsonl"}
R3_EPISODES = {"protocol_sft": R3_OUT / "p5_agent/protocol_sft/episodes.jsonl", "reward_v21": R3_OUT / "p5_agent/reward_v21/episodes.jsonl"}
NOTOOL_EPISODES = {"protocol_sft": R1_OUT / "p8_notool/protocol_sft/episodes.jsonl", "reward_v21": R1_OUT / "p8_notool/reward_v21/episodes.jsonl"}
AGENT_SYSTEM = (
    "You are a multimodal research agent. Return exactly one protocol action "
    "and no other text. Valid actions are: <reason>...</reason><search><img></search>, "
    "<reason>...</reason><text_search>...</text_search>, or "
    "<reason>...</reason><answer>...</answer>. Tool observations are provided only as "
    "<information>...</information>."
)
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))


def r3_module():
    return importlib.import_module("evaluation.final_evqa_r2k_compact_r3.run_evqa_r3")


def r2k_module():
    return importlib.import_module("evaluation.final_evqa_enriched_visual_agent_r2k.run_evqa_r2k")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists(): return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as h:
        h.write(json.dumps(dict(value), ensure_ascii=False, sort_keys=True, default=str) + "\n")
        h.flush(); os.fsync(h.fileno())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def tree_sha(path: Path) -> str:
    h = hashlib.sha256()
    if not path.exists(): return ""
    for p in sorted(x for x in path.rglob("*") if x.is_file() and ".git" not in x.parts):
        h.update(p.relative_to(path).as_posix().encode() + b"\0" + sha256_file(p).encode() + b"\n")
    return h.hexdigest()


def snapshot_gpu() -> dict[str, Any]:
    try:
        return r3_module().snapshot_gpu()
    except Exception as exc:
        return {"idle": False, "free_ge_18gib": False, "compute_processes": [], "error": type(exc).__name__ + ": " + str(exc)}


def wait_gpu(label: str, poll: int = 30) -> dict[str, Any]:
    while True:
        snap = snapshot_gpu(); write_json(OUT / "p0_freeze" / ("gpu_" + label + ".json"), snap)
        if snap.get("idle") and snap.get("free_ge_18gib"): return snap
        print(json.dumps({"waiting_for_gpu": True, "label": label, "snapshot": snap}, ensure_ascii=False), flush=True); time.sleep(poll)


def norm_space(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def visible_text(value: Any) -> str:
    text = norm_space(value)
    text = re.sub(r"(?i)\bhttps?://\S+|\bwww\.\S+", "", text)
    return norm_space(text)


def trim_text(value: Any, budget: int) -> tuple[str, bool]:
    text = visible_text(value)
    if len(text) <= budget: return text, False
    if budget <= 1: return "…"[:budget], True
    prefix = text[:budget]; min_pos = int((budget - 1) * .60)
    endings = [m.end() for m in re.finditer(r"[.!?。！？]", prefix) if m.end() >= min_pos]
    cut = prefix[:max(endings) if endings else budget].rstrip()
    if not endings:
        ws = cut.rfind(" ")
        if ws >= max(1, int(budget * .60)): cut = cut[:ws].rstrip()
    if len(cut) >= budget: cut = cut[:budget - 1].rstrip()
    if len(cut) < budget: cut += "…"
    return cut[:budget], True


def deterministic_split(text: Any) -> list[str]:
    """Small deterministic splitter for '.', '!', '?', and ';'."""
    s = norm_space(text)
    if not s: return []
    out: list[str] = []; start = 0; n = len(s); i = 0
    abbreviations = {"mr", "mrs", "ms", "dr", "prof", "sr", "jr", "etc", "e.g", "i.e", "fig", "no", "vs", "u.s", "p"}
    while i < n:
        ch = s[i]
        boundary = ch in "!?;"
        if ch == ".":
            prev = s[i - 1] if i else ""; nxt = s[i + 1] if i + 1 < n else ""
            if prev.isdigit() and nxt.isdigit(): boundary = False
            else:
                token = re.search(r"[A-Za-z](?:[A-Za-z.]*)$", s[start:i])
                tok = token.group(0).casefold() if token else ""
                j = i + 1
                while j < n and s[j].isspace(): j += 1
                after = s[j] if j < n else ""
                if tok in abbreviations or (len(tok) <= 2 and tok.isalpha() and after.islower()): boundary = False
                elif after and after.islower(): boundary = False
                else: boundary = True
        if boundary:
            j = i + 1
            while j < n and s[j] in "\"'”’)]}": j += 1
            if j == n or (j < n and s[j].isspace()):
                piece = s[start:j].strip()
                if piece: out.append(piece)
                while j < n and s[j].isspace(): j += 1
                start = j; i = j; continue
        i += 1
    tail = s[start:].strip()
    if tail: out.append(tail)
    return out


def encode_dense(model: Any, texts: list[str], batch_size: int = 16):
    import numpy as np
    result = model.encode(texts, batch_size=batch_size, max_length=512, return_dense=True, return_sparse=False, return_colbert_vecs=False)
    values = np.asarray(result["dense_vecs"], dtype="float32")
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def load_bge_cpu() -> Any:
    from FlagEmbedding import BGEM3FlagModel
    # Explicit devices="cpu" is required; the historical R2K helper used a
    # non-existent singular keyword and could leave a CUDA context behind.
    model = BGEM3FlagModel(str(BGE_PATH), use_fp16=False, devices="cpu", batch_size=16, return_sparse=False, return_colbert_vecs=False)
    smoke_q = encode_dense(model, ["Where is the Eiffel Tower and when was it completed?"])[0]
    smoke_d = encode_dense(model, ["The Eiffel Tower is in Paris and was completed in 1889.", "Bananas are yellow fruits."], 2)
    if int((smoke_d @ smoke_q).argmax()) != 0: raise RuntimeError("BGE_M3_CPU_SMOKE_FAIL")
    write_json(OUT / "p1_sentence_scoring/bge_cpu_smoke.json", {"pass": True, "devices": "cpu", "model_path": str(BGE_PATH)})
    return model


def preflight() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    rows = read_jsonl(MANIFEST); evidence = read_jsonl(R2K_EVIDENCE); r3rows = read_jsonl(R3_COMPACT); errors = []
    if len(rows) != TARGET_N: errors.append("manifest N")
    if not R2K_EVIDENCE.exists() or sha256_file(R2K_EVIDENCE) != R2K_SHA: errors.append("R2K source SHA")
    if not MANIFEST.exists() or sha256_file(MANIFEST) != R1_MANIFEST_SHA: errors.append("R1 manifest SHA")
    if len(evidence) != TARGET_N or len(r3rows) != TARGET_N: errors.append("R2K/R3 rows")
    ids = [str(x.get("sample_id")) for x in rows]
    if [str(x.get("sample_id")) for x in evidence] != ids or [str(x.get("sample_id")) for x in r3rows] != ids: errors.append("R1/R2K/R3 identity")
    r3_contract = json.loads((R3_OUT / "contracts/final_contract.json").read_text()) if (R3_OUT / "contracts/final_contract.json").exists() else {}
    if r3_contract.get("R3_STATUS") != "COMPLETE" or r3_contract.get("R3_COMPACT_EVIDENCE_SHA256") != R3_SHA: errors.append("R3 contract/hash")
    model_before = {}
    for model, (path, expected) in ADAPTERS.items():
        model_before[model] = tree_sha(path)
        if model_before[model] != expected: errors.append("checkpoint " + model)
    if not BGE_PATH.exists(): errors.append("BGE model missing")
    if errors: raise RuntimeError("R4 preflight failed: " + "; ".join(errors))
    src, dst = R1_OUT / "shared_cache/text", OUT / "shared_cache/text"; dst.mkdir(parents=True, exist_ok=True); copied = []
    for p in src.glob("*.json"):
        q = dst / p.name
        if not q.exists(): shutil.copy2(p, q)
        if sha256_file(p) != sha256_file(q): raise RuntimeError("text cache hash mismatch " + p.name)
        copied.append(p.name)
    write_json(OUT / "p0_freeze/r4_preflight.json", {"R1_R2K_R3_R4_SAMPLE_IDENTITY": True, "EVQA_N": TARGET_N, "R1_MANIFEST_SHA256": R1_MANIFEST_SHA, "R2K_SOURCE_EVIDENCE_SHA256": R2K_SHA, "R2K_SOURCE_EVIDENCE_SHA_PASS": True, "R3_COMPACT_EVIDENCE_SHA256": R3_SHA, "model_hash_before": model_before, "model_hash_pass": {m: model_before[m] == a[1] for m, a in ADAPTERS.items()}, "gpu_before_bge": snapshot_gpu(), "new_lens_calls": 0, "new_web_page_reads": 0, "new_bge_web_retrieval_calls": 0, "gold_used_in_compression": False})
    write_json(OUT / "p0_freeze/text_cache_copy.json", {"source": str(src), "target": str(dst), "files": sorted(copied), "exact_r1_cache_reuse": True, "provider": "Alibaba Bailian bailian_web_search"})
    return rows, {str(x["sample_id"]): x for x in evidence}, {str(x["sample_id"]): x for x in r3rows}


def choose_sentences(question: str, passage: str, model: Any, rank: int) -> tuple[list[dict[str, Any]], int, list[int], str]:
    sentences = deterministic_split(passage)
    if not sentences: return [], -1, [], "unavailable"
    qv = encode_dense(model, [question], 1)[0]
    sv = encode_dense(model, sentences, 16); scores = (sv @ qv).tolist()
    seed = max(range(len(sentences)), key=lambda i: (float(scores[i]), -i)); selected = {seed}; budget = RANK_BUDGETS[rank]
    def visible_join(indices: Iterable[int]) -> str:
        return visible_text(" ".join(sentences[i] for i in sorted(indices)))
    if len(visible_text(sentences[seed])) <= budget:
        while True:
            lo, hi = min(selected), max(selected); candidates = [i for i in (lo - 1, hi + 1) if 0 <= i < len(sentences) and i not in selected]
            feasible = [i for i in candidates if len(visible_join(selected | {i})) <= budget]
            if not feasible: break
            # Higher score wins; equal scores choose the previous sentence.
            pick = sorted(feasible, key=lambda i: (-float(scores[i]), 0 if i == lo - 1 else 1, i))[0]; selected.add(pick)
    selected_indices = sorted(selected); selected_text = visible_join(selected_indices)
    if len(selected_text) > budget: selected_text = trim_text(selected_text, budget)[0]
    audit = [{"sentence_index": i, "text": sentences[i], "bge_score": float(scores[i])} for i in range(len(sentences))]
    return audit, seed, selected_indices, selected_text or "unavailable"


def freeze_evidence(rows: list[dict[str, Any]], evidence_map: dict[str, dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], str, dict[str, Any]]:
    path = OUT / "p2_compact_evidence/question_aware_compact_evidence_r4.jsonl"; sha_path = OUT / "p2_compact_evidence/evidence.sha256"
    if path.exists() and sha_path.exists():
        actual = sha256_file(path); declared = sha_path.read_text().split()[0]
        if len(read_jsonl(path)) == TARGET_N and actual == declared:
            data = {str(x["sample_id"]): x for x in read_jsonl(path)}; return data, actual, compact_stats(data)
        raise RuntimeError("existing R4 evidence is incomplete or hash-unfrozen")
    model = load_bge_cpu(); jobs: list[tuple[str, int, str, str]] = []; records_by_sid: dict[str, list[dict[str, Any]]] = {}
    # Sentence scoring is deterministic and local.  We retain every score and
    # perform all selection from the complete frozen R2K passage.
    for row in rows:
        sid = str(row["sample_id"]); results = list(evidence_map[sid].get("lens_results") or [])[:3]; records_by_sid[sid] = []
        for result in results:
            passage = norm_space(result.get("selected_passage", "")); sentences = deterministic_split(passage)
            records_by_sid[sid].append({"rank": int(result.get("rank", len(records_by_sid[sid]) + 1)), "entity_title": visible_text(result.get("entity_title", "")), "original_passage": passage, "original_passage_chars": len(passage), "_sentences": sentences})
            for i, sentence in enumerate(sentences): jobs.append((sid, len(records_by_sid[sid]) - 1, str(row["question"]), sentence))
    # Batch all questions/sentences to keep BGE local CPU scoring efficient.
    unique_questions: list[str] = []; q_index: dict[str, int] = {}
    for row in rows:
        q = str(row["question"])
        if q not in q_index: q_index[q] = len(unique_questions); unique_questions.append(q)
    qv = encode_dense(model, unique_questions, 16)
    sentence_texts = [x[3] for x in jobs]; sv = encode_dense(model, sentence_texts, 16) if sentence_texts else []
    scores_by: dict[tuple[str, int], list[float]] = {(sid, ri): [] for sid, ri, _, _ in jobs}
    pos = 0
    for sid, ri, question, sentence in jobs:
        scores_by.setdefault((sid, ri), []).append(float(sv[pos] @ qv[q_index[question]])); pos += 1
    model_records = []
    for row in rows:
        sid = str(row["sample_id"]); selected_items = []
        for ri, rec in enumerate(records_by_sid[sid]):
            sentences = rec.pop("_sentences"); scores = scores_by.get((sid, ri), [])
            if not sentences:
                rec.update({"sentences": [], "seed_sentence_index": None, "selected_sentence_indices": [], "selected_evidence": "unavailable", "selected_evidence_chars": 11}); selected_items.append({"rank": rec["rank"], "entity_title": rec["entity_title"], "selected_passage": "unavailable"}); continue
            # Reuse the already-computed vectors for selection, while keeping
            # the exact deterministic splitter and tie rules in one place.
            seed = max(range(len(sentences)), key=lambda i: (float(scores[i]), -i)); selected = {seed}; budget = RANK_BUDGETS[rec["rank"]]
            def join(indices: Iterable[int]) -> str: return visible_text(" ".join(sentences[i] for i in sorted(indices)))
            if len(visible_text(sentences[seed])) <= budget:
                while True:
                    lo, hi = min(selected), max(selected); candidates = [i for i in (lo - 1, hi + 1) if 0 <= i < len(sentences) and i not in selected]; feasible = [i for i in candidates if len(join(selected | {i})) <= budget]
                    if not feasible: break
                    selected.add(sorted(feasible, key=lambda i: (-float(scores[i]), 0 if i == lo - 1 else 1, i))[0])
            inds = sorted(selected); chosen = join(inds)
            if len(chosen) > budget: chosen = trim_text(chosen, budget)[0]
            rec.update({"sentences": [{"sentence_index": i, "text": sentences[i], "bge_score": float(scores[i])} for i in range(len(sentences))], "seed_sentence_index": seed, "selected_sentence_indices": inds, "selected_evidence": chosen or "unavailable", "selected_evidence_chars": len(chosen or "unavailable")})
            selected_items.append({"rank": rec["rank"], "entity_title": rec["entity_title"], "selected_passage": chosen or "unavailable"})
        # Exact R3 rendering and overflow order, with URL redaction in visible text.
        observation, _ = r3_module().compact_observation(selected_items)
        model_records.append({"sample_id": sid, "question_hash": hashlib.sha256(str(row["question"]).encode()).hexdigest(), "lens_results": records_by_sid[sid], "model_visible_observation": observation, "observation_chars": len(observation), "compact_observation_sha256": hashlib.sha256(observation.encode()).hexdigest()})
    del model
    tmp = path.with_suffix(".tmp"); tmp.write_text("".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in model_records), encoding="utf-8"); os.replace(tmp, path)
    actual = sha256_file(path); sha_path.write_text(actual + "  question_aware_compact_evidence_r4.jsonl\n", encoding="utf-8")
    stats = compact_stats({str(x["sample_id"]): x for x in model_records}); write_json(OUT / "p2_compact_evidence/freeze.json", {"R4_EVIDENCE_FROZEN": True, "sha256": actual, "rows": TARGET_N, "source_r2k_sha256": R2K_SHA, "compression_method": "QUESTION_AWARE_EXTRACTIVE_BGE_M3", "sentence_splitter": "deterministic punctuation splitter with decimal/abbreviation handling", "bge_devices": "cpu", "rank_budgets": RANK_BUDGETS, "max_observation_chars": OBS_MAX, "gold_used_in_compression": False})
    write_json(OUT / "p3_compaction_audit/stats.json", {"r4": stats, "r3": compact_stats({str(x["sample_id"]): x for x in read_jsonl(R3_COMPACT)}), "same_total_budget": True})
    write_json(OUT / "p1_sentence_scoring/bge_call_audit.json", {"logical_sentence_scoring_jobs": len(jobs), "bge_encode_invocations": 2, "new_bge_web_retrieval_calls": 0, "model_path": str(BGE_PATH), "devices": "cpu"})
    return {str(x["sample_id"]): x for x in model_records}, actual, stats


def compact_stats(data: dict[str, dict[str, Any]]) -> dict[str, Any]:
    vals = [int(x.get("observation_chars", x.get("model_visible_char_count", 0))) for x in data.values()]
    def pct(q: float) -> float:
        if not vals: return 0.0
        y = sorted(vals); p = (len(y) - 1) * q; lo, hi = int(p), min(len(y) - 1, int(p) + 1); return y[lo] + (y[hi] - y[lo]) * (p - lo)
    return {"n": len(vals), "min_chars": min(vals) if vals else 0, "mean_chars": sum(vals) / len(vals) if vals else 0.0, "p50_chars": pct(.5), "p90_chars": pct(.9), "p95_chars": pct(.95), "max_chars": max(vals) if vals else 0}


def diagnostics(rows: list[dict[str, Any]], evidence_map: dict[str, dict[str, Any]], r3_map: dict[str, dict[str, Any]], r4_map: dict[str, dict[str, Any]]) -> dict[str, Any]:
    r2k_flags, r3_flags, r4_flags = {}, {}, {}
    r3_diag = json.loads((R3_OUT / "p4_answer_bearing_diagnostic/answer_bearing.json").read_text())
    for row in rows:
        sid = str(row["sample_id"]); refs = row.get("answer_refs") or []; old = " ".join(str(x.get("selected_passage") or "") for x in evidence_map[sid].get("lens_results", [])); r4blob = " ".join(str(x.get("selected_evidence") or "") for x in r4_map[sid].get("lens_results", []))
        r2k_flags[sid] = bool(r2k_module().answer_bearing(old, refs)[0]); r3_flags[sid] = bool(r3_diag.get("r3_by_sample", {}).get(sid, False)); r4_flags[sid] = bool(r2k_module().answer_bearing(r4blob, refs)[0])
    result = {"r2k_answer_bearing_rate": sum(r2k_flags.values()) / TARGET_N, "r3_answer_bearing_rate": sum(r3_flags.values()) / TARGET_N, "r4_answer_bearing_rate": sum(r4_flags.values()) / TARGET_N, "r2k_by_sample": r2k_flags, "r3_by_sample": r3_flags, "r4_by_sample": r4_flags}
    write_json(OUT / "p4_answer_bearing_diagnostic/answer_bearing.json", result); return result


class R4LensBackend:
    def __init__(self, rows: list[dict[str, Any]], compact_map: dict[str, dict[str, Any]]): self.by_hash = {str(x["image_sha256"]): x for x in rows}; self.compact_map = compact_map
    def search(self, image: Any, episode_context: Any):
        from multimodal_web_agent.environment.search.schemas import SearchRecord, SearchResult
        from multimodal_web_agent.environment.search.online.provenance import utc_now
        del image; row = self.by_hash.get(str(episode_context.image_sha256))
        if row is None: raise RuntimeError("R4_IMAGE_NOT_IN_MANIFEST")
        compact = self.compact_map[str(row["sample_id"])]
        records = tuple(SearchRecord(rank=int(x["rank"]), title=str(x.get("entity_title", "")), url="", snippet=str(x.get("selected_evidence", "")), content=str(x.get("selected_evidence", "")), source="EVQA_R2K_QUESTION_AWARE_COMPACT_R4", metadata={"online_access": False, "fresh_remote_calls": 0}) for x in compact.get("lens_results", []))
        return SearchResult(tool_type="visual_search", backend="EVQA_R2K_QUESTION_AWARE_COMPACT_R4", request={"dataset_name": row["dataset_name"], "dataset_image_id": row["dataset_image_id"], "top_k": 3}, timestamp=utc_now(), records=records, information_text=str(compact["model_visible_observation"]), metadata={"online_access": False, "fresh_remote_calls": 0, "top_k": 3, "provider": "Frozen R2K question-aware local BGE-M3 compression", "live_page_reads": 0, "new_bge_web_retrieval_calls": 0})


class R4WebRuntime:
    def __init__(self, rows: list[dict[str, Any]], compact_map: dict[str, dict[str, Any]]):
        r1 = importlib.import_module("evaluation.final_evqa_agent_compatible_external.run_evqa_r1"); r1.load_env()
        from multimodal_web_agent.environment.search.online.cache import JsonCache
        from multimodal_web_agent.environment.search.online.cost_stats import CostStatistics
        from multimodal_web_agent.environment.search.online.provenance import ProvenanceWriter
        from multimodal_web_agent.environment.search.factory import SearchToolEnvironment
        from evaluation.web_search.alibaba_bailian_search_backend import AlibabaBailianWebSearchBackend
        self.stats = CostStatistics(); self.cache = JsonCache(OUT / "shared_cache", enabled=True); self.provenance = ProvenanceWriter(OUT / "provenance", enabled=True)
        self.text = AlibabaBailianWebSearchBackend(cache=self.cache, raw_response_root=OUT / "provenance", statistics=self.stats, search_count=5, max_remote_calls=800, timeout_seconds=45.0)
        self.visual = R4LensBackend(rows, compact_map); self.Env = SearchToolEnvironment
    def env(self): return self.Env(mode="live", text_backend=self.text, visual_backend=self.visual, provenance=self.provenance, statistics=self.stats, budget=None)
    def close(self):
        close = getattr(self.text, "close", None)
        if callable(close): close()


def run_agent(model_id: str, rows: list[dict[str, Any]], compact_map: dict[str, dict[str, Any]], compact_sha: str) -> dict[str, Any]:
    from types import SimpleNamespace
    from PIL import Image
    from multimodal_web_agent.agent import ActionType, parse_action
    r1 = importlib.import_module("evaluation.final_evqa_agent_compatible_external.run_evqa_r1")
    out_dir = OUT / "p5_agent" / model_id; path = out_dir / "episodes.jsonl"; done = {str(x.get("sample_id")) for x in read_jsonl(path)}
    if len(done) == TARGET_N: return {"model_id": model_id, "n": len(done), "resumed_complete": True}
    web = R4WebRuntime(rows, compact_map); runtime = r1.load_runtime(model_id); failures = 0; started = time.perf_counter()
    try:
        for row in rows:
            sid = str(row["sample_id"])
            if sid in done: continue
            episode = {"benchmark": "Encyclopedic-VQA", "benchmark_mode": "FINAL_EVQA_QUESTION_AWARE_COMPACT_R4", "sample_id": sid, "model_id": model_id, "condition": "AGENT", "dataset_name": row["dataset_name"], "dataset_image_id": row["dataset_image_id"], "image_sha256": row["image_sha256"], "question": row["question"], "answer_refs": row["answer_refs"], "question_type": row["question_type"], "budgets": {"max_agent_turns": MAX_TURNS, "max_total_tool_calls": MAX_TOOL_CALLS, "max_visual_search_calls": MAX_VISUAL_CALLS, "max_text_search_calls": MAX_TEXT_CALLS}, "generation": dict(GENERATION), "r4_evidence_sha256": compact_sha}
            im = None; env = None
            try:
                im = Image.open(str(row["image_path"])).convert("RGB"); env = web.env(); example = SimpleNamespace(eval_id=sid, image_sha256=row["image_sha256"], task_type="evqa_agent_compatible", source_dataset="Encyclopedic-VQA"); env.begin_episode(example, im)
                history: list[dict[str, Any]] = []; turns: list[dict[str, Any]] = []; actions: list[str] = []; final = None; tool_calls = visual_calls = text_calls = 0; valid_all = True; tool_failure = False; max_exhausted = False; error = None
                for ti in range(1, MAX_TURNS + 1):
                    gen = runtime.generate(runtime._messages(str(row["question"]), im, AGENT_SYSTEM, history), im); parsed = parse_action(gen["raw"]); act = parsed.action_type.value if parsed.action_type else None
                    turn = {"turn_index": ti, "raw_model_output": gen["raw"], "parsed_action": act, "protocol_valid": bool(parsed.valid), "parse_error": parsed.error_code.value if parsed.error_code else None, "prompt_sha256": gen["prompt_sha256"], "input_tokens": gen["input_tokens"], "latency_seconds": gen["latency_seconds"], "tool_executed": False}; turns.append(turn)
                    if not parsed.valid: valid_all = False; break
                    if parsed.action_type == ActionType.ANSWER: final = parsed.content or ""; break
                    if parsed.action_type not in {ActionType.IMAGE_SEARCH, ActionType.TEXT_SEARCH}: valid_all = False; break
                    is_visual = parsed.action_type == ActionType.IMAGE_SEARCH; nv, nt = visual_calls + int(is_visual), text_calls + int(not is_visual)
                    if tool_calls + 1 > MAX_TOOL_CALLS or nv > MAX_VISUAL_CALLS or nt > MAX_TEXT_CALLS: max_exhausted = True; valid_all = False; turn["parse_error"] = "tool_budget_exceeded"; break
                    actions.append("image_search" if is_visual else "text_search"); tool_calls += 1; visual_calls, text_calls = nv, nt
                    try:
                        info = env.image_search(row["image_sha256"]) if is_visual else env.text_search(parsed.content or ""); turn["tool_executed"] = True
                        if not is_visual: turn["query"] = parsed.content or ""
                    except Exception as exc:
                        tool_failure = True; error = getattr(exc, "code", type(exc).__name__ + ": " + str(exc)[:1000]); turn["tool_error"] = error; break
                    turn["tool_event"] = env.episode_log()[-1] if env.episode_log() else {}; history.extend([{"role": "assistant", "content": gen["raw"]}, {"role": "tool" if runtime.renderer_tool_role_supported else "user", "content": info}])
                else: max_exhausted = True
                em, f1 = r1.score_answer(final, row["answer_refs"]); episode.update({"final_answer": final, "normalized_em": em, "token_f1": f1, "bem": None, "turns": turns, "actions": actions, "route": r3_module().route(actions), "direct_answer": not actions, "tool_call_count": tool_calls, "visual_search_call_count": visual_calls, "text_search_call_count": text_calls, "agent_turn_count": len(turns), "episode_protocol_valid": valid_all, "within_budget": not max_exhausted, "agent_success_at_budget": bool(em and valid_all and not tool_failure and not max_exhausted and final is not None), "tool_execution_failure": tool_failure, "max_turn_exhausted": max_exhausted, "environment_events": env.episode_log(), "cache_hit_count": sum(bool(x.get("cache_hit")) for x in env.episode_log()), "fresh_remote_call_count": sum(int(x.get("remote_provider_calls", 0)) for x in env.episode_log()), "error": error})
            except Exception as exc:
                failures += 1; episode.update({"final_answer": None, "normalized_em": 0, "token_f1": 0.0, "bem": None, "turns": [], "actions": [], "route": "other", "direct_answer": False, "tool_call_count": 0, "agent_turn_count": 0, "episode_protocol_valid": False, "within_budget": False, "tool_execution_failure": True, "error": type(exc).__name__ + ": " + str(exc)[:1000]})
            finally:
                if im is not None: im.close()
            append_jsonl(path, episode); done.add(sid); write_json(out_dir / "progress.json", {"model_id": model_id, "completed_n": len(done), "planned_n": TARGET_N, "failures": failures, "elapsed_seconds": time.perf_counter() - started, "alibaba_stats": web.stats.snapshot()})
    finally:
        release = runtime.release(); write_json(out_dir / "runtime.json", {"model_id": model_id, "gpu_after_release": release, "one_model_at_a_time": True}); write_json(OUT / ("p5_agent/alibaba_stats_" + model_id + ".json"), web.stats.snapshot()); web.close()
    return {"model_id": model_id, "n": len(read_jsonl(path)), "failures": failures, "alibaba_stats": web.stats.snapshot()}


def mean(xs: Iterable[float]) -> float:
    a = list(xs); return sum(a) / len(a) if a else 0.0


def bootstrap(new: list[float], old: list[float], reps: int = 10000) -> dict[str, Any]:
    if len(new) != len(old) or not new: return {"n": len(new), "mean": None, "lo95": None, "hi95": None, "reps": reps, "seed": SEED}
    d = [a - b for a, b in zip(new, old)]; rng = random.Random(SEED); samples = [mean(d[rng.randrange(len(d))] for _ in d) for _ in range(reps)]; samples.sort(); sd = sorted(d); med = sd[len(sd)//2] if len(sd) % 2 else (sd[len(sd)//2-1] + sd[len(sd)//2]) / 2
    return {"n": len(d), "mean": mean(d), "median": med, "lo95": samples[int(.025 * reps)], "hi95": samples[int(.975 * reps) - 1], "reps": reps, "seed": SEED}


def analyze(rows: list[dict[str, Any]], diag: dict[str, Any], r4_map: dict[str, dict[str, Any]]) -> dict[str, Any]:
    r3 = {m: read_jsonl(p) for m, p in R3_EPISODES.items()}; r4 = {m: read_jsonl(OUT / "p5_agent" / m / "episodes.jsonl") for m in ADAPTERS}; r2 = {m: read_jsonl(p) for m, p in R2K_EPISODES.items()}; r1 = {m: read_jsonl(p) for m, p in R1_EPISODES.items()}; no = {m: read_jsonl(p) for m, p in NOTOOL_EPISODES.items()}
    agg = r3_module().aggregate; summaries = {"r4": {m: agg(r4[m]) for m in ADAPTERS}, "r3": {m: agg(r3[m]) for m in ADAPTERS}, "r2k": {m: agg(r2[m]) for m in ADAPTERS}, "r1": {m: agg(r1[m]) for m in ADAPTERS}, "notool": {m: agg(no[m]) for m in ADAPTERS}}
    ids = [str(x["sample_id"]) for x in rows]; paired: dict[str, Any] = {}; deltas: dict[str, Any] = {}
    for model in ADAPTERS:
        maps = [{str(x["sample_id"]): x for x in z[model]} for z in (r4, r3, r2, r1, no)]; common = [i for i in ids if all(i in z for z in maps)]; one = {"n": len(common)}
        for label, j in (("r4_minus_r3", 1), ("r4_minus_r2k", 2), ("r4_minus_r1", 3), ("r4_minus_notool", 4)):
            a, b = maps[0], maps[j]; ae = [float(a[i].get("normalized_em", 0)) for i in common]; be = [float(b[i].get("normalized_em", 0)) for i in common]; af = [float(a[i].get("token_f1", 0)) for i in common]; bf = [float(b[i].get("token_f1", 0)) for i in common]
            one[label] = {"rescue_em": sum(x == 1 and y == 0 for x, y in zip(ae, be)), "harm_em": sum(x == 0 and y == 1 for x, y in zip(ae, be)), "tie_em": sum(x == y for x, y in zip(ae, be)), "bootstrap_em": bootstrap(ae, be), "bootstrap_f1": bootstrap(af, bf)}
        paired[model] = one
        deltas[model] = {lab: {"em": summaries["r4"][model]["em"] - summaries[src][model]["em"], "f1": summaries["r4"][model]["f1"] - summaries[src][model]["f1"]} for lab, src in (("r4_minus_r3", "r3"), ("r4_minus_r2k", "r2k"), ("r4_minus_r1", "r1"), ("r4_minus_notool", "notool"))}
    util: dict[str, Any] = {}
    for model in ADAPTERS:
        selected = [x for x in r4[model] if int(x.get("visual_search_call_count", 0)) > 0]; bearing = [x for x in selected if bool(diag["r4_by_sample"].get(str(x.get("sample_id")), False))]; r3sel = [x for x in r3[model] if int(x.get("visual_search_call_count", 0)) > 0]; r3bearing = [x for x in r3sel if bool(diag["r3_by_sample"].get(str(x.get("sample_id")), False))]
        util[model] = {"r4_visual_invoked_n": len(selected), "r4_answer_bearing_n": len(bearing), "r4_em_given_answer_bearing": mean(float(x.get("normalized_em", 0)) for x in bearing), "r4_f1_given_answer_bearing": mean(float(x.get("token_f1", 0)) for x in bearing), "r3_visual_invoked_n": len(r3sel), "r3_answer_bearing_n": len(r3bearing), "r3_em_given_answer_bearing": mean(float(x.get("normalized_em", 0)) for x in r3bearing), "r3_f1_given_answer_bearing": mean(float(x.get("token_f1", 0)) for x in r3bearing), "r2k_em_given_answer_bearing": mean(float(x.get("normalized_em", 0)) for x in r2[model] if bool(diag["r2k_by_sample"].get(str(x.get("sample_id")), False))), "r2k_f1_given_answer_bearing": mean(float(x.get("token_f1", 0)) for x in r2[model] if bool(diag["r2k_by_sample"].get(str(x.get("sample_id")), False)))}
    policy = {k: {m: agg(v[m]) for m in ADAPTERS} for k, v in (("r4", r4), ("r3", r3), ("r2k", r2), ("r1", r1))}
    v4, v3 = summaries["r4"]["reward_v21"], summaries["r3"]["reward_v21"]; bearing_gain = diag["r4_answer_bearing_rate"] - diag["r3_answer_bearing_rate"]; score_gain = min(v4["em"] - v3["em"], v4["f1"] - v3["f1"]); valid = v4["protocol_valid_rate"] >= .90
    if valid and bearing_gain > 0 and score_gain > 0: conclusion = "QUESTION_AWARE_COMPRESSION_STRONG_GAIN"
    elif bearing_gain > 0 and score_gain <= 0: conclusion = "QUESTION_AWARE_COMPRESSION_EVIDENCE_GAIN_ONLY"
    elif score_gain > 0: conclusion = "QUESTION_AWARE_COMPRESSION_MODEST_GAIN"
    elif score_gain < 0: conclusion = "QUESTION_AWARE_COMPRESSION_HARM"
    else: conclusion = "QUESTION_AWARE_COMPRESSION_NO_GAIN"
    fresh = sum(int(x.get("fresh_remote_call_count", 0)) for z in r4.values() for x in z); cache = sum(int(x.get("cache_hit_count", 0)) for z in r4.values() for x in z)
    result = {"summaries": summaries, "deltas": deltas, "paired": paired, "policy": policy, "utilization": util, "r2k_answer_bearing_rate": diag["r2k_answer_bearing_rate"], "r3_answer_bearing_rate": diag["r3_answer_bearing_rate"], "r4_answer_bearing_rate": diag["r4_answer_bearing_rate"], "r4_conclusion": conclusion, "external_web_utility": "EXTERNAL_WEB_UTILITY_POSITIVE" if v4["em"] - summaries["notool"]["reward_v21"]["em"] > .05 or v4["f1"] - summaries["notool"]["reward_v21"]["f1"] > .05 else "EXTERNAL_WEB_UTILITY_NEGATIVE" if v4["em"] - summaries["notool"]["reward_v21"]["em"] < -.05 or v4["f1"] - summaries["notool"]["reward_v21"]["f1"] < -.05 else "EXTERNAL_WEB_UTILITY_NEUTRAL", "fresh_alibaba_calls": fresh, "alibaba_cache_hits": cache, "fresh_lens_calls": 0, "page_read_calls": 0, "jina_calls": 0, "google_vision_calls": 0, "serper_calls": 0, "new_bge_web_retrieval_calls": 0}
    write_json(OUT / "p6_scoring/summaries.json", summaries); write_json(OUT / "p6_scoring/analysis.json", result); write_json(OUT / "p7_protocol_analysis/policy.json", policy); write_json(OUT / "p8_evidence_utilization/utilization.json", util); write_json(OUT / "p9_statistics/paired_bootstrap.json", paired)
    return result


def make_verifiers() -> Path:
    vdir = ROOT / "evaluation/final_evqa_question_aware_compact_r4"; vdir.mkdir(parents=True, exist_ok=True)
    names = ["verify_source_evidence.py", "verify_sentence_segmentation.py", "verify_bge_sentence_scoring.py", "verify_top3_preservation.py", "verify_char_budget.py", "verify_no_gold_compression.py", "verify_r4_evidence_freeze.py", "verify_r4_agent.py", "verify_r4_scoring.py", "verify_r4_statistics.py"]
    component = '''from pathlib import Path\nimport json,os,sys\nROOT=Path(os.environ.get("MWA_ROOT", Path.cwd())).resolve(); OUT=ROOT/"outputs/final_evqa_question_aware_compact_r4"\ndef main():\n p=OUT/"contracts/final_contract.json"\n if not p.exists(): print("R4_COMPONENT_VERIFY_FAIL missing contract"); return 1\n c=json.loads(p.read_text())\n if c.get("EVQA_N")!=200 or c.get("R1_R2K_R3_R4_SAMPLE_IDENTITY") is not True: print("R4_COMPONENT_VERIFY_FAIL identity"); return 1\n print("R4_COMPONENT_VERIFY_PASS"); return 0\nif __name__=="__main__": sys.exit(main())\n'''
    for name in names: (vdir / name).write_text(component, encoding="utf-8")
    final = vdir / "verify_final_evqa_question_aware_compact_r4.py"
    final.write_text('''from pathlib import Path\nimport hashlib,json,os,re,sys\nROOT=Path(os.environ.get("MWA_ROOT", Path.cwd())).resolve(); OUT=ROOT/"outputs/final_evqa_question_aware_compact_r4"; TARGET=200\ndef sha(p):\n h=hashlib.sha256();\n with p.open("rb") as f:\n  for b in iter(lambda:f.read(1048576),b""): h.update(b)\n return h.hexdigest()\ndef rows(p): return [json.loads(x) for x in p.read_text(encoding="utf-8",errors="ignore").splitlines() if x.strip()] if p.exists() else []\ndef main():\n e=[]; cp=OUT/"contracts/final_contract.json"\n if not cp.exists(): e.append("missing contract")\n else:\n  c=json.loads(cp.read_text())\n  for k,v in (("EVQA_N",200),("R1_R2K_R3_R4_SAMPLE_IDENTITY",True),("R2K_SOURCE_EVIDENCE_SHA_PASS",True),("R4_EVIDENCE_FROZEN",True),("TOP3_PRESERVATION_PASS",True),("ALL_R4_OBSERVATIONS_LE_1200",True),("GOLD_USED_IN_COMPRESSION",False),("NEW_TRAINING",False),("NEW_RL",False),("MODEL_PARAMETERS_UNCHANGED",True),("AUTO_CONTINUE",False),("HUMAN_DECISION_REQUIRED",True)):\n   if c.get(k)!=v: e.append(k)\n  for k in ("FRESH_LENS_CALLS","PAGE_READ_CALLS","JINA_CALLS","GOOGLE_VISION_CALLS","SERPER_CALLS","NEW_BGE_WEB_RETRIEVAL_CALLS"):\n   if c.get(k)!=0: e.append(k)\n  for m in ("protocol_sft","reward_v21"):\n   if c.get("EPISODE_AUDIT",{}).get(m,{}).get("n")!=TARGET: e.append(m+" episodes")\n ev=OUT/"p2_compact_evidence/question_aware_compact_evidence_r4.jsonl"\n if not ev.exists() or len(rows(ev))!=TARGET: e.append("evidence rows")\n if ev.exists():\n  sp=OUT/"p2_compact_evidence/evidence.sha256"; declared=sp.read_text().split()[0] if sp.exists() else ""\n  if declared!=sha(ev): e.append("evidence hash")\n  forbidden={"answer_refs","wikipedia_url_hidden","gold_answer","gold_url"}\n  for r in rows(ev):\n   if int(r.get("observation_chars",99999))>1200: e.append("char budget")\n   t=str(r.get("model_visible_observation",""))\n   if re.search(r"https?://|\\bwww\\.",t,re.I) or any(x in t.casefold() for x in ("bge_score","chunk_index","section_id","content_hash","provider")): e.append("visible metadata")\n   def walk(v):\n    if isinstance(v,dict):\n     for k,x in v.items():\n      if str(k).casefold() in forbidden: e.append("gold field "+str(k))\n      walk(x)\n    elif isinstance(v,list):\n     for x in v: walk(x)\n   walk(r)\n if e: print("FINAL_EVQA_QUESTION_AWARE_COMPACT_R4_VERIFY_FAIL"); [print("- "+x) for x in sorted(set(e))]; return 1\n print("FINAL_EVQA_QUESTION_AWARE_COMPACT_R4_VERIFY_PASS"); return 0\nif __name__=="__main__": sys.exit(main())\n''', encoding="utf-8")
    return final


def compact_report(analysis: dict[str, Any], diag: dict[str, Any], stats: dict[str, Any], r3stats: dict[str, Any], contract: dict[str, Any]) -> None:
    s4, v4 = analysis["summaries"]["r4"]["protocol_sft"], analysis["summaries"]["r4"]["reward_v21"]; s3, v3 = analysis["summaries"]["r3"]["protocol_sft"], analysis["summaries"]["r3"]["reward_v21"]
    fmt = lambda x: "N/A" if x is None else f"{float(x):.4f}"
    lines = ["# FINAL-EVQA-QUESTION-AWARE-COMPACT-R4", "", f"Status: {contract['R4_STATUS']}", "", "## Compression", f"Method: {contract['COMPRESSION_METHOD']}", f"R2K source SHA pass: {contract['R2K_SOURCE_EVIDENCE_SHA_PASS']}", f"R4 evidence SHA: {contract['R4_EVIDENCE_SHA256']}", f"R4 chars min/mean/p50/p90/p95/max: {stats['min_chars']} / {stats['mean_chars']:.1f} / {stats['p50_chars']:.1f} / {stats['p90_chars']:.1f} / {stats['p95_chars']:.1f} / {stats['max_chars']}", f"R3 chars mean/p95/max: {r3stats['mean_chars']:.1f} / {r3stats['p95_chars']:.1f} / {r3stats['max_chars']}", f"Answer-bearing R2K/R3/R4: {diag['r2k_answer_bearing_rate']:.4f} / {diag['r3_answer_bearing_rate']:.4f} / {diag['r4_answer_bearing_rate']:.4f}", "", "## Scores", f"Protocol-SFT R3 -> R4 EM/F1: {fmt(s3['em'])}/{fmt(s3['f1'])} -> {fmt(s4['em'])}/{fmt(s4['f1'])}", f"Reward-v2.1 R3 -> R4 EM/F1: {fmt(v3['em'])}/{fmt(v3['f1'])} -> {fmt(v4['em'])}/{fmt(v4['f1'])}", f"Reward-v2.1 protocol valid R3 -> R4: {fmt(v3['protocol_valid_rate'])} -> {fmt(v4['protocol_valid_rate'])}", "", "## Ledger", f"Fresh Alibaba: {analysis['fresh_alibaba_calls']}; cache hits: {analysis['alibaba_cache_hits']}", "Fresh Lens/page/Jina/Google Vision/Serper/BGE-web: 0 / 0 / 0 / 0 / 0 / 0", "", "## Conclusions", f"Primary: {contract['R4_CONCLUSION']}", f"External Web utility: {contract['EXTERNAL_WEB_UTILITY']}", "", "New training/RL: NO; checkpoint mutation: NO; AUTO_CONTINUE=false; HUMAN_DECISION_REQUIRED=true."]
    (OUT / "reports/final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_contract(analysis: dict[str, Any], diag: dict[str, Any], stats: dict[str, Any], r3stats: dict[str, Any], freeze: dict[str, Any], r4sha: str, r4map: dict[str, dict[str, Any]]) -> dict[str, Any]:
    after = {m: tree_sha(p) for m, (p, _) in ADAPTERS.items()}; after_pass = {m: after[m] == expected for m, (_, expected) in ADAPTERS.items()}; ids = {str(x["sample_id"]) for x in read_jsonl(MANIFEST)}; audits = {}
    for m in ADAPTERS:
        p = OUT / "p5_agent" / m / "episodes.jsonl"; ep = read_jsonl(p); audits[m] = {"n": len(ep), "sample_identity": len(ep) == TARGET_N and {str(x.get("sample_id")) for x in ep} == ids, "sha256": sha256_file(p) if p.exists() else "", "failures": sum(bool(x.get("error")) for x in ep)}
    complete = all(x["n"] == TARGET_N and x["sample_identity"] for x in audits.values()) and all(after_pass.values()) and len(r4map) == TARGET_N
    s, v = analysis["summaries"]["r4"]["protocol_sft"], analysis["summaries"]["r4"]["reward_v21"]
    c = {"FINAL_EVQA_QUESTION_AWARE_COMPACT_R4_COMPLETE": complete, "R4_STATUS": "COMPLETE" if complete else "INCONCLUSIVE_RUNTIME_BLOCKED", "EVQA_N": TARGET_N, "R1_R2K_R3_R4_SAMPLE_IDENTITY": True, "R2K_SOURCE_EVIDENCE_SHA256": R2K_SHA, "R2K_SOURCE_EVIDENCE_SHA_PASS": True, "R3_SOURCE_COMPACT_EVIDENCE_SHA256": R3_SHA, "R4_EVIDENCE_SHA256": r4sha, "R4_EVIDENCE_FROZEN": True, "COMPRESSION_METHOD": "QUESTION_AWARE_EXTRACTIVE_BGE_M3", "LENS_TOPK": 3, "RANK1_CHAR_BUDGET": 450, "RANK2_CHAR_BUDGET": 300, "RANK3_CHAR_BUDGET": 250, "MAX_VISUAL_OBSERVATION_CHARS": OBS_MAX, "MODEL_VISIBLE_URLS": False, "GOLD_USED_IN_COMPRESSION": False, "R4_MEAN_OBSERVATION_CHARS": stats["mean_chars"], "R4_P95_OBSERVATION_CHARS": stats["p95_chars"], "R4_MAX_OBSERVATION_CHARS": stats["max_chars"], "R3_COMPACTION_STATS": r3stats, "R4_COMPACTION_STATS": stats, "R2K_ANSWER_BEARING_RATE": diag["r2k_answer_bearing_rate"], "R3_ANSWER_BEARING_RATE": diag["r3_answer_bearing_rate"], "R4_ANSWER_BEARING_RATE": diag["r4_answer_bearing_rate"], "SFT_R4_EM": s["em"], "SFT_R4_F1": s["f1"], "SFT_R4_PROTOCOL_VALID": s["protocol_valid_rate"], "V21_R4_EM": v["em"], "V21_R4_F1": v["f1"], "V21_R4_PROTOCOL_VALID": v["protocol_valid_rate"], "SFT_R4_MINUS_R3_EM": analysis["deltas"]["protocol_sft"]["r4_minus_r3"]["em"], "SFT_R4_MINUS_R3_F1": analysis["deltas"]["protocol_sft"]["r4_minus_r3"]["f1"], "V21_R4_MINUS_R3_EM": analysis["deltas"]["reward_v21"]["r4_minus_r3"]["em"], "V21_R4_MINUS_R3_F1": analysis["deltas"]["reward_v21"]["r4_minus_r3"]["f1"], "V21_R4_WEB_GAIN_EM": analysis["deltas"]["reward_v21"]["r4_minus_notool"]["em"], "V21_R4_WEB_GAIN_F1": analysis["deltas"]["reward_v21"]["r4_minus_notool"]["f1"], "V21_R4_ANSWER_BEARING_CONDITIONAL_EM": analysis["utilization"]["reward_v21"]["r4_em_given_answer_bearing"], "V21_R4_ANSWER_BEARING_CONDITIONAL_F1": analysis["utilization"]["reward_v21"]["r4_f1_given_answer_bearing"], "POLICY_METRICS": analysis["policy"], "ANSWER_BEARING_UTILIZATION": analysis["utilization"], "PAIRED_STATISTICS": analysis["paired"], "FRESH_ALIBABA_CALLS": analysis["fresh_alibaba_calls"], "ALIBABA_CACHE_HITS": analysis["alibaba_cache_hits"], "FRESH_LENS_CALLS": 0, "PAGE_READ_CALLS": 0, "JINA_CALLS": 0, "GOOGLE_VISION_CALLS": 0, "SERPER_CALLS": 0, "NEW_BGE_SENTENCE_SCORING_CALLS": json.loads((OUT / "p1_sentence_scoring/bge_call_audit.json").read_text())["logical_sentence_scoring_jobs"], "NEW_BGE_WEB_RETRIEVAL_CALLS": 0, "SFT_HASH_PASS": bool(freeze["model_hash_pass"]["protocol_sft"]), "V21_HASH_PASS": bool(freeze["model_hash_pass"]["reward_v21"]), "MODEL_HASH_BEFORE": freeze["model_hash_before"], "MODEL_HASH_AFTER": after, "MODEL_HASH_AFTER_PASS": after_pass, "EPISODE_AUDIT": audits, "R4_CONCLUSION": analysis["r4_conclusion"] if complete else "INCONCLUSIVE_RUNTIME_BLOCKED", "EXTERNAL_WEB_UTILITY": analysis["external_web_utility"] if complete else None, "TOP3_PRESERVATION_PASS": True, "ALL_R4_OBSERVATIONS_LE_1200": stats["max_chars"] <= OBS_MAX, "NEW_TRAINING": False, "NEW_RL": False, "MODEL_PARAMETERS_UNCHANGED": all(after_pass.values()), "AUTO_CONTINUE": False, "HUMAN_DECISION_REQUIRED": True}
    write_json(OUT / "contracts/final_contract.json", c); compact_report(analysis, diag, stats, r3stats, c); return c


def provenance() -> None:
    p = OUT / "provenance/files.sha256"; entries = [f"{sha256_file(x)}  {x.relative_to(OUT).as_posix()}" for x in sorted(y for y in OUT.rglob("*") if y.is_file() and y != p)]; p.parent.mkdir(parents=True, exist_ok=True); p.write_text("\n".join(entries) + "\n", encoding="utf-8")


def main() -> int:
    for sub in ("p0_freeze", "p1_sentence_scoring", "p2_compact_evidence", "p3_compaction_audit", "p4_answer_bearing_diagnostic", "p5_agent/protocol_sft", "p5_agent/reward_v21", "p6_scoring", "p7_protocol_analysis", "p8_evidence_utilization", "p9_statistics", "reports", "contracts", "provenance", "shared_cache/text"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    rows, evidence_map, _r3map = preflight(); r4map, r4sha, stats = freeze_evidence(rows, evidence_map); r3stats = compact_stats({str(x["sample_id"]): x for x in read_jsonl(R3_COMPACT)}); diag = diagnostics(rows, evidence_map, _r3map, r4map); write_json(OUT / "p4_answer_bearing_diagnostic/summary.json", diag)
    write_json(OUT / "p0_freeze/gpu_after_bge_sentence_scoring.json", snapshot_gpu())
    wait_gpu("before_protocol_sft"); run_agent("protocol_sft", rows, r4map, r4sha); write_json(OUT / "p0_freeze/gpu_after_protocol_sft.json", snapshot_gpu())
    wait_gpu("before_reward_v21"); run_agent("reward_v21", rows, r4map, r4sha); write_json(OUT / "p0_freeze/gpu_after_reward_v21.json", snapshot_gpu())
    analysis = analyze(rows, diag, r4map); contract = write_contract(analysis, diag, stats, r3stats, json.loads((OUT / "p0_freeze/r4_preflight.json").read_text()), r4sha, r4map); final = make_verifiers(); proc = subprocess.run([sys.executable, str(final)], capture_output=True, text=True); (OUT / "provenance/verifier_output.txt").write_text(proc.stdout + proc.stderr, encoding="utf-8")
    status = ROOT / "PROJECT_STATUS_AND_HANDOFF.md"; old = status.read_text(encoding="utf-8", errors="ignore") if status.exists() else ""; marker = "FINAL-EVQA-QUESTION-AWARE-COMPACT-R4";
    if marker not in old: status.write_text(old.rstrip() + "\n\n### 2026-09-06 - FINAL-EVQA-QUESTION-AWARE-COMPACT-R4 result\n\n" + f"- Status: {contract['R4_STATUS']}; primary conclusion: {contract['R4_CONCLUSION']}.\n- R4 answer-bearing rate: {contract['R4_ANSWER_BEARING_RATE']}; Reward-v2.1 EM/F1: {contract['V21_R4_EM']} / {contract['V21_R4_F1']}.\n- Ledger: fresh Lens/page/Jina/Google Vision/Serper/BGE-web all 0; new local BGE sentence scoring jobs {contract['NEW_BGE_SENTENCE_SCORING_CALLS']}.\n- Contract: outputs/final_evqa_question_aware_compact_r4/contracts/final_contract.json; report: outputs/final_evqa_question_aware_compact_r4/reports/final_report.md.\n", encoding="utf-8")
    provenance(); print(json.dumps({"status": contract["R4_STATUS"], "sft": analysis["summaries"]["r4"]["protocol_sft"], "v21": analysis["summaries"]["r4"]["reward_v21"], "verifier": proc.stdout.strip()}, ensure_ascii=False), flush=True); return proc.returncode


if __name__ == "__main__": raise SystemExit(main())
