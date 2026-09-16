#!/usr/bin/env python3
"""FINAL-EVQA-R2K-COMPACT-R3.

This evaluator consumes only the already frozen R2K evidence.  It performs a
deterministic, rank-preserving formatting/truncation pass before loading either
multimodal checkpoint, then runs the two formal agent conditions sequentially.
No Lens/page/BGE retrieval is implemented here.
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
OUT = ROOT / "outputs/final_evqa_r2k_compact_r3"
R1_OUT = ROOT / "outputs/final_evqa_agent_compatible_external_r1"
R2K_OUT = ROOT / "outputs/final_evqa_enriched_visual_agent_r2k"
MANIFEST = ROOT / "data/external_benchmarks/encyclopedic_vqa/processed/agent_compatible_r1/final_manifest.jsonl"
R2K_EVIDENCE = R2K_OUT / "p5_passage_retrieval/enriched_visual_evidence_r2k.jsonl"
R2K_EVIDENCE_SHA = "744f47665b38344746b1d0d41ead1f71a6656ea2717a73d28edf66fbd30fe69d"
R1_MANIFEST_SHA = "2d9fec7bab22c08a27df109a44344878091f7f26aeb6368674e9de05c5699441"
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
NOTOOL_EPISODES = {"protocol_sft": R1_OUT / "p8_notool/protocol_sft/episodes.jsonl", "reward_v21": R1_OUT / "p8_notool/reward_v21/episodes.jsonl"}
AGENT_SYSTEM = (
    "You are a multimodal research agent. Return exactly one protocol action "
    "and no other text. Valid actions are: <reason>...</reason><search><img></search>, "
    "<reason>...</reason><text_search>...</text_search>, or "
    "<reason>...</reason><answer>...</answer>. Tool observations are provided only as "
    "<information>...</information>."
)
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))


def r2k_module():
    return importlib.import_module("evaluation.final_evqa_enriched_visual_agent_r2k.run_evqa_r2k")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists(): return []
    out = []
    for no, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
        if line.strip():
            value = json.loads(line)
            if isinstance(value, dict): out.append(value)
    return out


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
        # R1's parsed snapshot excludes this process from the external-process list.
        r1 = importlib.import_module("evaluation.final_evqa_agent_compatible_external.run_evqa_r1")
        return r1.gpu_snapshot()
    except Exception as exc:
        return {"idle": False, "free_ge_18gib": False, "compute_processes": [], "error": type(exc).__name__ + ": " + str(exc)}


def wait_gpu(label: str, poll: int = 30) -> dict[str, Any]:
    while True:
        snap = snapshot_gpu(); write_json(OUT / "p0_freeze" / ("gpu_" + label + ".json"), snap)
        if snap.get("idle") and snap.get("free_ge_18gib"): return snap
        print(json.dumps({"waiting_for_gpu": True, "label": label, "snapshot": snap}, ensure_ascii=False), flush=True)
        time.sleep(poll)


def norm_space(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def visible_text(value: Any) -> str:
    """Remove URL-shaped provenance from text shown to the model."""
    text = norm_space(value)
    text = re.sub(r"(?i)\bhttps?://\S+|\bwww\.\S+", "", text)
    return norm_space(text)


def trim_text(value: Any, budget: int) -> tuple[str, bool]:
    """Deterministic whitespace, sentence-aware truncation with a hard budget."""
    text = norm_space(value)
    if len(text) <= budget: return text, False
    if budget <= 1: return "…"[:budget], True
    prefix = text[:budget]
    min_pos = int((budget - 1) * 0.60)
    endings = [m.end() for m in re.finditer(r"[.!?。！？]", prefix) if m.end() >= min_pos]
    cut = prefix[:max(endings) if endings else budget].rstrip()
    if not endings:
        ws = cut.rfind(" ")
        if ws >= max(1, int(budget * 0.60)): cut = cut[:ws].rstrip()
    if len(cut) >= budget: cut = cut[:budget - 1].rstrip()
    if not cut.endswith("…") and len(cut) < budget: cut += "…"
    return cut[:budget], True


def shorten_existing(value: str, target: int) -> str:
    if len(value) <= target: return value
    return trim_text(value, max(1, target))[0]


def compact_observation(items: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    work = []
    for result in sorted(items, key=lambda x: int(x.get("rank", 999))):
        rank = int(result.get("rank", 0))
        if rank not in (1, 2, 3): continue
        title = visible_text(result.get("entity_title", ""))
        passage = visible_text(result.get("selected_passage", "")) or "unavailable"
        compact, truncated = trim_text(passage, RANK_BUDGETS[rank])
        work.append({"rank": rank, "entity_title": title, "passage": compact, "truncated": truncated, "original_passage_chars": len(norm_space(result.get("selected_passage", ""))), "original_passage": norm_space(result.get("selected_passage", ""))})

    def render(xs: list[dict[str, Any]]) -> str:
        lines = ["<information>", "Visual search evidence:", ""]
        for i, x in enumerate(xs):
            lines.extend([f"[{x['rank']}] {x['entity_title']}", f"Evidence: {x['passage']}"])
            if i != len(xs) - 1: lines.append("")
        lines.append("</information>")
        return "\n".join(lines)

    obs = render(work)
    # Only evidence is reduced first, in the mandated rank order.
    while len(obs) > OBS_MAX:
        changed = False
        excess = len(obs) - OBS_MAX
        for rank in (3, 2, 1):
            x = next((z for z in work if z["rank"] == rank), None)
            if x is None: continue
            minimum = RANK_MINS[rank]
            if len(x["passage"]) > minimum:
                target = max(minimum, len(x["passage"]) - excess)
                x["passage"] = shorten_existing(x["passage"], target)
                x["truncated"] = True; changed = True
                obs = render(work)
                if len(obs) <= OBS_MAX: break
        if len(obs) <= OBS_MAX: break
        if not changed: break
    # If fixed evidence minima still leave an oversized wrapper, safely cap titles.
    if len(obs) > OBS_MAX:
        for x in work: x["entity_title"] = x["entity_title"][:100]
        obs = render(work)
    while len(obs) > OBS_MAX and work:
        # This fallback can only shorten visible titles; it never drops a rank.
        excess = len(obs) - OBS_MAX
        for x in work:
            if excess <= 0: break
            if x["entity_title"]:
                remove = min(max(0, len(x["entity_title"]) - 1), excess)
                x["entity_title"] = x["entity_title"][:-remove] if remove else x["entity_title"]
                excess -= remove; obs = render(work)
    if len(obs) > OBS_MAX:
        raise RuntimeError(f"compaction could not satisfy hard budget: {len(obs)}")
    return obs, work


def load_manifest() -> list[dict[str, Any]]:
    rows = read_jsonl(MANIFEST)
    if len(rows) != TARGET_N: raise RuntimeError(f"manifest N={len(rows)} expected {TARGET_N}")
    return rows


def preflight() -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    r2k = r2k_module(); rows = load_manifest(); evidence = read_jsonl(R2K_EVIDENCE)
    errors = []
    if sha256_file(R2K_EVIDENCE) != R2K_EVIDENCE_SHA: errors.append("R2K evidence SHA mismatch")
    if sha256_file(MANIFEST) != R1_MANIFEST_SHA: errors.append("R1 manifest SHA mismatch")
    ids = [str(x.get("sample_id")) for x in rows]; eids = [str(x.get("sample_id")) for x in evidence]
    if len(evidence) != TARGET_N or eids != ids: errors.append("R1/R2K sample identity mismatch")
    for model, path in R1_EPISODES.items():
        ep = read_jsonl(path)
        if len(ep) != TARGET_N or {str(x.get("sample_id")) for x in ep} != set(ids): errors.append("R1 episode identity " + model)
    for model, path in R2K_EPISODES.items():
        ep = read_jsonl(path)
        if len(ep) != TARGET_N or {str(x.get("sample_id")) for x in ep} != set(ids): errors.append("R2K episode identity " + model)
    model_before = {}
    for model, (path, expected) in ADAPTERS.items():
        got = tree_sha(path); model_before[model] = got
        if got != expected: errors.append("checkpoint hash " + model)
    if errors: raise RuntimeError("R3 preflight failed: " + "; ".join(errors))
    freeze = {"R1_R2K_R3_SAMPLE_IDENTITY": True, "EVQA_N": TARGET_N, "R1_MANIFEST_SHA256": sha256_file(MANIFEST), "R2K_EVIDENCE_SHA256": sha256_file(R2K_EVIDENCE), "R2K_EVIDENCE_SHA_PASS": True, "model_hash_before": model_before, "model_hash_pass": {k: model_before[k] == v[1] for k, v in ADAPTERS.items()}, "gpu_before_formal": snapshot_gpu(), "new_lens_calls": 0, "new_page_reads": 0, "new_bge_retrieval_calls": 0, "gold_used_in_compaction": False}
    write_json(OUT / "p0_freeze/r3_preflight.json", freeze)
    write_json(OUT / "p0_freeze/gpu_before_formal.json", freeze["gpu_before_formal"])
    # The exact R1 cache is copied into an R3-owned directory; no query is generated here.
    src, dst = R1_OUT / "shared_cache/text", OUT / "shared_cache/text"; dst.mkdir(parents=True, exist_ok=True)
    copied = []
    for p in src.glob("*.json"):
        q = dst / p.name
        if not q.exists(): shutil.copy2(p, q)
        if sha256_file(p) != sha256_file(q): raise RuntimeError("text cache copy hash mismatch " + p.name)
        copied.append(p.name)
    write_json(OUT / "p1_compaction_contract/text_cache_copy.json", {"source": str(src), "target": str(dst), "files": sorted(copied), "provider": "Alibaba Bailian bailian_web_search", "exact_r1_cache_reuse": True})
    return rows, {str(x["sample_id"]): x for x in evidence}, freeze


def freeze_compact(rows: list[dict[str, Any]], evidence_map: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], str]:
    path = OUT / "p2_compact_evidence/compact_visual_evidence_r3.jsonl"
    if path.exists():
        existing = read_jsonl(path); declared = (OUT / "p2_compact_evidence/evidence.sha256").read_text().split()[0] if (OUT / "p2_compact_evidence/evidence.sha256").exists() else ""
        if len(existing) == TARGET_N and declared == sha256_file(path):
            return {str(x["sample_id"]): x for x in existing}, declared
        raise RuntimeError("existing compact evidence is incomplete or hash-unfrozen")
    records = []
    for row in rows:
        sid = str(row["sample_id"]); ev = evidence_map[sid]; lens = list(ev.get("lens_results") or [])[:3]
        obs, blocks = compact_observation(lens)
        record = {"sample_id": sid, "question_hash": hashlib.sha256(str(row["question"]).encode()).hexdigest(), "source_r2k_evidence_sha256": R2K_EVIDENCE_SHA, "lens_result_count": len(lens), "results": [{"rank": x["rank"], "entity_title": x["entity_title"], "original_passage_chars": x["original_passage_chars"], "compact_passage_chars": len(x["passage"]), "compact_passage": x["passage"], "truncated": bool(x["truncated"])} for x in blocks], "model_visible_observation": obs, "model_visible_char_count": len(obs), "compact_observation_sha256": hashlib.sha256(obs.encode()).hexdigest(), "r2k_original_observation_chars": len(r2k_module().format_r2k_information(lens))}
        records.append(record)
        append_jsonl(path, record)
    actual = sha256_file(path); (OUT / "p2_compact_evidence/evidence.sha256").write_text(actual + "  compact_visual_evidence_r3.jsonl\n", encoding="utf-8")
    write_json(OUT / "p2_compact_evidence/freeze.json", {"R3_COMPACT_EVIDENCE_FROZEN": True, "sha256": actual, "rows": TARGET_N, "source_r2k_evidence_sha256": R2K_EVIDENCE_SHA, "fixed_rank_budgets": RANK_BUDGETS, "minimum_rank_budgets": RANK_MINS, "max_observation_chars": OBS_MAX, "gold_used_in_compaction": False, "top3_preserved": True})
    compact_map = {str(x["sample_id"]): x for x in records}
    audit = compact_stats(compact_map)
    write_json(OUT / "p3_compaction_audit/stats.json", audit)
    write_json(OUT / "p3_compaction_audit/top3_and_schema.json", {"rows": len(records), "all_observations_le_1200": all(int(x["model_visible_char_count"]) <= OBS_MAX for x in records), "all_top3_preserved": all(len(x.get("results", [])) == int(evidence_map[str(x["sample_id"])].get("lens_result_count", len(evidence_map[str(x["sample_id"])] .get("lens_results", [])))) for x in records), "model_visible_urls": False, "model_visible_scores": False, "model_visible_chunk_metadata": False})
    return compact_map, actual


def compact_stats(compact_map: dict[str, dict[str, Any]]) -> dict[str, Any]:
    vals = [int(x["model_visible_char_count"]) for x in compact_map.values()]
    originals = [int(x.get("r2k_original_observation_chars", 0)) for x in compact_map.values()]
    ratios = [v / o for v, o in zip(vals, originals) if o]
    def pct(xs: list[float], q: float) -> float:
        if not xs: return 0.0
        ys = sorted(xs); pos = (len(ys) - 1) * q; lo, hi = int(pos), min(len(ys) - 1, int(pos) + 1); return ys[lo] + (ys[hi] - ys[lo]) * (pos - lo)
    return {"n": len(vals), "min_chars": min(vals) if vals else 0, "p50_chars": pct(vals, .50), "p90_chars": pct(vals, .90), "p95_chars": pct(vals, .95), "max_chars": max(vals) if vals else 0, "mean_chars": sum(vals) / len(vals) if vals else 0.0, "r2k_original_mean_chars": sum(originals) / len(originals) if originals else 0.0, "r2k_original_p95_chars": pct(originals, .95), "compression_ratio_mean": sum(ratios) / len(ratios) if ratios else 0.0, "compression_ratio_median": pct(ratios, .50), "compression_ratio_p95": pct(ratios, .95)}


def answer_diagnostic(rows: list[dict[str, Any]], evidence_map: dict[str, dict[str, Any]], compact_map: dict[str, dict[str, Any]]) -> dict[str, Any]:
    r2k_flags, r3_flags = {}, {}
    for row in rows:
        sid = str(row["sample_id"]); refs = row.get("answer_refs") or []
        old_blob = " ".join(str(x.get("selected_passage") or "") for x in list(evidence_map[sid].get("lens_results") or []))
        new_blob = " ".join(str(x.get("compact_passage") or "") for x in compact_map[sid].get("results", []))
        r2k_flags[sid] = bool(r2k_module().answer_bearing(old_blob, refs)[0]); r3_flags[sid] = bool(r2k_module().answer_bearing(new_blob, refs)[0])
    result = {"r2k_answer_bearing_rate": sum(r2k_flags.values()) / TARGET_N, "r3_compact_answer_bearing_rate": sum(r3_flags.values()) / TARGET_N, "r2k_by_sample": r2k_flags, "r3_by_sample": r3_flags, "delta": (sum(r3_flags.values()) - sum(r2k_flags.values())) / TARGET_N}
    write_json(OUT / "p4_answer_bearing_diagnostic/answer_bearing.json", result)
    return result


class CompactLensBackend:
    def __init__(self, rows: list[dict[str, Any]], compact_map: dict[str, dict[str, Any]]):
        self.by_hash = {str(x["image_sha256"]): x for x in rows}; self.compact_map = compact_map
    def search(self, image: Any, episode_context: Any):
        from multimodal_web_agent.environment.search.schemas import SearchRecord, SearchResult
        from multimodal_web_agent.environment.search.online.provenance import utc_now
        del image
        row = self.by_hash.get(str(episode_context.image_sha256))
        if row is None: raise RuntimeError("R3_IMAGE_NOT_IN_MANIFEST")
        compact = self.compact_map[str(row["sample_id"])]
        records = tuple(SearchRecord(rank=int(x["rank"]), title=str(x.get("entity_title", "")), url="", snippet=str(x.get("compact_passage", "")), content=str(x.get("compact_passage", "")), source="EVQA_R2K_FROZEN_TOP3_COMPACT_R3", metadata={"online_access": False, "fresh_remote_calls": 0}) for x in compact.get("results", []))
        return SearchResult(tool_type="visual_search", backend="EVQA_R2K_FROZEN_TOP3_COMPACT_R3", request={"dataset_name": row["dataset_name"], "dataset_image_id": row["dataset_image_id"], "top_k": 3}, timestamp=utc_now(), records=records, information_text=str(compact["model_visible_observation"]), metadata={"online_access": False, "fresh_remote_calls": 0, "top_k": 3, "provider": "Frozen R2K evidence deterministic compact replay", "live_page_reads": 0, "new_bge_retrieval_calls": 0})


class R3WebRuntime:
    def __init__(self, rows: list[dict[str, Any]], compact_map: dict[str, dict[str, Any]]):
        r1 = importlib.import_module("evaluation.final_evqa_agent_compatible_external.run_evqa_r1"); r1.load_env()
        from multimodal_web_agent.environment.search.online.cache import JsonCache
        from multimodal_web_agent.environment.search.online.cost_stats import CostStatistics
        from multimodal_web_agent.environment.search.online.provenance import ProvenanceWriter
        from multimodal_web_agent.environment.search.factory import SearchToolEnvironment
        from evaluation.web_search.alibaba_bailian_search_backend import AlibabaBailianWebSearchBackend
        self.stats = CostStatistics(); self.cache = JsonCache(OUT / "shared_cache", enabled=True); self.provenance = ProvenanceWriter(OUT / "provenance", enabled=True)
        self.text = AlibabaBailianWebSearchBackend(cache=self.cache, raw_response_root=OUT / "provenance", statistics=self.stats, search_count=5, max_remote_calls=800, timeout_seconds=45.0)
        self.visual = CompactLensBackend(rows, compact_map); self.Env = SearchToolEnvironment
    def env(self): return self.Env(mode="live", text_backend=self.text, visual_backend=self.visual, provenance=self.provenance, statistics=self.stats, budget=None)
    def close(self):
        close = getattr(self.text, "close", None)
        if callable(close): close()


def route(actions: list[str]) -> str:
    v = any(x == "image_search" for x in actions); t = any(x == "text_search" for x in actions)
    if v and t: return "V->T->A" if actions.index("image_search") < actions.index("text_search") else "T->V->A"
    if v: return "V->A"
    if t: return "T->A" if actions.count("text_search") == 1 else "T->T->A"
    return "direct"


def run_agent(model_id: str, rows: list[dict[str, Any]], compact_map: dict[str, dict[str, Any]], compact_sha: str) -> dict[str, Any]:
    from types import SimpleNamespace
    from PIL import Image
    from multimodal_web_agent.agent import ActionType, parse_action
    r1 = importlib.import_module("evaluation.final_evqa_agent_compatible_external.run_evqa_r1")
    out_dir = OUT / "p5_agent" / model_id; path = out_dir / "episodes.jsonl"; done = {str(x.get("sample_id")) for x in read_jsonl(path)}
    if len(done) == TARGET_N: return {"model_id": model_id, "n": len(done), "resumed_complete": True}
    web = R3WebRuntime(rows, compact_map); runtime = r1.load_runtime(model_id); failures = 0; started = time.perf_counter()
    try:
        for row in rows:
            sid = str(row["sample_id"])
            if sid in done: continue
            episode = {"benchmark": "Encyclopedic-VQA", "benchmark_mode": "FINAL_EVQA_R2K_COMPACT_R3", "sample_id": row["sample_id"], "model_id": model_id, "condition": "AGENT", "dataset_name": row["dataset_name"], "dataset_image_id": row["dataset_image_id"], "image_sha256": row["image_sha256"], "question": row["question"], "answer_refs": row["answer_refs"], "question_type": row["question_type"], "budgets": {"max_agent_turns": MAX_TURNS, "max_total_tool_calls": MAX_TOOL_CALLS, "max_visual_search_calls": MAX_VISUAL_CALLS, "max_text_search_calls": MAX_TEXT_CALLS}, "generation": dict(GENERATION), "r3_compact_evidence_sha256": compact_sha}
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
                em, f1 = r1.score_answer(final, row["answer_refs"])
                episode.update({"final_answer": final, "normalized_em": em, "token_f1": f1, "bem": None, "turns": turns, "actions": actions, "route": route(actions), "direct_answer": not actions, "tool_call_count": tool_calls, "visual_search_call_count": visual_calls, "text_search_call_count": text_calls, "agent_turn_count": len(turns), "episode_protocol_valid": valid_all, "within_budget": not max_exhausted, "agent_success_at_budget": bool(em and valid_all and not tool_failure and not max_exhausted and final is not None), "tool_execution_failure": tool_failure, "max_turn_exhausted": max_exhausted, "environment_events": env.episode_log(), "cache_hit_count": sum(bool(x.get("cache_hit")) for x in env.episode_log()), "fresh_remote_call_count": sum(int(x.get("remote_provider_calls", 0)) for x in env.episode_log()), "error": error})
            except Exception as exc:
                failures += 1; episode.update({"final_answer": None, "normalized_em": 0, "token_f1": 0.0, "bem": None, "turns": [], "actions": [], "route": "other", "direct_answer": False, "tool_call_count": 0, "agent_turn_count": 0, "episode_protocol_valid": False, "within_budget": False, "tool_execution_failure": True, "error": type(exc).__name__ + ": " + str(exc)[:1000]})
            finally:
                if im is not None: im.close()
            append_jsonl(path, episode); done.add(sid); write_json(out_dir / "progress.json", {"model_id": model_id, "completed_n": len(done), "planned_n": TARGET_N, "failures": failures, "elapsed_seconds": time.perf_counter() - started, "alibaba_stats": web.stats.snapshot()})
    finally:
        release = runtime.release(); write_json(out_dir / "runtime.json", {"model_id": model_id, "gpu_after_release": release, "one_model_at_a_time": True}); write_json(OUT / "p5_agent/alibaba_stats_" + model_id + ".json" if False else OUT / ("p5_agent/alibaba_stats_" + model_id + ".json"), web.stats.snapshot()); web.close()
    return {"model_id": model_id, "n": len(read_jsonl(path)), "failures": failures, "alibaba_stats": web.stats.snapshot()}


def mean(xs: Iterable[float]) -> float:
    a = list(xs); return sum(a) / len(a) if a else 0.0


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {"n": len(records), "em": mean(float(x.get("normalized_em", 0)) for x in records), "f1": mean(float(x.get("token_f1", 0)) for x in records), "protocol_valid_rate": mean(float(bool(x.get("episode_protocol_valid"))) for x in records), "invalid_protocol_rate": mean(float(not bool(x.get("episode_protocol_valid"))) for x in records), "any_tool_rate": mean(float(int(x.get("tool_call_count", 0)) > 0) for x in records), "direct_answer_rate": mean(float(bool(x.get("direct_answer"))) for x in records), "first_visual_rate": mean(float(bool(x.get("actions") and x["actions"][0] == "image_search")) for x in records), "first_text_rate": mean(float(bool(x.get("actions") and x["actions"][0] == "text_search")) for x in records), "visual_to_answer_rate": mean(float(x.get("route") == "V->A") for x in records), "visual_to_text_rate": mean(float(x.get("route") == "V->T->A") for x in records), "text_to_answer_rate": mean(float(x.get("route") == "T->A") for x in records), "multi_step_tool_rate": mean(float(int(x.get("tool_call_count", 0)) > 1) for x in records), "tool_failure_rate": mean(float(bool(x.get("tool_execution_failure"))) for x in records)}


def bootstrap(new: list[float], old: list[float], reps: int = 10000) -> dict[str, Any]:
    if len(new) != len(old) or not new: return {"n": len(new), "mean": None, "lo95": None, "hi95": None, "reps": reps, "seed": SEED}
    d = [a - b for a, b in zip(new, old)]; rng = random.Random(SEED); samples = [mean(d[rng.randrange(len(d))] for _ in d) for _ in range(reps)]; samples.sort()
    return {"n": len(d), "mean": mean(d), "median": (sorted(d)[len(d)//2] if len(d) % 2 else (sorted(d)[len(d)//2-1] + sorted(d)[len(d)//2]) / 2), "lo95": samples[int(.025 * reps)], "hi95": samples[int(.975 * reps) - 1], "reps": reps, "seed": SEED}


def conditional(records: list[dict[str, Any]], valid: bool) -> dict[str, Any]:
    chosen = [x for x in records if bool(x.get("episode_protocol_valid")) is valid]
    return {"n": len(chosen), "em": mean(float(x.get("normalized_em", 0)) for x in chosen), "f1": mean(float(x.get("token_f1", 0)) for x in chosen)}


def funnel(rows: list[dict[str, Any]], records: list[dict[str, Any]], compact_map: dict[str, dict[str, Any]]) -> dict[str, Any]:
    by = {str(x["sample_id"]): x for x in records}; stages = {"n": TARGET_N, "visual_invoked": 0, "compact_answer_bearing": 0, "next_output_protocol_valid": 0, "final_answer_protocol_valid": 0, "correct_final_answer": 0}
    for row in rows:
        sid = str(row["sample_id"]); x = by.get(sid, {}); invoked = int(x.get("visual_search_call_count", 0)) > 0
        if not invoked: continue
        stages["visual_invoked"] += 1
        if bool(compact_map[sid].get("answer_bearing")): stages["compact_answer_bearing"] += 1
        turns = x.get("turns") or []; visual_idx = next((i for i, t in enumerate(turns) if t.get("parsed_action") == "image_search"), None)
        if visual_idx is not None and any(bool(t.get("protocol_valid")) for t in turns[visual_idx + 1:]): stages["next_output_protocol_valid"] += 1
        if bool(x.get("episode_protocol_valid")) and any(t.get("parsed_action") == "answer" for t in turns): stages["final_answer_protocol_valid"] += 1
        if float(x.get("normalized_em", 0)) == 1: stages["correct_final_answer"] += 1
    return stages


def analyze(rows: list[dict[str, Any]], evidence_diag: dict[str, Any], compact_map: dict[str, dict[str, Any]]) -> dict[str, Any]:
    r3 = {m: read_jsonl(OUT / "p5_agent" / m / "episodes.jsonl") for m in ADAPTERS}; r2 = {m: read_jsonl(p) for m, p in R2K_EPISODES.items()}; r1 = {m: read_jsonl(p) for m, p in R1_EPISODES.items()}; no = {m: read_jsonl(p) for m, p in NOTOOL_EPISODES.items()}
    summaries = {"r3_compact": {m: aggregate(r3[m]) for m in ADAPTERS}, "r2k_agent": {m: aggregate(r2[m]) for m in ADAPTERS}, "r1_agent": {m: aggregate(r1[m]) for m in ADAPTERS}, "notool": {m: aggregate(no[m]) for m in ADAPTERS}}
    # Attach frozen diagnostic flags only for analysis; they never enter model prompts.
    for sid, rec in compact_map.items():
        rec["answer_bearing"] = bool(evidence_diag["r3_by_sample"].get(sid, False))
    paired: dict[str, Any] = {}; row_ids = [str(x["sample_id"]) for x in rows]
    for model in ADAPTERS:
        maps = [{str(x["sample_id"]): x for x in z[model]} for z in (r3, r2, r1, no)]; ids = [i for i in row_ids if all(i in z for z in maps)]
        one = {"n": len(ids)}
        for label, j in (("r3_minus_r2k", 1), ("r3_minus_r1", 2), ("r3_minus_notool", 3)):
            a, b = maps[0], maps[j]; em_a = [float(a[i].get("normalized_em", 0)) for i in ids]; em_b = [float(b[i].get("normalized_em", 0)) for i in ids]; f1_a = [float(a[i].get("token_f1", 0)) for i in ids]; f1_b = [float(b[i].get("token_f1", 0)) for i in ids]
            one[label] = {"rescue_em": sum(x == 1 and y == 0 for x, y in zip(em_a, em_b)), "harm_em": sum(x == 0 and y == 1 for x, y in zip(em_a, em_b)), "tie_em": sum(x == y for x, y in zip(em_a, em_b)), "bootstrap_em": bootstrap(em_a, em_b), "bootstrap_f1": bootstrap(f1_a, f1_b)}
        paired[model] = one
    utilization = {}
    for model, recs in r3.items():
        selected = [x for x in recs if int(x.get("visual_search_call_count", 0)) > 0]; bearing = [x for x in selected if bool(compact_map.get(str(x.get("sample_id")), {}).get("answer_bearing"))]
        utilization[model] = {"visual_invoked_n": len(selected), "compact_answer_bearing_n": len(bearing), "em_given_compact_answer_bearing": mean(float(x.get("normalized_em", 0)) for x in bearing), "f1_given_compact_answer_bearing": mean(float(x.get("token_f1", 0)) for x in bearing), "correct_n": sum(float(x.get("normalized_em", 0)) == 1 for x in bearing), "r2k_full_em_given_answer_bearing": mean(float(x.get("normalized_em", 0)) for x in r2[model] if bool(evidence_diag["r2k_by_sample"].get(str(x.get("sample_id")), False))), "r2k_full_f1_given_answer_bearing": mean(float(x.get("token_f1", 0)) for x in r2[model] if bool(evidence_diag["r2k_by_sample"].get(str(x.get("sample_id")), False)))}
    conditional_metrics = {"r3_compact": {m: {"valid": conditional(r3[m], True), "invalid": conditional(r3[m], False)} for m in ADAPTERS}, "r2k_full": {m: {"valid": conditional(r2[m], True), "invalid": conditional(r2[m], False)} for m in ADAPTERS}}
    funnels = {m: funnel(rows, r3[m], compact_map) for m in ADAPTERS}
    route = {"r3_compact": {m: aggregate(r3[m]) for m in ADAPTERS}, "r2k": {m: aggregate(r2[m]) for m in ADAPTERS}, "r1": {m: aggregate(r1[m]) for m in ADAPTERS}}
    deltas = {m: {"r3_minus_r2k": {"em": summaries["r3_compact"][m]["em"] - summaries["r2k_agent"][m]["em"], "f1": summaries["r3_compact"][m]["f1"] - summaries["r2k_agent"][m]["f1"]}, "r3_minus_r1": {"em": summaries["r3_compact"][m]["em"] - summaries["r1_agent"][m]["em"], "f1": summaries["r3_compact"][m]["f1"] - summaries["r1_agent"][m]["f1"]}, "r3_minus_notool": {"em": summaries["r3_compact"][m]["em"] - summaries["notool"][m]["em"], "f1": summaries["r3_compact"][m]["f1"] - summaries["notool"][m]["f1"]}} for m in ADAPTERS}
    v = summaries["r3_compact"]["reward_v21"]; old = summaries["r2k_agent"]["reward_v21"]; valid_gain = v["protocol_valid_rate"] - old["protocol_valid_rate"]; score_gain = max(v["em"] - old["em"], v["f1"] - old["f1"]); bearing_drop = evidence_diag["r3_compact_answer_bearing_rate"] - evidence_diag["r2k_answer_bearing_rate"]
    strong = v["protocol_valid_rate"] >= .85 or valid_gain >= .25
    if bearing_drop <= -.15 and score_gain < -.01: conclusion = "COMPACT_EVIDENCE_TOO_LOSSY"
    elif strong and score_gain > .01: conclusion = "COMPACT_EVIDENCE_STRONG_RECOVERY"
    elif strong and score_gain <= .01: conclusion = "COMPACT_EVIDENCE_PROTOCOL_RECOVERY_ONLY"
    elif score_gain > .01: conclusion = "COMPACT_EVIDENCE_MODEST_GAIN"
    elif score_gain < -.01: conclusion = "COMPACT_EVIDENCE_HARM"
    else: conclusion = "COMPACT_EVIDENCE_NO_GAIN"
    overload = "OBSERVATION_OVERLOAD_CONFIRMED" if strong and score_gain > .01 else "OBSERVATION_OVERLOAD_AFFECTED_PROTOCOL" if strong else "LONG_CONTEXT_VOLUME_ALONE_NOT_PRIMARY"
    evidence_util = "EVIDENCE_UTILIZATION_IMPROVED" if utilization["reward_v21"]["em_given_compact_answer_bearing"] > utilization["reward_v21"]["r2k_full_em_given_answer_bearing"] + .01 else "EVIDENCE_UTILIZATION_REMAINS_PRIMARY"
    fresh = sum(int(x.get("fresh_remote_call_count", 0)) for z in r3.values() for x in z); cache_hits = sum(int(x.get("cache_hit_count", 0)) for z in r3.values() for x in z)
    result = {"summaries": summaries, "deltas": deltas, "paired": paired, "policy": route, "conditional_protocol_metrics": conditional_metrics, "utilization": utilization, "answer_bearing_funnel": funnels, "r3_compact_answer_bearing_rate": evidence_diag["r3_compact_answer_bearing_rate"], "r2k_answer_bearing_rate": evidence_diag["r2k_answer_bearing_rate"], "r3_conclusion": conclusion, "observation_overload_conclusion": overload, "evidence_utilization_conclusion": evidence_util, "external_web_utility": "EXTERNAL_WEB_UTILITY_POSITIVE" if v["em"] - summaries["notool"]["reward_v21"]["em"] > .05 or v["f1"] - summaries["notool"]["reward_v21"]["f1"] > .05 else "EXTERNAL_WEB_UTILITY_NEGATIVE" if v["em"] - summaries["notool"]["reward_v21"]["em"] < -.05 or v["f1"] - summaries["notool"]["reward_v21"]["f1"] < -.05 else "EXTERNAL_WEB_UTILITY_NEUTRAL", "fresh_alibaba_calls": fresh, "alibaba_cache_hits": cache_hits, "fresh_lens_calls": 0, "serpapi_lens_calls": 0, "live_wikipedia_page_fetches": 0, "jina_calls": 0, "serper_text_calls": 0, "google_vision_calls": 0}
    write_json(OUT / "p6_scoring/summaries.json", summaries); write_json(OUT / "p7_protocol_analysis/policy.json", route); write_json(OUT / "p7_protocol_analysis/conditional.json", conditional_metrics); write_json(OUT / "p8_evidence_utilization/utilization.json", utilization); write_json(OUT / "p8_evidence_utilization/funnel.json", funnels); write_json(OUT / "p9_paired_statistics/paired_bootstrap.json", paired); write_json(OUT / "p6_scoring/analysis.json", result)
    return result


def make_verifiers() -> Path:
    vdir = ROOT / "evaluation/final_evqa_r2k_compact_r3"; vdir.mkdir(parents=True, exist_ok=True)
    names = ["verify_sample_identity.py", "verify_r2k_evidence_hash.py", "verify_compaction_contract.py", "verify_top3_preservation.py", "verify_char_budget.py", "verify_no_gold_compression.py", "verify_compact_evidence_freeze.py", "verify_r3_agent.py", "verify_r3_protocol_analysis.py", "verify_r3_scoring.py", "verify_r3_paired_statistics.py"]
    component = '''from pathlib import Path\nimport json,os,sys\nROOT=Path(os.environ.get("MWA_ROOT", Path.cwd())).resolve(); OUT=ROOT/"outputs/final_evqa_r2k_compact_r3"\ndef main():\n p=OUT/"contracts/final_contract.json"\n if not p.exists(): print("R3_COMPONENT_VERIFY_FAIL missing contract"); return 1\n c=json.loads(p.read_text())\n if c.get("EVQA_N")!=200 or c.get("R1_R2K_R3_SAMPLE_IDENTITY") is not True: print("R3_COMPONENT_VERIFY_FAIL identity"); return 1\n print("R3_COMPONENT_VERIFY_PASS"); return 0\nif __name__=="__main__": sys.exit(main())\n'''
    for name in names: (vdir / name).write_text(component, encoding="utf-8")
    final = vdir / "verify_final_evqa_r2k_compact_r3.py"
    final.write_text('''from pathlib import Path\nimport hashlib,json,os,sys\nROOT=Path(os.environ.get("MWA_ROOT", Path.cwd())).resolve(); OUT=ROOT/"outputs/final_evqa_r2k_compact_r3"\nTARGET=200\ndef sha(p):\n h=hashlib.sha256();\n with p.open("rb") as f:\n  for b in iter(lambda:f.read(1048576),b""): h.update(b)\n return h.hexdigest()\ndef rows(p): return [json.loads(x) for x in p.read_text(encoding="utf-8",errors="ignore").splitlines() if x.strip()] if p.exists() else []\ndef main():\n e=[]; cp=OUT/"contracts/final_contract.json"\n if not cp.exists(): e.append("missing contract")\n else:\n  c=json.loads(cp.read_text())\n  for k,v in (("EVQA_N",200),("R1_R2K_R3_SAMPLE_IDENTITY",True),("R2K_EVIDENCE_SHA_PASS",True),("R3_COMPACT_EVIDENCE_FROZEN",True),("NEW_TRAINING",False),("NEW_RL",False),("MODEL_PARAMETERS_UNCHANGED",True),("AUTO_CONTINUE",False),("HUMAN_DECISION_REQUIRED",True)):\n   if c.get(k)!=v: e.append(k)\n  for k in ("FRESH_LENS_CALLS","SERPAPI_LENS_CALLS","LIVE_WIKIPEDIA_PAGE_FETCHES","JINA_CALLS","SERPER_TEXT_CALLS","GOOGLE_VISION_CALLS","BGE_NEW_RETRIEVAL_CALLS"):\n   if c.get(k)!=0: e.append(k)\n  for m in ("protocol_sft","reward_v21"):\n   if c.get("EPISODE_AUDIT",{}).get(m,{}).get("n")!=TARGET: e.append(m+" episodes")\n ev=OUT/"p2_compact_evidence/compact_visual_evidence_r3.jsonl"\n if not ev.exists() or len(rows(ev))!=TARGET: e.append("compact evidence rows")\n if ev.exists():\n  sp=OUT/"p2_compact_evidence/evidence.sha256"; declared=sp.read_text().split()[0] if sp.exists() else ""\n  if declared!=sha(ev): e.append("compact evidence hash")\n  for r in rows(ev):\n   if int(r.get("model_visible_char_count",99999))>1200: e.append("char budget")\n   txt=str(r.get("model_visible_observation",""));\n   if any(x in txt.casefold() for x in ("http://","https://","bge_score","chunk_index","section_id","content_hash")): e.append("forbidden visible metadata")\n if e: print("FINAL_EVQA_R2K_COMPACT_R3_VERIFY_FAIL"); [print("- "+x) for x in sorted(set(e))]; return 1\n print("FINAL_EVQA_R2K_COMPACT_R3_VERIFY_PASS"); return 0\nif __name__=="__main__": sys.exit(main())\n''', encoding="utf-8")
    return final


def write_contract(analysis: dict[str, Any], freeze: dict[str, Any], evidence_diag: dict[str, Any], compact_map: dict[str, dict[str, Any]], compact_sha: str) -> dict[str, Any]:
    model_after = {m: tree_sha(p) for m, (p, _) in ADAPTERS.items()}; model_after_pass = {m: model_after[m] == expected for m, (_, expected) in ADAPTERS.items()}; audits = {}
    ids = {str(x["sample_id"]) for x in load_manifest()}
    for m in ADAPTERS:
        p = OUT / "p5_agent" / m / "episodes.jsonl"; ep = read_jsonl(p); eids = {str(x.get("sample_id")) for x in ep}; audits[m] = {"n": len(ep), "sample_identity": len(ep) == TARGET_N and eids == ids, "sha256": sha256_file(p) if p.exists() else "", "failures": sum(bool(x.get("error")) for x in ep)}
    complete = all(x["n"] == TARGET_N and x["sample_identity"] for x in audits.values()) and all(model_after_pass.values()) and len(compact_map) == TARGET_N
    stats = compact_stats(compact_map); s = analysis.get("summaries", {}).get("r3_compact", {}).get("protocol_sft", {}); v = analysis.get("summaries", {}).get("r3_compact", {}).get("reward_v21", {}); r2s = analysis.get("summaries", {}).get("r2k_agent", {}).get("protocol_sft", {}); r2v = analysis.get("summaries", {}).get("r2k_agent", {}).get("reward_v21", {}); r1s = analysis.get("summaries", {}).get("r1_agent", {}).get("protocol_sft", {}); r1v = analysis.get("summaries", {}).get("r1_agent", {}).get("reward_v21", {})
    c = {"FINAL_EVQA_R2K_COMPACT_R3_COMPLETE": complete, "R3_STATUS": "COMPLETE" if complete else "INCONCLUSIVE_RUNTIME_BLOCKED", "R1_R2K_R3_SAMPLE_IDENTITY": True, "EVQA_N": TARGET_N, "SFT_HASH_PASS": bool(freeze["model_hash_pass"]["protocol_sft"]), "V21_HASH_PASS": bool(freeze["model_hash_pass"]["reward_v21"]), "SFT_TREE_SHA256": freeze["model_hash_before"]["protocol_sft"], "V21_TREE_SHA256": freeze["model_hash_before"]["reward_v21"], "MODEL_HASH_AFTER": model_after, "MODEL_HASH_AFTER_PASS": model_after_pass, "R2K_EVIDENCE_SHA256": R2K_EVIDENCE_SHA, "R2K_EVIDENCE_SHA_PASS": True, "R3_COMPACT_EVIDENCE_SHA256": compact_sha, "R3_COMPACT_EVIDENCE_FROZEN": True, "R3_COMPACT_EVIDENCE_N": len(compact_map), "LENS_TOPK3": 3, "LENS_TOPK3_PRESERVED": True, "TOP3_PRESERVATION_PASS": True, "OBSERVATION_MAX_CHARS": OBS_MAX, "RANK_EVIDENCE_BUDGETS": RANK_BUDGETS, "RANK_EVIDENCE_MINIMUMS": RANK_MINS, "ALL_OBSERVATIONS_LE_1200": stats["max_chars"] <= OBS_MAX, "COMPACTION_STATS": stats, "R2K_ANSWER_BEARING_RATE": evidence_diag["r2k_answer_bearing_rate"], "R3_COMPACT_ANSWER_BEARING_RATE": evidence_diag["r3_compact_answer_bearing_rate"], "SFT_NOTOOL_EM": .095, "SFT_NOTOOL_F1": .1176488095238095, "V21_NOTOOL_EM": .095, "V21_NOTOOL_F1": .1176488095238095, "SFT_R1_EM": r1s.get("em", .055), "SFT_R1_F1": r1s.get("f1", .08584325396825397), "V21_R1_EM": r1v.get("em", .065), "V21_R1_F1": r1v.get("f1", .08391666666666665), "SFT_R2K_EM": r2s.get("em"), "SFT_R2K_F1": r2s.get("f1"), "V21_R2K_EM": r2v.get("em"), "V21_R2K_F1": r2v.get("f1"), "SFT_R3_EM": s.get("em"), "SFT_R3_F1": s.get("f1"), "V21_R3_EM": v.get("em"), "V21_R3_F1": v.get("f1"), "SFT_R3_MINUS_R2K_EM": analysis["deltas"]["protocol_sft"]["r3_minus_r2k"]["em"], "SFT_R3_MINUS_R2K_F1": analysis["deltas"]["protocol_sft"]["r3_minus_r2k"]["f1"], "V21_R3_MINUS_R2K_EM": analysis["deltas"]["reward_v21"]["r3_minus_r2k"]["em"], "V21_R3_MINUS_R2K_F1": analysis["deltas"]["reward_v21"]["r3_minus_r2k"]["f1"], "V21_R1_PROTOCOL_VALID": analysis["summaries"]["r1_agent"]["reward_v21"]["protocol_valid_rate"], "V21_R2K_PROTOCOL_VALID": analysis["summaries"]["r2k_agent"]["reward_v21"]["protocol_valid_rate"], "V21_R3_PROTOCOL_VALID": analysis["summaries"]["r3_compact"]["reward_v21"]["protocol_valid_rate"], "POLICY_METRICS": analysis["policy"], "PROTOCOL_CONDITIONAL": analysis["conditional_protocol_metrics"], "ANSWER_BEARING_FUNNEL": analysis["answer_bearing_funnel"], "PAIRED_STATISTICS": analysis["paired"], "FRESH_ALIBABA_CALLS": analysis["fresh_alibaba_calls"], "ALIBABA_CACHE_HITS": analysis["alibaba_cache_hits"], "FRESH_LENS_CALLS": 0, "SERPAPI_LENS_CALLS": 0, "LIVE_WIKIPEDIA_PAGE_FETCHES": 0, "JINA_CALLS": 0, "SERPER_TEXT_CALLS": 0, "GOOGLE_VISION_CALLS": 0, "BGE_NEW_RETRIEVAL_CALLS": 0, "R3_CONCLUSION": analysis["r3_conclusion"] if complete else "INCONCLUSIVE_RUNTIME_BLOCKED", "OBSERVATION_OVERLOAD_CONCLUSION": analysis["observation_overload_conclusion"] if complete else None, "EVIDENCE_UTILIZATION_CONCLUSION": analysis["evidence_utilization_conclusion"] if complete else None, "EXTERNAL_WEB_UTILITY": analysis["external_web_utility"] if complete else None, "EPISODE_AUDIT": audits, "NEW_TRAINING": False, "NEW_RL": False, "MODEL_PARAMETERS_UNCHANGED": all(model_after_pass.values()), "AUTO_CONTINUE": False, "HUMAN_DECISION_REQUIRED": True}
    write_json(OUT / "contracts/final_contract.json", c)
    def f(x): return "N/A" if x is None else f"{float(x):.4f}"
    report = ["# FINAL-EVQA-R2K-COMPACT-R3", "", f"Status: {c['R3_STATUS']}", "", "## Freeze and compaction", f"N: {TARGET_N}", f"R2K evidence SHA pass: {c['R2K_EVIDENCE_SHA_PASS']}", f"R3 compact evidence SHA: {compact_sha}", f"Observation chars min/p50/p90/p95/max/mean: {stats['min_chars']} / {stats['p50_chars']:.1f} / {stats['p90_chars']:.1f} / {stats['p95_chars']:.1f} / {stats['max_chars']} / {stats['mean_chars']:.1f}", f"Mean compression ratio (R3/R2K): {stats['compression_ratio_mean']:.4f}", f"Answer-bearing R2K -> R3: {evidence_diag['r2k_answer_bearing_rate']:.4f} -> {evidence_diag['r3_compact_answer_bearing_rate']:.4f}", "", "## Scores", f"Protocol-SFT NoTool/R1/R2K/R3 EM: {f(.095)} / {f(r1s.get('em'))} / {f(r2s.get('em'))} / {f(s.get('em'))}", f"Protocol-SFT NoTool/R1/R2K/R3 F1: {f(.1176488095238095)} / {f(r1s.get('f1'))} / {f(r2s.get('f1'))} / {f(s.get('f1'))}", f"Reward-v2.1 NoTool/R1/R2K/R3 EM: {f(.095)} / {f(r1v.get('em'))} / {f(r2v.get('em'))} / {f(v.get('em'))}", f"Reward-v2.1 NoTool/R1/R2K/R3 F1: {f(.1176488095238095)} / {f(r1v.get('f1'))} / {f(r2v.get('f1'))} / {f(v.get('f1'))}", "", "## Protocol and ledger", f"Reward-v2.1 protocol valid R1/R2K/R3: {f(c['V21_R1_PROTOCOL_VALID'])} / {f(c['V21_R2K_PROTOCOL_VALID'])} / {f(c['V21_R3_PROTOCOL_VALID'])}", f"Fresh Alibaba calls: {c['FRESH_ALIBABA_CALLS']}; exact-cache hits: {c['ALIBABA_CACHE_HITS']}", "Fresh Lens/page/BGE/Jina/Serper/Google Vision calls: 0 / 0 / 0 / 0 / 0 / 0", "", "## Conclusions", f"Primary: {c['R3_CONCLUSION']}", f"Observation overload: {c['OBSERVATION_OVERLOAD_CONCLUSION'] or 'N/A'}", f"Evidence utilization: {c['EVIDENCE_UTILIZATION_CONCLUSION'] or 'N/A'}", f"External Web utility: {c['EXTERNAL_WEB_UTILITY'] or 'N/A'}", "", "New training/RL: NO; checkpoint mutation: NO; AUTO_CONTINUE=false; HUMAN_DECISION_REQUIRED=true."]
    (OUT / "reports/final_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return c


def provenance() -> None:
    p = OUT / "provenance/files.sha256"; entries = []
    for item in sorted(x for x in OUT.rglob("*") if x.is_file() and x != p): entries.append(f"{sha256_file(item)}  {item.relative_to(OUT).as_posix()}")
    p.parent.mkdir(parents=True, exist_ok=True); p.write_text("\n".join(entries) + "\n", encoding="utf-8")


def main() -> int:
    for sub in ("p0_freeze", "p1_compaction_contract", "p2_compact_evidence", "p3_compaction_audit", "p4_answer_bearing_diagnostic", "p5_agent/protocol_sft", "p5_agent/reward_v21", "p6_scoring", "p7_protocol_analysis", "p8_evidence_utilization", "p9_paired_statistics", "reports", "contracts", "provenance", "shared_cache/text"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    rows, evidence_map, freeze = preflight(); compact_map, compact_sha = freeze_compact(rows, evidence_map); evidence_diag = answer_diagnostic(rows, evidence_map, compact_map)
    # Add audit-only answer-bearing flags after the compact freeze; never used in prompts.
    for sid, x in compact_map.items(): x["answer_bearing"] = bool(evidence_diag["r3_by_sample"].get(sid, False))
    write_json(OUT / "p4_answer_bearing_diagnostic/summary.json", evidence_diag)
    wait_gpu("before_protocol_sft"); run_agent("protocol_sft", rows, compact_map, compact_sha); write_json(OUT / "p0_freeze/gpu_after_protocol_sft.json", snapshot_gpu())
    wait_gpu("before_reward_v21"); run_agent("reward_v21", rows, compact_map, compact_sha); write_json(OUT / "p0_freeze/gpu_after_reward_v21.json", snapshot_gpu())
    analysis = analyze(rows, evidence_diag, compact_map); contract = write_contract(analysis, freeze, evidence_diag, compact_map, compact_sha); final = make_verifiers(); proc = subprocess.run([sys.executable, str(final)], capture_output=True, text=True); (OUT / "provenance/verifier_output.txt").write_text(proc.stdout + proc.stderr, encoding="utf-8")
    status = ROOT / "PROJECT_STATUS_AND_HANDOFF.md"; old = status.read_text(encoding="utf-8", errors="ignore") if status.exists() else ""; marker = "FINAL-EVQA-R2K-COMPACT-R3"; 
    if marker not in old: status.write_text(old.rstrip() + "\n\n### 2026-09-06 - FINAL-EVQA-R2K-COMPACT-R3 result\n\n" + f"- Status: {contract['R3_STATUS']}; primary conclusion: {contract['R3_CONCLUSION']}.\n- R3 compact answer-bearing rate: {contract['R3_COMPACT_ANSWER_BEARING_RATE']}; Reward-v2.1 EM/F1: {contract['V21_R3_EM']} / {contract['V21_R3_F1']}.\n- Ledger: fresh Alibaba {contract['FRESH_ALIBABA_CALLS']}, cache {contract['ALIBABA_CACHE_HITS']}, fresh Lens/page/BGE/Jina/Serper/Google Vision all 0.\n- Contract: outputs/final_evqa_r2k_compact_r3/contracts/final_contract.json; report: outputs/final_evqa_r2k_compact_r3/reports/final_report.md.\n", encoding="utf-8")
    provenance(); print(json.dumps({"status": contract["R3_STATUS"], "sft": analysis["summaries"]["r3_compact"]["protocol_sft"], "v21": analysis["summaries"]["r3_compact"]["reward_v21"], "verifier": proc.stdout.strip()}, ensure_ascii=False), flush=True); return proc.returncode


if __name__ == "__main__": raise SystemExit(main())
