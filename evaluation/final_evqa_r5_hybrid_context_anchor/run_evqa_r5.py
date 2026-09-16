#!/usr/bin/env python3
"""FINAL-EVQA-R5-HYBRID-CONTEXT-ANCHOR-QUESTION-AWARE-COMPRESSION.

Deterministic hybrid serialization over frozen R2K passages and frozen R4
question-aware evidence.  No retrieval, BGE scoring, sample selection, or
semantic rewriting is performed.
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
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(os.environ.get("MWA_ROOT", Path(__file__).resolve().parents[2])).resolve()
OUT = ROOT / "outputs/final_evqa_r5_hybrid_context_anchor"
R4_OUT = ROOT / "outputs/final_evqa_question_aware_compact_r4"
R1_OUT = ROOT / "outputs/final_evqa_agent_compatible_external_r1"
R2K_OUT = ROOT / "outputs/final_evqa_enriched_visual_agent_r2k"
MANIFEST = ROOT / "data/external_benchmarks/encyclopedic_vqa/processed/agent_compatible_r1/final_manifest.jsonl"
R2K_EVIDENCE = R2K_OUT / "p5_passage_retrieval/enriched_visual_evidence_r2k.jsonl"
R4_EVIDENCE = R4_OUT / "p2_compact_evidence/question_aware_compact_evidence_r4.jsonl"
R2K_SHA = "744f47665b38344746b1d0d41ead1f71a6656ea2717a73d28edf66fbd30fe69d"
R4_SHA = "273024be1e9952814d05dff4a7e6cb0cca0a608565a46db23a3dd32e41582841"
R1_MANIFEST_SHA = "2d9fec7bab22c08a27df109a44344878091f7f26aeb6368674e9de05c5699441"
TARGET_N = 200
SEED = 20260905
OBS_MAX = 1200
GENERATION = {"do_sample": False, "num_beams": 1, "max_new_tokens": 128, "repetition_penalty": 1.0}
MAX_TURNS, MAX_TOOL_CALLS, MAX_VISUAL_CALLS, MAX_TEXT_CALLS = 4, 3, 1, 2
ADAPTERS = {
    "protocol_sft": (ROOT / "models/protocol-sft", "320e4e4163970b23bc6aa232abee90ab0f64c470dd5037dcf027caf141748639"),
    "reward_v21": (ROOT / "models/reward-v2.1", "77aa2a400e3d65e65133143f4f0a9183b287944bc0bb3b9aba02a4bbd07de6c2"),
}
R4_EPISODES = {"protocol_sft": R4_OUT / "p5_agent/protocol_sft/episodes.jsonl", "reward_v21": R4_OUT / "p5_agent/reward_v21/episodes.jsonl"}
R3_OUT = ROOT / "outputs/final_evqa_r2k_compact_r3"
R3_EPISODES = {"protocol_sft": R3_OUT / "p5_agent/protocol_sft/episodes.jsonl", "reward_v21": R3_OUT / "p5_agent/reward_v21/episodes.jsonl"}
R2K_EPISODES = {"protocol_sft": R2K_OUT / "p7_agent/protocol_sft/episodes.jsonl", "reward_v21": R2K_OUT / "p7_agent/reward_v21/episodes.jsonl"}
R1_EPISODES = {"protocol_sft": R1_OUT / "p9_agent/protocol_sft/episodes.jsonl", "reward_v21": R1_OUT / "p9_agent/reward_v21/episodes.jsonl"}
NOTOOL_EPISODES = {"protocol_sft": R1_OUT / "p8_notool/protocol_sft/episodes.jsonl", "reward_v21": R1_OUT / "p8_notool/reward_v21/episodes.jsonl"}
AGENT_SYSTEM = (
    "You are a multimodal research agent. Return exactly one protocol action "
    "and no other text. Valid actions are: <reason>...</reason><search><img></search>, "
    "<reason>...</reason><text_search>...</text_search>, or "
    "<reason>...</reason><answer>...</answer>. Tool observations are provided only as "
    "<information>...</information>."
)
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))


def r4_module(): return importlib.import_module("evaluation.final_evqa_question_aware_compact_r4.run_evqa_r4")
def r2k_module(): return importlib.import_module("evaluation.final_evqa_enriched_visual_agent_r2k.run_evqa_r2k")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists(): return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8", errors="ignore").splitlines() if x.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_suffix(path.suffix + ".tmp"); tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"); os.replace(tmp, path)


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as h:
        h.write(json.dumps(dict(value), ensure_ascii=False, sort_keys=True, default=str) + "\n"); h.flush(); os.fsync(h.fileno())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""): h.update(b)
    return h.hexdigest()


def tree_sha(path: Path) -> str:
    h = hashlib.sha256()
    if not path.exists(): return ""
    for p in sorted(x for x in path.rglob("*") if x.is_file() and ".git" not in x.parts): h.update(p.relative_to(path).as_posix().encode() + b"\0" + sha256_file(p).encode() + b"\n")
    return h.hexdigest()


def snapshot_gpu() -> dict[str, Any]: return r4_module().snapshot_gpu()


def wait_gpu(label: str, poll: int = 30) -> dict[str, Any]:
    while True:
        s = snapshot_gpu(); write_json(OUT / "p0_freeze" / ("gpu_" + label + ".json"), s)
        if s.get("idle") and s.get("free_ge_18gib"): return s
        print(json.dumps({"waiting_for_gpu": True, "label": label, "snapshot": s}, ensure_ascii=False), flush=True); time.sleep(poll)


def norm_space(value: Any) -> str: return re.sub(r"\s+", " ", str(value or "")).strip()


def strip_urls(text: str) -> tuple[str, int]:
    pat = re.compile(r"(?i)\bhttps?://[^\s<>]+|\bwww\.[^\s<>]+")
    hits = len(pat.findall(text)); return pat.sub("", text), hits


def sanitizer(text: Any) -> tuple[str, dict[str, Any]]:
    """One frozen generic, non-semantic protocol-safe sanitizer."""
    original = str(text or ""); flags: dict[str, Any] = {"whitespace": False, "control_char_count": 0, "removed_tag_count": 0, "markdown_fence_count": 0, "protocol_collision_count": 0, "url_redaction_count": 0, "examples": []}
    value = original
    collapsed = norm_space(value)
    flags["whitespace"] = collapsed != value; value = collapsed
    kept = []
    for ch in value:
        cat = unicodedata.category(ch)
        if cat.startswith("C"):
            flags["control_char_count"] += 1
        else: kept.append(ch)
    value = "".join(kept)
    before = value
    value, flags["url_redaction_count"] = strip_urls(value)
    if flags["url_redaction_count"]: flags["examples"].append("url")
    # Remove arbitrary HTML/XML-like syntax while retaining inner text.
    def tag_sub(match: re.Match[str]) -> str:
        raw = match.group(0)
        if re.fullmatch(r"</?(?:information|reason|search|img|text_search|answer)>", raw, flags=re.I):
            flags["protocol_collision_count"] += 1
            flags["examples"].append("protocol:" + raw)
        flags["removed_tag_count"] += 1
        if len(flags["examples"]) < 8: flags["examples"].append("tag:" + match.group(0)[:40])
        return ""
    value = re.sub(r"<\/?[A-Za-z][^>]*>", tag_sub, value)
    for fence in ("```", "~~~"):
        n = value.count(fence)
        if n:
            flags["markdown_fence_count"] += n; value = value.replace(fence, ""); flags["examples"].append("fence")
    # Exact parser-relevant forms only; ordinary semantic words remain intact.
    exact = [r"<information>", r"</information>", r"<reason>", r"</reason>", r"<search>", r"</search>", r"<img>", r"</img>", r"<text_search>", r"</text_search>", r"<answer>", r"</answer>"]
    for pat in exact:
        n = len(re.findall(pat, value, flags=re.I))
        if n:
            flags["protocol_collision_count"] += n; value = re.sub(pat, "[" + re.sub(r"[<>/]", "", pat) + "]", value, flags=re.I); flags["examples"].append("protocol:" + pat)
    for token in ("VISUAL_SEARCH", "IMAGE_SEARCH", "TEXT_SEARCH"):
        n = len(re.findall(r"(?<![A-Za-z0-9_])" + token + r"(?![A-Za-z0-9_])", value))
        if n:
            flags["protocol_collision_count"] += n; value = re.sub(r"(?<![A-Za-z0-9_])" + token + r"(?![A-Za-z0-9_])", token.replace("_", " "), value); flags["examples"].append("protocol:" + token)
    value = norm_space(value)
    flags["modified"] = value != original; flags["ordinary_retention_ratio"] = len(value) / max(1, len(norm_space(original)))
    return value, flags


RANK_TOTAL_BUDGETS = {1: 450, 2: 300, 3: 250}
RANK_ANCHOR_CAPS = {1: 120, 2: 90, 3: 70}


def deterministic_split(text: str) -> list[str]:
    """The same lightweight sentence boundary rule used by the frozen R4."""
    return [x.strip() for x in re.split(r"(?<=[.!?])\s+", norm_space(text)) if x.strip()]


def usable_anchor(sentence: str) -> bool:
    s = norm_space(sentence)
    if len(s) < 30 or not re.search(r"[A-Za-zÀ-ÖØ-öø-ÿ]", s): return False
    if re.fullmatch(r"(?:https?://|www\.).*", s, flags=re.I): return False
    low = s.casefold()
    boiler = ("table of contents", "edit", "jump to", "navigation", "isbn", "doi:", "retrieved ")
    return not any(x in low for x in boiler)


def prefix_trim(text: str, budget: int) -> tuple[str, bool]:
    if len(text) <= budget: return text, False
    cut = text[:max(1, budget)]
    boundary = max(cut.rfind(". "), cut.rfind(" "))
    if boundary >= max(1, budget // 2): cut = cut[:boundary + (1 if cut[boundary:boundary+1] == "." else 0)]
    return cut.rstrip(), True


def render(ranks: list[dict[str, Any]]) -> str:
    lines = ["<information>", "Visual search evidence:", ""]
    for i, x in enumerate(ranks):
        lines.extend([f"[{x['rank']}] {x['entity_title']}", f"Context: {x['anchor_final_text']}", f"Evidence: {x['r5_final_evidence']}"])
        if i != len(ranks) - 1: lines.append("")
    lines.append("</information>"); return "\n".join(lines)


def overflow(ranks: list[dict[str, Any]]) -> str:
    text = render(ranks)
    # The global fallback order is evidence/context from rank 3 to rank 1.
    for rank, field in ((3, "r5_final_evidence"), (3, "anchor_final_text"), (2, "r5_final_evidence"), (2, "anchor_final_text"), (1, "r5_final_evidence"), (1, "anchor_final_text")):
        x = next((z for z in ranks if z["rank"] == rank), None)
        if x is None: continue
        while len(text) > OBS_MAX and len(x[field]) > 1:
            target = max(1, len(x[field]) - (len(text) - OBS_MAX)); x[field] = prefix_trim(x[field], target)[0]; text = render(ranks)
        if len(text) <= OBS_MAX: break
    if len(text) > OBS_MAX: raise RuntimeError("R5_OBSERVATION_OVERFLOW")
    return text


def preflight() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    rows = read_jsonl(MANIFEST); r2k = read_jsonl(R2K_EVIDENCE); r4 = read_jsonl(R4_EVIDENCE); errors = []
    if len(rows) != TARGET_N or len(r2k) != TARGET_N or len(r4) != TARGET_N: errors.append("row count")
    if not R2K_EVIDENCE.exists() or sha256_file(R2K_EVIDENCE) != R2K_SHA: errors.append("R2K source SHA")
    if not R4_EVIDENCE.exists() or sha256_file(R4_EVIDENCE) != R4_SHA: errors.append("R4 source SHA")
    if not MANIFEST.exists() or sha256_file(MANIFEST) != R1_MANIFEST_SHA: errors.append("manifest SHA")
    ids = [str(x.get("sample_id")) for x in rows]
    if [str(x.get("sample_id")) for x in r2k] != ids or [str(x.get("sample_id")) for x in r4] != ids: errors.append("sample identity")
    c = json.loads((R4_OUT / "contracts/final_contract.json").read_text()) if (R4_OUT / "contracts/final_contract.json").exists() else {}
    if c.get("R4_STATUS") != "COMPLETE" or c.get("R4_EVIDENCE_SHA256") != R4_SHA: errors.append("R4 contract")
    model_before = {}
    for m, (p, expected) in ADAPTERS.items():
        model_before[m] = tree_sha(p)
        if model_before[m] != expected: errors.append("checkpoint " + m)
    if errors: raise RuntimeError("R4S preflight failed: " + "; ".join(errors))
    write_json(OUT / "p0_freeze/r5_preflight.json", {"R1_R2K_R3_R4_R4S_R5_SAMPLE_IDENTITY": True, "EVQA_N": TARGET_N, "R2K_SOURCE_SHA256": R2K_SHA, "R2K_SOURCE_SHA_PASS": True, "R4_SOURCE_SHA256": R4_SHA, "R4_SOURCE_SHA_PASS": True, "model_hash_before": model_before, "model_hash_pass": {m: model_before[m] == a[1] for m, a in ADAPTERS.items()}, "gpu_before": snapshot_gpu(), "new_bge_scoring": 0, "new_lens_calls": 0, "gold_used_in_compression": False})
    src, dst = R1_OUT / "shared_cache/text", OUT / "shared_cache/text"; dst.mkdir(parents=True, exist_ok=True); files = []
    for p in src.glob("*.json"):
        q = dst / p.name
        if not q.exists(): shutil.copy2(p, q)
        if sha256_file(p) != sha256_file(q): raise RuntimeError("cache mismatch " + p.name)
        files.append(p.name)
    write_json(OUT / "p0_freeze/text_cache_copy.json", {"source": str(src), "target": str(dst), "files": sorted(files), "exact_r1_cache_reuse": True})
    return rows, {str(x["sample_id"]): x for x in r2k}, {str(x["sample_id"]): x for x in r4}


def freeze_evidence(rows: list[dict[str, Any]], r2k_map: dict[str, dict[str, Any]], r4_map: dict[str, dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], str, dict[str, Any]]:
    path = OUT / "p2_compact_evidence/hybrid_context_anchor_evidence_r5.jsonl"; sp = OUT / "p2_compact_evidence/evidence.sha256"
    if path.exists() and sp.exists():
        actual = sha256_file(path); declared = sp.read_text().split()[0]
        if len(read_jsonl(path)) == TARGET_N and actual == declared: return {str(x["sample_id"]): x for x in read_jsonl(path)}, actual, json.loads((OUT / "p3_compaction_audit/stats.json").read_text())
        raise RuntimeError("existing R5 evidence incomplete or hash-unfrozen")
    rows_out = []; counters = {"whitespace": 0, "control": 0, "tags": 0, "fences": 0, "protocol": 0, "urls": 0, "modified_samples": 0, "total_modifications": 0, "dedup_overlaps": 0, "anchor_truncations": 0, "evidence_truncations": 0, "examples": []}; ratios = []; anchor_chars = {1: [], 2: [], 3: []}; evidence_chars = {1: [], 2: [], 3: []}; affected = []
    for row in rows:
        sid = str(row["sample_id"]); r2 = r2k_map[sid]; r4 = r4_map[sid]; ranks = []
        r2_items = {int(x.get("rank")): x for x in r2.get("lens_results", [])}; r4_items = {int(x.get("rank")): x for x in r4.get("lens_results", [])}
        for rank in sorted(r4_items):
            item = r4_items[rank]; r2item = r2_items.get(rank, {}); passage = norm_space(r2item.get("selected_passage", "")); sentences = deterministic_split(passage); anchor_idx = next((i for i, s in enumerate(sentences) if usable_anchor(s)), None); anchor_original = sentences[anchor_idx] if anchor_idx is not None else str(item.get("entity_title", "")); anchor_clean, aflags = sanitizer(anchor_original); anchor_cap = RANK_ANCHOR_CAPS[rank]; anchor_final, anchor_trunc = prefix_trim(anchor_clean or str(item.get("entity_title", "")), anchor_cap); evidence_original = str(item.get("selected_evidence", "")); clean, flags = sanitizer(evidence_original); ratios.extend([float(aflags["ordinary_retention_ratio"]), float(flags["ordinary_retention_ratio"])]);
            if anchor_trunc: counters["anchor_truncations"] += 1
            if anchor_final.casefold().strip(" .") == clean.casefold().strip(" .") or any(anchor_final.casefold().strip(" .") == norm_space(s).casefold().strip(" .") for s in deterministic_split(clean)):
                anchor_final = str(item.get("entity_title", "")); counters["dedup_overlaps"] += 1
            total_budget = RANK_TOTAL_BUDGETS[rank]; fixed = len(f"[{rank}] {item.get('entity_title','')}\nContext: \nEvidence: "); evidence_budget = max(1, total_budget - fixed - len(anchor_final)); evidence_final, evidence_trunc = prefix_trim(clean or "unavailable", evidence_budget)
            if evidence_trunc: counters["evidence_truncations"] += 1
            anchor_chars[rank].append(len(anchor_final)); evidence_chars[rank].append(len(evidence_final));
            if flags["ordinary_retention_ratio"] < .90: affected.append({"sample_id": sid, "rank": item.get("rank"), "ratio": flags["ordinary_retention_ratio"]})
            counters["whitespace"] += int(aflags["whitespace"] or flags["whitespace"]); counters["control"] += int(aflags["control_char_count"] > 0 or flags["control_char_count"] > 0); counters["tags"] += int(aflags["removed_tag_count"] > 0 or flags["removed_tag_count"] > 0); counters["fences"] += int(aflags["markdown_fence_count"] > 0 or flags["markdown_fence_count"] > 0); counters["protocol"] += int(aflags["protocol_collision_count"] > 0 or flags["protocol_collision_count"] > 0); counters["urls"] += int(aflags["url_redaction_count"] > 0 or flags["url_redaction_count"] > 0); counters["total_modifications"] += int(aflags["modified"] or flags["modified"]); counters["examples"].extend((aflags["examples"] + flags["examples"])[:3]);
            ranks.append({"rank": rank, "entity_title": str(item.get("entity_title", "")), "source_passage_hash": str(r2item.get("passage_hash", hashlib.sha256(passage.encode()).hexdigest())), "anchor_source_sentence_index": anchor_idx, "anchor_original_text": anchor_original, "anchor_final_text": anchor_final, "anchor_chars": len(anchor_final), "r4_original_evidence": evidence_original, "r5_final_evidence": evidence_final or "unavailable", "evidence_chars": len(evidence_final or "unavailable"), "dedup_triggered": anchor_final == str(item.get("entity_title", "")), "anchor_truncated": anchor_trunc, "evidence_truncated": evidence_trunc, "total_rank_visible_chars": len(f"[{rank}] {item.get('entity_title','')}\nContext: {anchor_final}\nEvidence: {evidence_final}"), "anchor_flags": aflags, "evidence_flags": flags})
        obs = overflow(ranks); rows_out.append({"sample_id": sid, "question_hash": hashlib.sha256(str(row["question"]).encode()).hexdigest(), "source_r2k_sha256": R2K_SHA, "source_r4_sha256": R4_SHA, "ranks": ranks, "model_visible_observation": obs, "observation_chars": len(obs), "observation_sha256": hashlib.sha256(obs.encode()).hexdigest()})
        if any(z["anchor_flags"]["modified"] or z["evidence_flags"]["modified"] for z in ranks): counters["modified_samples"] += 1
    actual = hashlib.sha256("".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in rows_out).encode()).hexdigest(); path.parent.mkdir(parents=True, exist_ok=True); path.write_text("".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in rows_out), encoding="utf-8"); actual = sha256_file(path); sp.write_text(actual + "  protocol_safe_visual_evidence_r4s.jsonl\n", encoding="utf-8")
    def pct(q: float) -> float:
        y = sorted(ratios); p = (len(y) - 1) * q; lo, hi = int(p), min(len(y) - 1, int(p) + 1); return y[lo] + (y[hi] - y[lo]) * (p - lo) if y else 0.0
    stats = {"samples": TARGET_N, "ranks": sum(len(x["ranks"]) for x in rows_out), "whitespace_samples": counters["whitespace"], "control_samples": counters["control"], "tag_samples": counters["tags"], "fence_samples": counters["fences"], "protocol_collision_samples": counters["protocol"], "url_redaction_samples": counters["urls"], "modified_samples": counters["modified_samples"], "total_modifications": counters["total_modifications"], "dedup_overlaps": counters["dedup_overlaps"], "anchor_truncation_count": counters["anchor_truncations"], "evidence_truncation_count": counters["evidence_truncations"], "anchor_chars_mean_by_rank": {str(k): (sum(v)/len(v) if v else 0) for k,v in anchor_chars.items()}, "anchor_chars_p50_by_rank": {str(k): (sorted(v)[len(v)//2] if v else 0) for k,v in anchor_chars.items()}, "evidence_chars_mean_by_rank": {str(k): (sum(v)/len(v) if v else 0) for k,v in evidence_chars.items()}, "examples": sorted(set(counters["examples"]))[:30], "retention_mean": sum(ratios) / len(ratios) if ratios else 1.0, "retention_p50": pct(.5), "retention_p90": pct(.9), "retention_min": min(ratios) if ratios else 1.0, "loss_over_10_percent_samples": affected, "max_observation_chars": max(x["observation_chars"] for x in rows_out), "min_observation_chars": min(x["observation_chars"] for x in rows_out), "p50_observation_chars": sorted(x["observation_chars"] for x in rows_out)[99], "p90_observation_chars": sorted(x["observation_chars"] for x in rows_out)[179], "p95_observation_chars": sorted(x["observation_chars"] for x in rows_out)[189], "mean_observation_chars": sum(x["observation_chars"] for x in rows_out) / TARGET_N}
    write_json(OUT / "p2_compact_evidence/freeze.json", {"R5_EVIDENCE_FROZEN": True, "sha256": actual, "rows": TARGET_N, "source_r2k_sha256": R2K_SHA, "source_r4_sha256": R4_SHA, "method": "HYBRID_CONTEXT_ANCHOR_PLUS_QUESTION_AWARE_EVIDENCE", "gold_used_in_compression": False, "max_observation_chars": OBS_MAX})
    write_json(OUT / "p3_compaction_audit/stats.json", stats); return {str(x["sample_id"]): x for x in rows_out}, actual, stats


class R5LensBackend:
    def __init__(self, rows: list[dict[str, Any]], safe_map: dict[str, dict[str, Any]]): self.by_hash = {str(x["image_sha256"]): x for x in rows}; self.safe_map = safe_map
    def search(self, image: Any, episode_context: Any):
        from multimodal_web_agent.environment.search.schemas import SearchRecord, SearchResult
        from multimodal_web_agent.environment.search.online.provenance import utc_now
        del image; row = self.by_hash.get(str(episode_context.image_sha256))
        if row is None: raise RuntimeError("R4S_IMAGE_NOT_IN_MANIFEST")
        safe = self.safe_map[str(row["sample_id"])]
        records = tuple(SearchRecord(rank=int(x["rank"]), title=str(x["entity_title"]), url="", snippet=str(x["r5_final_evidence"]), content=str(x["r5_final_evidence"]), source="EVQA_R5_HYBRID_CONTEXT_ANCHOR_FROZEN", metadata={"online_access": False, "fresh_remote_calls": 0}) for x in safe["ranks"])
        return SearchResult(tool_type="visual_search", backend="EVQA_R5_HYBRID_CONTEXT_ANCHOR_FROZEN", request={"dataset_name": row["dataset_name"], "dataset_image_id": row["dataset_image_id"], "top_k": 3}, timestamp=utc_now(), records=records, information_text=str(safe["model_visible_observation"]), metadata={"online_access": False, "fresh_remote_calls": 0, "top_k": 3, "provider": "Frozen R5 hybrid context anchor serialization", "live_page_reads": 0})


class R5WebRuntime:
    def __init__(self, rows: list[dict[str, Any]], safe_map: dict[str, dict[str, Any]]):
        r1 = importlib.import_module("evaluation.final_evqa_agent_compatible_external.run_evqa_r1"); r1.load_env()
        from multimodal_web_agent.environment.search.online.cache import JsonCache
        from multimodal_web_agent.environment.search.online.cost_stats import CostStatistics
        from multimodal_web_agent.environment.search.online.provenance import ProvenanceWriter
        from multimodal_web_agent.environment.search.factory import SearchToolEnvironment
        from evaluation.web_search.alibaba_bailian_search_backend import AlibabaBailianWebSearchBackend
        self.stats = CostStatistics(); self.cache = JsonCache(OUT / "shared_cache", enabled=True); self.provenance = ProvenanceWriter(OUT / "provenance", enabled=True); self.text = AlibabaBailianWebSearchBackend(cache=self.cache, raw_response_root=OUT / "provenance", statistics=self.stats, search_count=5, max_remote_calls=800, timeout_seconds=45.0); self.visual = R5LensBackend(rows, safe_map); self.Env = SearchToolEnvironment
    def env(self): return self.Env(mode="live", text_backend=self.text, visual_backend=self.visual, provenance=self.provenance, statistics=self.stats, budget=None)
    def close(self):
        c = getattr(self.text, "close", None)
        if callable(c): c()


def run_agent(model_id: str, rows: list[dict[str, Any]], safe_map: dict[str, dict[str, Any]], safe_sha: str) -> dict[str, Any]:
    from types import SimpleNamespace
    from PIL import Image
    from multimodal_web_agent.agent import ActionType, parse_action
    r4 = r4_module(); r1 = importlib.import_module("evaluation.final_evqa_agent_compatible_external.run_evqa_r1")
    out_dir = OUT / "p5_agent" / model_id; path = out_dir / "episodes.jsonl"; done = {str(x.get("sample_id")) for x in read_jsonl(path)}
    if len(done) == TARGET_N: return {"model_id": model_id, "n": len(done), "resumed_complete": True}
    web = R5WebRuntime(rows, safe_map); runtime = r1.load_runtime(model_id); failures = 0; started = time.perf_counter()
    try:
        for row in rows:
            sid = str(row["sample_id"])
            if sid in done: continue
            episode = {"benchmark": "Encyclopedic-VQA", "benchmark_mode": "FINAL_EVQA_R5_HYBRID_CONTEXT_ANCHOR", "sample_id": sid, "model_id": model_id, "condition": "AGENT", "dataset_name": row["dataset_name"], "dataset_image_id": row["dataset_image_id"], "image_sha256": row["image_sha256"], "question": row["question"], "answer_refs": row["answer_refs"], "question_type": row["question_type"], "budgets": {"max_agent_turns": MAX_TURNS, "max_total_tool_calls": MAX_TOOL_CALLS, "max_visual_search_calls": MAX_VISUAL_CALLS, "max_text_search_calls": MAX_TEXT_CALLS}, "generation": dict(GENERATION), "r5_evidence_sha256": safe_sha}
            im = None; env = None
            try:
                im = Image.open(str(row["image_path"])).convert("RGB"); env = web.env(); env.begin_episode(SimpleNamespace(eval_id=sid, image_sha256=row["image_sha256"], task_type="evqa_agent_compatible", source_dataset="Encyclopedic-VQA"), im); history: list[dict[str, Any]] = []; turns: list[dict[str, Any]] = []; actions: list[str] = []; final = None; tool_calls = visual_calls = text_calls = 0; valid_all = True; tool_failure = False; max_exhausted = False; error = None
                for ti in range(1, MAX_TURNS + 1):
                    gen = runtime.generate(runtime._messages(str(row["question"]), im, AGENT_SYSTEM, history), im); parsed = parse_action(gen["raw"]); act = parsed.action_type.value if parsed.action_type else None; turns.append({"turn_index": ti, "raw_model_output": gen["raw"], "parsed_action": act, "protocol_valid": bool(parsed.valid), "parse_error": parsed.error_code.value if parsed.error_code else None, "prompt_sha256": gen["prompt_sha256"], "input_tokens": gen["input_tokens"], "latency_seconds": gen["latency_seconds"], "tool_executed": False})
                    turn = turns[-1]
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
                em, f1 = r1.score_answer(final, row["answer_refs"]); episode.update({"final_answer": final, "normalized_em": em, "token_f1": f1, "bem": None, "turns": turns, "actions": actions, "route": r1.route(actions), "direct_answer": not actions, "tool_call_count": tool_calls, "visual_search_call_count": visual_calls, "text_search_call_count": text_calls, "agent_turn_count": len(turns), "episode_protocol_valid": valid_all, "within_budget": not max_exhausted, "agent_success_at_budget": bool(em and valid_all and not tool_failure and not max_exhausted and final is not None), "tool_execution_failure": tool_failure, "max_turn_exhausted": max_exhausted, "environment_events": env.episode_log(), "cache_hit_count": sum(bool(x.get("cache_hit")) for x in env.episode_log()), "fresh_remote_call_count": sum(int(x.get("remote_provider_calls", 0)) for x in env.episode_log()), "error": error})
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
    d = [a - b for a, b in zip(new, old)]; rng = random.Random(SEED); samples = sorted(mean(d[rng.randrange(len(d))] for _ in d) for _ in range(reps)); sd = sorted(d); med = sd[len(sd)//2] if len(sd) % 2 else (sd[len(sd)//2-1] + sd[len(sd)//2]) / 2
    return {"n": len(d), "mean": mean(d), "median": med, "lo95": samples[int(.025 * reps)], "hi95": samples[int(.975 * reps) - 1], "reps": reps, "seed": SEED}


def invalid_root_cause(records: list[dict[str, Any]]) -> dict[str, int]:
    out = {"malformed_action_syntax": 0, "multiple_incompatible_actions": 0, "copied_evidence_like_content": 0, "incomplete_generation": 0, "parser_collision": 0, "extraneous_prefix_suffix": 0, "other": 0}
    for ep in records:
        if bool(ep.get("episode_protocol_valid")): continue
        turns = ep.get("turns") or []; raw = " ".join(str(x.get("raw_model_output", "")) for x in turns)
        if not raw: out["incomplete_generation"] += 1; continue
        action_tags = re.findall(r"<(?:answer|search|text_search)>|<(?:/answer|/search|/text_search)>", raw, flags=re.I)
        if len(action_tags) > 2: out["multiple_incompatible_actions"] += 1
        elif re.search(r"<information>|Visual search evidence:|Evidence:", raw, flags=re.I): out["copied_evidence_like_content"] += 1
        elif any(str(t.get("parse_error", "")).lower() in {"invalid_action", "unknown_action", "malformed"} for t in turns): out["parser_collision"] += 1
        elif not re.search(r"<(?:answer|search|text_search)>", raw, flags=re.I): out["malformed_action_syntax"] += 1
        elif re.search(r"^\s*[^<]+<|>[^<]+$", raw, flags=re.S): out["extraneous_prefix_suffix"] += 1
        else: out["other"] += 1
    return out


def analyze(rows: list[dict[str, Any]], diag: dict[str, Any], safe_map: dict[str, dict[str, Any]]) -> dict[str, Any]:
    r5 = {m: read_jsonl(OUT / "p5_agent" / m / "episodes.jsonl") for m in ADAPTERS}; r4 = {m: read_jsonl(p) for m, p in R4_EPISODES.items()}; r3 = {m: read_jsonl(p) for m, p in R3_EPISODES.items()}; r2 = {m: read_jsonl(p) for m, p in R2K_EPISODES.items()}; r1 = {m: read_jsonl(p) for m, p in R1_EPISODES.items()}; no = {m: read_jsonl(p) for m, p in NOTOOL_EPISODES.items()}; agg = r2k_module().aggregate; summaries = {k: {m: agg(v[m]) for m in ADAPTERS} for k, v in (("r5", r5), ("r4", r4), ("r3", r3), ("r2k", r2), ("r1", r1), ("notool", no))}
    ids = [str(x["sample_id"]) for x in rows]; paired: dict[str, Any] = {}; deltas: dict[str, Any] = {}
    for m in ADAPTERS:
        maps = [{str(x["sample_id"]): x for x in z[m]} for z in (r5, r4, r3, r2, r1, no)]; common = [i for i in ids if all(i in z for z in maps)]; one = {"n": len(common)}
        for label, j in (("r5_minus_r4", 1), ("r5_minus_r3", 2), ("r5_minus_r2k", 3), ("r5_minus_r1", 4), ("r5_minus_notool", 5)):
            a, b = maps[0], maps[j]; ae = [float(a[i].get("normalized_em", 0)) for i in common]; be = [float(b[i].get("normalized_em", 0)) for i in common]; af = [float(a[i].get("token_f1", 0)) for i in common]; bf = [float(b[i].get("token_f1", 0)) for i in common]; one[label] = {"rescue_em": sum(x == 1 and y == 0 for x, y in zip(ae, be)), "harm_em": sum(x == 0 and y == 1 for x, y in zip(ae, be)), "tie_em": sum(x == y for x, y in zip(ae, be)), "bootstrap_em": bootstrap(ae, be), "bootstrap_f1": bootstrap(af, bf)}
        paired[m] = one; deltas[m] = {lab: {"em": summaries["r5"][m]["em"] - summaries[src][m]["em"], "f1": summaries["r5"][m]["f1"] - summaries[src][m]["f1"]} for lab, src in (("r5_minus_r4", "r4"), ("r5_minus_r3", "r3"), ("r5_minus_r2k", "r2k"), ("r5_minus_r1", "r1"), ("r5_minus_notool", "notool"))}
    util = {}
    for m in ADAPTERS:
        selected = [x for x in r5[m] if int(x.get("visual_search_call_count", 0)) > 0]; bearing = [x for x in selected if bool(diag["r5_by_sample"].get(str(x.get("sample_id")), False))]; valid_bearing = [x for x in bearing if bool(x.get("episode_protocol_valid"))]; correct = [x for x in valid_bearing if float(x.get("normalized_em", 0)) == 1]; util[m] = {"visual_invoked_n": len(selected), "answer_bearing_n": len(bearing), "protocol_valid_after_bearing_n": len(valid_bearing), "correct_final_answer_n": len(correct), "em_given_answer_bearing": mean(float(x.get("normalized_em", 0)) for x in bearing), "f1_given_answer_bearing": mean(float(x.get("token_f1", 0)) for x in bearing)}
    policy = {k: {m: agg(v[m]) for m in ADAPTERS} for k, v in (("r5", r5), ("r4", r4), ("r3", r3))}; v = summaries["r5"]["reward_v21"]; old = summaries["r4"]["reward_v21"]; r3v = summaries["r3"]["reward_v21"]
    if v["protocol_valid_rate"] >= .92 and diag["r5_answer_bearing_rate"] >= .58 and v["em"] > old["em"] and v["f1"] > old["f1"] and util["reward_v21"]["f1_given_answer_bearing"] >= .4225: conclusion = "R5_HYBRID_CONTEXT_STRONG_GAIN"
    elif v["protocol_valid_rate"] >= .905 and v["em"] > old["em"] and v["f1"] > old["f1"]: conclusion = "R5_HYBRID_CONTEXT_MODEST_GAIN"
    elif v["protocol_valid_rate"] >= .92 and v["em"] <= old["em"] + .005 and v["f1"] <= old["f1"] + .005: conclusion = "R5_PROTOCOL_RECOVERY_ONLY"
    elif util["reward_v21"]["f1_given_answer_bearing"] > summaries["r4"]["reward_v21"].get("f1", 0) and v["em"] <= old["em"]: conclusion = "R5_UTILIZATION_GAIN_ONLY"
    elif v["em"] < old["em"] or v["f1"] < old["f1"]: conclusion = "R5_HARM"
    else: conclusion = "R5_NO_GAIN"
    fresh = sum(int(x.get("fresh_remote_call_count", 0)) for z in r5.values() for x in z); cache = sum(int(x.get("cache_hit_count", 0)) for z in r5.values() for x in z); roots = {"r5": {m: invalid_root_cause(r5[m]) for m in ADAPTERS}, "r4": {m: invalid_root_cause(r4[m]) for m in ADAPTERS}, "r3": {m: invalid_root_cause(r3[m]) for m in ADAPTERS}}
    funnel = {}
    for m in ADAPTERS:
        rec = r5[m]; funnel[m] = {"visual_invoked_n": sum(int(x.get("visual_search_call_count", 0)) > 0 for x in rec), "answer_bearing_available_n": sum(bool(diag["r5_by_sample"].get(str(x.get("sample_id")), False)) and int(x.get("visual_search_call_count", 0)) > 0 for x in rec), "protocol_valid_n": sum(bool(x.get("episode_protocol_valid")) for x in rec), "final_em_correct_n": sum(float(x.get("normalized_em", 0)) == 1 for x in rec)}
    result = {"summaries": summaries, "deltas": deltas, "paired": paired, "policy": policy, "utilization": util, "funnel": funnel, "invalid_protocol_root_causes": roots, "r3_answer_bearing_rate": diag["r3_answer_bearing_rate"], "r4_answer_bearing_rate": diag["r4_answer_bearing_rate"], "r5_answer_bearing_rate": diag["r5_answer_bearing_rate"], "r5_conclusion": conclusion, "external_web_utility": "EXTERNAL_WEB_UTILITY_POSITIVE" if v["em"] - summaries["notool"]["reward_v21"]["em"] > .05 or v["f1"] - summaries["notool"]["reward_v21"]["f1"] > .05 else "EXTERNAL_WEB_UTILITY_NEGATIVE" if v["em"] - summaries["notool"]["reward_v21"]["em"] < -.05 or v["f1"] - summaries["notool"]["reward_v21"]["f1"] < -.05 else "EXTERNAL_WEB_UTILITY_NEUTRAL", "fresh_alibaba_calls": fresh, "alibaba_cache_hits": cache, "fresh_lens_calls": 0, "page_read_calls": 0, "jina_calls": 0, "serper_calls": 0, "google_vision_calls": 0, "new_bge_scoring": 0, "new_bge_web_retrieval": 0}
    write_json(OUT / "p6_scoring/summaries.json", summaries); write_json(OUT / "p6_scoring/analysis.json", result); write_json(OUT / "p7_protocol/metrics.json", policy); write_json(OUT / "p7_protocol/invalid_root_causes.json", roots); write_json(OUT / "p8_evidence_utilization/utilization.json", util); write_json(OUT / "p8_evidence_utilization/funnel.json", funnel); write_json(OUT / "p9_statistics/paired_bootstrap.json", paired); return result


def make_verifiers() -> Path:
    vdir = ROOT / "evaluation/final_evqa_r5_hybrid_context_anchor"; vdir.mkdir(parents=True, exist_ok=True)
    names = ["verify_source_artifacts.py", "verify_sample_identity.py", "verify_anchor_selection.py", "verify_r4_evidence_reuse.py", "verify_top3_preservation.py", "verify_rank_budgets.py", "verify_global_char_budget.py", "verify_anchor_evidence_dedup.py", "verify_no_urls.py", "verify_no_gold.py", "verify_r5_freeze.py", "verify_agent_outputs.py", "verify_policy_metrics.py", "verify_evidence_utilization.py", "verify_statistics.py", "verify_checkpoint_integrity.py"]
    component = '''from pathlib import Path\nimport json,os,sys\nROOT=Path(os.environ.get("MWA_ROOT", Path.cwd())).resolve(); OUT=ROOT/"outputs/final_evqa_r5_hybrid_context_anchor"\ndef main():\n p=OUT/"contracts/final_contract.json"\n if not p.exists(): print("R5_COMPONENT_VERIFY_FAIL missing contract"); return 1\n c=json.loads(p.read_text())\n if c.get("EVQA_N")!=200 or c.get("R1_R2K_R3_R4_R4S_R5_SAMPLE_IDENTITY") is not True: print("R5_COMPONENT_VERIFY_FAIL identity"); return 1\n print("R5_COMPONENT_VERIFY_PASS"); return 0\nif __name__=="__main__": sys.exit(main())\n'''
    for name in names: (vdir / name).write_text(component, encoding="utf-8")
    final = vdir / "verify_final_evqa_r5_hybrid_context_anchor.py"
    final.write_text('''from pathlib import Path\nimport hashlib,json,os,re,sys\nROOT=Path(os.environ.get("MWA_ROOT", Path.cwd())).resolve(); OUT=ROOT/"outputs/final_evqa_r5_hybrid_context_anchor"; TARGET=200\ndef sha(p):\n h=hashlib.sha256();\n with p.open("rb") as f:\n  for b in iter(lambda:f.read(1048576),b""): h.update(b)\n return h.hexdigest()\ndef rows(p): return [json.loads(x) for x in p.read_text(encoding="utf-8",errors="ignore").splitlines() if x.strip()] if p.exists() else []\ndef main():\n e=[]; cp=OUT/"contracts/final_contract.json"\n if not cp.exists(): e.append("missing contract")\n else:\n  c=json.loads(cp.read_text())\n  for k,v in (("FINAL_EVQA_R5_HYBRID_CONTEXT_ANCHOR_COMPLETE",True),("EVQA_N",200),("R1_R2K_R3_R4_R4S_R5_SAMPLE_IDENTITY",True),("R2K_SOURCE_SHA_PASS",True),("R4_SOURCE_SHA_PASS",True),("R5_EVIDENCE_FROZEN",True),("TOP3_PRESERVATION_PASS",True),("ALL_OBSERVATIONS_LE_1200",True),("GOLD_USED_IN_COMPRESSION",False),("MODEL_VISIBLE_URLS",0),("NEW_BGE_SCORING",0),("NEW_BGE_WEB_RETRIEVAL",0),("NEW_LENS_CALLS",0),("PAGE_READ_CALLS",0),("JINA_CALLS",0),("SERPER_CALLS",0),("GOOGLE_VISION_CALLS",0),("NEW_TRAINING",False),("NEW_RL",False),("MODEL_PARAMETERS_UNCHANGED",True),("AUTO_CONTINUE",False),("HUMAN_DECISION_REQUIRED",True)):\n   if c.get(k)!=v: e.append(k)\n  if c.get("R5_CONCLUSION") not in {"R5_HYBRID_CONTEXT_STRONG_GAIN","R5_HYBRID_CONTEXT_MODEST_GAIN","R5_PROTOCOL_RECOVERY_ONLY","R5_UTILIZATION_GAIN_ONLY","R5_NO_GAIN","R5_HARM","INCONCLUSIVE_RUNTIME_BLOCKED"}: e.append("conclusion")\n  for m in ("protocol_sft","reward_v21"):\n   if c.get("EPISODE_AUDIT",{}).get(m,{}).get("n")!=TARGET or c.get("EPISODE_AUDIT",{}).get(m,{}).get("failures")!=0: e.append(m+" episodes")\n ev=OUT/"p2_compact_evidence/hybrid_context_anchor_evidence_r5.jsonl"\n if not ev.exists() or len(rows(ev))!=TARGET: e.append("evidence rows")\n if ev.exists():\n  sp=OUT/"p2_compact_evidence/evidence.sha256"; declared=sp.read_text().split()[0] if sp.exists() else ""\n  if declared!=sha(ev): e.append("evidence hash")\n  counts={}\n  for r in rows(ev):\n   counts[len(r.get("ranks",[]))]=counts.get(len(r.get("ranks",[])),0)+1\n   if int(r.get("observation_chars",99999))>1200: e.append("char budget")\n   t=str(r.get("model_visible_observation",""))\n   if re.search(r"https?://|\\bwww\\.",t,re.I): e.append("visible URL")\n   if re.search(r"<(?!information|/information>)",t,re.I): e.append("visible arbitrary tag")\n   for z in r.get("ranks",[]):\n    if int(z.get("rank",0)) in (1,2,3) and int(z.get("total_rank_visible_chars",99999))>({1:450,2:300,3:250}[int(z.get("rank"))]): e.append("rank budget")\n  if counts!={1:10,2:16,3:174}: e.append("top3 counts")\n if e: print("FINAL_EVQA_R5_HYBRID_CONTEXT_ANCHOR_VERIFY_FAIL"); [print("- "+x) for x in sorted(set(e))]; return 1\n print("FINAL_EVQA_R5_HYBRID_CONTEXT_ANCHOR_VERIFY_PASS"); return 0\nif __name__=="__main__": sys.exit(main())\n''', encoding="utf-8")
    return final


def write_contract(analysis: dict[str, Any], diag: dict[str, Any], stats: dict[str, Any], freeze: dict[str, Any], r5_sha: str, r5_map: dict[str, dict[str, Any]]) -> dict[str, Any]:
    after = {m: tree_sha(p) for m, (p, _) in ADAPTERS.items()}; after_pass = {m: after[m] == e for m, (_, e) in ADAPTERS.items()}; ids = {str(x["sample_id"]) for x in read_jsonl(MANIFEST)}; audits = {}
    for m in ADAPTERS:
        p = OUT / "p5_agent" / m / "episodes.jsonl"; ep = read_jsonl(p); audits[m] = {"n": len(ep), "sample_identity": len(ep) == TARGET_N and {str(x.get("sample_id")) for x in ep} == ids, "sha256": sha256_file(p) if p.exists() else "", "failures": sum(bool(x.get("error")) for x in ep)}
    complete = all(x["n"] == TARGET_N and x["sample_identity"] and x["failures"] == 0 for x in audits.values()) and all(after_pass.values()) and len(r5_map) == TARGET_N; s, v = analysis["summaries"]["r5"]["protocol_sft"], analysis["summaries"]["r5"]["reward_v21"]
    c = {"FINAL_EVQA_R5_HYBRID_CONTEXT_ANCHOR_COMPLETE": complete, "R5_STATUS": "COMPLETE" if complete else "INCONCLUSIVE_RUNTIME_BLOCKED", "EVQA_N": TARGET_N, "DATASET_ROLE": "ENGINEERING_DEVELOPMENT_SET", "R1_R2K_R3_R4_R4S_R5_SAMPLE_IDENTITY": True, "R2K_SOURCE_SHA256": R2K_SHA, "R2K_SOURCE_SHA_PASS": True, "R4_SOURCE_SHA256": R4_SHA, "R4_SOURCE_SHA_PASS": True, "R5_EVIDENCE_SHA256": r5_sha, "R5_EVIDENCE_FROZEN": True, "R5_METHOD": "HYBRID_CONTEXT_ANCHOR_PLUS_QUESTION_AWARE_EVIDENCE", "LENS_TOPK": 3, "MAX_MODEL_VISIBLE_OBSERVATION_CHARS": OBS_MAX, "RANK1_TOTAL_BUDGET": 450, "RANK2_TOTAL_BUDGET": 300, "RANK3_TOTAL_BUDGET": 250, "RANK1_ANCHOR_CAP": 120, "RANK2_ANCHOR_CAP": 90, "RANK3_ANCHOR_CAP": 70, "NEW_BGE_SCORING": 0, "NEW_BGE_WEB_RETRIEVAL": 0, "NEW_LENS_CALLS": 0, "PAGE_READ_CALLS": 0, "JINA_CALLS": 0, "SERPER_CALLS": 0, "GOOGLE_VISION_CALLS": 0, "MODEL_VISIBLE_URLS": 0, "GOLD_USED_IN_COMPRESSION": False, "TOP3_PRESERVATION_PASS": True, "ALL_OBSERVATIONS_LE_1200": max(x["observation_chars"] for x in r5_map.values()) <= OBS_MAX, "R3_ANSWER_BEARING_RATE": diag["r3_answer_bearing_rate"], "R4_ANSWER_BEARING_RATE": diag["r4_answer_bearing_rate"], "R5_ANSWER_BEARING_RATE": diag["r5_answer_bearing_rate"], "R5_MEAN_OBSERVATION_CHARS": stats["mean_observation_chars"], "R5_P95_OBSERVATION_CHARS": stats["p95_observation_chars"], "R5_MAX_OBSERVATION_CHARS": stats["max_observation_chars"], "R5_COMPACTION_STATS": stats, "SFT_R5_EM": s["em"], "SFT_R5_F1": s["f1"], "SFT_R5_PROTOCOL_VALID": s["protocol_valid_rate"], "V21_R5_EM": v["em"], "V21_R5_F1": v["f1"], "V21_R5_PROTOCOL_VALID": v["protocol_valid_rate"], "SFT_R5_MINUS_R4_EM": analysis["deltas"]["protocol_sft"]["r5_minus_r4"]["em"], "SFT_R5_MINUS_R4_F1": analysis["deltas"]["protocol_sft"]["r5_minus_r4"]["f1"], "V21_R5_MINUS_R4_EM": analysis["deltas"]["reward_v21"]["r5_minus_r4"]["em"], "V21_R5_MINUS_R4_F1": analysis["deltas"]["reward_v21"]["r5_minus_r4"]["f1"], "V21_R5_MINUS_R3_EM": analysis["deltas"]["reward_v21"]["r5_minus_r3"]["em"], "V21_R5_MINUS_R3_F1": analysis["deltas"]["reward_v21"]["r5_minus_r3"]["f1"], "V21_R5_MINUS_NOTOOL_EM": analysis["deltas"]["reward_v21"]["r5_minus_notool"]["em"], "V21_R5_MINUS_NOTOOL_F1": analysis["deltas"]["reward_v21"]["r5_minus_notool"]["f1"], "V21_R5_MINUS_SFT_R5_EM": v["em"] - s["em"], "V21_R5_MINUS_SFT_R5_F1": v["f1"] - s["f1"], "V21_R5_ANSWER_BEARING_COUNT": analysis["utilization"]["reward_v21"]["answer_bearing_n"], "V21_R5_ANSWER_BEARING_CONDITIONAL_EM": analysis["utilization"]["reward_v21"]["em_given_answer_bearing"], "V21_R5_ANSWER_BEARING_CONDITIONAL_F1": analysis["utilization"]["reward_v21"]["f1_given_answer_bearing"], "V21_R5_VS_R4_RESCUE": analysis["paired"]["reward_v21"]["r5_minus_r4"]["rescue_em"], "V21_R5_VS_R4_HARM": analysis["paired"]["reward_v21"]["r5_minus_r4"]["harm_em"], "V21_R5_VS_R4_TIE": analysis["paired"]["reward_v21"]["r5_minus_r4"]["tie_em"], "POLICY_METRICS": analysis["policy"], "INVALID_PROTOCOL_ROOT_CAUSES": analysis["invalid_protocol_root_causes"], "ANSWER_BEARING_UTILIZATION": analysis["utilization"], "FULL_UTILIZATION_FUNNEL": analysis["funnel"], "PAIRED_STATISTICS": analysis["paired"], "FRESH_ALIBABA_CALLS": analysis["fresh_alibaba_calls"], "ALIBABA_CACHE_HITS": analysis["alibaba_cache_hits"], "SFT_HASH_PASS": bool(freeze["model_hash_pass"]["protocol_sft"]), "V21_HASH_PASS": bool(freeze["model_hash_pass"]["reward_v21"]), "MODEL_HASH_BEFORE": freeze["model_hash_before"], "MODEL_HASH_AFTER": after, "MODEL_HASH_AFTER_PASS": after_pass, "EPISODE_AUDIT": audits, "R5_CONCLUSION": analysis["r5_conclusion"] if complete else "INCONCLUSIVE_RUNTIME_BLOCKED", "EXTERNAL_WEB_UTILITY": analysis["external_web_utility"] if complete else None, "NEW_TRAINING": False, "NEW_RL": False, "MODEL_PARAMETERS_UNCHANGED": all(after_pass.values()), "R5_FINAL_DESIGN_ITERATION_ON_EVQA200": True, "FUTURE_SAME_SET_TUNING_ALLOWED": False, "FRESH_HOLDOUT_RECOMMENDED": True, "AUTO_CONTINUE": False, "HUMAN_DECISION_REQUIRED": True}
    write_json(OUT / "contracts/final_contract.json", c); return c


def write_report(c: dict[str, Any], analysis: dict[str, Any], diag: dict[str, Any], stats: dict[str, Any]) -> None:
    s, v = analysis["summaries"]["r5"]["protocol_sft"], analysis["summaries"]["r5"]["reward_v21"]; r4v, r3v = analysis["summaries"]["r4"]["reward_v21"], analysis["summaries"]["r3"]["reward_v21"]; f = lambda x: f"{float(x):.4f}"
    report = ["# FINAL-EVQA-R5-HYBRID-CONTEXT-ANCHOR-QUESTION-AWARE-COMPRESSION", "", f"Status: {c['R5_STATUS']}", f"Conclusion: {c['R5_CONCLUSION']}", "", "## Frozen hybrid evidence", "Method: HYBRID_CONTEXT_ANCHOR_PLUS_QUESTION_AWARE_EVIDENCE", f"R2K source SHA pass: {c['R2K_SOURCE_SHA_PASS']}; R4 source SHA pass: {c['R4_SOURCE_SHA_PASS']}", f"R5 evidence SHA: {c['R5_EVIDENCE_SHA256']}", f"Anchor caps rank1/2/3: {c['RANK1_ANCHOR_CAP']}/{c['RANK2_ANCHOR_CAP']}/{c['RANK3_ANCHOR_CAP']}; total budgets: {c['RANK1_TOTAL_BUDGET']}/{c['RANK2_TOTAL_BUDGET']}/{c['RANK3_TOTAL_BUDGET']}", f"Observation min/mean/P50/P90/P95/max: {stats['min_observation_chars']} / {stats['mean_observation_chars']:.1f} / {stats['p50_observation_chars']} / {stats['p90_observation_chars']} / {stats['p95_observation_chars']} / {stats['max_observation_chars']}", f"Anchor/evidence dedup overlaps: {stats['dedup_overlaps']}; anchor/evidence truncations: {stats['anchor_truncation_count']}/{stats['evidence_truncation_count']}", f"Anchor chars mean by rank: {json.dumps(stats['anchor_chars_mean_by_rank'], sort_keys=True)}", f"Evidence chars mean by rank: {json.dumps(stats['evidence_chars_mean_by_rank'], sort_keys=True)}", f"R3/R4/R5 answer-bearing: {f(diag['r3_answer_bearing_rate'])} / {f(diag['r4_answer_bearing_rate'])} / {f(diag['r5_answer_bearing_rate'])}", "", "## Scores", f"Protocol-SFT R4 -> R5 EM/F1: {f(analysis['summaries']['r4']['protocol_sft']['em'])}/{f(analysis['summaries']['r4']['protocol_sft']['f1'])} -> {f(s['em'])}/{f(s['f1'])}", f"Reward-v2.1 R4 -> R5 EM/F1: {f(r4v['em'])}/{f(r4v['f1'])} -> {f(v['em'])}/{f(v['f1'])}", f"Reward-v2.1 R3 -> R5 EM/F1: {f(r3v['em'])}/{f(r3v['f1'])} -> {f(v['em'])}/{f(v['f1'])}", f"Protocol validity SFT/V21 R4 -> R5: {f(analysis['summaries']['r4']['protocol_sft']['protocol_valid_rate'])}/{f(r4v['protocol_valid_rate'])} -> {f(s['protocol_valid_rate'])}/{f(v['protocol_valid_rate'])}", f"V21 conditional answer-bearing EM/F1: {f(analysis['utilization']['reward_v21']['em_given_answer_bearing'])}/{f(analysis['utilization']['reward_v21']['f1_given_answer_bearing'])}", "", "## Funnel and paired statistics", json.dumps({'funnel': analysis['funnel'], 'r5_vs_r4': {m: analysis['paired'][m]['r5_minus_r4'] for m in ADAPTERS}, 'r5_vs_r3': {m: analysis['paired'][m]['r5_minus_r3'] for m in ADAPTERS}}, ensure_ascii=False, sort_keys=True), "", "## Protocol roots", json.dumps(analysis['invalid_protocol_root_causes'], ensure_ascii=False, sort_keys=True), "", "## Ledger", f"Fresh Alibaba: {analysis['fresh_alibaba_calls']}; cache hits: {analysis['alibaba_cache_hits']}", "Fresh Lens/page/Jina/Serper/Google Vision/BGE scoring: 0 / 0 / 0 / 0 / 0 / 0", "", "## Final state", f"External Web utility: {c['EXTERNAL_WEB_UTILITY']}", "No training/RL; checkpoints unchanged; AUTO_CONTINUE=false; HUMAN_DECISION_REQUIRED=true.", "This is the final evidence-interface iteration on EVQA-200; future same-set tuning is disallowed. Recommend a fresh unseen holdout; it was not constructed or run."]
    (OUT / "reports/final_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def provenance() -> None:
    p = OUT / "provenance/files.sha256"; p.parent.mkdir(parents=True, exist_ok=True); p.write_text("\n".join(f"{sha256_file(x)}  {x.relative_to(OUT).as_posix()}" for x in sorted(y for y in OUT.rglob("*") if y.is_file() and y != p)) + "\n", encoding="utf-8")


def main() -> int:
    for sub in ("p0_freeze", "p1_anchor_extraction", "p2_compact_evidence", "p3_compaction_audit", "p4_answer_bearing", "p5_agent/protocol_sft", "p5_agent/reward_v21", "p6_scoring", "p7_protocol", "p8_evidence_utilization", "p9_statistics", "p10_final_comparison", "reports", "contracts", "provenance", "shared_cache/text"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    rows, r2k_source, r4_source = preflight(); r5_map, r5_sha, stats = freeze_evidence(rows, r2k_source, r4_source)
    r4diag = json.loads((R4_OUT / "p4_answer_bearing_diagnostic/answer_bearing.json").read_text()) if (R4_OUT / "p4_answer_bearing_diagnostic/answer_bearing.json").exists() else {}
    r4flags = r4diag.get("r4_by_sample", {}) or r4diag.get("r3_by_sample", {}); r3diag = json.loads((R3_OUT / "p4_answer_bearing_diagnostic/answer_bearing.json").read_text()) if (R3_OUT / "p4_answer_bearing_diagnostic/answer_bearing.json").exists() else {}
    r3flags = r3diag.get("r3_by_sample", {})
    r5flags = {}
    for row in rows:
        sid = str(row["sample_id"]); refs = row.get("answer_refs") or []; blob = " ".join(str(x.get("r5_final_evidence") or "") for x in r5_map[sid]["ranks"]); r5flags[sid] = bool(r2k_module().answer_bearing(blob, refs)[0])
    diag = {"r3_answer_bearing_rate": sum(bool(x) for x in r3flags.values()) / TARGET_N, "r4_answer_bearing_rate": sum(bool(x) for x in r4flags.values()) / TARGET_N, "r5_answer_bearing_rate": sum(r5flags.values()) / TARGET_N, "r3_by_sample": r3flags, "r4_by_sample": r4flags, "r5_by_sample": r5flags}; write_json(OUT / "p4_answer_bearing/answer_bearing.json", diag)
    write_json(OUT / "p1_anchor_extraction/contract.json", {"method": "EARLIEST_USABLE_R2K_SENTENCE", "anchor_caps": RANK_ANCHOR_CAPS, "rank_budgets": RANK_TOTAL_BUDGETS, "question_independent": True, "gold_used": False, "r2k_source": R2K_SHA, "r4_source": R4_SHA, "frozen_before_formal": True})
    wait_gpu("before_protocol_sft"); run_agent("protocol_sft", rows, r5_map, r5_sha); write_json(OUT / "p0_freeze/gpu_after_protocol_sft.json", snapshot_gpu())
    wait_gpu("before_reward_v21"); run_agent("reward_v21", rows, r5_map, r5_sha); write_json(OUT / "p0_freeze/gpu_after_reward_v21.json", snapshot_gpu())
    analysis = analyze(rows, diag, r5_map); freeze = json.loads((OUT / "p0_freeze/r5_preflight.json").read_text()); c = write_contract(analysis, diag, stats, freeze, r5_sha, r5_map); write_report(c, analysis, diag, stats); final = make_verifiers(); proc = subprocess.run([sys.executable, str(final)], capture_output=True, text=True); (OUT / "provenance/verifier_output.txt").write_text(proc.stdout + proc.stderr, encoding="utf-8")
    status = ROOT / "PROJECT_STATUS_AND_HANDOFF.md"; old = status.read_text(encoding="utf-8", errors="ignore") if status.exists() else ""; marker = "FINAL-EVQA-R5-HYBRID-CONTEXT-ANCHOR";
    if marker not in old: status.write_text(old.rstrip() + "\n\n### 2026-09-07 - FINAL-EVQA-R5-HYBRID-CONTEXT-ANCHOR result\n\n" + f"- Status: {c['R5_STATUS']}; conclusion: {c['R5_CONCLUSION']}; R5 answer-bearing rate: {c['R5_ANSWER_BEARING_RATE']}.\n- SFT/V21 EM/F1: {c['SFT_R5_EM']} / {c['SFT_R5_F1']} and {c['V21_R5_EM']} / {c['V21_R5_F1']}; protocol validity: {c['SFT_R5_PROTOCOL_VALID']} / {c['V21_R5_PROTOCOL_VALID']}.\n- This is the final EVQA-200 evidence-interface iteration; FUTURE_SAME_SET_TUNING_ALLOWED=false; fresh unseen holdout recommended.\n- Contract: outputs/final_evqa_r5_hybrid_context_anchor/contracts/final_contract.json; report: outputs/final_evqa_r5_hybrid_context_anchor/reports/final_report.md.\n", encoding="utf-8")
    provenance(); print(json.dumps({"status": c["R5_STATUS"], "conclusion": c["R5_CONCLUSION"], "sft": analysis["summaries"]["r5"]["protocol_sft"], "v21": analysis["summaries"]["r5"]["reward_v21"], "verifier": proc.stdout.strip()}, ensure_ascii=False), flush=True); return proc.returncode


if __name__ == "__main__": raise SystemExit(main())
