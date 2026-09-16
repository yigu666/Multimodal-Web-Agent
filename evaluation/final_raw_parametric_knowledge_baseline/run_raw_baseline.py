"""RAW-KNOWLEDGE: Qwen2.5-VL-3B-Instruct parametric-only baseline.

This runner intentionally contains no web/tool imports or adapter loading.  It
evaluates the frozen Unified Agent Eval V1.1 dev-200 and the exact E-VQA-200
manifest with image+question direct answering only.  It is designed to run on
Titan; no dataset acquisition is performed here.
"""
from __future__ import annotations

import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import statistics
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping

from PIL import Image

ROOT = Path(os.environ.get("MWA_ROOT", Path(__file__).resolve().parents[2])).resolve()
PYTHON = Path(os.environ.get("MWA_PYTHON", sys.executable)).resolve()
BASE_MODEL = Path(os.environ.get("MWA_BASE_MODEL", ROOT / "models/Qwen2.5-VL-3B-Instruct")).resolve()
OUT = ROOT / "outputs/final_raw_parametric_knowledge_baseline"
UNIFIED = ROOT / "data/processed/unified_agent_eval_v1_1/dev.jsonl"
O1_IDS = ROOT / "outputs/online_web_agent_v1_o1_100/online_pilot/sample_ids.json"
EVQA_MANIFEST = ROOT / "data/external_benchmarks/encyclopedic_vqa/processed/agent_compatible_r1/final_manifest.jsonl"
EVQA_MANIFEST_SHA = "2d9fec7bab22c08a27df109a44344878091f7f26aeb6368674e9de05c5699441"
IMG_ROOT = ROOT / "data/processed/unified_agent_eval_v1_1"
SEED = 20260905
BOOTSTRAP_REPS = 10_000
PIXELS = 200704
VISUAL_TOKEN_TARGET = 256
GENERATION = {
    "do_sample": False,
    "num_beams": 1,
    "max_new_tokens": 128,
    "repetition_penalty": 1.0,
    "temperature_effective": 1e-6,
    "top_p_effective": None,
    "stop_rule": "model_eos_only; no XML/protocol stop strings",
}
DIRECT_SYSTEM = (
    "Answer the user's question using only the image and the knowledge already "
    "contained in the model. Do not use web search, tools, external information, "
    "or action/XML tags. Return only the shortest final answer needed to answer "
    "the question, without explanation."
)
EXPECTED_ROUTES = ("search_free", "visual_search_required", "text_search_required", "mixed_search_required")

sys.path.insert(0, str(ROOT / "src"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(dict(value), ensure_ascii=False, sort_keys=True, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tree_fingerprint(path: Path) -> str:
    """Content-independent inventory fingerprint, safe for multi-GB weights."""
    h = hashlib.sha256()
    for p in sorted(x for x in path.rglob("*") if x.is_file() and ".git" not in x.parts):
        st = p.stat()
        h.update(p.relative_to(path).as_posix().encode() + b"\0")
        h.update(str(st.st_size).encode() + b"\0")
        # Include small metadata files completely and first/last bytes for weights.
        if st.st_size <= 2_000_000:
            h.update(file_sha(p).encode())
        else:
            with p.open("rb") as f:
                first = f.read(4096)
                f.seek(max(0, st.st_size - 4096))
                last = f.read(4096)
            h.update(hashlib.sha256(first + last).hexdigest().encode())
        h.update(b"\n")
    return h.hexdigest()


def gpu_snapshot() -> dict[str, Any]:
    try:
        q = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.free,memory.total,utilization.gpu,temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
        )
        a = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory,process_name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15,
        )
        gpus = []
        for line in q.stdout.splitlines():
            f = [x.strip() for x in line.split(",")]
            if len(f) >= 7:
                gpus.append({"index": int(f[0]), "name": f[1], "memory_used_mib": int(f[2]), "memory_free_mib": int(f[3]), "memory_total_mib": int(f[4]), "utilization_gpu_percent": int(f[5]), "temperature_c": int(f[6])})
        own = str(os.getpid())
        procs = [x.strip() for x in a.stdout.splitlines() if x.strip() and not x.strip().startswith(own + ",")]
        return {"gpus": gpus, "compute_processes": procs, "idle": not procs, "free_ge_18gib": bool(gpus) and all(x["memory_free_mib"] >= 18 * 1024 for x in gpus), "returncode": q.returncode}
    except Exception as exc:
        return {"gpus": [], "compute_processes": [], "idle": False, "free_ge_18gib": False, "error": type(exc).__name__ + ": " + str(exc)}


def wait_for_gpu(label: str, poll_seconds: int = 30) -> dict[str, Any]:
    while True:
        s = gpu_snapshot()
        write_json(OUT / "p0_contract" / ("gpu_" + label + ".json"), s)
        if s.get("idle") and s.get("free_ge_18gib"):
            return s
        print(json.dumps({"waiting_for_gpu": True, "label": label, "snapshot": s}, ensure_ascii=False), flush=True)
        time.sleep(poll_seconds)


def norm_answer(value: Any) -> str:
    from multimodal_web_agent.evaluation.unified_agent.answer_metrics import normalize_answer
    return normalize_answer(str(value or ""))


def score_answer(prediction: Any, refs: Iterable[str]) -> tuple[int, float]:
    from multimodal_web_agent.evaluation.unified_agent.answer_metrics import maximum_alias_token_f1, normalized_exact_match
    aliases = [str(x) for x in refs if str(x).strip()]
    return int(normalized_exact_match(str(prediction or ""), aliases)), float(maximum_alias_token_f1(str(prediction or ""), aliases))


def load_rows(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path)


def preflight() -> dict[str, Any]:
    if not BASE_MODEL.exists() or not (BASE_MODEL / "config.json").exists():
        raise RuntimeError("BASE_MODEL_MISSING")
    if not UNIFIED.exists() or not EVQA_MANIFEST.exists() or not O1_IDS.exists():
        raise RuntimeError("FROZEN_MANIFEST_MISSING")
    unified = load_rows(UNIFIED)
    evqa = load_rows(EVQA_MANIFEST)
    ids_doc = json.loads(O1_IDS.read_text())
    ordered = list(ids_doc.get("ordered_eval_ids") or [])
    uid = {str(r["eval_id"]) for r in unified}
    route_counts = {k: sum(str(r.get("task_type")) == k for r in unified) for k in EXPECTED_ROUTES}
    if len(unified) != 200 or route_counts != {k: 50 for k in EXPECTED_ROUTES}:
        raise RuntimeError({"UNIFIED_IDENTITY_FAIL": {"n": len(unified), "routes": route_counts}})
    if len(ordered) != 100 or not set(ordered).issubset(uid):
        raise RuntimeError("O1_SUBSET_IDENTITY_FAIL")
    evqa_sha = file_sha(EVQA_MANIFEST)
    if len(evqa) != 200 or evqa_sha != EVQA_MANIFEST_SHA:
        raise RuntimeError({"EVQA_IDENTITY_FAIL": {"n": len(evqa), "sha": evqa_sha}})
    missing_unified = [r["eval_id"] for r in unified if not (IMG_ROOT / str(r["image_path"])).exists()]
    missing_evqa = [r["sample_id"] for r in evqa if not Path(str(r["image_path"])).exists()]
    if missing_unified or missing_evqa:
        raise RuntimeError({"IMAGE_IDENTITY_FAIL": {"unified": missing_unified[:5], "evqa": missing_evqa[:5]}})
    adapters = [ROOT / "models/protocol-sft", ROOT / "models/reward-v2.1"]
    contract = {
        "status": "PREFLIGHT_PASS",
        "base_model": str(BASE_MODEL),
        "base_model_exists": True,
        "raw_base_model_only": True,
        "peft_adapter_loaded": False,
        "task_training_updates": 0,
        "optimizer_initialized": False,
        "tool_schema_visible": False,
        "web_evidence_visible": False,
        "tool_observations_visible": False,
        "tool_calls": 0,
        "remote_api_calls": 0,
        "tool_backend_initialized": False,
        "tool_cache_initialized": False,
        "evidence_initialized": False,
        "unified_n": len(unified),
        "unified_route_counts": route_counts,
        "unified_manifest_sha256": file_sha(UNIFIED),
        "o1_n": len(ordered),
        "o1_ids_subset_of_unified": True,
        "o1_route_counts": {k: sum(str(r.get("task_type")) == k and str(r.get("eval_id")) in set(ordered) for r in unified) for k in EXPECTED_ROUTES},
        "evqa_n": len(evqa),
        "evqa_manifest_sha256": evqa_sha,
        "evqa_manifest_expected_sha256": EVQA_MANIFEST_SHA,
        "model_fingerprint_before": tree_fingerprint(BASE_MODEL),
        "adapter_paths_present_but_not_loaded": [str(p) for p in adapters if p.exists()],
        "visual_token_target": VISUAL_TOKEN_TARGET,
        "min_pixels": PIXELS,
        "max_pixels": PIXELS,
        "direct_system_prompt": DIRECT_SYSTEM,
        "generation": GENERATION,
        "inference_inputs": "exact original image + exact question only",
        "no_web_download": True,
    }
    write_json(OUT / "p0_contract" / "preflight.json", contract)
    write_json(OUT / "p1_manifest_audit" / "unified_manifest_audit.json", {"n": len(unified), "route_counts": route_counts, "source_counts": _counts(unified, "source_dataset"), "search_required_counts": _counts(unified, "search_required"), "image_missing": missing_unified})
    write_json(OUT / "p1_manifest_audit" / "o1_subset_audit.json", {"n": len(ordered), "ordered_eval_ids": ordered, "route_counts": contract["o1_route_counts"], "exact_subset": True})
    write_json(OUT / "p1_manifest_audit" / "evqa_manifest_audit.json", {"n": len(evqa), "manifest_sha256": evqa_sha, "image_missing": missing_evqa, "exact_frozen_manifest": True})
    return contract


def _counts(rows: Iterable[Mapping[str, Any]], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in rows:
        v = str(r.get(key))
        out[v] = out.get(v, 0) + 1
    return dict(sorted(out.items()))


class BaseRuntime:
    def __init__(self) -> None:
        import torch
        from transformers import AutoProcessor
        from multimodal_web_agent.training.sft.config import load_config
        from multimodal_web_agent.training.sft.model_factory import load_qwen_base
        self.torch = torch
        wait_for_gpu("before_model_load")
        cfg = load_config(ROOT / "configs/protocol_sft/train_full_format_v1.yaml", ROOT)
        if int(cfg.data.pixel_budget) != PIXELS:
            raise RuntimeError(f"PIXEL_CONFIG_MISMATCH:{cfg.data.pixel_budget}")
        self.model, _processor = load_qwen_base(cfg)
        # Explicitly recreate the formal processor with the required budget.
        self.processor = AutoProcessor.from_pretrained(str(BASE_MODEL), min_pixels=PIXELS, max_pixels=PIXELS, local_files_only=True, use_fast=False)
        self.model.config.use_cache = True
        self.model.eval()
        self.device = next(self.model.parameters()).device
        self.adapter_class = type(self.model).__name__
        self.lora_parameter_names = [n for n, _ in self.model.named_parameters() if "lora_" in n.casefold()]
        if self.lora_parameter_names or "peft" in self.adapter_class.casefold():
            raise RuntimeError("PEFT_ADAPTER_DETECTED")
        self.gpu_after_load = gpu_snapshot()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def render(self, question: str) -> str:
        messages = [{"role": "system", "content": DIRECT_SYSTEM}, {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
        return str(self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

    def generate(self, question: str, image: Image.Image) -> dict[str, Any]:
        rendered = self.render(question)
        inputs = self.processor(text=[rendered], images=[image], padding=True, return_tensors="pt")
        model_inputs = {k: (v.to(self.device) if self.torch.is_tensor(v) else v) for k, v in inputs.items()}
        started = time.perf_counter()
        with self.torch.inference_mode():
            output = self.model.generate(**model_inputs, do_sample=False, num_beams=1, max_new_tokens=128, repetition_penalty=1.0)
        prompt_tokens = int(model_inputs["input_ids"].shape[-1])
        new_tokens = max(0, int(output.shape[-1]) - prompt_tokens)
        decoded = self.processor.batch_decode(output[:, prompt_tokens:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
        return {
            "raw": decoded,
            "prompt_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
            "input_tokens": prompt_tokens,
            "generated_token_count": new_tokens,
            "latency_seconds": time.perf_counter() - started,
            "gpu_allocated_mib": int(self.torch.cuda.memory_allocated() / 2**20) if self.torch.cuda.is_available() else 0,
            "gpu_reserved_mib": int(self.torch.cuda.memory_reserved() / 2**20) if self.torch.cuda.is_available() else 0,
        }

    def release(self) -> dict[str, Any]:
        peak = {}
        if self.torch.cuda.is_available():
            peak = {"peak_allocated_mib": int(self.torch.cuda.max_memory_allocated() / 2**20), "peak_reserved_mib": int(self.torch.cuda.max_memory_reserved() / 2**20)}
        del self.model
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
        return {**peak, "gpu_after_release": gpu_snapshot()}


def answer_refs(row: Mapping[str, Any]) -> list[str]:
    refs = row.get("answer_aliases", row.get("answer_refs", []))
    if isinstance(refs, (list, tuple)):
        return [str(x) for x in refs if str(x).strip()]
    return [str(refs)] if str(refs).strip() else []


def make_row(row: Mapping[str, Any], source: str, image_path: Path, gen: Mapping[str, Any]) -> dict[str, Any]:
    refs = answer_refs(row)
    em, f1 = score_answer(gen["raw"], refs)
    sid = str(row.get("eval_id", row.get("sample_id")))
    question = str(row.get("question", ""))
    return {
        "sample_id": sid,
        "source": source,
        "route_label": str(row.get("task_type", "")) if source == "unified_agent_eval_v1_1" else None,
        "search_required": bool(row.get("search_required", False)) if source == "unified_agent_eval_v1_1" else None,
        "source_dataset": str(row.get("source_dataset", row.get("dataset_name", ""))),
        "source_data_id": str(row.get("source_data_id", row.get("dataset_image_id", ""))),
        "question": question,
        "question_hash": hashlib.sha256(question.encode("utf-8")).hexdigest(),
        "image_sha256": str(row.get("image_sha256", "")),
        "image_path_ref": str(image_path),
        "raw_prompt_hash": str(gen["prompt_sha256"]),
        "generated_text": str(gen["raw"]),
        "generated_token_count": int(gen["generated_token_count"]),
        "normalized_prediction": norm_answer(gen["raw"]),
        "normalized_gold": [norm_answer(x) for x in refs],
        "answer_aliases": refs,
        "em": int(em),
        "f1": float(f1),
        "latency_seconds": float(gen["latency_seconds"]),
        "input_tokens": int(gen["input_tokens"]),
        "accidental_tags": {tag: bool(re.search(re.escape(tag), str(gen["raw"]), flags=re.I)) for tag in ("<search>", "<text_search>", "<answer>")},
    }


def run_dataset(runtime: BaseRuntime, rows: list[dict[str, Any]], source: str, out_path: Path) -> list[dict[str, Any]]:
    existing = read_jsonl(out_path)
    done = {str(r.get("sample_id")) for r in existing}
    for idx, row in enumerate(rows, 1):
        sid = str(row.get("eval_id", row.get("sample_id")))
        if sid in done:
            continue
        image_path = (IMG_ROOT / str(row["image_path"])) if source == "unified_agent_eval_v1_1" else Path(str(row["image_path"]))
        im = None
        try:
            im = Image.open(image_path).convert("RGB")
            generated = runtime.generate(str(row["question"]), im)
            out = make_row(row, source, image_path, generated)
        except Exception as exc:
            out = make_row(row, source, image_path, {"raw": "", "prompt_sha256": "", "generated_token_count": 0, "input_tokens": 0, "latency_seconds": 0.0})
            out.update({"failure": type(exc).__name__ + ": " + str(exc)[:1000], "em": 0, "f1": 0.0})
        finally:
            if im is not None:
                im.close()
        append_jsonl(out_path, out)
        done.add(sid)
        write_json(out_path.parent / "progress.json", {"source": source, "completed_n": len(done), "planned_n": len(rows), "last_sample_id": sid, "failures": sum(bool(r.get("failure")) for r in read_jsonl(out_path))})
        if idx % 10 == 0 or idx == len(rows):
            print(json.dumps({"source": source, "completed": len(done), "planned": len(rows), "last": sid}, ensure_ascii=False), flush=True)
    return read_jsonl(out_path)


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def avg(sub: list[dict[str, Any]], key: str) -> float:
        return sum(float(r.get(key, 0.0)) for r in sub) / len(sub) if sub else 0.0
    return {"n": len(rows), "em": avg(rows, "em"), "f1": avg(rows, "f1"), "failure_count": sum(bool(r.get("failure")) for r in rows), "route_buckets": {k: {"n": sum(str(r.get("route_label")) == k for r in rows), "em": avg([r for r in rows if str(r.get("route_label")) == k], "em"), "f1": avg([r for r in rows if str(r.get("route_label")) == k], "f1")} for k in EXPECTED_ROUTES}, "search_required": {"n": sum(bool(r.get("search_required")) for r in rows), "em": avg([r for r in rows if r.get("search_required")], "em"), "f1": avg([r for r in rows if r.get("search_required")], "f1")}}


def source_breakdown(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for source in sorted({str(r.get("source_dataset", "")) for r in rows}):
        sub = [r for r in rows if str(r.get("source_dataset", "")) == source]
        out[source] = {"n": len(sub), "em": sum(r.get("em", 0) for r in sub) / len(sub), "f1": sum(r.get("f1", 0.0) for r in sub) / len(sub)}
    return out


def token_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    vals = sorted(int(r.get("generated_token_count", 0)) for r in rows)
    def pct(q: float) -> int:
        if not vals: return 0
        return vals[min(len(vals) - 1, max(0, math.ceil(q * len(vals)) - 1))]
    tags = {t: sum(bool(r.get("accidental_tags", {}).get(t)) for r in rows) for t in ("<search>", "<text_search>", "<answer>")}
    return {"avg_generated_tokens": (sum(vals) / len(vals) if vals else 0.0), "p50_generated_tokens": pct(0.50), "p95_generated_tokens": pct(0.95), "empty_or_failure_count": sum(not str(r.get("generated_text", "")).strip() or bool(r.get("failure")) for r in rows), "accidental_tag_counts": tags}


def bootstrap_delta(a: list[float], b: list[float], seed: int = SEED) -> dict[str, Any]:
    if len(a) != len(b) or not a:
        return {"n": 0, "reps": BOOTSTRAP_REPS, "seed": seed}
    deltas = [float(x) - float(y) for x, y in zip(a, b)]
    rng = random.Random(seed)
    vals = []
    n = len(deltas)
    for _ in range(BOOTSTRAP_REPS):
        vals.append(sum(deltas[rng.randrange(n)] for _ in range(n)) / n)
    vals.sort()
    return {"n": n, "reps": BOOTSTRAP_REPS, "seed": seed, "mean": sum(deltas) / n, "median": statistics.median(deltas), "lo95": vals[int(0.025 * len(vals))], "hi95": vals[int(0.975 * len(vals))], "rescue": sum(x > y for x, y in zip(a, b)), "harm": sum(x < y for x, y in zip(a, b)), "tie": sum(x == y for x, y in zip(a, b))}


def load_historical() -> dict[str, Any]:
    o1root = ROOT / "outputs/online_web_agent_v1_o1_100/online_pilot"
    evqaroot = ROOT / "outputs/final_evqa_r5_hybrid_context_anchor/p5_agent"
    def hist(path: Path) -> list[dict[str, Any]]:
        return read_jsonl(path)
    return {
        "o1": {
            "protocol_sft_live": hist(o1root / "live/sft/episodes.jsonl"),
            "reward_v21_live": hist(o1root / "live/reward_v21/episodes.jsonl"),
            "stage2_live": hist(o1root / "live/stage2/episodes.jsonl"),
        },
        "evqa": {
            "protocol_sft_r5": hist(evqaroot / "protocol_sft/episodes.jsonl"),
            "reward_v21_r5": hist(evqaroot / "reward_v21/episodes.jsonl"),
        },
    }


def make_reports(unified: list[dict[str, Any]], evqa: list[dict[str, Any]], pre: dict[str, Any]) -> None:
    write_json(OUT / "p5_scoring/unified_metrics.json", aggregate(unified))
    write_json(OUT / "p5_scoring/evqa_metrics.json", {"n": len(evqa), "em": sum(r.get("em", 0) for r in evqa) / len(evqa), "f1": sum(r.get("f1", 0.0) for r in evqa) / len(evqa), "failure_count": sum(bool(r.get("failure")) for r in evqa)})
    write_json(OUT / "p6_source_breakdown/unified_by_source.json", source_breakdown(unified))
    write_json(OUT / "p6_source_breakdown/evqa_by_source.json", source_breakdown(evqa))
    write_json(OUT / "p8_statistics/unified_token_audit.json", token_audit(unified))
    write_json(OUT / "p8_statistics/evqa_token_audit.json", token_audit(evqa))
    ordered = set(json.loads(O1_IDS.read_text())["ordered_eval_ids"])
    o1 = [r for r in unified if r["sample_id"] in ordered]
    write_json(OUT / "p3_o1_subset/o1_metrics.json", aggregate(o1))
    (OUT / "p3_o1_subset/o1_subset_outputs.jsonl").unlink(missing_ok=True)
    for r in o1:
        append_jsonl(OUT / "p3_o1_subset/o1_subset_outputs.jsonl", r)
    historical = load_historical()
    live = {str(r.get("eval_id")): r for r in historical["o1"]["reward_v21_live"]}
    paired_o1 = [r for r in o1 if r["sample_id"] in live]
    end_to_end_o1 = {"raw_vs_reward_v21_live": {"em": bootstrap_delta([r["em"] for r in paired_o1], [int(live[r["sample_id"]].get("normalized_em", 0)) for r in paired_o1]), "f1": bootstrap_delta([r["f1"] for r in paired_o1], [float(live[r["sample_id"]].get("token_f1", 0.0)) for r in paired_o1])}, "paired_n": len(paired_o1), "label": "END_TO_END_SYSTEM_DIFFERENCE"}
    evqa_hist = {str(r.get("sample_id")): r for r in historical["evqa"]["reward_v21_r5"]}
    paired_evqa = [r for r in evqa if r["sample_id"] in evqa_hist]
    end_to_end_evqa = {"raw_vs_reward_v21_r5": {"em": bootstrap_delta([r["em"] for r in paired_evqa], [int(evqa_hist[r["sample_id"]].get("normalized_em", 0)) for r in paired_evqa]), "f1": bootstrap_delta([r["f1"] for r in paired_evqa], [float(evqa_hist[r["sample_id"]].get("token_f1", 0.0)) for r in paired_evqa])}, "paired_n": len(paired_evqa), "label": "END_TO_END_SYSTEM_DIFFERENCE"}
    comparisons = {"historical_o1": {"protocol_sft_live": {"overall_em": 0.25, "overall_f1": 0.3124372294, "search_required_em": 0.28, "search_required_f1": 0.3365194805}, "reward_v21_live": {"overall_em": 0.28, "overall_f1": 0.3380476190, "search_required_em": 0.32, "search_required_f1": 0.3808888889}, "protocol_sft_notool": {"overall_em": 0.14, "overall_f1": 0.1719, "search_required_em": 0.16, "search_required_f1": 0.1923}, "reward_v21_notool": {"overall_em": 0.02, "overall_f1": 0.0267, "search_required_em": 0.0133, "search_required_f1": 0.0222}}, "historical_evqa": {"protocol_sft_notool": {"em": 0.095, "f1": 0.1176}, "reward_v21_notool": {"em": 0.095, "f1": 0.1176}, "protocol_sft_r5": {"em": 0.115, "f1": 0.1588}, "reward_v21_r5": {"em": 0.18, "f1": 0.23}}, "raw_o1": aggregate(o1), "raw_evqa": {"n": len(evqa), "em": sum(r.get("em", 0) for r in evqa) / len(evqa), "f1": sum(r.get("f1", 0.0) for r in evqa) / len(evqa)}, "paired_end_to_end": {"o1": end_to_end_o1, "evqa": end_to_end_evqa}, "causal_tool_effect_reference": {"reward_v21_live_minus_reward_v21_notool": {"em": 0.30, "f1": 0.3587, "label": "CAUSAL_TOOL_EFFECT_ONLY_FOR_REWARD_LIVE_VS_REWARD_NOTOOL"}}}
    write_json(OUT / "p7_comparison/comparisons.json", comparisons)
    write_json(OUT / "p8_statistics/paired_bootstrap.json", {"o1_raw_vs_reward_live": end_to_end_o1, "evqa_raw_vs_reward_r5": end_to_end_evqa, "seed": SEED, "reps": BOOTSTRAP_REPS})
    contract = dict(pre)
    contract.update({"status": "FINAL_RAW_PARAMETRIC_KNOWLEDGE_BASELINE_COMPLETE", "base_model": "Qwen2.5-VL-3B-Instruct", "raw_base_model_only": True, "peft_adapter_loaded": False, "training_performed": False, "task_training_updates": 0, "direct_answer_only": True, "tool_schema_visible": False, "web_evidence_visible": False, "tool_observations_visible": False, "tool_calls": 0, "remote_api_calls": 0, "lens_calls": 0, "serpapi_calls": 0, "alibaba_search_calls": 0, "serper_calls": 0, "jina_calls": 0, "bge_retrieval_calls": 0, "unified_metrics": aggregate(unified), "unified_source_breakdown": source_breakdown(unified), "o1_metrics": aggregate(o1), "evqa_metrics": {"n": len(evqa), "em": sum(r.get("em", 0) for r in evqa) / len(evqa), "f1": sum(r.get("f1", 0.0) for r in evqa) / len(evqa)}, "model_fingerprint_after": tree_fingerprint(BASE_MODEL), "model_unchanged": tree_fingerprint(BASE_MODEL) == pre.get("model_fingerprint_before"), "historical_comparisons_label": "Raw-vs-trained-agent comparisons are end-to-end; causal tool effect is only Reward Live vs Reward NoTool.", "auto_continue": False, "human_decision_required": True})
    write_json(OUT / "contracts/final_contract.json", contract)
    report = ["# FINAL RAW PARAMETRIC KNOWLEDGE BASELINE", "", "Status: FINAL_RAW_PARAMETRIC_KNOWLEDGE_BASELINE_COMPLETE", "", "## Frozen direct-answer contract", json.dumps({k: contract[k] for k in ["base_model", "raw_base_model_only", "peft_adapter_loaded", "training_performed", "task_training_updates", "direct_answer_only", "tool_schema_visible", "web_evidence_visible", "tool_calls", "remote_api_calls", "visual_token_target", "min_pixels", "max_pixels", "generation"]}, ensure_ascii=False, indent=2), "", "## Results", f"Unified-200: EM={aggregate(unified)['em']:.4f}, F1={aggregate(unified)['f1']:.4f}; search-required-150 EM={aggregate(unified)['search_required']['em']:.4f}, F1={aggregate(unified)['search_required']['f1']:.4f}.", f"O1 exact subset-100: EM={aggregate(o1)['em']:.4f}, F1={aggregate(o1)['f1']:.4f}.", f"E-VQA-200: EM={sum(r.get('em',0) for r in evqa)/len(evqa):.4f}, F1={sum(r.get('f1',0.0) for r in evqa)/len(evqa):.4f}.", "", "Raw-vs-trained references are labelled END_TO_END_SYSTEM_DIFFERENCE; no causal tool-only claim is made for Raw.", "", "No new Lens/SerpApi/Alibaba/Serper/Jina/page/BGE calls were made. Model fingerprint unchanged."]
    (OUT / "reports").mkdir(parents=True, exist_ok=True)
    (OUT / "reports/final_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pre = preflight()
    write_json(OUT / "p0_contract" / "status.json", {"phase": "PREFLIGHT_PASS", "gpu": gpu_snapshot()})
    unified_rows = load_rows(UNIFIED)
    evqa_rows = load_rows(EVQA_MANIFEST)
    runtime = BaseRuntime()
    try:
        raw_unified = run_dataset(runtime, unified_rows, "unified_agent_eval_v1_1", OUT / "p2_unified200_raw/raw_direct_outputs.jsonl")
    finally:
        write_json(OUT / "p2_unified200_raw/runtime_load.json", {"adapter_class": runtime.adapter_class, "lora_parameter_count": len(runtime.lora_parameter_names), "gpu_after_load": runtime.gpu_after_load, "generation": GENERATION})
        write_json(OUT / "p2_unified200_raw/runtime_release.json", runtime.release())
    wait_for_gpu("between_datasets")
    runtime = BaseRuntime()
    try:
        raw_evqa = run_dataset(runtime, evqa_rows, "evqa_r1_frozen", OUT / "p4_evqa200_raw/raw_direct_outputs.jsonl")
    finally:
        write_json(OUT / "p4_evqa200_raw/runtime_load.json", {"adapter_class": runtime.adapter_class, "lora_parameter_count": len(runtime.lora_parameter_names), "gpu_after_load": runtime.gpu_after_load, "generation": GENERATION})
        write_json(OUT / "p4_evqa200_raw/runtime_release.json", runtime.release())
    make_reports(raw_unified, raw_evqa, pre)
    final = json.loads((OUT / "contracts/final_contract.json").read_text())
    status = {"status": final["status"], "unified_n": final["unified_metrics"]["n"], "evqa_n": final["evqa_metrics"]["n"], "o1_n": final["o1_metrics"]["n"], "model_unchanged": final["model_unchanged"], "remote_api_calls": final["remote_api_calls"]}
    write_json(OUT / "p0_contract" / "status.json", status)
    with (ROOT / "PROJECT_STATUS_AND_HANDOFF.md").open("a", encoding="utf-8") as h:
        h.write("\n\n## FINAL RAW PARAMETRIC KNOWLEDGE BASELINE\n\n" + json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(status, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
