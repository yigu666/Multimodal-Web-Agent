from __future__ import annotations
import json
import os
from pathlib import Path
import sys

ROOT = Path(os.environ.get("MWA_ROOT", Path(__file__).resolve().parents[2])).resolve()
OUT = ROOT / "outputs/final_raw_parametric_knowledge_baseline"

def read_json(p: Path):
    return json.loads(p.read_text())

def read_jsonl(p: Path):
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]

def check(condition: bool, message: str, errors: list[str]):
    if not condition: errors.append(message)

def verify() -> tuple[bool, dict]:
    errors: list[str] = []
    contract_path = OUT / "contracts/final_contract.json"
    check(contract_path.exists(), "final_contract_missing", errors)
    if errors:
        return False, {"errors": errors}
    c = read_json(contract_path)
    for key, value in {
        "status": "FINAL_RAW_PARAMETRIC_KNOWLEDGE_BASELINE_COMPLETE",
        "base_model": "Qwen2.5-VL-3B-Instruct",
        "raw_base_model_only": True,
        "peft_adapter_loaded": False,
        "training_performed": False,
        "task_training_updates": 0,
        "direct_answer_only": True,
        "tool_schema_visible": False,
        "web_evidence_visible": False,
        "tool_observations_visible": False,
        "tool_calls": 0,
        "remote_api_calls": 0,
        "model_unchanged": True,
    }.items():
        check(c.get(key) == value, f"contract_{key}={c.get(key)!r}_expected_{value!r}", errors)
    check(c.get("min_pixels") == 200704 and c.get("max_pixels") == 200704, "pixel_budget_mismatch", errors)
    check(c.get("visual_token_target") == 256, "visual_token_target_mismatch", errors)
    unified = read_jsonl(OUT / "p2_unified200_raw/raw_direct_outputs.jsonl") if (OUT / "p2_unified200_raw/raw_direct_outputs.jsonl").exists() else []
    evqa = read_jsonl(OUT / "p4_evqa200_raw/raw_direct_outputs.jsonl") if (OUT / "p4_evqa200_raw/raw_direct_outputs.jsonl").exists() else []
    check(len(unified) == 200, f"unified_n={len(unified)}", errors)
    check(len(evqa) == 200, f"evqa_n={len(evqa)}", errors)
    check(len({r.get("sample_id") for r in unified}) == len(unified), "unified_duplicate_ids", errors)
    check(len({r.get("sample_id") for r in evqa}) == len(evqa), "evqa_duplicate_ids", errors)
    required = {"sample_id", "source", "question_hash", "image_sha256", "raw_prompt_hash", "generated_text", "generated_token_count", "normalized_prediction", "normalized_gold", "em", "f1"}
    for name, rows in [("unified", unified), ("evqa", evqa)]:
        for i, row in enumerate(rows):
            check(required.issubset(row), f"{name}_row_{i}_missing_fields", errors)
            check(not row.get("tool_calls"), f"{name}_row_{i}_tool_calls", errors)
    check((OUT / "p3_o1_subset/o1_subset_outputs.jsonl").exists(), "o1_subset_index_missing", errors)
    check((OUT / "p5_scoring/unified_metrics.json").exists(), "unified_scoring_missing", errors)
    check((OUT / "p5_scoring/evqa_metrics.json").exists(), "evqa_scoring_missing", errors)
    result = {"errors": errors, "unified_n": len(unified), "evqa_n": len(evqa), "status": c.get("status")}
    return not errors, result

def main() -> int:
    ok, result = verify()
    result["marker"] = "FINAL_RAW_PARAMETRIC_KNOWLEDGE_BASELINE_VERIFY_PASS" if ok else "FINAL_RAW_PARAMETRIC_KNOWLEDGE_BASELINE_VERIFY_FAIL"
    (OUT / "contracts/final_verification.json").parent.mkdir(parents=True, exist_ok=True)
    (OUT / "contracts/final_verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))
    return 0 if ok else 1

if __name__ == "__main__":
    raise SystemExit(main())
