#!/usr/bin/env python3
from pathlib import Path
import json, os, time

ROOT = Path(os.environ.get("MWA_ROOT", Path(__file__).resolve().parents[2])).resolve()
MODEL_PATH = ROOT / 'references/runtime/huggingface/bge-m3'
OUT = ROOT / 'outputs/final_evqa_enriched_visual_agent_r2k'

def main():
    from FlagEmbedding import BGEM3FlagModel
    started = time.time()
    model = BGEM3FlagModel(str(MODEL_PATH), use_fp16=False)
    docs = ['The Eiffel Tower is in Paris and was completed in 1889.', 'Bananas are yellow fruits rich in potassium.']
    query = 'Where is the Eiffel Tower and when was it completed?'
    q = model.encode([query], batch_size=1, max_length=512, return_dense=True, return_sparse=False, return_colbert_vecs=False)['dense_vecs'][0]
    d = model.encode(docs, batch_size=2, max_length=512, return_dense=True, return_sparse=False, return_colbert_vecs=False)['dense_vecs']
    import numpy as np
    q = q / max(np.linalg.norm(q), 1e-12)
    d = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
    scores = (d @ q).tolist()
    result = {'model_path': str(MODEL_PATH), 'scores': scores, 'rank': sorted(range(len(scores)), key=lambda i: scores[i], reverse=True), 'pass': int(max(range(len(scores)), key=lambda i: scores[i])) == 0, 'elapsed_seconds': time.time()-started, 'cpu_only': True}
    (OUT/'p4_bge_audit').mkdir(parents=True, exist_ok=True)
    (OUT/'p4_bge_audit/smoke.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))
    return 0 if result['pass'] else 1
if __name__ == '__main__': raise SystemExit(main())
