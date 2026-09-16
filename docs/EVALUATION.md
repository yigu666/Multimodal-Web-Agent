# Evaluation

## Frozen, replay, and live O1-100

The public O1 configuration evaluates only the successful SFT and Reward-v2.1 models. It never trains and never reads frozen test data.

```bash
export MWA_ROOT="$PWD"
export MWA_PYTHON="${MWA_PYTHON:-$(command -v python)}"
python scripts/prepare_online_o1_100.py --config configs/evaluation/online_web_agent_o1_100.yaml
python scripts/run_online_web_agent_v1.py \
  --config configs/evaluation/online_web_agent_o1_100.yaml \
  --phase o1 --backend-mode live \
  --models sft reward_v21 \
  --output-root outputs/online_web_agent_o1_public
```

Live mode requires `SERPER_API_KEY` and `SERPAPI_API_KEY`. Use replay mode for zero new remote requests after a live evidence snapshot, or frozen mode for the prebuilt environment.

## E-VQA staged evaluation through R5

The public E-VQA code is a staged, fail-closed pipeline. R5 consumes the exact frozen artifacts produced by the preceding stages; do not skip a stage or edit a frozen artifact between stages.

Download the public metadata first. This does not download the full image datasets; R1 later retrieves only images referenced by its deterministic candidate plan.

```bash
export MWA_ROOT="$PWD"
export MWA_PYTHON="${MWA_PYTHON:-$(command -v python)}"

bash scripts/download_evqa_public_inputs.sh

# Required only when a stage performs fresh text-Web requests.
export DASHSCOPE_API_KEY="..."

python evaluation/final_evqa_agent_compatible_external/run_evqa_r1.py prepare
python evaluation/final_evqa_agent_compatible_external/run_evqa_r1.py acquire
python evaluation/final_evqa_agent_compatible_external/run_evqa_r1.py notool protocol_sft
python evaluation/final_evqa_agent_compatible_external/run_evqa_r1.py notool reward_v21
python evaluation/final_evqa_agent_compatible_external/run_evqa_r1.py agent protocol_sft
python evaluation/final_evqa_agent_compatible_external/run_evqa_r1.py agent reward_v21
python evaluation/final_evqa_agent_compatible_external/run_evqa_r1.py finalize
python evaluation/final_evqa_enriched_visual_agent_r2k/download_kb.py
python evaluation/final_evqa_enriched_visual_agent_r2k/extract_kb.py
python evaluation/final_evqa_enriched_visual_agent_r2k/run_evqa_r2k.py
python evaluation/final_evqa_r2k_compact_r3/run_evqa_r3.py
python evaluation/final_evqa_question_aware_compact_r4/run_evqa_r4.py
python evaluation/final_evqa_r5_hybrid_context_anchor/run_evqa_r5.py
```

The E-VQA manifest, benchmark images, downloaded KB, generated evidence, and episode files stay outside Git. R5 uses the same 200 IDs as the Raw baseline and consumes frozen R2K/R4 evidence. It performs no training or RL and must leave both adapters unchanged. A byte-identical rerun requires the same pinned benchmark inputs and frozen intermediate evidence; a fresh remote search may return different Web content.

`prepare` checks the official E-VQA metadata hashes before selecting any sample. `acquire` may need to scan more than 200 candidates because unavailable upstream image URLs are rejected; it stops only after freezing an exact, valid subset.

## Raw parametric baseline

```bash
export MWA_BASE_MODEL="$PWD/models/Qwen2.5-VL-3B-Instruct"
python evaluation/final_raw_parametric_knowledge_baseline/run_raw_baseline.py
```
