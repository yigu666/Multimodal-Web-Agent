# Multimodal Web-Agent

**English** | [简体中文](README_zh-CN.md)

[![License: Apache-2.0](https://img.shields.io/badge/Code-Apache--2.0-blue.svg)](LICENSE)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-green.svg)](environment/environment.yml)
[![Base model: Qwen2.5-VL-3B](https://img.shields.io/badge/Base-Qwen2.5--VL--3B-purple.svg)](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)

> A reproducible multimodal web-agent pipeline built with protocol-format SFT and Reward-v2.1 GRPO on Qwen2.5-VL-3B-Instruct.

Multimodal Web-Agent receives an image and a question, then autonomously chooses one of three protocol actions:

```text
<answer>...</answer>
<search><img></search>
<text_search>...</text_search>
```

Search results are converted into tool observations and returned to the model for the next decision. The repository covers the successful path from public-data construction through Protocol-SFT, GRPO, optional Stage2 S2-A continuation, and frozen/replay/live evaluation.

## Highlights

- **Autonomous multimodal tool use:** answer directly, invoke visual search, or invoke text search.
- **Protocol-SFT + GRPO:** stable action serialization first, followed by answer-dominant policy optimization.
- **Real and replayable Web search:** live evaluation plus frozen/replay modes for controlled comparisons.
- **Search-aware metrics:** overall and search-required EM/F1 are reported separately.
- **Compact evidence interface:** question-aware evidence selection for small multimodal agents.
- **Auditable release:** fixed seeds, frozen contracts, checksums, selected LoRA adapters, and contract tests.

## Agent workflow

```text
Image + Question
       |
       v
Multimodal Web-Agent
       |
       +-------- answer -------------------------------> Final answer
       |
       +-------- visual search ----+
       |                            |
       +-------- text search -------+--> Web evidence
                                             |
                                             v
                                  Compact tool observation
                                             |
                                             +--> Next agent decision
```

The flow is not a fixed `image -> search -> answer` pipeline. The model decides whether to search, which tool to use, how to consume the returned evidence, and when to answer.

## Training pipeline

```text
Qwen2.5-VL-3B-Instruct
          |
          v
   Protocol-SFT v0.1
          |
          v
 Reward-v2.1 GRPO
          |
          v
Multimodal Web-Agent v0.1
          |
          +---- optional Stage2 S2-A short continuation
```

Protocol-SFT teaches action serialization, query generation, observation consumption, and answer termination. Reward-v2.1 GRPO then optimizes the tool-use and answering policy. The optional S2-A run is retained only as a successful frozen-dev research checkpoint.

## Released checkpoints

| Directory | Stage | Intended use |
|---|---|---|
| `models/protocol-sft` | Protocol-format SFT | Protocol cold start and ablation |
| `models/reward-v2.1` | Reward-v2.1 GRPO | Recommended public Web-Agent checkpoint |
| `models/stage2-s2a-step16` | Stage2 S2-A, step 16 | Successful frozen-dev research checkpoint |

Only LoRA adapters are included. Download the base model separately. Exact hashes are in [`models/CHECKSUMS.sha256`](models/CHECKSUMS.sha256).

## Verified results

Only completed, contract-passing results are shown. Values are taken from [`results/key_results.json`](results/key_results.json).

### Protocol-SFT, Dev-100

| Protocol valid | Exactly one action | Action accuracy | Macro action F1 | Malformed |
|---:|---:|---:|---:|---:|
| **100.0%** | **100.0%** | **70.0%** | **78.65%** | **0.0%** |

These are protocol-format diagnostics, not a claim of policy improvement.

### Frozen Dev-200: Reward-v2.1 and Stage2

| Model | EM | Token F1 | Search EM | Search F1 |
|---|---:|---:|---:|---:|
| Reward-v2.1 | 37.50% | 42.73% | 47.33% | 52.49% |
| Stage2 S2-A step 16 | **38.50%** | **44.00%** | **48.67%** | **54.18%** |

Stage2 is reported only as a frozen-dev result; this release does not claim that it is better than Reward-v2.1 in live deployment.

### O1-100 Live: Raw vs Reward-v2.1

All rows use the same sample IDs. Values are `EM / Token F1`.

| Subset | N | Raw | Reward Live | Delta |
|---|---:|---:|---:|---:|
| Overall | 100 | 0.2500 / 0.3141 | **0.2800 / 0.3380** | +0.0300 / +0.0239 |
| Search-free | 25 | 0.1600 / 0.1800 | **0.1600 / 0.2095** | +0.0000 / +0.0295 |
| Visual-search-required | 25 | 0.2400 / 0.3298 | **0.3200 / 0.4100** | +0.0800 / +0.0802 |
| Text-search-required | 25 | 0.3200 / 0.4033 | **0.4400 / 0.4600** | +0.1200 / +0.0567 |
| Mixed-search-required | 25 | **0.2800 / 0.3434** | 0.2000 / 0.2727 | -0.0800 / -0.0708 |
| Search-required total | 75 | 0.2800 / 0.3588 | **0.3200 / 0.3809** | +0.0400 / +0.0220 |

For 10,000 paired bootstrap samples (seed `20260905`), the Raw-minus-Reward 95% intervals are `[-0.1300, 0.0800]` for EM and `[-0.1222, 0.0738]` for F1. Both include zero, so the overall O1 improvement is reported descriptively rather than as a statistically conclusive gain.

Raw vs Reward measures the complete trained-system difference. Tool causality is instead assessed by holding Reward-v2.1 fixed:

| Subset | Reward NoTool | Reward Live | Live - NoTool |
|---|---:|---:|---:|
| Overall | 0.0200 / 0.0267 | 0.2800 / 0.3380 | +0.2600 / +0.3113 |
| Search-required | 0.0133 / 0.0222 | 0.3200 / 0.3809 | +0.3067 / +0.3587 |

### External E-VQA-200 R5: Raw vs Reward-v2.1

| System | EM | Token F1 |
|---|---:|---:|
| Raw Qwen2.5-VL-3B-Instruct | 9.00% | 12.11% |
| Reward-v2.1 R5 hybrid context anchor | **18.00%** | **23.00%** |
| Delta | **+9.00 pp** | **+10.89 pp** |

Paired bootstrap 95% intervals for Reward-minus-Raw are `[+3.50, +14.50]` percentage points for EM and `[+4.85, +16.83]` points for F1. R5 protocol validity is 90.5%, tool-use rate is 91.0%, and 169/200 episodes follow the visual-search-to-answer route.

## Agent-facing evidence interface

For a small multimodal agent, retrieving relevant evidence is not sufficient: the evidence must also fit the model's usable context. R5 uses a hybrid context anchor and question-aware selection:

```text
Web retrieval
      |
      v
Evidence selection
      |
      v
Question-aware compression + short context anchor
      |
      v
Compact observation (<= 1200 characters)
```

R5 performs no new training or reinforcement learning and does not mutate model parameters. E-VQA-200 was used for repeated evidence-interface engineering, so a fresh unseen holdout is recommended for future generalization claims.

See [`docs/RESULTS.md`](docs/RESULTS.md) for metric definitions and the full result boundary.

## Reproduce the project

All commands below run from the repository root on Linux. The verified runtime used Python 3.10.18 and CUDA 12.1 PyTorch packages. Keep model and dataset caches inside the repository if the home/system disk is space-constrained.

### 1. Clone the repository and fetch model files

```bash
git clone https://github.com/yigu666/Multimodal-Web-Agent.git
cd Multimodal-Web-Agent
git lfs install
git lfs pull
```

### 2. Create the environment and run tests

```bash
conda env create -f environment/environment.yml
conda activate multimodal-web-agent
python -m pip install -e . --no-deps

pytest -q
sha256sum -c models/CHECKSUMS.sha256
```

The curated release suite is expected to report `15 passed`.

### 3. Download the base model into the project

```bash
export MWA_ROOT="$PWD"
export HF_HOME="$MWA_ROOT/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"

hf download Qwen/Qwen2.5-VL-3B-Instruct \
  --local-dir "$MWA_ROOT/models/Qwen2.5-VL-3B-Instruct"
```

The base model is intentionally not committed to Git.

### 4. Download FVQA training inputs

```bash
mkdir -p "$MWA_ROOT/data/raw/fvqa"
hf download lmms-lab/FVQA \
  fvqa_train.parquet \
  fvqa_train_image_search_results_cache.pkl \
  --repo-type dataset \
  --revision bb4a4ff4c9c3fd0382d11f5d7fccd66d0b8428b5 \
  --local-dir "$MWA_ROOT/data/raw/fvqa"
```

No dataset is distributed in this repository. Do not unpickle files obtained from untrusted sources.

### 5. Build the Protocol-SFT dataset

```bash
python scripts/audit_fvqa_cache.py \
  --root data/raw/fvqa \
  --output-dir data/manifests/fvqa_cache_audit \
  --splits train

python scripts/build_protocol_sft_v0.py --config configs/protocol_sft/data_v0_server.yaml
python scripts/build_protocol_sft_v0.py --config configs/protocol_sft/data_v0_1_server.yaml
python scripts/build_protocol_sft_v0.py --config configs/protocol_sft/data_v0_2_server.yaml
python scripts/build_protocol_sft_v0.py --config configs/protocol_sft/data_v0_3_server.yaml
python scripts/build_protocol_sft_v0_4.py --config configs/protocol_sft/data_v0_4_server.yaml
python scripts/build_validated_master_pool.py --config configs/data_quality/validated_master_pool_v0_2_server.yaml
python scripts/build_protocol_sft_v0_5.py --config configs/protocol_sft/data_v0_5_server.yaml
python scripts/build_protocol_format_sft_v1.py --config configs/protocol_sft/data_format_v1.yaml
python scripts/audit_protocol_format_sft_v1.py \
  --data-dir data/processed/protocol_format_sft_v1 \
  --output data/manifests/protocol_format_sft_v1_audit.json
```

The final format-only view contains 900 training and 100 development state-action examples. Training does not create or read a test split.

### 6. Run Protocol-SFT

```bash
export CUDA_VISIBLE_DEVICES=0

python scripts/inspect_protocol_sft_mask.py \
  --project-root "$MWA_ROOT" \
  --config configs/protocol_sft/train_full_format_v1.yaml \
  --all-splits

python scripts/train_protocol_sft.py \
  --project-root "$MWA_ROOT" \
  --config configs/protocol_sft/train_full_format_v1.yaml
```

### 7. Run Reward-v2.1 GRPO

```bash
python scripts/build_grpo_prompt_pool_v1.py
python scripts/audit_grpo_prompt_pool.py \
  --input data/processed/grpo_prompt_pool_v1/train.jsonl \
  --output data/manifests/grpo_prompt_pool_v1_audit.json
python scripts/build_reward_v2_coverage_cache.py \
  --config configs/grpo/reward_v2_1_answer_dominant_positive.yaml

python scripts/run_grpo_reward_v2_text_exploration_smoke.py \
  --config configs/grpo/reward_v2_hierarchical_grounded_search.yaml \
  --output-dir outputs/grpo_reward_v2_text_exploration_smoke_128

python scripts/prepare_reward_v21_contract.py
python scripts/run_grpo_reward_v2_full.py \
  --training-config configs/grpo/reward_v21_full_server.yaml \
  --output-dir outputs/grpo_reward_v21_full
```

The formal GRPO run uses 2,048 prompts, group size 4, 8,192 rollouts, and 512 optimizer updates. It fails closed if a frozen artifact changes, a non-finite value appears, the protocol alignment check fails, or a visual/projector parameter changes.

### 8. Run evaluation

For Live O1-100, set the required credentials locally and never commit them:

```bash
export SERPER_API_KEY="..."
export SERPAPI_API_KEY="..."

python scripts/prepare_online_o1_100.py \
  --config configs/evaluation/online_web_agent_o1_100.yaml
python scripts/run_online_web_agent_v1.py \
  --config configs/evaluation/online_web_agent_o1_100.yaml \
  --phase o1 \
  --backend-mode live \
  --models sft reward_v21 \
  --output-root outputs/online_web_agent_o1_public
```

To reproduce the public external E-VQA path through R5, download the official metadata into this project and run every frozen stage in order:

```bash
bash scripts/download_evqa_public_inputs.sh
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

Use `--backend-mode replay` with a previously frozen evidence snapshot to avoid new remote requests. Live-Web retrieval can change over time; deterministic decoding does not make external search responses immutable. Dataset preparation, replay requirements, the optional Stage2 S2-A path, and E-VQA commands are documented in:

- [`docs/DATA.md`](docs/DATA.md)
- [`docs/TRAINING.md`](docs/TRAINING.md)
- [`docs/EVALUATION.md`](docs/EVALUATION.md)

## Repository layout

```text
Multimodal-Web-Agent/
├── configs/       # Data, training, environment, and evaluation contracts
├── docs/          # Detailed reproduction and release documentation
├── environment/   # Conda and pip dependency pins
├── evaluation/    # External E-VQA and raw-baseline pipelines
├── models/        # Three selected LoRA adapters and checksums
├── results/       # Machine-readable verified headline results
├── scripts/       # Public data/training/evaluation entry points
├── src/           # Agent, environment, training, and metric implementations
└── tests/         # Synthetic and contract tests
```

## Release boundary

Included: deterministic public-data construction, Protocol-SFT, Reward-v2.1 GRPO, the successful S2-A short checkpoint, frozen/replay/live evaluation code, compact E-VQA evaluation, selected adapters, and tests.

Excluded: raw/processed datasets, generated trajectories, Web caches, API credentials, host-specific paths, conversations, private logs, the Qwen base model, OPD/OPD2, later unsuccessful reward variants, failed checkpoints, and blocked or inconclusive branches. See [`docs/RELEASE_AUDIT.md`](docs/RELEASE_AUDIT.md).

## Acknowledgements

This project builds on [Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct), [Multimodal Search-R1](https://github.com/EvolvingLMMs-Lab/multimodal-search-r1), [InfoSeek](https://github.com/open-vision-language/infoseek), FVQA, and BGE-M3.

## License

Original project code is released under Apache-2.0; see [`LICENSE`](LICENSE). The included LoRA adapters are derivatives of Qwen2.5-VL-3B-Instruct and are governed by the Qwen Research License, including its non-commercial restriction and redistribution requirements. Read [`MODEL_LICENSE-QWEN`](MODEL_LICENSE-QWEN) and [`NOTICE`](NOTICE) before using or redistributing model files.

Improved using Qwen.

## Project status

```text
Base model:     Qwen2.5-VL-3B-Instruct
Protocol model: Protocol-SFT v0.1
Final agent:    Multimodal Web-Agent v0.1 (Reward-v2.1)
Training:       Protocol-SFT -> Reward-v2.1 GRPO
Tools:          Visual Web Search + Text Web Search
Evaluation:     FVQA + O1 Live/Frozen/Replay + E-VQA R5
```
