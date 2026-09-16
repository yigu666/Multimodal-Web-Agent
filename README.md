# Multimodal Web-Agent

[English](README.md) | [简体中文](README_zh-CN.md)

> Built with **Qwen2.5-VL-3B**, **Protocol-SFT**, and **GRPO**, Multimodal Web-Agent enables a multimodal model to autonomously call real visual/text Web Search and use external knowledge for visual question answering.

**Multimodal Web-Agent** targets knowledge-intensive multimodal question answering.

Unlike a fixed Image → Search → Answer pipeline, the model decides from the current image, question, and tool context whether to:

- answer directly;
- issue a visual Web Search;
- issue a text Web Search;
- consume the Tool Observation, continue reasoning, and produce the final answer.

The project covers the complete path:

~~~text
Tool-Use data construction
          ↓
     Protocol-SFT
          ↓
          GRPO
          ↓
Multimodal Web-Agent
          ↓
      Real Web Search
          ↓
    Agent Evaluation
~~~

For 3B-scale multimodal models, the project also provides a compact **Agent-facing Evidence Interface** that makes retrieved Web evidence easier to consume.

---

## ✨ Highlights

- **Multimodal Web-Agent:** supports three autonomous actions: <code>ANSWER / VISUAL_SEARCH / TEXT_SEARCH</code>.
- **Real Web Search:** executes real visual and text Web Search rather than simulated tool calling.
- **Protocol-SFT + GRPO:** establishes reliable agent-protocol behavior first, then optimizes tool use and search-assisted answering.
- **Search-aware evaluation:** reports Search-free and Search-required subsets separately to check whether search gains preserve base capability.
- **External multimodal evaluation:** evaluates the final system on knowledge-intensive visual questions from E-VQA.
- **Agent-facing Evidence Interface:** applies question-aware evidence selection, context anchoring, and compression to reduce the observation burden of a small model.
- **Reproducible pipeline:** includes data builders, training entry points, Raw/NoTool/Frozen/Replay/Live-Web evaluation, and execution-contract tests.

---

## 🧠 Agent Overview

The agent action space is:

~~~text
ANSWER
VISUAL_SEARCH
TEXT_SEARCH
~~~

Overall execution:

~~~mermaid
flowchart LR
    A[Image + Question] --> B[Multimodal Web-Agent]
    B --> C{Action}
    C -->|ANSWER| G[Final Answer]
    C -->|VISUAL_SEARCH| D[Visual Web Search]
    C -->|TEXT_SEARCH| E[Text Web Search]
    D --> F[Tool Observation]
    E --> F
    F --> B
~~~

When the model chooses a search action, the Agent Runtime executes the corresponding Web Search, organizes the returned evidence as a Tool Observation, and sends it back for the next decision. Search policy is learned by the model rather than hard-coded as an external search sequence.

---

## 🚀 Training Pipeline

The public training path has two stages:

~~~text
Qwen2.5-VL-3B-Instruct
          │
          ▼
     Protocol-SFT
          │
          ▼
          GRPO
          │
          ▼
Multimodal Web-Agent v0.1
~~~

The final GRPO agent is referred to publicly as **Multimodal Web-Agent v0.1**.

Historical scripts, configurations, and artifacts may still contain identifiers such as <code>reward_v21</code>. Those identifiers preserve reproducibility of the original runs and are not public model names.

### Stage 1 — Protocol-SFT

The first stage builds structured Tool-Use data from **FVQA** and applies Protocol-SFT to the base model.

It teaches:

- Tool Action formatting;
- visual/text search-query generation;
- Tool Observation consumption;
- continued decisions after search;
- final-answer generation.

Public Protocol-SFT Dev-100 results:

| Metric | Protocol-SFT |
|---|---:|
| Protocol Validity | **100.0%** |
| Exactly One Action | **100.0%** |
| Action Accuracy | **70.0%** |
| Macro Action F1 | **78.65%** |
| Malformed Rate | **0.0%** |

These metrics verify structured Tool-Use behavior and are not a claim of answer-policy improvement.

### Stage 2 — GRPO

Starting from Protocol-SFT, GRPO further optimizes:

- Tool-Use Policy;
- search/answer decisions;
- answer quality on Search-required examples;
- Web-assisted answering.

The formal GRPO run uses:

~~~text
Prompts:           2,048
Group Size:        4
Total Rollouts:    8,192
Optimizer Updates: 512
~~~

Relative to Protocol-SFT, the largest gains appear on examples that require external information:

| Metric | Protocol-SFT | Multimodal Web-Agent v0.1 | Δ |
|---|---:|---:|---:|
| Overall EM | 33.50% | **37.50%** | +4.00pp |
| Overall Token-F1 | 39.16% | **42.73%** | +3.57pp |
| **Search-required EM** | 40.67% | **47.33%** | **+6.66pp** |
| **Search-required Token-F1** | 45.97% | **52.49%** | **+6.52pp** |

This indicates that the main GRPO benefit is concentrated in tool-use and external-knowledge scenarios rather than being only an overall-score shift.

See [docs/TRAINING.md](docs/TRAINING.md) for the complete training contract and commands.

---

## 🌐 Real Web Search Evaluation

Calling a tool does not by itself show that the tool improves the task. We therefore report two comparisons:

1. **Raw → Final Agent:** the change from the complete training and agent system;
2. **Live Web → NoTool:** the task-level utility of Web access while holding the same trained agent fixed.

### O1-100 Live-Web

O1-100 contains Search-free, Visual-search-required, Text-search-required, and Mixed-search-required examples.

The headline table below focuses on Search-free and the two single-tool search-required subsets. All rows use the same sample IDs and report **EM / Token-F1 (%)**.

| Subset | Raw | Multimodal Web-Agent v0.1 | Δ |
|---|---:|---:|---:|
| Search-free | 16.00 / 18.00 | **16.00 / 20.95** | +0.00 / +2.95 |
| Visual-search-required | 24.00 / 32.98 | **32.00 / 41.00** | **+8.00 / +8.02** |
| Text-search-required | 32.00 / 40.33 | **44.00 / 46.00** | **+12.00 / +5.67** |
| **Single-tool Search-required** | 28.00 / 36.66 | **38.00 / 43.50** | **+10.00 / +6.85** |

Single-tool Search-required combines the 25 visual-search-required and 25 text-search-required examples.

The final agent keeps Search-free EM unchanged at **16.0%**, while improving visual-search-required EM from **24.0% to 32.0%** and text-search-required EM from **32.0% to 44.0%**.

> Full O1 results—including Mixed-search-required, all Search-required samples, source splits, Frozen/Replay conditions, and bootstrap statistics—are reported in [docs/RESULTS.md](docs/RESULTS.md).

### Web Tool Utility: Live vs NoTool

Raw versus the final agent includes both training and system changes, so it is not a tool-only causal comparison. To isolate Web utility, we hold **Multimodal Web-Agent v0.1** fixed and only disable Web access:

| Condition | EM | Token-F1 |
|---|---:|---:|
| NoTool | 1.33% | 2.22% |
| Live Web | **32.00%** | **38.09%** |
| Gain | **+30.67pp** | **+35.87pp** |

Thus, Raw → Final Agent measures the complete training/system difference, while Same Agent: Live Web → NoTool measures task-level Web-tool utility.

---

## 🖼️ E-VQA: External-Knowledge Multimodal Evaluation

To test the system on an external visual-question distribution, we use a 200-sample single-hop Agent-compatible subset of E-VQA.

E-VQA emphasizes:

- fine-grained visual-entity understanding;
- external encyclopedic knowledge;
- joint use of image information and retrieved evidence.

We therefore call it **External-Knowledge Multimodal Evaluation**, rather than labeling it as another Search-required split.

### Raw → Final Agent

On the same 200 examples:

| Model | EM | Token-F1 |
|---|---:|---:|
| Raw Qwen2.5-VL-3B | 9.00% | 12.11% |
| **Multimodal Web-Agent v0.1** | **18.00%** | **23.00%** |
| Δ | **+9.00pp** | **+10.89pp** |

Paired bootstrap 95% confidence intervals:

~~~text
EM:
+9.00pp
95% CI: [+3.50pp, +14.50pp]

Token-F1:
+10.89pp
95% CI: [+4.85pp, +16.83pp]
~~~

The final system obtains:

- EM: **9.0% → 18.0%**;
- Token-F1: **12.11% → 23.0%**;
- Protocol Validity: **90.5%**;
- Tool-use Rate: **91.0%**.

See [docs/RESULTS.md](docs/RESULTS.md) and [docs/EVALUATION.md](docs/EVALUATION.md) for full statistics and staged commands.

> **Evaluation note.** E-VQA-200 was also used during Evidence Interface engineering. We therefore report it as an external Agent-compatible evaluation/development set, not as an untouched final test set. A new unseen holdout should be used for a stricter generalization claim after the system is frozen.

---

## 🔎 Agent-facing Evidence Interface

For a small multimodal agent, retrieving correct evidence does not guarantee that the model can consume it effectively.

The final interface is:

~~~mermaid
flowchart LR
    A[Web Retrieval] --> B[Question-Aware Evidence Selection]
    B --> C[Short Context Anchor]
    C --> D[Compact Tool Observation]
    D --> E[Multimodal Web-Agent]
~~~

The model-visible Tool Observation is reduced from approximately:

~~~text
~2.5K characters
        ↓
~0.9K characters
~~~

| Metric | Long Evidence | Compact Evidence Interface |
|---|---:|---:|
| Protocol Validity | 51.0% | **90.5%** |
| EM | 6.0% | **18.0%** |
| Token-F1 | 7.78% | **23.0%** |

R5 performs no new training or reinforcement learning and does not modify model parameters. It demonstrates that tool quality is both a retrieval problem and an interface problem: evidence selection, context compression, and semantic grounding are needed after Web retrieval.

---

## 📊 Key Results

| Stage | Metric | Before / Raw | Final |
|---|---|---:|---:|
| Protocol-SFT | Protocol Validity | — | **100.0%** |
| Protocol-SFT | Action Accuracy | — | **70.0%** |
| GRPO | Search-required EM | 40.67% | **47.33%** |
| GRPO | Search-required Token-F1 | 45.97% | **52.49%** |
| O1 Live-Web | Visual-search-required EM | 24.0% | **32.0%** |
| O1 Live-Web | Text-search-required EM | 32.0% | **44.0%** |
| O1 Live-Web | Single-tool Search-required EM | 28.0% | **38.0%** |
| O1 Live-Web | Search-free EM | 16.0% | **16.0%** |
| E-VQA-200 | EM | 9.0% | **18.0%** |
| E-VQA-200 | Token-F1 | 12.11% | **23.0%** |
| Evidence Interface | Protocol Validity | 51.0% | **90.5%** |

Detailed results and comparison boundaries are in [docs/RESULTS.md](docs/RESULTS.md).

---

## 🏷️ Model Naming

Public model names are:

| Public Name | Description |
|---|---|
| **Protocol-SFT v0.1** | The first-stage model for Tool-Use Protocol cold-start |
| **Multimodal Web-Agent v0.1** | The final agent trained from Protocol-SFT with GRPO |

To preserve historical runs, code and configuration may still contain:

~~~text
reward_v21
grpo_reward_v21
reward_v2_1
~~~

These are implementation identifiers, not separate public model names. The optional Stage2 research checkpoint is documented separately in [docs/MODELS.md](docs/MODELS.md) and [docs/RESULTS.md](docs/RESULTS.md); it is not part of the recommended public path.

---

## ⚡ Quick Start

### 1. Clone

~~~bash
git clone https://github.com/yigu666/Multimodal-Web-Agent.git
cd Multimodal-Web-Agent
~~~

### 2. Create the environment

~~~bash
conda env create -f environment/environment.yml
conda activate multimodal-web-agent

python -m pip install -e . --no-deps
~~~

### 3. Verify the public code

~~~bash
pytest -q
~~~

See [docs/INSTALL.md](docs/INSTALL.md) for dependency details and runtime notes.

---

## 📥 Base Model

The base model is not redistributed with this repository. Download it separately into the project directory:

~~~bash
export MWA_ROOT="$PWD"
export HF_HOME="$MWA_ROOT/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"

hf download Qwen/Qwen2.5-VL-3B-Instruct \
  --local-dir "$MWA_ROOT/models/Qwen2.5-VL-3B-Instruct"
~~~

If disk space is limited, point the Hugging Face cache explicitly at a large project volume.

---

## 📚 Data

Raw datasets are not redistributed.

Protocol-SFT and GRPO training inputs are built from **FVQA** and can be downloaded from Hugging Face:

~~~bash
mkdir -p "$MWA_ROOT/data/raw/fvqa"

hf download lmms-lab/FVQA \
  fvqa_train.parquet \
  fvqa_train_image_search_results_cache.pkl \
  --repo-type dataset \
  --revision bb4a4ff4c9c3fd0382d11f5d7fccd66d0b8428b5 \
  --local-dir "$MWA_ROOT/data/raw/fvqa"
~~~

See [docs/DATA.md](docs/DATA.md) for downloads, cache audits, Protocol-SFT builders, and E-VQA public inputs.

> Never unpickle files from an untrusted source.

---

## 🏋️ Training

### Protocol-SFT

Inspect the loss mask before training:

~~~bash
export CUDA_VISIBLE_DEVICES=0

python scripts/inspect_protocol_sft_mask.py \
  --project-root "$PWD" \
  --config configs/protocol_sft/train_full_format_v1.yaml \
  --all-splits
~~~

Run Protocol-SFT:

~~~bash
python scripts/train_protocol_sft.py \
  --project-root "$PWD" \
  --config configs/protocol_sft/train_full_format_v1.yaml
~~~

### GRPO

Build and audit the prompt pool:

~~~bash
python scripts/build_grpo_prompt_pool_v1.py

python scripts/audit_grpo_prompt_pool.py \
  --input data/processed/grpo_prompt_pool_v1/train.jsonl \
  --output data/manifests/grpo_prompt_pool_v1_audit.json
~~~

Prepare the reward/training contract:

~~~bash
python scripts/build_reward_v2_coverage_cache.py \
  --config configs/grpo/reward_v2_1_answer_dominant_positive.yaml

python scripts/run_grpo_reward_v2_text_exploration_smoke.py \
  --config configs/grpo/reward_v2_hierarchical_grounded_search.yaml \
  --output-dir outputs/grpo_reward_v2_text_exploration_smoke_128

python scripts/prepare_reward_v21_contract.py
~~~

Run formal GRPO:

~~~bash
python scripts/run_grpo_reward_v2_full.py \
  --training-config configs/grpo/reward_v21_full_server.yaml \
  --output-dir outputs/grpo_reward_v21_full
~~~

Script names retain historical experiment identifiers so the verified runs remain reproducible. The public final model name is **Multimodal Web-Agent v0.1**.

See [docs/TRAINING.md](docs/TRAINING.md) for the complete procedure.

---

## 📏 Evaluation

### O1 Live-Web

Live mode requires local Web Search credentials. **Never commit API keys to Git.**

~~~bash
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
~~~

If a frozen evidence snapshot is available, use Replay or Frozen mode to reduce new remote requests. Live-Web responses change over time; deterministic decoding cannot make third-party search responses immutable.

### Raw baseline

~~~bash
export MWA_BASE_MODEL="$PWD/models/Qwen2.5-VL-3B-Instruct"

python evaluation/final_raw_parametric_knowledge_baseline/run_raw_baseline.py
~~~

### E-VQA

Download the public inputs first:

~~~bash
bash scripts/download_evqa_public_inputs.sh
~~~

E-VQA is a staged, fail-closed evaluation. R5 consumes the frozen artifacts produced by preceding stages.

See [docs/EVALUATION.md](docs/EVALUATION.md) for the complete R1 → R5 procedure. Benchmark images, KB files, generated evidence, and episode outputs remain outside Git.

---

## 🧪 Reproducibility

The release keeps execution contracts and audits at key stages:

- fixed data and artifact hashes;
- Protocol and loss-mask checks;
- GRPO prompt-pool audit;
- Frozen and Replay evaluation;
- Raw baseline;
- fail-closed staged E-VQA;
- synthetic and contract tests;
- machine-readable headline results.

Headline results are also stored in:

~~~text
results/key_results.json
~~~

Detailed interpretation is in [docs/RESULTS.md](docs/RESULTS.md).

---

## 📁 Repository Structure

~~~text
Multimodal-Web-Agent/
├── configs/                 # Data, training, and evaluation configurations
├── docs/                    # Install, data, training, evaluation, and results docs
├── environment/             # Conda and pip environment definitions
├── evaluation/              # E-VQA and Raw-baseline evaluation flows
├── models/                  # LoRA adapters and checksums
├── results/                 # Machine-readable verified headline results
├── scripts/                 # Data, training, and evaluation entry points
├── src/
│   └── multimodal_web_agent/
│                              # Agent, environment, training, and metrics
├── tests/                   # Synthetic and contract tests
├── LICENSE
├── MODEL_LICENSE-QWEN
├── NOTICE
├── pyproject.toml
├── README.md
└── README_zh-CN.md
~~~

---

## 📖 Documentation

Recommended reading order:

1. [docs/INSTALL.md](docs/INSTALL.md) — environment and dependencies
2. [docs/DATA.md](docs/DATA.md) — data downloads and construction
3. [docs/TRAINING.md](docs/TRAINING.md) — Protocol-SFT and GRPO
4. [docs/EVALUATION.md](docs/EVALUATION.md) — O1, Raw, and E-VQA
5. [docs/RESULTS.md](docs/RESULTS.md) — complete results and comparison boundaries
6. [docs/RELEASE_AUDIT.md](docs/RELEASE_AUDIT.md) — release audit information

---

## 📌 Public Release Scope

This repository publishes the reproducible core training and evaluation path:

- public-data acquisition and construction code;
- Protocol-SFT;
- GRPO;
- the Multimodal Web-Agent runtime;
- Real/Frozen/Replay Web evaluation;
- the Raw parametric baseline;
- staged E-VQA evaluation;
- result aggregation and evaluation code;
- synthetic and contract tests.

The public release does not include:

- private API credentials;
- raw or fully processed datasets;
- raw Web caches;
- private logs or conversations;
- host-specific files and paths;
- Qwen base-model weights;
- non-public intermediate research artifacts.

See [docs/RELEASE_AUDIT.md](docs/RELEASE_AUDIT.md) for release boundaries. The recommended public model is Multimodal Web-Agent v0.1; historical reward identifiers remain only where needed by reproducibility scripts.

---

## 🙏 Acknowledgements

This project uses or references:

- [Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
- [Multimodal Search-R1](https://github.com/EvolvingLMMs-Lab/multimodal-search-r1)
- [Search-R1](https://github.com/PeterGriffinJin/Search-R1)
- [FVQA](https://huggingface.co/datasets/lmms-lab/FVQA)
- InfoSeek
- E-VQA
- [BGE-M3](https://huggingface.co/BAAI/bge-m3)

Where applicable, external code is identified as adapted from its upstream project; related work and benchmarks are cited as inspiration or evaluation sources.

---

## 📄 License

Original repository code is released under the **Apache License 2.0**; see [LICENSE](LICENSE).

The base model and derived model files are governed by the applicable Qwen terms, including the [official Qwen Research License](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/blob/main/LICENSE). See [MODEL_LICENSE-QWEN](MODEL_LICENSE-QWEN) and [NOTICE](NOTICE) before using or redistributing model-related files.

---

## 🌟 Current Release

~~~text
Base Model:
Qwen2.5-VL-3B-Instruct

Protocol Model:
Protocol-SFT v0.1

Final Agent:
Multimodal Web-Agent v0.1

Training:
Protocol-SFT → GRPO

Tools:
Visual Web Search + Text Web Search

Main Evaluation:
FVQA + O1 Live/Frozen/Replay + E-VQA
~~~

If this project is useful for your research or engineering work, please consider starring the repository, opening an issue, or contributing improvements.

