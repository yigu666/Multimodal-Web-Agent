# Multimodal Web-Agent

[English](README.md) | **简体中文**

[![License: Apache-2.0](https://img.shields.io/badge/Code-Apache--2.0-blue.svg)](LICENSE)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-green.svg)](environment/environment.yml)
[![Base model: Qwen2.5-VL-3B](https://img.shields.io/badge/Base-Qwen2.5--VL--3B-purple.svg)](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)

> 基于 Protocol-SFT 与 Reward-v2.1 GRPO，训练 Qwen2.5-VL-3B 自主调用视觉与文本 Web Search 的多模态智能体。

**Multimodal Web-Agent** 面向多模态问答与外部知识获取。模型接收图像与问题后，可自主选择直接回答、视觉搜索或文本搜索，并利用工具返回的 Evidence 继续决策，最终生成答案。

本项目公开从 **Tool-Use 数据构建 → Protocol-SFT → Reward-v2.1 GRPO → Real Web Search → Agent Evaluation** 的成功链路，并针对小参数多模态 Agent 实现紧凑的 Tool Observation Interface。

## ✨ Highlights

- **Multimodal Web-Agent：** 支持 `ANSWER / VISUAL_SEARCH / TEXT_SEARCH` 三类自主动作。
- **Real Web Search：** 支持真实视觉与文本 Web Search，而不仅是模拟 Tool Calling。
- **Protocol-SFT + GRPO：** 先完成 Agent 协议冷启动，再优化工具使用和最终回答。
- **Search-aware Evaluation：** 分别评估 Search-required 与 Search-free 样本。
- **Agent-facing Evidence Interface：** 对 Web Evidence 进行问题相关筛选与上下文压缩。
- **Reproducible Evaluation：** 支持 Raw、NoTool、Frozen、Replay 与 Live-Web 等评估条件。
- **Auditable Release：** 固定随机种子、冻结执行契约、Adapter 校验和与合成测试。

## 🧠 Agent Overview

核心协议动作如下：

```text
<answer>...</answer>
<search><img></search>
<text_search>...</text_search>
```

基本运行流程：

```text
          Image + Question
                 |
                 v
        Multimodal Web-Agent
                 |
            Action Decision
          +------+------+ 
          |      |      |
          v      v      v
       ANSWER  VISUAL   TEXT
               SEARCH  SEARCH
                  |      |
                  +--+---+
                     |
                     v
                  Real Web
                     |
                     v
             Compact Observation
                     |
                     v
             Next Agent Decision
```

这不是固定的 `Image → Search → Answer` 流水线。是否搜索、使用哪种工具、如何利用搜索结果以及何时回答，都由模型自主决定。

## 🚀 Training Pipeline

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

### Stage 1 — Protocol-SFT

第一阶段基于 FVQA 构建结构化 Tool-Use 数据，使基础模型学习动作格式、搜索 Query 生成、Tool Observation 消费、搜索后的后续决策与最终答案生成。

在 Protocol-SFT Dev-100 上：

| Metric | Protocol-SFT v0.1 |
|---|---:|
| Protocol Validity | **100.0%** |
| Exactly One Action | **100.0%** |
| Action Accuracy | **70.0%** |
| Macro Action F1 | **78.65%** |
| Malformed Rate | **0.0%** |

这些数值用于验证协议格式学习，不代表回答策略已经提升。

### Stage 2 — Reward-v2.1 GRPO

第二阶段从 Protocol-SFT Adapter 初始化，使用 Answer-dominant Reward-v2.1 优化 Tool-Use Policy 与 Web-assisted Answering。正式训练规模为 2,048 个 Prompt、group size 4、8,192 条 rollout 和 512 次优化更新。

Frozen Dev-200：

| Model | EM | Token-F1 | Search EM | Search F1 | Protocol Valid | Missing |
|---|---:|---:|---:|---:|---:|---:|
| Reward-v2.1 | 37.50% | 42.73% | 47.33% | 52.49% | 92.50% | 9.50% |
| Stage2 S2-A step 16 | **38.50%** | **44.00%** | **48.67%** | **54.18%** | **93.50%** | **8.50%** |

Stage2 只作为通过 Frozen Dev 的研究检查点公开；本项目不声称它在真实在线部署中优于 Reward-v2.1。

## 🌐 Real Web Search Evaluation

### O1-100 Live：Raw vs Reward-v2.1

以下比较使用完全相同的 100 个样本 ID，所有单元格均为 `EM / Token-F1`：

| 子集 | N | Raw | Reward Live | Reward - Raw |
|---|---:|---:|---:|---:|
| Overall | 100 | 0.2500 / 0.3141 | **0.2800 / 0.3380** | +0.0300 / +0.0239 |
| Search-free | 25 | 0.1600 / 0.1800 | **0.1600 / 0.2095** | +0.0000 / +0.0295 |
| Visual-search-required | 25 | 0.2400 / 0.3298 | **0.3200 / 0.4100** | +0.0800 / +0.0802 |
| Text-search-required | 25 | 0.3200 / 0.4033 | **0.4400 / 0.4600** | +0.1200 / +0.0567 |
| Mixed-search-required | 25 | **0.2800 / 0.3434** | 0.2000 / 0.2727 | -0.0800 / -0.0708 |
| Search-required 合计 | 75 | 0.2800 / 0.3588 | **0.3200 / 0.3809** | +0.0400 / +0.0220 |

Reward-v2.1 在视觉搜索和文本搜索子集上提升明显，但在 Mixed-search-required 上低于 Raw；Search-free 的 EM 不变，仅 F1 提升。

基于相同 100 个 O1 ID、10,000 次 bootstrap、seed `20260905`，Raw-minus-Reward 的区间为：

| Metric | Mean Difference | 95% CI |
|---|---:|---:|
| EM | -0.0300 | [-0.1300, 0.0800] |
| Token-F1 | -0.0239 | [-0.1222, 0.0738] |

两个区间都跨过 0，因此 Overall O1 提升只作描述性报告，不表述为统计显著结论。

### Tool Utility：Reward Live vs Reward NoTool

Raw 与 Reward 的差异反映完整训练系统差异，不能单独归因于工具。工具的因果参考需要固定同一个 Reward-v2.1 模型：

| 子集 | Reward NoTool | Reward Live | Live - NoTool |
|---|---:|---:|---:|
| Overall | 0.0200 / 0.0267 | 0.2800 / 0.3380 | +0.2600 / +0.3113 |
| Search-required | 0.0133 / 0.0222 | 0.3200 / 0.3809 | +0.3067 / +0.3587 |

Frozen 与 Replay 是历史参考条件，也不能作为 Raw 的直接工具因果对照。完整分路由、数据来源和配对统计见 [`docs/RESULTS.md`](docs/RESULTS.md)。

## 🖼️ External Multimodal Evaluation — E-VQA R5

R5 使用 Hybrid Context Anchor + Question-Aware Evidence 构造不超过 1,200 字符的紧凑 Tool Observation：

```text
Visual Web Search
       |
       v
Retrieved Web Evidence
       |
       v
Question-Aware Evidence + Context Anchor
       |
       v
Compact Tool Observation
       |
       v
Multimodal Web-Agent
```

在完全相同的 200 个 E-VQA 样本上：

| Model | EM | Token-F1 |
|---|---:|---:|
| Raw Qwen2.5-VL-3B | 9.00% | 12.11% |
| Reward-v2.1 R5 | **18.00%** | **23.00%** |
| Delta | **+9.00 pp** | **+10.89 pp** |

对应 Reward-minus-Raw 的 paired bootstrap 95% CI：

```text
EM:       [+3.50pp, +14.50pp]
Token-F1: [+4.85pp, +16.83pp]
```

Reward R5 的 Protocol Validity 为 90.5%，工具使用率为 91.0%，主要执行路线为 `Visual Search → Answer`（169/200）。同一 E-VQA-200 已用于多轮 Evidence Interface 工程开发，因此项目建议后续使用新的未见 Holdout 验证泛化能力。

## 🔎 Agent-facing Evidence Interface

对于小参数多模态 Agent，检索到 Evidence 并不等于模型能够正确利用 Evidence。R5 的接口采用：

```text
Web Retrieval
      |
      v
Evidence Selection
      |
      v
Question-aware Compression
      +
Short Context Anchor
      |
      v
Compact Observation (<= 1200 chars)
```

R5 不进行新训练或新强化学习，不修改模型参数；它研究的是如何把冻结检索证据组织为模型更容易消费的 Observation。

## 📊 Key Results

| Stage | Comparison | EM | Token-F1 |
|---|---|---:|---:|
| Protocol-SFT | Protocol Validity / Action Accuracy | 100.0% | 70.0% |
| O1 Live | Raw | 25.00% | 31.41% |
| O1 Live | Reward-v2.1 | **28.00%** | **33.80%** |
| O1 Search-required | Raw | 28.00% | 35.88% |
| O1 Search-required | Reward-v2.1 | **32.00%** | **38.09%** |
| E-VQA-200 | Raw | 9.00% | 12.11% |
| E-VQA-200 R5 | Reward-v2.1 | **18.00%** | **23.00%** |

## 📦 Released Models

| Model | Base | Training Stage | 目录 |
|---|---|---|---|
| **Protocol-SFT v0.1** | Qwen2.5-VL-3B-Instruct | Protocol-SFT | `models/protocol-sft` |
| **Multimodal Web-Agent v0.1** | Protocol-SFT v0.1 | Reward-v2.1 GRPO | `models/reward-v2.1` |
| **Stage2 S2-A step 16** | Reward-v2.1 | Short continuation | `models/stage2-s2a-step16` |

仓库只包含 LoRA Adapter，不包含基础模型。精确哈希见 [`models/CHECKSUMS.sha256`](models/CHECKSUMS.sha256)。推荐公开 Web-Agent 使用 `models/reward-v2.1`；Stage2 作为研究检查点提供。

## 🛠️ Installation and Reproduction

以下命令均在 Linux 仓库根目录执行。已验证环境为 Python 3.10.18 与 CUDA 12.1 对应的 PyTorch。若系统盘空间有限，下述设置会将模型、数据和 Hugging Face 缓存全部放在项目目录。

### 1. 克隆仓库并获取 Git LFS 模型

```bash
git clone https://github.com/yigu666/Multimodal-Web-Agent.git
cd Multimodal-Web-Agent
git lfs install
git lfs pull
```

### 2. 创建环境并验证发布包

```bash
conda env create -f environment/environment.yml
conda activate multimodal-web-agent
python -m pip install -e . --no-deps

pytest -q
sha256sum -c models/CHECKSUMS.sha256
```

精选测试应输出 `15 passed`。

### 3. 将基础模型下载到项目目录

```bash
export MWA_ROOT="$PWD"
export HF_HOME="$MWA_ROOT/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"

hf download Qwen/Qwen2.5-VL-3B-Instruct \
  --local-dir "$MWA_ROOT/models/Qwen2.5-VL-3B-Instruct"
```

### 4. 下载 FVQA 训练输入

```bash
mkdir -p "$MWA_ROOT/data/raw/fvqa"
hf download lmms-lab/FVQA \
  fvqa_train.parquet \
  fvqa_train_image_search_results_cache.pkl \
  --repo-type dataset \
  --revision bb4a4ff4c9c3fd0382d11f5d7fccd66d0b8428b5 \
  --local-dir "$MWA_ROOT/data/raw/fvqa"
```

仓库不包含数据集。请勿反序列化来源不可信的 Pickle 文件。

### 5. 构建 Protocol-SFT 数据

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

最终格式数据包含 900 条训练与 100 条 Dev state-action 样本；训练不生成或读取测试集。

### 6. 运行 Protocol-SFT

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

### 7. 运行 Reward-v2.1 GRPO

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

训练契约会持续检查冻结输入、非有限数值、行为 Log-prob 对齐、环境信息泄漏、视觉/投影层冻结和检查点保存/重载；任一工程门控失败都会停止训练。

### 8. 运行 O1 Live / Replay 评估

Live 模式需要本地搜索服务凭据，禁止提交到 Git：

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

如需复现公开的 E-VQA R1→R5 外部评估，先把官方元数据下载到项目盘，然后严格按顺序运行全部冻结阶段：

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

已有冻结证据快照时，可改用 `--backend-mode replay`，避免产生新的远程请求。Live-Web 搜索结果会随时间变化；确定性解码不能使外部搜索响应永久不变。

数据细节、可选 Stage2 S2-A 和 E-VQA 复现前置条件见：

- [`docs/DATA.md`](docs/DATA.md)
- [`docs/TRAINING.md`](docs/TRAINING.md)
- [`docs/EVALUATION.md`](docs/EVALUATION.md)
- [`docs/RESULTS.md`](docs/RESULTS.md)

## 📁 Repository Structure

```text
Multimodal-Web-Agent/
├── configs/       # 数据、训练、环境与评估契约
├── docs/          # 详细复现和发布文档
├── environment/   # Conda 与 pip 依赖版本
├── evaluation/    # 外部 E-VQA 与 Raw Baseline 流程
├── models/        # 三个 LoRA Adapter 及校验和
├── results/       # 可机器读取的已验证关键结果
├── scripts/       # 数据、训练和评估入口
├── src/           # Agent、环境、训练与指标实现
└── tests/         # 合成测试与执行契约测试
```

## 📌 Release Scope

公开内容：确定性公共数据构建、Protocol-SFT、Reward-v2.1 GRPO、成功的 S2-A 短续训检查点、Frozen/Replay/Live 评估代码、E-VQA 评估、筛选后的 Adapter 和测试。

不公开内容：原始/处理后数据、生成轨迹、Web Cache、API 凭据、机器路径、对话、私有日志、Qwen 基础模型、OPD/OPD2、后续未成功 Reward 变体、失败检查点及被阻塞或结论不充分的实验。详见 [`docs/RELEASE_AUDIT.md`](docs/RELEASE_AUDIT.md)。

## 🙏 Acknowledgements

本项目建立在 [Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)、[Multimodal Search-R1](https://github.com/EvolvingLMMs-Lab/multimodal-search-r1)、[InfoSeek](https://github.com/open-vision-language/infoseek)、FVQA 和 BGE-M3 等工作之上。

## 📄 License

本项目原创代码以 Apache-2.0 发布，见 [`LICENSE`](LICENSE)。仓库内 LoRA Adapter 是 Qwen2.5-VL-3B-Instruct 的衍生模型，受 Qwen Research License 的非商业限制和再分发要求约束。使用或再分发模型前，请阅读 [`MODEL_LICENSE-QWEN`](MODEL_LICENSE-QWEN) 与 [`NOTICE`](NOTICE)。

Improved using Qwen.

## 🌟 Project Status

```text
Base Model:     Qwen2.5-VL-3B-Instruct
Protocol Model: Protocol-SFT v0.1
Final Agent:    Multimodal Web-Agent v0.1 (Reward-v2.1)
Training:       Protocol-SFT -> Reward-v2.1 GRPO
Tools:          Visual Web Search + Text Web Search
Evaluation:     FVQA + O1 Live/Frozen/Replay + E-VQA R5
```
