# Model artifacts

The repository contains three LoRA adapters only; it does not include the Qwen base model.

| Directory | Stage | Weight SHA-256 |
|---|---|---|
| `models/protocol-sft` | Protocol-format SFT | `0e3c73c2c8f1ba23de4fb1e9a640eb3ffdea4a12a61224a52ad84c02b71a68f1` |
| `models/reward-v2.1` | Reward-v2.1 GRPO | `c566231f2a8df47146d04a4ff4ffeeb549f176cb229b897f00dc7fd257386dd9` |
| `models/stage2-s2a-step16` | Stage2 S2-A step 16 | `224bc4353cd4b3861190b85d60b62e2c90c97031e84695a8bfb6b98d267d4768` |

Each adapter is rank 16, alpha 32, dropout 0.05, with the visual modules excluded from training.

Download the base model into the project directory, never an implicit home-directory cache:

```bash
export HF_HOME="$PWD/.cache/huggingface"
hf download Qwen/Qwen2.5-VL-3B-Instruct \
  --local-dir "$PWD/models/Qwen2.5-VL-3B-Instruct"
```

The adapters are Qwen derivatives. Use and redistribution are governed by `MODEL_LICENSE-QWEN`, not by the repository's Apache-2.0 code license. Commercial use requires a separate license from Alibaba Cloud.

GitHub rejects ordinary files above 100 MB, so `.gitattributes` places Safetensors under Git LFS. A separate Hugging Face model repository is preferable for public model hosting.
