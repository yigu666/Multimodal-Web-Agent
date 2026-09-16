# Training

All commands run from the repository root. Set `CUDA_VISIBLE_DEVICES` explicitly and keep the frozen test split inaccessible during training.

## Protocol-format SFT

```bash
export CUDA_VISIBLE_DEVICES=0
python scripts/inspect_protocol_sft_mask.py \
  --project-root "$PWD" \
  --config configs/protocol_sft/train_full_format_v1.yaml \
  --all-splits
python scripts/train_protocol_sft.py \
  --project-root "$PWD" \
  --config configs/protocol_sft/train_full_format_v1.yaml
```

This stage learns action syntax/serialization. Its policy metrics are diagnostic and are not used to claim answer or routing improvement.

## Reward-v2.1 GRPO

Build the prompt pool and deterministic reward caches:

```bash
python scripts/build_grpo_prompt_pool_v1.py
python scripts/audit_grpo_prompt_pool.py \
  --input data/processed/grpo_prompt_pool_v1/train.jsonl \
  --output data/manifests/grpo_prompt_pool_v1_audit.json
python scripts/build_reward_v2_coverage_cache.py \
  --config configs/grpo/reward_v2_1_answer_dominant_positive.yaml
```

Run the successful exploration smoke, freeze the one-run contract, then start the full run:

```bash
python scripts/run_grpo_reward_v2_text_exploration_smoke.py \
  --config configs/grpo/reward_v2_hierarchical_grounded_search.yaml \
  --output-dir outputs/grpo_reward_v2_text_exploration_smoke_128
python scripts/prepare_reward_v21_contract.py
python scripts/run_grpo_reward_v2_full.py \
  --training-config configs/grpo/reward_v21_full_server.yaml \
  --output-dir outputs/grpo_reward_v21_full
```

The formal scale is 2,048 prompts, group size 4, 8,192 rollouts, and 512 optimizer updates. The visual encoder/projector remain frozen.

## Stage2 S2-A short continuation

The selected S2-A step-16 adapter is published for audit and downstream evaluation. The original continuation was gated by a byte-frozen Unified-Agent Dev bank and a historical Reward-v2.1 episode bank. Those run-specific frozen artifacts are not bundled as source code or model weights, so this optional continuation is not advertised as a clean-clone reproduction.

When the exact frozen assets are available, first run the step-0 reproduction and only then freeze the continuation contract:

```bash
python scripts/run_grpo_stage2_step0_frozen_dev.py \
  --config configs/evaluation/stage2_short_frozen_dev.yaml
python scripts/prepare_stage2_s2a_contract.py
python scripts/run_grpo_stage2.py \
  --config configs/grpo/stage2_v21_continue_short_256.yaml \
  --output-dir outputs/grpo_stage2_v21_continue_short_256
```

The evaluation config and frozen episode bank are intentionally not included in this public package; the command therefore fails closed in a clean clone instead of silently rebuilding a different selection environment. The fully reproducible public GRPO route is Reward-v2.1 above. S2-A used 256 prompts, 1,024 rollouts, 64 updates, fresh optimizer/scheduler state, half the Stage1 learning rate, and no restarted exploration; the released selection is checkpoint step 16.
