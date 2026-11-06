# Data download and construction

No dataset is included in this repository. The commands below deliberately place both files and Hugging Face cache metadata inside the current project directory.

## FVQA training inputs

The formal training chain uses the official `lmms-lab/FVQA` release, pinned to the revision used by the project. It includes the official cached image-search results.

```bash
export MWA_ROOT="$PWD"
export HF_HOME="$MWA_ROOT/.cache/huggingface"
mkdir -p "$MWA_ROOT/data/raw/fvqa"
hf download lmms-lab/FVQA \
  fvqa_train.parquet \
  fvqa_train_image_search_results_cache.pkl \
  --repo-type dataset \
  --revision bb4a4ff4c9c3fd0382d11f5d7fccd66d0b8428b5 \
  --local-dir "$MWA_ROOT/data/raw/fvqa"
```

Do not unpickle files from untrusted sources. Verify the downloaded inputs before construction (Linux, from the repository root):

```bash
sha256sum data/raw/fvqa/fvqa_train.parquet data/raw/fvqa/fvqa_train_image_search_results_cache.pkl
```

Expected SHA-256 values for the pinned revision:

| File | SHA-256 |
|---|---|
| `fvqa_train.parquet` | `d23be97f4493846381f71c6953a29777fe1522aaf37942a26393605ffd78171f` |
| `fvqa_train_image_search_results_cache.pkl` | `b5e39e5504c1f9215c77a21b839cb923614bd26f405d383a944c754a725d9be6` |

Stop if either value differs. The master-pool builder records input hashes, but does not enforce the historical `expected_*_sha256` YAML fields itself.

For evaluation only, download the held-out FVQA files separately:

```bash
hf download lmms-lab/FVQA \
  fvqa_test.parquet \
  fvqa_test_image_search_results_cache.pkl \
  --repo-type dataset \
  --revision bb4a4ff4c9c3fd0382d11f5d7fccd66d0b8428b5 \
  --local-dir "$MWA_ROOT/data/raw/fvqa"
```

## Build the protocol-format view

Run the versioned builders and audits in order from the repository root. Inspect each command's result before continuing; retain existing generated artifacts and use a separate working copy when rebuilding.

```bash
python scripts/audit_fvqa_cache.py --root data/raw/fvqa --output-dir data/manifests/fvqa_cache_audit --splits train
python scripts/build_protocol_sft_v0.py --config configs/protocol_sft/data_v0_server.yaml
python scripts/build_protocol_sft_v0.py --config configs/protocol_sft/data_v0_1_server.yaml
python scripts/build_protocol_sft_v0.py --config configs/protocol_sft/data_v0_2_server.yaml
python scripts/audit_protocol_sft_v0.py --data-dir data/processed/protocol_sft_v0_2 --output data/manifests/protocol_sft_v0_2_audit.json
python scripts/build_protocol_sft_v0.py --config configs/protocol_sft/data_v0_3_server.yaml

# v0.4 requires both the v0.3 history and a validated master pool.
test -s data/processed/protocol_sft_v0_3/trajectories.jsonl
test -s data/processed/protocol_sft_v0_3/test.jsonl
python scripts/build_validated_master_pool.py --config configs/data_quality/validated_master_pool_v0_1_server.yaml
python scripts/audit_validated_master_pool.py --config configs/data_quality/validated_master_pool_v0_1_server.yaml
python scripts/build_protocol_sft_v0_4.py --config configs/protocol_sft/data_v0_4_server.yaml
python scripts/build_validated_master_pool.py --config configs/data_quality/validated_master_pool_v0_2_server.yaml
python scripts/audit_validated_master_pool.py --config configs/data_quality/validated_master_pool_v0_2_server.yaml
python scripts/build_protocol_sft_v0_5.py --config configs/protocol_sft/data_v0_5_server.yaml
python scripts/build_protocol_format_sft_v1.py --config configs/protocol_sft/data_format_v1.yaml
python scripts/audit_protocol_format_sft_v1.py --data-dir data/processed/protocol_format_sft_v1 --output data/manifests/protocol_format_sft_v1_audit.json
```

The final format-only view contains 900 train and 100 dev state-action examples; no test split is produced or read by training.

### Missing `validated_master_pool_v0_1/accepted_candidates.jsonl`

This is a generated prerequisite, not a file to download separately. The earlier public instructions omitted the v0.1 pool configuration and its build step. If v0.3 is already built, check that its `trajectories.jsonl` and `test.jsonl` exist, then run:

```bash
python scripts/build_validated_master_pool.py --config configs/data_quality/validated_master_pool_v0_1_server.yaml
python scripts/audit_validated_master_pool.py --config configs/data_quality/validated_master_pool_v0_1_server.yaml
test -s data/processed/validated_master_pool_v0_1/accepted_candidates.jsonl
test -s data/processed/validated_master_pool_v0_1/source_groups.jsonl
python scripts/build_protocol_sft_v0_4.py --config configs/protocol_sft/data_v0_4_server.yaml
```

Proceed to v0.4 only after the pool build and audit succeed. If the v0.1 configuration is absent from your checkout, obtain the updated configuration from the maintainer first. If v0.3 is absent, follow the preceding versioned builders first. An existing nonempty pool should be audited and reused rather than overwritten.

The public v0.1 configuration uses the **v0.1 pool format with the current v0.2 executability gate**, matching the current implementation. It is a current-code reconstruction and does not establish historical byte-identical data or identical metrics. The v0.4 builder still enforces route capacity and exact 800/100/100 state-action split counts. Real-data completion of this reconstruction must be checked separately; do not relabel a v0.2 pool as v0.1 or bypass an audit/count failure.

## E-VQA public evaluation inputs

E-VQA is evaluation-only and is never mixed into the training pool. Download its official public question metadata, frozen Google Lens entities, GLDv2 URL metadata, and the pinned official evaluator reference with:

```bash
export MWA_ROOT="$PWD"
bash scripts/download_evqa_public_inputs.sh
```

All files and images remain under `$MWA_ROOT/data` or `$MWA_ROOT/references`; the script does not use a home-directory cache. It verifies the two E-VQA inputs consumed directly by subset construction against the hashes frozen by this project. The later R1 `acquire` phase downloads only the exact original images selected by the frozen plan.
