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

Do not unpickle files from untrusted sources. The pipeline checks the pinned files before construction.

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

Run the versioned builders in order; each refuses unsafe overwrite and writes manifests/checksums:

```bash
python scripts/audit_fvqa_cache.py --root data/raw/fvqa --output-dir data/manifests/fvqa_cache_audit --splits train
python scripts/build_protocol_sft_v0.py --config configs/protocol_sft/data_v0_server.yaml
python scripts/build_protocol_sft_v0.py --config configs/protocol_sft/data_v0_1_server.yaml
python scripts/build_protocol_sft_v0.py --config configs/protocol_sft/data_v0_2_server.yaml
python scripts/build_protocol_sft_v0.py --config configs/protocol_sft/data_v0_3_server.yaml
python scripts/build_protocol_sft_v0_4.py --config configs/protocol_sft/data_v0_4_server.yaml
python scripts/build_validated_master_pool.py --config configs/data_quality/validated_master_pool_v0_2_server.yaml
python scripts/build_protocol_sft_v0_5.py --config configs/protocol_sft/data_v0_5_server.yaml
python scripts/build_protocol_format_sft_v1.py --config configs/protocol_sft/data_format_v1.yaml
python scripts/audit_protocol_format_sft_v1.py --data-dir data/processed/protocol_format_sft_v1 --output data/manifests/protocol_format_sft_v1_audit.json
```

The final format-only view contains 900 train and 100 dev state-action examples; no test split is produced or read by training.

## E-VQA public evaluation inputs

E-VQA is evaluation-only and is never mixed into the training pool. Download its official public question metadata, frozen Google Lens entities, GLDv2 URL metadata, and the pinned official evaluator reference with:

```bash
export MWA_ROOT="$PWD"
bash scripts/download_evqa_public_inputs.sh
```

All files and images remain under `$MWA_ROOT/data` or `$MWA_ROOT/references`; the script does not use a home-directory cache. It verifies the two E-VQA inputs consumed directly by subset construction against the hashes frozen by this project. The later R1 `acquire` phase downloads only the exact original images selected by the frozen plan.
