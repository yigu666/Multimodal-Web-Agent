#!/usr/bin/env bash
set -euo pipefail

# Download only public E-VQA metadata into the current project. Benchmark
# images are fetched later, by the frozen R1 acquisition plan.
MWA_ROOT="${MWA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RAW_DIR="$MWA_ROOT/data/external_benchmarks/encyclopedic_vqa/raw"
REF_DIR="$MWA_ROOT/references/benchmarks/encyclopedic_vqa/encyclopedic_vqa"
SOURCE_COMMIT="932d4685e23f671b9e8c2abc72dd228ba5ff9252"

mkdir -p "$RAW_DIR/gld_metadata" "$REF_DIR"

curl --location --fail --retry 3 \
  https://storage.googleapis.com/encyclopedic-vqa/test.csv \
  --output "$RAW_DIR/test.csv"
curl --location --fail --retry 3 \
  https://storage.googleapis.com/encyclopedic-vqa/lens_entities.csv \
  --output "$RAW_DIR/lens_entities.csv"
curl --location --fail --retry 3 \
  https://s3.amazonaws.com/google-landmark/metadata/train.csv \
  --output "$RAW_DIR/gld_metadata/train.csv"

printf '%s  %s\n' \
  'dbf3cf7336b7904cb0f996d2cea1762f0ae5186cd42f0c9f6a74c1c16d1d9bb5' \
  "$RAW_DIR/test.csv" \
  '348c7043c51184e327337538e889c26832081c6dc16f0f349d903f884793dd68' \
  "$RAW_DIR/lens_entities.csv" \
  | sha256sum --check --strict

# Keep the official evaluator source used for provenance/BEM reference pinned
# to the exact upstream commit. The released headline metrics are EM/F1.
curl --location --fail --retry 3 \
  "https://raw.githubusercontent.com/google-research/google-research/$SOURCE_COMMIT/encyclopedic_vqa/evaluation_utils.py" \
  --output "$REF_DIR/evaluation_utils.py"

echo "E_VQA_PUBLIC_INPUTS_READY"
