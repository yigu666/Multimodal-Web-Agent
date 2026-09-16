# Release audit

The project history was reviewed across four workstreams: dataset/SFT, GRPO, later optimization experiments, and real-web evaluation.

Included:

- Deterministic public-data acquisition and construction.
- Protocol-format SFT and its selected adapter.
- Reward-v2.1 GRPO and its selected adapter.
- Stage2 S2-A short continuation and selected step-16 adapter.
- Contract-passing frozen, live O1, raw-baseline, and staged E-VQA R1-to-R5 evaluation code/results.

Excluded:

- Raw or processed datasets and generated trajectories.
- Private credentials, host identifiers, SSH material, conversations, and local handoff reports.
- Caches, logs, temporary files, bytecode, tarballs, and third-party repository snapshots.
- OPD/OPD2, later Reward variants, controlled continuation, and other failed, blocked, or inconclusive branches.
- Failed checkpoints and negative-result reports from those branches.
- The Qwen base model.

The original workspace is untouched. This directory is a standalone publication candidate and should receive one final owner/license review before pushing to GitHub.

## Release verification

- All 322 Python files passed AST parsing.
- The curated suite passed: `15 passed` under the verified Python 3.10.18 server interpreter.
- Credential, private-key, server-address, and absolute-project-path scans returned no hits.
- No dataset, generated JSONL, pickle, Parquet, log, archive, bytecode, or cache file is present.
- All six adapter/configuration hashes match `models/CHECKSUMS.sha256`.
- The temporary server-side validation directory and archive were removed after testing.
