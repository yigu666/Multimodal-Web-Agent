# Key results and comparison boundaries

Only completed results are reported. Comparisons are made only when the dataset and sample IDs are identical. Every score cell below is `EM / Token-F1` unless noted otherwise. Machine-readable values are in `results/key_results.json`.

## Protocol-SFT Dev-100

| Metric | Value |
|---|---:|
| Protocol valid rate | 1.0000 |
| Exactly one action | 1.0000 |
| Action type accuracy | 0.7000 |
| Macro action F1 | 0.7865 |
| Answer exact match | 0.4615 |
| Malformed rate | 0.0000 |

These are protocol-format diagnostics, not a policy-improvement claim.

## Frozen Dev-200

| Model | EM | Token F1 | Search EM | Search F1 | Protocol | Missing |
|---|---:|---:|---:|---:|---:|---:|
| Reward-v2.1 | 0.3750 | 0.4273 | 0.4733 | 0.5249 | 0.9250 | 0.0950 |
| Stage2 S2-A step 16 | 0.3850 | 0.4400 | 0.4867 | 0.5418 | 0.9350 | 0.0850 |

Stage2 is a successful frozen-dev research checkpoint. It is not presented as a better live deployment checkpoint.

## O1-100: Raw vs Reward-v2.1

### Route subsets

| Subset | N | Raw | Reward Live | Reward Frozen | Reward Replay |
|---|---:|---:|---:|---:|---:|
| Overall | 100 | 0.2500 / 0.3141 | 0.2800 / 0.3380 | 0.3600 / 0.4418 | 0.2800 / 0.3447 |
| Search-free | 25 | 0.1600 / 0.1800 | 0.1600 / 0.2095 | 0.0800 / 0.1641 | 0.1600 / 0.2095 |
| Visual-search-required | 25 | 0.2400 / 0.3298 | 0.3200 / 0.4100 | 0.4800 / 0.5737 | 0.3200 / 0.4100 |
| Text-search-required | 25 | 0.3200 / 0.4033 | 0.4400 / 0.4600 | 0.4800 / 0.5973 | 0.4400 / 0.4867 |
| Mixed-search-required | 25 | 0.2800 / 0.3434 | 0.2000 / 0.2727 | 0.4000 / 0.4320 | 0.2000 / 0.2727 |
| Search-required total | 75 | 0.2800 / 0.3588 | 0.3200 / 0.3809 | 0.4533 / 0.5343 | 0.3200 / 0.3898 |

Frozen and Replay are historical reference conditions, not direct causal controls against Raw. The formal online result is Reward Live.

### Reward Live minus Raw

| Subset | EM delta | F1 delta |
|---|---:|---:|
| Overall | +0.0300 | +0.0239 |
| Search-free | +0.0000 | +0.0295 |
| Visual-search-required | +0.0800 | +0.0802 |
| Text-search-required | +0.1200 | +0.0567 |
| Mixed-search-required | -0.0800 | -0.0708 |
| Search-required total | +0.0400 | +0.0220 |

Reward-v2.1 improves the visual- and text-search-required subsets, while Raw performs better on the mixed-search subset.

### Source subsets

| Source | N | Raw | Reward Live | Reward - Raw |
|---|---:|---:|---:|---:|
| `fvqa_test` | 52 | 0.3462 / 0.4244 | 0.4038 / 0.4626 | +0.0577 / +0.0382 |
| `visual_infoseek_2023` | 35 | 0.1714 / 0.2127 | 0.1714 / 0.2286 | +0.0000 / +0.0159 |
| `mmsearch` | 13 | 0.0769 / 0.1462 | 0.0769 / 0.1346 | +0.0000 / -0.0115 |

### Paired statistics

Using the same 100 O1 IDs, 10,000 bootstrap samples, and seed `20260905`:

| Comparison | Mean EM difference | 95% CI | Mean F1 difference | 95% CI |
|---|---:|---:|---:|---:|
| Raw - Reward Live | -0.0300 | [-0.1300, 0.0800] | -0.0239 | [-0.1222, 0.0738] |

The intervals include zero. Win/tie counts were Raw/Reward/tie = 13/16/71 for EM and 17/23/60 for F1.

## O1 NoTool control

Raw vs Reward measures the complete trained-system difference. Only a same-model comparison such as Reward Live vs Reward NoTool is a tool-utility causal reference.

| Subset | Raw | Reward NoTool | Reward Live |
|---|---:|---:|---:|
| Overall | 0.2500 / 0.3141 | 0.0200 / 0.0267 | 0.2800 / 0.3380 |
| Search-required | 0.2800 / 0.3588 | 0.0133 / 0.0222 | 0.3200 / 0.3809 |

Reward Live minus Reward NoTool is `+0.2600 / +0.3113` overall and `+0.3067 / +0.3587` on search-required samples.

## E-VQA-200 R5

### Overall and question type

| Subset | N | Raw | Reward-v2.1 R5 | Reward - Raw |
|---|---:|---:|---:|---:|
| Overall | 200 | 0.0900 / 0.1211 | 0.1800 / 0.2300 | +0.0900 / +0.1089 |
| Automatic | 169 | 0.1006 / 0.1280 | 0.2012 / 0.2428 | +0.1006 / +0.1148 |
| Templated | 31 | 0.0323 / 0.0830 | 0.0645 / 0.1601 | +0.0323 / +0.0771 |

Reward R5 protocol validity is 0.9050 and tool-use rate is 0.9100. Its main execution route is visual-search-to-answer, used for 169/200 samples.

### Reward-route-conditioned analysis

The route labels below are the routes actually chosen by Reward. Raw always answers directly, so these rows share sample IDs and gold answers but do not represent identical behavior conditions.

| Reward route | N | Raw | Reward-v2.1 R5 | Reward - Raw |
|---|---:|---:|---:|---:|
| `V -> A` | 169 | 0.0888 / 0.1133 | 0.2012 / 0.2604 | +0.1124 / +0.1471 |
| `T -> A` | 13 | 0.1538 / 0.1846 | 0.0769 / 0.0769 | -0.0769 / -0.1077 |
| Direct | 18 | 0.0556 / 0.1481 | 0.0556 / 0.0556 | +0.0000 / -0.0926 |

### Paired statistics

| Comparison | Mean EM difference | 95% CI | Mean F1 difference | 95% CI |
|---|---:|---:|---:|---:|
| Raw - Reward R5 | -0.0900 | [-0.1450, -0.0350] | -0.1089 | [-0.1683, -0.0485] |

Win/tie counts were Raw/Reward/tie = 8/26/166 for EM and 21/46/133 for F1.

## Incomplete comparison boundaries

- **Unified-200:** Raw completed 200 samples at `0.2050 / 0.2660`; there is no complete valid Reward-v2.1 Unified-200 output. Do not compare Unified-200 Raw with O1-100 Reward.
- **MMSearch-300:** Reward-v2.1 completed at `0.0933 / 0.1350`, but the matching Raw parametric baseline was not run. No Raw-vs-Reward delta is reported.
- **Live-Web drift:** deterministic decoding does not freeze external search results. Exact future reproduction requires the frozen/replay evidence snapshot; a fresh live run tests the current Web, not a byte-identical historical Web state.
