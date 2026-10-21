# v3: test on ATUS 2025

**Single held-out test of v3** (v1 two-latent VAE + self-budgeted decoding + 3-model ensembles for groups).
The pipeline and criteria were frozen in `docs/preregistration_v3.md` (approved 2026-10-21, before any
model was run on 2025). Checksums of the document, code, config and all six model checkpoints:
`docs/locks/preregistration_v3_2025_lock.sha256`.

Full output:
- `docs/v3_final_test.md`;
- `outputs/v3/final_test/` (per-model CSV, summary JSON);
- validation equivalents: `docs/v3_final_val.md`.

Setup:
- 6,146 diaries from 2025, unweighted. No October 2025 data: no collection during the federal shutdown,
  per the official data dictionary.
- Six models trained on the 2022 train split, 40 epochs.
- Probes, scalers and centroids fitted on 2022 train-split rows only.

## Result: 15 of 16 checks met — exactly the validation prediction

| # | Claim | 2025 | Threshold | Met | 2022 val |
| --- | --- | ---: | ---: | :---: | ---: |
| S1 | Amount latent holds totals (tree R²) | 0.923 | ≥ 0.85 | yes | 0.916 |
| S2 | Amount latent excludes timing (clock gain beyond totals) | 0.015 | ≤ 0.05 | yes | 0.014 |
| S3 | Timing latent holds timing beyond totals | 0.618 | ≥ 0.50 | yes | 0.627 |
| S4 | Timing latent does not duplicate amounts (totals R²) | 0.279 | ≤ 0.380 | yes | 0.285 |
| S5 | Decoder uses both latents (CE with the other diary's amount / timing) | 3.43 / 2.30 | ≥ 1.13 | yes | 3.48 / 2.18 |
| S6 | Amount latent ignores rearrangement (shift; swap) | 0.019; 0.029 | ≤ 0.091; 0.071 | yes | 0.019; 0.029 |
| P1 | Ensemble groups reproducible (A vs B ARI) | 0.360 | ≥ 0.30 | yes | 0.384 |
| P2 | Ensembling improves reliability (vs single-model 0.177) | 0.360 | ≥ 0.266 | yes | 0.384 |
| P3 | Groups split same-total days by timing (sleep-midpoint ratio A / B) | 2.00 / 1.95 | ≥ 1.5 | yes | 1.93 / 2.19 |
| P4 | Group splits track timing (AUC) | 0.589 | ≥ 0.58 | yes (narrowly) | 0.611 |
| P5 | As reliable as engineered features | 0.360 | ≥ 0.818 | **no** | 0.384 vs 0.819 |
| R1 | Realistic decoded days (transitions, true 15.1) | 12.8 | within 20% | yes | 12.6 |
| R2 | Accurate durations (self-budgeted vs argmax, min) | 10.3 | ≤ 15.6 | yes | 9.9 |

## What this means

1. **The separation holds on a year the models never saw, with essentially no drift.**
   - The amount latent captures totals (R² 0.92) and almost no timing (0.015).
   - The timing latent captures timing beyond totals (0.62).
   - The timing latent carries less totals information than real timing naturally does (0.28 vs 0.76).
   - The decoder relies on both latents.
   - Rearranging a day barely moves the amount latent (0.02-0.03) but moves the timing latent (0.29-0.37).
2. **The model recognizes schedule patterns among days with the same totals.**
   - Among 1,151 pairs of 2025 days whose totals differ by at most 60 minutes, the pairs the ensemble groups
     separate differ by 2.06 h in sleep midpoint, against 1.03-1.05 h for pairs kept together (ratio ≈ 2).
   - Groups from engineered timing features reach only a ratio of 1.28 on the same pairs. So the learned
     timing groups pick up differences in daily rhythm among same-total days more strongly than
     hand-built-feature groups.
3. **Groups are moderately reliable, not highly reliable.**
   - Two disjoint 3-model ensembles agree at ARI 0.36, double the single-model agreement (0.18) and above
     the joint model (0.29). Group sizes are consistent between ensembles: A {3, 8, 24, 26, 21, 17}% vs
     B {5, 8, 17, 23, 23, 25}%.
   - Engineered timing features remain far more reproducible (0.82) and track overall clock timing more
     directly (AUC 0.72 vs 0.59, partly by construction).
4. **Reconstruction:** self-budgeted decoding gives realistic days (12.8 vs 15.1 true transitions; 10.3-min
   duration error vs 25.6 with argmax).

## Caveats

- v3 was developed after v1 and v2 were tested on 2023 and 2024. Those years informed decisions and are not
  fresh tests for v3. 2025 was the last untouched year; there is no further held-out ATUS year until 2026
  data are released.
- 2025 has no October data. Group shares are unweighted and not population estimates; groups describe days,
  not people.
- One A-vs-B ensemble comparison (6 models). 40-epoch models. Probes are gradient-boosted trees on
  engineered targets. P4 passed narrowly (0.589 vs 0.58).
