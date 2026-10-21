# Pre-registration: model v2 test on ATUS 2024 (APPROVED by the project owner 2026-10-20, before 2024 was opened)

Written 2026-10-20 from 2022 validation results only (`docs/v2_results_val.md`). Once approved, nothing below
changes after any 2024 model result is seen. ATUS 2024 has been downloaded and integrity-checked only
(7,669 valid diaries, same schema as 2022 apart from one renamed respondent field we do not use). No model
has been run on it. 2025 stays in reserve.

## Frozen pipeline

- Data: as v1 (mapping `v1`, 5-minute bins, 2022 train/val split seed 42). 2024 prepared with the same code;
  nothing refitted on it.
- Model v2: `configs/vae_v2.yaml`:
  - amount encoder on counts only; timing encoder on sequence + counts;
  - Sinkhorn budget-constrained decoder, budgets predicted from z_amount;
  - β_amount = β_timing = 0.3 (chosen by the sweep rule in `docs/v2_validation_plan.md`);
  - pairs with split/merge;
  - 80 epochs, seeds 42/43/44, best validation checkpoint.
- Comparators: the frozen v1 final models (`outputs/checkpoint5_final`; guided λ=0.3, no_invariance,
  unguided, joint), untrained v1/v2 networks, and engineered-feature clusters.
- Decoding: v2 uses exact-budget decoding with budgets predicted from its own amount latent, and CE with
  predicted budgets. v1 uses per-slot argmax and softmax CE.
- Evaluation: `scripts/evaluate_v2.py --target test`. Probes, scalers and k=6 centroids are fitted on 2022
  train-split data only; VAE latents are unstandardized; matched-totals pairs ≤ 120 min.

## Criteria (2024, mean over 3 seeds; "v1 guided" = the frozen v1 main model)

| # | Claim | Criterion | 2022 validation |
| --- | --- | --- | --- |
| R1 | v2 models days better | v2 CE < CE of every v1 model | 0.402 vs ≥ 0.415 → met |
| R2 | Realistic structure (no oversmoothing) | v2 transitions/day within 10% of true | 15.0 vs 14.8 → met |
| R3 | Activity durations more accurate | v2 mean abs duration error < v1 guided | 9.6 vs 20.4 min → met |
| S1 | Amount latent ignores arrangement | v2 amount pair sensitivity = 0 (shift and swap) | 0 → met (by construction) |
| S2 | Timing latent controls decoded timing | v2 matched mix-and-match > v1 guided | 0.797 vs 0.770 → met |
| S3 | Timing latent responds to fragmentation | v2 split/merge sensitivity > v1 guided | 0.108 vs 0.077 → met |
| S4 | Timing latent excludes totals | v2 kNN totals R² from timing ≤ 0.5 × untrained v2 | 0.244 vs 0.393 → met |
| S5 | Fragmentation information kept | v2 tree fragmentation gain ≥ v1 guided − 0.02 | 0.108 vs 0.101 → met |
| S6 | Clock-timing information as readable as v1 | v2 tree clock gain ≥ v1 guided − 0.02 | 0.531 vs 0.604 → **not met** |
| G1 | Stable groups | v2 cross-seed ARI (k=6) ≥ 0.5 | 0.239 → **not met** |
| G2 | Groups split same-total days by timing | v2 timing-split AUC ≥ 0.60 | 0.554 → **not met** |

Expected from validation: R1-R3 and S1-S5 met; S6, G1 and G2 not met. Whatever 2024 shows is reported as
observed.

## Reporting rules

- 2024 results are reported separately from v1's 2023 test and from all development results.
- v2 was designed after the v1 test on 2023. It is a post-test redesign, labelled as such.
- Unweighted; group shares are not population estimates; groups describe days, not people. No BMI or health
  variables.
