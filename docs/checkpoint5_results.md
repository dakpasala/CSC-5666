# Checkpoint 5 results: final evaluation on ATUS 2023

**This is the one held-out test.** The pipeline, models and criteria were frozen in
`docs/preregistration.md` (approved 2026-10-20, before 2023 was opened). The SHA-256 checksums of the
pre-registration, final config and evaluation script at test time are in
`outputs/checkpoint5_final/preregistration_lock.sha256`. Full generated tables:
`docs/final_results_test.md` (2023) and `docs/final_results_val.md` (2022 validation reference).

Setup:
- **Data:** 8,548 diaries from 2023, unweighted.
- **Models:** trained on the 6,509 diaries of the 2022 train split, 80 epochs, seeds 42/43/44. Values are
  mean ± SD over seeds.
- **Fitting:** probes, scalers and cluster centroids were fitted on 2022 train-split data only.
- **Health data:** no BMI or health variables were used.

## Pre-registered criteria: 8 of 11 met, same pattern as validation

| # | Claim | 2023 value | Threshold | Met | Validation |
| --- | --- | ---: | ---: | :---: | :---: |
| A1 | Amount latent holds totals (kNN R²) | 0.849 | ≥ 0.764 | yes | yes |
| A2 | Amount latent excludes timing (kNN gain) | 0.008 | ≤ 0.050 | yes | yes |
| A3 | Timing latent holds timing beyond totals (kNN gain) | 0.237 | ≥ 0.102 | yes | yes |
| A4 | Decoded timing follows timing latent (matched totals) | 0.789 | ≥ 0.722 | yes | yes |
| B1a/b | Invariance: amount pair sensitivity (shift / swap) | 0.019 / 0.028 | ≤ 0.030 / 0.042 | yes | yes |
| B1c/d | …without losing timing (kNN gain / mix-and-match) | 0.237 / 0.789 | ≥ 0.222 / 0.773 | yes | yes |
| C1 | Split adds timing information beyond a joint latent | 0.237 | ≥ 0.249 | **no** | no |
| D1 | Timing-latent groups stable across seeds (ARI, k=6) | 0.257 | ≥ 0.500 | **no** | no |
| D2 | Groups split same-total days by timing (AUC) | 0.577 | ≥ 0.600 | **no** | no |

## What this means

1. **The separation works as designed and holds on a new year.** The amount latent:
   - predicts activity totals (kNN R² 0.85 vs 0.71 for an untrained network);
   - carries almost no timing information beyond totals (0.008);
   - barely moves when a day is shifted or reordered.

   The timing latent carries clock-timing information beyond totals (0.237 vs 0.002 untrained). When two
   diaries with near-identical totals swap timing latents, the decoded day's timing follows the timing
   latent 79% of the time (67% unguided, 50% untrained).
2. **The amount-prediction head does most of the separating; invariance adds a cleaner amount latent.**
   Against amount prediction alone, invariance (λ = 0.3) halves amount pair sensitivity and slightly
   raises totals R² (0.849 vs 0.829), with no measurable loss of timing information. It costs about
   2% reconstruction CE (0.454 vs 0.444).
3. **The split does not create more timing information than one joint latent (C1 not met).** The
   joint model's single 24-dim latent holds as much timing information (0.229) as the guided timing
   latent (0.237). What the split adds is an interpretable division of labour, not extra content.
4. **Learned groups are interpretable but not stable (D1, D2 not met).** k=6 groups from the guided
   timing latent are clearly organized by timing rather than totals. Their mean minutes per activity are
   similar, but their clock timing differs. Examples (seed 42):
   - daytime sleep (C1, 7%);
   - late start, sleep centred near 04:30 (C4, 14%);
   - full daytime work (C2, 20%);
   - morning work then afternoon leisure (C5, 21%);
   - long leisure from late afternoon (C3, 19%).

   However, different seeds partition the same diaries differently (ARI 0.26). Engineered timing features
   give more stable groups (ARI 0.76) whose splits track timing more closely (AUC 0.72 vs 0.58), partly by
   construction. **For finding recurring schedule groups, the engineered timing baseline is the better
   tool at this scale.** That is a valid negative result.
5. **Reconstruction transfers across years with no drop.** 2023 vs 2022 validation: CE 0.454 vs 0.457,
   accuracy 0.844 vs 0.843, macro-F1 0.671 vs 0.667. Oversmoothing remains: 7.1 predicted vs ~14.8 true
   transitions per day.

## Caveats

- Unweighted. About half of the diaries in every group are weekend days. This reflects the ATUS sample
  design (weekends are oversampled), not population prevalence. Group shares are not national estimates.
- Groups describe single days, not people or lifestyles.
- One architecture, one transform family (±2 h shifts, block swaps), 3 seeds; probes are kNN (k=25) and ridge.
- The final models were still slightly improving at 80 epochs. All compared models had the same budget.
- Choices made on validation before the test, all documented: λ re-swept at full data (1 → 0.3),
  80 epochs, unstandardized latent clustering, D2 redefined.
