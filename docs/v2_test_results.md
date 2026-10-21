# Model v2: final test on ATUS 2024

**Single held-out test of the post-test redesign (v2).** The pipeline and criteria were frozen in
`docs/preregistration_v2.md` (approved 2026-10-20, before 2024 was opened). Checksums at test time:
`docs/locks/preregistration_v2_2024_lock.sha256`.

Full tables:
- `docs/v2_results_test.md` (all metrics and groups);
- `docs/v2_criteria_test.md` (criteria, scored mechanically);
- 2022 validation equivalents: `docs/v2_results_val.md`, `docs/v2_criteria_val.md`.

Setup:
- 7,669 diaries from 2024, unweighted.
- All models trained on the 2022 train split, 80 epochs, seeds 42/43/44. Values are mean ± SD over seeds.
- Probes, scalers and cluster centroids fitted on 2022 train-split data only.
- v2 is scored fully generatively: budgets come from its own amount latent, not the true totals.

## Criteria: 9 of 12 checks met — exactly the validation prediction

| # | Claim | v2 (2024) | Reference | Met |
| --- | --- | ---: | ---: | :---: |
| R1 | Models days better than every v1 model (CE) | 0.409 | best v1 0.423 | yes |
| R2 | Realistic structure: transitions/day within 10% of true | 15.0 | true 14.8 | yes |
| R3 | Activity durations more accurate (mean abs error, min) | 9.8 | v1 guided 20.6 | yes |
| S1 | Amount latent ignores arrangement (shift / swap) | 0 / 0 | by construction | yes |
| S2 | Timing latent controls decoded timing (matched totals) | 0.812 | v1 guided 0.785 | yes |
| S3 | Timing latent responds to fragmentation (split/merge) | 0.107 | v1 guided 0.075 | yes |
| S4 | Timing latent excludes totals (kNN R²) | 0.241 | ≤ 0.395 | yes |
| S5 | Fragmentation information kept (tree probe gain) | 0.105 | ≥ 0.102 | yes |
| S6 | Clock-timing information as probe-readable as v1 | 0.554 | ≥ 0.620 | **no** |
| G1 | Stable groups across seeds (ARI, k=6) | 0.225 | ≥ 0.5 | **no** |
| G2 | Groups split same-total days by timing (AUC) | 0.585 | ≥ 0.60 | **no** |

## What this means

1. **v2 is the better model of a day, and the result transfers to a new year.**
   - Lowest reconstruction CE of any model (0.409), even though it must predict its own activity budgets.
   - About 15 transitions per day against 14.8 real: the oversmoothing that limited every v1 model
     (~7 transitions) is gone.
   - Half the duration error of v1, and better F1 for meals (0.49 vs 0.40-0.45).
   - Per-slot accuracy and macro-F1 match v1 guided (0.840 / 0.667). v1 unguided scores slightly higher on
     those two because predicting the common activity per slot is safe. v2 instead produces days with
     realistic episode structure.
2. **The separation does what it is supposed to do.**
   - The amount latent is exactly invariant to rearrangement.
   - Swapping timing latents between same-total days moves the decoded timing with the timing latent 81% of
     the time (v1 79%).
   - The timing latent reacts more to fragmentation changes than any v1 model.
   - The timing latent carries far less totals information than an untrained encoder or the joint model
     (0.24 vs 0.79 / 0.65).
3. **But v2's timing latent is not a better feature set.** A tree probe on [true totals + latent] reads
   less clock-timing information from v2's timing latent (0.55) than from v1's (0.64). The decoder uses the
   timing latent to place budgeted activities well, but that information is less directly readable by
   generic predictors. One plausible reason: in v2 the timing latent only has to work together with the
   budgets the decoder receives, rather than stand on its own.
4. **Schedule groups remain unstable, for v1 and v2 alike.** k=6 groups from v2's timing latent agree only
   weakly across seeds (ARI 0.23). Their splits of same-total days track timing only slightly (AUC 0.585).
   Engineered timing features give far more stable groups (ARI 0.76, AUC 0.73).
   **For discovering recurring daily schedule archetypes, engineered timing features remain the
   recommended tool**; the VAEs are better used as generative and analytic models of individual days.

## Caveats

- v2 was designed after v1's 2023 test, motivated by v1's results, and is labelled as a post-test redesign.
  It was developed on 2022 data only and tested once on 2024.
- One architecture family, 3 seeds, 80 epochs (still slowly improving); probes are kNN and gradient-boosted trees.
- Unweighted. Group shares are not population estimates (ATUS oversamples weekends). Groups describe days,
  not people. No BMI or health variables were used.
