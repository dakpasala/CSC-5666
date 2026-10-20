# Pre-registration for the final 2023 evaluation (DRAFT — needs approval before 2023 is opened)

Written 2026-10-19, using 2022 validation results only. Once approved, nothing below changes after any
2023 model result is seen. Deviations must be listed in the final report.

## 1. Frozen pipeline

- **Data:** ATUS 2022 (train/val, respondent split, seed 42) and 2023 (test). Category mapping `v1`,
  5-minute bins, 288 steps, diary window 04:00-04:00. 2023 is prepared with the same code and mapping;
  nothing is refitted on 2023.
- **Model:** two-latent sequence VAE as in `configs/vae_checkpoint5_robustness.yaml`: amount 8 dims,
  timing 16 dims, pairs = shift ±30 min-2 h or block swap, λ_amount = 1.
- **λ_invariance:** chosen from {0.1, 0.3, 1} on full-data validation (40 epochs, 3 seeds) by the rule
  fixed at Checkpoint 4: lowest mean amount pair-sensitivity whose mean validation CE is ≤ 1.05 × the
  no_invariance mean. *Result: to be filled in from the sweep before approval.*
- **Final training:** 80 epochs, seeds 42/43/44, checkpoint with best validation loss. The same
  budget is used for every compared model.
- **Compared models (all frozen the same way):** guided (chosen λ), no_invariance (amount prediction
  only), unguided, joint (single 24-dim latent), and an untrained network of the same architecture.
  Engineered-feature baselines from Checkpoint 2 (totals, timing-only, totals+timing).
- **Clustering:** k-means with **k = 6** as the primary setting for every representation (engineered
  features, guided timing latent, joint latent). Scaling and centroids are fitted on 2022 train-split
  embeddings only and then applied to 2022 val and 2023. The Checkpoint 2 stability rule's k is reported as
  a sensitivity check.
- **Probes:** ridge and kNN (k=25), fitted on 2022 train-split embeddings and scored on 2023.

## 2. Success criteria (evaluated on 2023, mean over 3 seeds)

Validation values (2022, 40 epochs) are shown for reference. A criterion is "met" if the mean over
seeds satisfies it; seed SDs are reported alongside.

| # | Claim | Criterion on 2023 | Validation reference |
| --- | --- | --- | --- |
| A1 | Amount latent holds totals | guided kNN totals R² from amount ≥ untrained + 0.05 | 0.849 vs 0.709 (λ=1) |
| A2 | Amount latent excludes timing | guided kNN timing gain over totals from amount ≤ 0.05 | 0.008 (λ=1) |
| A3 | Timing latent holds timing beyond totals | guided kNN timing gain from timing latent ≥ untrained + 0.10 | 0.220 vs 0.003 |
| A4 | Decoded timing follows the timing latent | matched-totals mix-and-match ≥ unguided + 0.05 | 0.714 vs 0.636 |
| B1 | Invariance adds value over amount prediction alone | guided amount pair sensitivity ≤ 0.7 × no_invariance (shift and swap) **and** guided kNN timing gain and matched mix-and-match each ≥ no_invariance − 0.02 | λ=1: sensitivity met; mix 0.714 vs 0.732 met at −0.018 |
| C1 | Split adds timing information beyond one joint latent | guided timing-latent kNN timing gain ≥ joint + 0.02 | 0.220 vs 0.230 (not met) |
| D1 | Timing-latent groups are stable | mean pairwise ARI of k=6 cluster assignments across the 3 seeds on the 2023 diaries ≥ 0.5 | to be computed on 2022 val before approval |
| D2 | Timing-latent groups separate same-total days | share of matched-totals 2023 pairs placed in different clusters ≥ engineered timing-only clusters | to be computed on 2022 val before approval |
| R1 | Reconstruction holds across years (descriptive) | report 2023 vs 2022-val CE, accuracy, macro-F1 and transitions; no threshold | — |

Expected from validation: A1-A4 likely met; B1 borderline; C1 likely **not** met (the split gives an
interpretable division of labour, not more timing information than a joint latent). Any outcome is
reported as observed.

## 3. Reporting rules

- 2023 cluster occupancy is reported unweighted and labelled as such, never as national prevalence.
  Weekday/weekend composition of each cluster is reported (`TUDIARYDAY`).
- No BMI or health variables are used anywhere in this evaluation.
- Results on 2023 are kept in a separate section from all earlier exploratory results.
