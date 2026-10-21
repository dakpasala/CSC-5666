# Project status

Last updated: 2026-10-21

## Completed

- **Checkpoint 1: understand and prepare the data.** Real 2022 data loaded, schema checked against the
  dictionary, joins and durations validated, mapping `v1` implemented, 288-bin sequences built,
  deterministic split saved, discretization and boundary errors reported, example days plotted.
  2023 integrity-checked only. Details: `docs/data_decisions.md`, `docs/data_report_2022.md`.

- **Checkpoint 2: baselines.** Totals, timing-only and totals+timing k-means baselines on 2022
  train/val, with k selection, bootstrap stability, cluster profiles and representative diaries. The timing
  checks confirm that the evaluation detects arrangement changes when totals are fixed. Notes:
  `docs/checkpoint2_notes.md`; numbers: `docs/baselines_report.md`. Total-preserving transforms (circular
  shift, block swap) are implemented in `src/data/transforms.py` with count assertions, ready for Checkpoint 4.

- **Checkpoint 3: end-to-end model.** Two-latent sequence VAE trained on a 400-diary 2022 train
  subset (reconstruction + KL only). Evaluated on the full validation split: accuracy 0.73 (position-majority
  0.53), macro-F1 0.43 (0.12), but strong oversmoothing (4.7 vs 14.8 transitions/day). Checkpoint reload
  reproduces embeddings and evaluation exactly; embeddings exported. Notes: `docs/checkpoint3_notes.md`.

- **Checkpoint 4: separation guidance.** Paired training (shift/swap), amount invariance and amount
  prediction; 7 variants plus Checkpoint 3 and untrained references, one seed, 400-diary subset.
  Invariance works (amount pair-sensitivity 0.08-0.14 → 0.01-0.02) and amount prediction prevents collapse,
  at ~4% reconstruction cost. Evidence that the timing latent captures timing better than alternatives is
  weak (59% vs 54-58% on matched-totals mix-and-match; linear probes at the untrained floor).
  λ_invariance = 1 by the pre-registered rule. Notes: `docs/checkpoint4_notes.md`.

- **Checkpoint 5: evaluate the research claim.** Full-data robustness (3 seeds), kNN probes, full-data
  λ sweep (→ 0.3), 80-epoch frozen models, pre-registration approved by the project owner, then a single
  2023 test. 8 of 11 criteria met, the same pattern as validation: separation works and transfers across
  years; the split adds no timing information beyond a joint latent; learned groups are interpretable but
  unstable, and engineered timing features give more stable groups. `docs/checkpoint5_results.md`.

- **Model v2 (post-test redesign).** The amount pathway sees counts only, the timing pathway is conditioned on
  totals, and the decoder is budget-constrained (Sinkhorn) with exact-budget decoding; a split/merge transform
  was added. Swept on 2022 validation (β = 0.3), trained 3 seeds × 80 epochs, pre-registered, then tested once
  on ATUS 2024: 9 of 12 checks met, exactly as validation predicted. v2 is the best model of individual days
  (lowest CE, realistic ~15 transitions/day, half the duration error) and its separation works by construction.
  Its timing latent is less probe-readable than v1's, and groups remain unstable; engineered features stay the
  best grouping tool. `docs/v2_test_results.md`. 2025 data remain untouched.

- **v3 (improving v1; 2022 development only).** Diagnosis, then one-factor experiments, then seed ensembles.
  Adopted: v1 + self-budgeted decoding + 3-model ensembles for groups (ARI across disjoint ensembles 0.38 vs 0.19
  single; more timing information, not less). Rejected: realistic local transforms, timing-feature supervision;
  stronger timing KL gives a smaller, information-costly gain. Engineered features remain the most reliable
  grouping tool. `docs/v3_plan.md`.

- **v3 test on ATUS 2025** (pre-registered, approved 2026-10-21): 15 of 16 checks met, exactly as validation
  predicted. Separation holds on unseen data; ensemble groups separate same-total days by daily rhythm
  (sleep-midpoint ratio ≈ 2 vs 1.28 for engineered-feature groups) and are moderately reliable (ARI 0.36 vs
  0.82 for engineered features). `docs/v3_test_results.md`. No untouched ATUS year remains until 2026 data.

- **How many groups (development, 2022 val):** a first sweep suggested k = 4 (ARI 0.52, one comparison). With
  80-epoch models and three ensembles (error bars), this **did not replicate**: k = 4 gives 0.25 ± 0.01, and
  reproducibility stays around 0.3-0.4 for k = 2-10 with no clear optimum. Longer training improves
  reconstruction (accuracy ~0.84) but not group reproducibility. `docs/v3_k_sweep.md`,
  `docs/v3_k_sweep_ensembles.md`.

## Next

- **Checkpoint 6: report and presentation.** Results table, model diagram, example timelines,
  limitations, reproducible run guide. Verify deadlines and format against the course assignment PDF.

## Open decisions

- Whether sleeplessness (`010102`) should be sleep or personal care (currently sleep).
- Whether `unknown` minutes should ever trigger exclusion (currently report only).
- Weekday/weekend handling: keep pooled, or stratify or condition later (currently pooled; `TUDIARYDAY`
  stored in metadata).
- Whether to add a non-linear totals->timing predictor and a block-weighted totals+timing baseline.
- How to address oversmoothing (more data first; then beta/decoder changes only if diagnostics justify).
- v2 choices for validation: β values, conditional prior on/off, budget warm-up, Sinkhorn temperature.
- Survey weights: prototype is unweighted; weighted sensitivity analysis needed before any
  population-facing statement.
