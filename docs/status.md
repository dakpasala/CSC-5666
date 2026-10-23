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

- **v4 mixture prior (development, 2022 val):** learning groups inside the model (VaDE-style prior on z_timing)
  did not help. Cross-seed ARI is 0.16-0.19, below v1 + k-means (0.25) and the v1 ensembles (0.25-0.41). The
  same model is *more* reproducible on null data with no groups (0.31), so reproducibility is not evidence of
  real types. Separation is unaffected. Fallback adopted: engineered features for groups, the VAE for
  separation and rhythm analysis. Daily rhythms behave like a continuum. `docs/v4_mixture_plan.md`.

- **Continuum description (development, 2022 val):** instead of hard groups, days are placed on rhythm axes
  from the v1 timing latent (9 models, train-fitted Procrustes alignment). Axis 1 (earlier- ↔ later-shifted
  day) is reproducible: consensus ρ = 0.90 and 78% same tercile band (κ 0.67), far above hard-group ARI. Axis 2
  (early/busy ↔ late/consolidated morning) is moderate (ρ 0.70). Axes are about timing, not amounts.
  `docs/continuum_axes.md`.

- **Weekday vs weekend (development, 2022 val):** day type explains ≤ 1.1% of the rhythm axes (weekends differ
  mainly in work *amounts*); stratifying by day type does not reliably make groups more reproducible.
  `docs/weekday_weekend.md`.

- **Survey weights (development, 2022 val):** with TUFINLWGT and 160 replicate weights (Var = 4/160 Σ…), rhythm-band
  shares, rhythm measures, group shares and separation metrics barely move. Amounts do (weekends are 48% of
  diaries but 29% of days; average work 153 → 195 min), so population-facing amounts must be weighted.
  `docs/weighted_sensitivity.md`.

- **Two-level schedule types (development, 2022):** 5 stable composition day types (work, leisure, home & chores,
  religious/volunteer, school; ARI 0.93). Within types, timing is a continuum (no density cores; positive
  control confirms sensitivity to ≥ ~3 SD clusters), except a small real night-work group (~3-4% of work days)
  that the VAE timing latent encodes almost perfectly (AUC 0.998) but clustering cannot surface.
  `docs/two_level_types.md`.

- **Rare patterns, dashboard, typology (development, 2022 val):** ICA on the timing latent finds strongly
  non-Gaussian directions (kurtosis up to 13 vs null 0.41). Every one of 9 models has an unsupervised "nocturnal
  day" direction ranked top 1-7 that separates night work at AUC 0.85-0.92, and a nap direction appears (AUC 0.91).
  Exact components are not one-to-one reproducible across models. Probes: the timing latent knows clock patterns
  far beyond totals (night work 0.98, naps 0.95, late riser 0.98) but not fragmentation. Combined typology =
  composition type × rhythm band × pattern flags. `docs/patterns.md`.

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
- Survey weights: sensitivity check done (`docs/weighted_sensitivity.md`); population-facing amounts must be weighted.
