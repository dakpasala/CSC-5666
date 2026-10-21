# Project status

Last updated: 2026-10-20

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

## In progress: model v2 (post-test redesign)

- After the 2023 test, the project owner asked for a model that separates the factors more cleanly. Design:
  `docs/model_v2_proposal.md`. The amount pathway sees counts only, the timing pathway is conditioned on
  totals, the decoder is budget-constrained (Sinkhorn) with exact-budget decoding, and a new split/merge
  transform covers fragmentation.
- Implemented and tested (27 tests pass); smoke run only. **Not trained yet.**
- Plan: train/tune on 2022 validation only → new pre-registration → one test on ATUS 2024 (2025 in
  reserve). The 2023 result stays as the record for v1.

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
