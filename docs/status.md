# Project status

Last updated: 2026-10-19

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

## Next

- **Checkpoint 3: end-to-end model.** Install PyTorch. Train a small two-latent sequence VAE on a
  100-500 diary subset, reload its checkpoint, reconstruct held-out diaries, and export posterior-mean
  embeddings. Joint-latent baseline afterwards.

## Open decisions

- Whether sleeplessness (`010102`) should be sleep or personal care (currently sleep).
- Whether `unknown` minutes should ever trigger exclusion (currently report only).
- Weekday/weekend handling: keep pooled, or stratify or condition later (currently pooled; `TUDIARYDAY`
  stored in metadata).
- Final k-selection rule (current rule gives k=2-3; see checkpoint 2 notes). Fix before Checkpoint 5.
- Whether to add a non-linear totals->timing predictor and a block-weighted totals+timing baseline.
- Survey weights: prototype is unweighted; weighted sensitivity analysis needed before any
  population-facing statement.
