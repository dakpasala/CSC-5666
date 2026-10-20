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

- **Checkpoint 3: end-to-end model.** Two-latent sequence VAE trained on a 400-diary 2022 train
  subset (reconstruction + KL only). Evaluated on the full validation split: accuracy 0.73 (position-majority
  0.53), macro-F1 0.43 (0.12), but strong oversmoothing (4.7 vs 14.8 transitions/day). Checkpoint reload
  reproduces embeddings and evaluation exactly; embeddings exported. Notes: `docs/checkpoint3_notes.md`.

## Next

- **Checkpoint 4: separation guidance.** Train on paired diaries (circular shift and block swap from
  `src/data/transforms.py`), turn on amount invariance and amount prediction, and compare with the
  constraint removed and with a joint-latent model of equal total size. Report reconstruction and collapse
  diagnostics side by side.

## Open decisions

- Whether sleeplessness (`010102`) should be sleep or personal care (currently sleep).
- Whether `unknown` minutes should ever trigger exclusion (currently report only).
- Weekday/weekend handling: keep pooled, or stratify or condition later (currently pooled; `TUDIARYDAY`
  stored in metadata).
- Final k-selection rule (current rule gives k=2-3; see checkpoint 2 notes). Fix before Checkpoint 5.
- Whether to add a non-linear totals->timing predictor and a block-weighted totals+timing baseline.
- How to address oversmoothing (more data first; then beta/decoder changes only if diagnostics justify).
- Survey weights: prototype is unweighted; weighted sensitivity analysis needed before any
  population-facing statement.
