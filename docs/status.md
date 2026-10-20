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

- **Checkpoint 4: separation guidance.** Paired training (shift/swap), amount invariance and amount
  prediction; 7 variants plus Checkpoint 3 and untrained references, one seed, 400-diary subset.
  Invariance works (amount pair-sensitivity 0.08-0.14 → 0.01-0.02) and amount prediction prevents collapse,
  at ~4% reconstruction cost. Evidence that the timing latent captures timing better than alternatives is
  weak (59% vs 54-58% on matched-totals mix-and-match; linear probes at the untrained floor).
  λ_invariance = 1 by the pre-registered rule. Notes: `docs/checkpoint4_notes.md`.

## Next

- **Before Checkpoint 5:** 3 seeds on the full 2022 train split for guided (λ=1), no_invariance, unguided
  and joint; add non-linear probes; fix the k-selection rule and success criteria on validation before
  opening 2023.

## Open decisions

- Whether sleeplessness (`010102`) should be sleep or personal care (currently sleep).
- Whether `unknown` minutes should ever trigger exclusion (currently report only).
- Weekday/weekend handling: keep pooled, or stratify or condition later (currently pooled; `TUDIARYDAY`
  stored in metadata).
- Final k-selection rule (current rule gives k=2-3; see checkpoint 2 notes). Fix before Checkpoint 5.
- Whether to add a non-linear totals->timing predictor and a block-weighted totals+timing baseline.
- How to address oversmoothing (more data first; then beta/decoder changes only if diagnostics justify).
- Whether to broaden transforms (larger shifts) or try timing-target supervision as a documented experiment.
- Survey weights: prototype is unweighted; weighted sensitivity analysis needed before any
  population-facing statement.
