# Project status

Last updated: 2026-10-19

## Completed

- **Checkpoint 1: understand and prepare the data.** Real 2022 data loaded, schema checked against the
  dictionary, joins and durations validated, mapping `v1` implemented, 288-bin sequences built,
  deterministic split saved, discretization and boundary errors reported, example days plotted.
  2023 integrity-checked only. Details: `docs/data_decisions.md`, `docs/data_report_2022.md`.

## Next

- **Checkpoint 2: baselines.** Activity-totals clustering, engineered timing features
  (presence, circular timing with absence masks, transitions, bouts), and a check that the evaluation can
  tell apart diaries with matching totals but different timing.

## Open decisions

- Whether sleeplessness (`010102`) should be sleep or personal care (currently sleep).
- Whether `unknown` minutes should ever trigger exclusion (currently report only).
- Weekday/weekend handling: keep pooled, or stratify or condition later (currently pooled; `TUDIARYDAY`
  stored in metadata).
- Survey weights: prototype is unweighted; weighted sensitivity analysis needed before any
  population-facing statement.
