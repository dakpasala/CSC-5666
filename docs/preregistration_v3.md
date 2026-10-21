# Pre-registration: v3 test on ATUS 2025 (APPROVED by the project owner 2026-10-21, before any model was run on 2025)

Written 2026-10-21. Thresholds were fixed in `scripts/evaluate_v3_final.py` before the validation references
were computed. Validation references come from 2022 only (`docs/v3_final_val.md`). Once approved, nothing
below changes after any 2025 model result is seen.

## 2025 data: availability and suitability (checked; no model run on it)

- Official BLS files downloaded and recorded in `data/raw/manifest.csv`. 6,146 valid diaries, none excluded.
  Schema identical to 2022, apart from one renamed respondent field we do not use (TREMODR → TRLVMODR).
  All activity codes map. Unknown minutes 1.0% (2022: 0.75%). Weekend share 49.9%.
- **No October 2025 data.** The official 2025 data dictionary states that no ATUS collection occurred in
  October because of the lapse in appropriations (Oct 1-Nov 12, 2025). The sample is therefore smaller and
  its month mix differs. That affects group *shares* (not reported as prevalence anyway), not the
  separation or pattern criteria. Suitable for this evaluation, with this caveat reported.

## Frozen v3 pipeline

- **Model:** v1 two-latent sequence VAE (amount 8-d, timing 16-d), both latents learned from the full
  schedule. λ_invariance = 0.3, λ_amount = 1, global shift/swap pairs, 40 epochs, best validation checkpoint.
  Six frozen models: seeds 42-44 (`outputs/checkpoint5_robustness/guided_l0.3_s*`) and seeds 45-47
  (`outputs/v3/base_s*`), all trained on the 2022 train split. No retraining.
- **Reconstruction:** self-budgeted decoding (integer budgets from the decoder's own expected minutes,
  then exact assignment; no observed totals).
- **Groups:** two disjoint 3-model ensembles (A = 42-44, B = 45-47). Concatenated timing posterior means,
  then k-means k = 6, n_init = 50, fitted on 2022 train rows.
- **References:** engineered timing features (standardized, k-means k = 6) and the joint-latent model
  (40 epochs, seeds 42-44).
- **Rejected in development and not used:** realistic local transforms, timing-feature supervision,
  stronger timing KL.
- **Evaluation:** `scripts/evaluate_v3_final.py --target test --confirm-preregistered`.

## Criteria (2025)

| # | Claim | Criterion | 2022 validation |
| --- | --- | --- | --- |
| S1 | Amount latent holds totals | mean amount totals R² ≥ 0.85 | 0.916 → met |
| S2 | Amount latent excludes timing | mean amount clock gain beyond totals ≤ 0.05 | 0.014 → met |
| S3 | Timing latent holds timing beyond totals | mean timing clock gain ≥ 0.50 | 0.627 → met |
| S4 | Timing latent does not duplicate amounts | timing totals R² ≤ 0.5 × engineered clock-only features' totals R² | 0.285 vs 0.376 → met |
| S5 | Decoder uses both latents | CE with another diary's amount / timing latent ≥ 2 × own CE (each) | 3.48 / 2.18 vs 1.09 → met |
| S6 | Amount latent ignores rearrangement | amount sensitivity ≤ 0.25 × timing sensitivity (global shift; block swap) | 0.019 vs 0.093; 0.029 vs 0.072 → met |
| P1 | Ensemble groups reproducible | ensemble A vs B ARI ≥ 0.30 | 0.384 → met |
| P2 | Ensembling improves reliability | A vs B ARI ≥ 1.5 × mean single-model ARI | 0.384 vs 0.278 → met |
| P3 | Groups split same-total days by timing | sleep-midpoint difference, split vs kept pairs (totals ≤ 60 min apart), ratio ≥ 1.5 (A and B) | 1.93 / 2.19 → met (engineered: 1.20) |
| P4 | Group splits track timing | mean ensemble timing-split AUC ≥ 0.58 | 0.611 → met (engineered 0.732, partly by construction) |
| P5 | As reliable as engineered features | A vs B ARI ≥ engineered k-means ARI | 0.384 vs 0.819 → **not met** |
| R1 | Realistic decoded days | self-budgeted transitions within 20% of true | 12.6 vs 14.8 → met |
| R2 | Accurate durations | self-budgeted duration error ≤ 0.6 × argmax | 9.9 vs 15.1 → met |

Expected from validation: 15 of 16 checks met; P5 not met.

All results are reported as observed. 2025 results are kept separate from all development results; 2023 and
2024 are not presented as fresh tests for v3. Unweighted; group shares are not population estimates; groups
describe days, not people. No BMI or health variables.
