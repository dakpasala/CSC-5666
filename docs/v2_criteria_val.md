# v2 pre-registered criteria — 2022 validation

Scored mechanically by `scripts/score_v2_criteria.py` from `outputs/v2/eval_val/`.

| # | Criterion | v2 value | Reference / threshold | Met |
| --- | --- | ---: | ---: | :---: |
| R1 | v2 CE < every v1 model | 0.402 | 0.415 | yes |
| R2 | transitions within 10% of true | 14.998 | 14.766 | yes |
| R3 | duration error < v1 guided | 9.641 | 20.424 | yes |
| S1a | amount sensitivity (shift) = 0 | 0.000 | 0.000 | yes |
| S1b | amount sensitivity (swap) = 0 | 0.000 | 0.000 | yes |
| S2 | matched mix-and-match > v1 guided | 0.797 | 0.770 | yes |
| S3 | split/merge sensitivity > v1 guided | 0.108 | 0.077 | yes |
| S4 | totals leakage ≤ 0.5 × untrained v2 | 0.244 | 0.393 | yes |
| S5 | tree fragmentation gain ≥ v1 guided − 0.02 | 0.108 | 0.101 | yes |
| S6 | tree clock gain ≥ v1 guided − 0.02 | 0.531 | 0.604 | no |
| G1 | cross-seed ARI ≥ 0.5 | 0.239 | 0.500 | no |
| G2 | timing-split AUC ≥ 0.60 | 0.554 | 0.600 | no |

Met: 9 of 12.
