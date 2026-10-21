# v2 pre-registered criteria — ATUS 2024 TEST

Scored mechanically by `scripts/score_v2_criteria.py` from `outputs/v2/eval_test/`.

| # | Criterion | v2 value | Reference / threshold | Met |
| --- | --- | ---: | ---: | :---: |
| R1 | v2 CE < every v1 model | 0.409 | 0.423 | yes |
| R2 | transitions within 10% of true | 15.046 | 14.799 | yes |
| R3 | duration error < v1 guided | 9.780 | 20.567 | yes |
| S1a | amount sensitivity (shift) = 0 | 0.000 | 0.000 | yes |
| S1b | amount sensitivity (swap) = 0 | 0.000 | 0.000 | yes |
| S2 | matched mix-and-match > v1 guided | 0.812 | 0.785 | yes |
| S3 | split/merge sensitivity > v1 guided | 0.107 | 0.075 | yes |
| S4 | totals leakage ≤ 0.5 × untrained v2 | 0.241 | 0.395 | yes |
| S5 | tree fragmentation gain ≥ v1 guided − 0.02 | 0.105 | 0.102 | yes |
| S6 | tree clock gain ≥ v1 guided − 0.02 | 0.554 | 0.620 | no |
| G1 | cross-seed ARI ≥ 0.5 | 0.225 | 0.500 | no |
| G2 | timing-split AUC ≥ 0.60 | 0.585 | 0.600 | no |

Met: 9 of 12.
