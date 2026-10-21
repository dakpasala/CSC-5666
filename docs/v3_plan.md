# v3: improving v1 — diagnosis and experiment plan

Started 2026-10-20 at the project owner's request. v3 develops **v1** further; it does not use v2's
counts-only amount encoder. Both latents stay learned from the full schedule. v1 (`outputs/checkpoint5_final`)
and v2 (`outputs/v2`) are frozen, reproducible baselines.

**Data rules for this iteration.** 2022 train/validation only. 2023 and 2024 have already informed
decisions, so they will not be presented as fresh tests for v3. 2025 stays untouched until availability and
suitability are verified, the approach is frozen, and new criteria are written.

## Diagnosis (2022 validation, frozen v1 guided λ=0.3, 80 epochs; `scripts/diagnose_v1.py`)

Raw numbers: `outputs/v3/diagnosis/diagnosis.json`.

**Q1. Does the timing latent store substantial activity-total information? No — less than real timing does.**
Tree-probe R² for predicting the 8 main activity totals:

| Representation | Mean R² |
| --- | ---: |
| v1 amount latent | 0.92 |
| joint latent | 0.84 |
| engineered clock-only timing features | 0.75 |
| v1 timing latent | 0.24 |

Real timing is naturally strongly tied to totals: knowing when work starts and ends nearly tells you work
minutes. The v1 timing latent holds far less totals information than real timing features do. So the split
is effectively: amount latent = totals plus totals-predictable timing; timing latent = timing beyond totals.
That matches "groups accounting for amounts", but it also means the timing latent alone is not a complete
timing description. This is not leakage, and forcing more independence is not indicated.

**Q2. Collapse or an ignored latent? No.**
- All 8 amount and 16 timing dimensions are active. KL is 17.5 nats (amount) and 19.6 nats (timing).
- The decoder depends heavily on both. Substituting another diary's amount latent raises validation CE from
  0.46 to 3.9-4.2; substituting the timing latent raises it to 2.5-2.7.
- The amount latent carries almost no timing beyond totals (tree gain 0.01 clock, 0.01 fragmentation).
- The timing latent carries clock timing beyond totals (0.57-0.63) and some fragmentation (0.08-0.09).

**Q3. Are the training transformations realistic and informative? Informative, but often implausible.**
Plausibility is measured as mean per-slot log-probability under the training time-of-day marginal
(original days: −1.474).

| Transform | Mean log-prob | Share less plausible than original | Changes |
| --- | ---: | ---: | --- |
| global circular shift (±30 min-2 h) | −1.564 | 73% | clock timing; wraps a non-sleep activity across 04:00 in 10% of shifts |
| block swap (any two interior runs) | −1.507 | 66% | order and fragmentation (0.66 bouts/day) |
| split/merge (v2) | −1.479 | 60% | fragmentation (1.5 bouts/day), almost no clock change |

Invariance is therefore taught largely on off-manifold days. That is a plausible reason the amount
encoder's invariance does not shape the timing side well.

**Q4. Does the reconstruction objective oversmooth and erase short activities? Mostly a decoding artefact.**
The decoder's probabilities know how much of each activity there is, but argmax throws it away:

| Activity | True min/day | Argmax min/day | Expected min from probabilities | Abs error: argmax vs expected |
| --- | ---: | ---: | ---: | ---: |
| eating | 71.9 | 37.4 | 73.6 | 41.3 vs 14.9 |
| travel | 60.9 | 41.9 | 58.6 | 33.1 vs 14.8 |
| personal care | 42.4 | 22.9 | 37.7 | 23.8 vs 13.7 |

- Placement of short activities is uncertain: the mean probability on true eating slots is 0.30.
- Independent per-slot decoding is unusable either way: argmax gives 7.0 transitions/day, sampling gives
  66.5, against 14.8 true.
- The fix belongs in **decoding** (structured decoding with the decoder's *own* expected budgets), not in a
  new loss.

**Q5. Is cluster instability caused by the representation, k-means initialisation, or both? Mainly the representation.**

| Representation | k-means ARI, n_init=1 | k-means ARI, n_init=50 | Across model seeds (n_init=50) | kNN-20 overlap across seeds | Linear CKA across seeds |
| --- | ---: | ---: | ---: | ---: | ---: |
| v1 timing (per seed) | 0.58-0.69 | 0.88-0.97 | 0.25 | 0.15 | 0.63 |
| joint (per seed) | 0.49-0.60 | 0.60-0.96 | 0.31 | 0.29 | 0.71 |
| engineered timing | 0.65 | 0.72 | — (deterministic) | 1.0 | 1.0 |

- k-means initialisation explains part of the earlier instability (n_init=10 was used). n_init=50 makes
  partitions of a *fixed* VAE representation nearly reproducible (ARI ~0.9-0.97).
- Different model seeds give genuinely different geometries: only 15% of a diary's 20 nearest neighbours
  are shared between seeds.
- The v1 timing latent's neighbourhoods barely resemble engineered-timing neighbourhoods (overlap 0.05).
- **The main target is representation consistency across seeds.** k-means needs only more initialisations.

## Experiment plan (one main factor at a time)

**Common setup.** Full 2022 train split, 40 epochs (same budget as the existing v1 robustness runs), seeds
42/43/44. **Baseline:** v1 guided λ=0.3 at 40 epochs (`outputs/checkpoint5_robustness/guided_l0.3_s*`,
already trained). **Comparators:** joint latent (40 epochs, already trained) and engineered timing features.

| ID | Single change from baseline | Motivated by | Supervision? |
| --- | --- | --- | --- |
| E0 | Evaluation only: self-budgeted structured decoding (integer budgets from the decoder's own expected minutes, then exact assignment) and k-means with n_init=50 | Q4, Q5 | no |
| E1 | Realistic local transforms instead of global wrap-shift and far swaps: move one interior block by up to ±1 h, swap adjacent interior blocks, split/merge | Q3 | no |
| E2 | Stronger timing KL (β_timing = 2): a more regular timing geometry | Q5 | no |
| E3 | Timing-feature supervision: an auxiliary head predicts engineered **clock** features (centre sin/cos, first onset, last offset per activity, masked when absent) from z_timing (λ = 0.3). Fragmentation is not supervised, so it serves as a held-out check | Q5 | **yes**, reported as such |

Only the components that clearly help on the evaluation below get combined later, with 3 seeds, as one
more one-factor step.

## Evaluation (`scripts/evaluate_v3.py`, 2022 validation, mean ± SD over seeds)

1. **Amount from each latent:** tree R² of totals.
2. **Timing from each latent:** tree gain beyond true totals (clock; fragmentation). For E3, fragmentation
   is the held-out check of value beyond the supervised features.
3. **Controlled transforms:** pair sensitivity of each latent (global shift, swap, split/merge, local shift).
4. **Collapse and decoder dependence:** KL, active dimensions, CE when one latent comes from another diary.
5. **Reconstruction:**
   - CE;
   - argmax vs self-budgeted decoding: transitions, duration error, eating/travel F1 (strict and ±15 min);
   - fragmentation error.
6. **Stability on the same reference diaries:** across model seeds, kNN-20 overlap and linear CKA
   (representation, no k-means), plus k-means (n_init=50) ARI. Within a seed, k-means ARI across
   initialisations.
7. **Groups among similar totals:** timing-split AUC on matched-totals pairs, and descriptive group profiles.

**What counts as improvement.** Better representation consistency (kNN overlap, cross-seed ARI) and timing
information beyond totals. Neither may come at the cost of collapse or of the amount latent's totals
information. Lower loss alone does not count. A variant that helps only through supervision must also beat
the engineered features it was trained on, otherwise that is reported as a limitation.

## Results, round 1 (2022 validation; full tables in `docs/v3_results.md`)

All variants: 3 seeds, 40 epochs, same data, decoder and budget as the v1 baseline.

| | kNN-20 overlap across seeds | k-means ARI across seeds (k=6, n_init=50) | Timing info beyond totals (tree, clock) | Fragmentation gain (held out for E3) | Validation CE |
| --- | ---: | ---: | ---: | ---: | ---: |
| v1 baseline | 0.141 | 0.169 | 0.633 | 0.114 | 0.540 |
| E1 realistic local transforms | 0.144 | 0.184 | 0.621 | 0.115 | 0.543 |
| E2 β_timing = 2 | 0.157 | 0.286 | 0.606 | 0.107 | 0.590 |
| E3 timing supervision (supervised) | 0.177 | 0.161 | 0.639 | 0.113 | 0.528 |
| joint latent | 0.290 | 0.340 | 0.610 | 0.114 | 0.499 |
| engineered timing features | 1.000 | 0.819 | — | — | — |

- **E0 (decoding only) works.** Self-budgeted decoding, using the decoder's own expected minutes and no
  observed totals, raises v1 transitions from 5.8 to 12.6 per day (true 14.8). Duration error falls from
  25.0 to 9.9 min; meal F1 rises from 0.30 to 0.35 (0.50 within ±15 min). Travel F1 and macro-F1 drop
  slightly (0.40 → 0.39, 0.61 → 0.60). It changes nothing in the representation.
- **k-means initialisation is solved by n_init=50.** Within-model ARI is 0.96-0.99 for E1-E3 and 0.86 for
  the baseline. The remaining instability is representational.
- **E1 (realistic transforms): no effect on stability or information.** The amount latent becomes less
  invariant to *global* shifts (0.039 vs 0.019), which it no longer trains on, and holds slightly more
  clock timing (0.036 vs 0.014). Not adopted.
- **E2 (β_timing = 2): the only stability gain** (ARI 0.17 → 0.29, CKA 0.67 → 0.72, timing-split AUC
  0.590 → 0.612). It costs information and reconstruction: timing clock gain 0.633 → 0.606, 13.3 of 16
  timing dimensions active, CE +9%, meal F1 lower. Still below the joint latent (0.34) and far below
  engineered features (0.82).
- **E3 (timing supervision): no stability gain** (ARI 0.16). Held-out fragmentation information is
  unchanged (0.113), so the head adds nothing beyond the features it was trained on. The timing latent
  absorbs more totals information (0.35 vs 0.29). Not adopted.
- **No variant collapses.** All keep both latents active, decoder-dependent and informative: amount totals
  R² is about 0.92 throughout.
- **Post-hoc dominant directions do not help.** Clustering the top 2/3/4/6 principal components of the
  timing latent instead of all dimensions lowers or leaves unchanged the cross-seed agreement (e.g. v1
  kNN overlap 0.02-0.09 vs 0.14). Seeds do not share stable dominant directions.

**Interpretation.** The split timing latent represents *timing beyond totals*. In these data, that residual
seems to have weak cluster structure, so different seeds organise it differently. The joint latent is
steadier because totals (e.g. work vs non-work) anchor it; engineered features are steadier because they are
deterministic. None of the targeted changes closes the gap. A stronger timing KL helps somewhat, at an
information cost.

## Round 2 plan: seed ensembles (written before training)

Question: does a consensus of several models give groups that are reliable across independent model sets?
- Train seeds 45/46/47 of the v1 baseline (`base`, settings identical to `guided_l0.3`) and of E2
  (β_timing = 2), 40 epochs.
- **Primary ensemble representation:** concatenate the timing posterior means of the 3 models, then k-means
  (k=6, n_init=50) fitted on train rows. **Secondary:** co-association consensus of the 3 models' k-means
  labels (average linkage, 6 groups).
- **Test:** ensemble A (seeds 42-44) vs ensemble B (seeds 45-47) on the same validation diaries:
  kNN-20 overlap and ARI. Reference: mean ARI and overlap over all pairs of single models.
- Also report timing information (tree clock gain beyond totals) and timing-split AUC of the ensemble
  representation, so stability is never judged without usefulness.

## Round 2 results: seed ensembles (`docs/v3_ensembles.md`)

Launcher bug, found and fixed: an empty settings string left a trailing blank that `xargs -L` treated as a
line continuation, so the first launch merged four jobs into one failed command. The 4 missing runs were
re-launched with identical settings; E2 seeds 46/47 from the first launch were valid and kept.

| Family | Single models: ARI / kNN overlap | 3-model ensemble A vs B: ARI / kNN overlap | Timing clock gain (single → ensemble) | Timing-split AUC (single → ensemble) |
| --- | --- | --- | --- | --- |
| v1 baseline | 0.186 / 0.148 | **0.384 / 0.295** | 0.627 → **0.691** | 0.589 → 0.611 |
| E2 (β_timing = 2) | 0.299 / 0.174 | 0.341 / 0.327 | 0.592 → 0.629 | 0.606 → 0.604 |

- Concatenating three v1 models' timing latents roughly **doubles** agreement between two disjoint
  ensembles. It also **raises** timing information beyond totals. Stability did not cost usefulness here.
- E2 gains less from ensembling. Co-association consensus helps E2 (ARI 0.386) but not the baseline (0.194).
- Only one A-vs-B comparison is possible with 6 models, so there is no error bar. 0.38 remains well below
  engineered timing features (0.82).

**Groups among similar totals** (v1 ensemble A, 2022 validation, k = 6). The groups have similar mean totals
(sleep 530-590 min, work 81-192 min) but differ in timing:
- a night-shift group (3%; sleep centred ~06:00, work ~02:00);
- a late group (16%; sleep ~04:20, work ~15:50);
- an early group (26%; sleep ~01:45, work ~12:00);
- a long-sleep, light-work group (8%);
- two mid-day groups.

Among 179 validation pairs whose totals differ by ≤ 60 min, the pairs placed in different groups differ in
sleep midpoint by 2.06 h, against 1.07 h for pairs kept together.

## v3 conclusion (development, 2022 only)

- **Adopted:** v1 as trained (λ_invariance = 0.3; E1 and E3 rejected; E2 optional and not adopted, because
  its single-model stability gain costs information and mostly disappears in ensembles), plus
  **self-budgeted decoding** for reconstruction, plus **3-model ensembles with k-means n_init = 50** for groups.
- **Evidence of separation:**
  - The amount latent holds totals (tree R² 0.92) and almost no timing beyond totals (≤ 0.04).
  - The timing latent holds timing beyond totals (0.63 single, 0.69 ensemble) and less totals information
    than real timing features naturally carry (0.29 vs 0.75).
  - Both latents are used by the decoder, and neither collapses.
- **Groups:** ensembles give interpretable, timing-defined groups among days with similar totals. They
  are about twice as reliable across model sets as single models (ARI 0.38). Engineered timing features
  remain more reliable (0.82) and track timing more directly (AUC 0.73, partly by construction).
- **Limitations:**
  - 40-epoch budget; one A-vs-B ensemble comparison.
  - The residual "timing beyond totals" appears to have weak natural cluster structure.
  - 2023 and 2024 informed decisions and are not fresh tests for v3.
  - 2025 is untouched. Before any test on it: verify availability and suitability, freeze this approach,
    and write new criteria.
