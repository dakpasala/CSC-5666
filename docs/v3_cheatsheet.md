# v3 cheat sheet

## What v3 is

v3 = **the v1 model, unchanged**, plus two things around it:

1. **Budgeted decoding** (fixes the oversmoothing when rebuilding days).
2. **Ensembles of 3 models** (makes the groups steadier).

The model, its training and the loss are the same as v1 (pair rule weight 0.3, amount head weight 1). The only
training difference: the six v3 models trained for **40 epochs** (the v1 models tested on 2023 trained for 80).
Seeds 42 to 47, all trained on the 2022 training split.

## Budgeted decoding, step by step

1. The decoder outputs a probability for each of the 13 activities in each of the 288 slots (same as v1).
2. Add up each activity's probability across all 288 slots. That is how many slots the decoder expects for it.
   Example: eating has about 0.3 probability in 10 slots around noon → about 3 slots (15 minutes).
3. Round to whole numbers that add up to 288. These are the **budgets**.
4. Hand out the slots so every activity gets exactly its budget, each placed where its probability is highest
   (solved exactly as an assignment problem).

v1 picked the most likely activity in each slot, so short activities like meals never won a slot and
disappeared. Now every activity gets its expected time. The budget comes from the decoder's own output, never
from the true day.

## Ensembles, step by step

1. Each of 3 models gives every day a 16-number timing vector (its mean).
2. Join them side by side → one 48-number vector per day.
3. Run k-means with 6 groups on that (fitted on 2022 training days).

Each model's random quirks partly cancel out. To test it, we built **two separate ensembles**: A = seeds 42-44
and B = seeds 45-47, and checked whether they produce the same groups.

## The test

Tested once on **ATUS 2025** (6,146 days, no October data because of the federal shutdown). The 16 checks were
written down and approved before running anything on 2025. **15 of 16 passed**, exactly as the 2022 validation
data predicted.

The probes in v3 are **tree models** (gradient boosting), stronger than v1's nearest-neighbor probes. So v3's
probe numbers are higher than v1's mainly because of the probe, not the model. Do not compare them directly.

## The 16 checks

### Separation (S): does the split still work?

| # | Plain question | How it is measured | Pass if | 2025 | Unit | |
| --- | --- | --- | --- | ---: | --- | :---: |
| S1 | Does the amount vector hold the totals? | Probe predicts minutes per activity from the amount vector (R²) | ≥ 0.85 | 0.92 | R² (0 to 1) | ✅ |
| S2 | Does the amount vector stay free of timing? | Extra clock-timing info the amount vector adds beyond the totals | ≤ 0.05 | 0.015 | R² gain (0 to 1) | ✅ |
| S3 | Does the timing vector know *when*? | Extra clock-timing info the timing vector adds beyond the totals | ≥ 0.50 | 0.62 | R² gain (0 to 1) | ✅ |
| S4 | Does the timing vector avoid copying the totals? | How well the totals can be read from the timing vector (R²), compared with real clock features (0.76) | ≤ half of 0.76 (0.38) | 0.28 | R² (0 to 1) | ✅ |
| S5a | Does the decoder really use the amount vector? | Rebuild a day with another day's amount vector: how much worse is the rebuild error? | ≥ 2 × normal (1.13) | 3.43 (normal 0.57) | cross-entropy (lower = better rebuild) | ✅ |
| S5b | Does the decoder really use the timing vector? | Same, with another day's timing vector | ≥ 2 × normal (1.13) | 2.30 | cross-entropy (lower = better rebuild) | ✅ |
| S6a | Does the amount vector ignore a shift? | Shift a day; how far does the amount vector move compared with the timing vector? | ≤ 25% of timing's move (0.091) | 0.019 (timing 0.36) | distance ÷ typical gap between days | ✅ |
| S6b | Does the amount vector ignore a block swap? | Same, with a block swap | ≤ 25% of timing's move (0.071) | 0.029 (timing 0.28) | distance ÷ typical gap between days | ✅ |

**Plain words:**
- S1 to S3 are the same ideas as v1's tests.
- S4 is new. Real timing naturally hints at the totals (someone who works nights also works a lot). The timing
  vector should carry *less* of that than real timing does, so it is not just a second copy of the totals.
- S5 is new. If swapping in another day's vector barely changed the rebuild, the decoder would be ignoring that
  vector. A rebuild error 4 to 6 times worse means both vectors matter.
- S6 is the rearrangement test from v1, but now compared with how far the *timing* vector moves (in v1 it was
  compared with the model without the pair rule).

### Patterns and groups (P): are the groups better?

| # | Plain question | How it is measured | Pass if | 2025 | Unit | |
| --- | --- | --- | --- | ---: | --- | :---: |
| P1 | Do two independent ensembles find the same groups? | ARI between ensemble A's and ensemble B's groups | ≥ 0.30 | 0.36 | ARI (0 = chance, 1 = identical) | ✅ |
| P2 | Are ensembles steadier than single models? | Ensemble ARI vs single models' ARI (0.18) | ≥ 1.5 × 0.18 (0.27) | 0.36 | ARI (0 = chance, 1 = identical) | ✅ |
| P3a | Do ensemble A's groups split same-total days by rhythm? | For pairs of days with similar totals: sleep-midpoint difference of pairs split apart ÷ pairs kept together | ≥ 1.5 | 2.00 | ratio (1 = no difference) | ✅ |
| P3b | Same for ensemble B | Same | ≥ 1.5 | 1.95 | ratio (1 = no difference) | ✅ |
| P4 | Do group splits follow timing in general? | AUC: do split pairs differ more in hand-built timing features than kept pairs? | ≥ 0.58 | 0.59 | AUC (0.5 = coin flip) | ✅ (barely) |
| P5 | Are the groups as reliable as hand-built features? | Ensemble ARI vs hand-built feature groups' ARI | ≥ 0.82 | 0.36 | ARI (0 = chance, 1 = identical) | ❌ |

**Plain words:**
- P1, P2: combining 3 models doubles group agreement (0.18 → 0.36). Moderate, not high.
- P3: among days with similar totals, the days the groups split apart differ in sleep timing about **twice** as
  much (2.06 h vs 1.03 h) as the days kept together. Hand-built feature groups only reach 1.28 here. So the
  learned groups pick up daily rhythm well.
- P4: the same idea as v1's group AUC (0.58 in v1). Only just passes.
- P5: the one failure. Hand-built timing features still give much steadier groups (0.82).

### Rebuilding (R): does budgeted decoding fix oversmoothing?

| # | Plain question | How it is measured | Pass if | 2025 | Unit | |
| --- | --- | --- | --- | ---: | --- | :---: |
| R1 | Do rebuilt days have realistic activity changes? | Activity changes per rebuilt day vs real days (15.1) | within 20% | 12.8 (v1 argmax: 5.8) | count per day | ✅ |
| R2 | Are activity minutes more accurate? | Average error in minutes per activity, budgeted vs argmax (26.0) | ≤ 0.6 × argmax (15.6) | 10.3 | minutes | ✅ |

**Plain words:** budgeted decoding more than doubles the activity changes (5.8 → 12.8, real 15.1) and cuts the
error in activity minutes by more than half (26 → 10 minutes). Short activities now show up.

## Reading the units

| Unit | Range | Meaning | Better is |
| --- | --- | --- | --- |
| % | 0-100% | share that came out right | higher |
| R² | 0 to 1 | share of the differences between days a probe explains (0 = guessing the average) | higher, except where a vector should be empty |
| R² gain | 0 to 1 | extra R² from adding a vector on top of the totals | higher for timing in the timing vector; near 0 for timing in the amount vector |
| Distance ÷ typical gap | 0 and up | how far a vector moved; 1.0 = as different as two unrelated days | low for the amount vector |
| ARI | about 0 to 1 | agreement between two groupings (0 = chance, 1 = identical) | higher |
| AUC | 0.5 to 1 | chance a random pair is ranked the right way round (0.5 = coin flip) | higher |
| Ratio | 1 and up | one average ÷ another (1 = no difference) | higher |
| Cross-entropy | 0 and up | rebuild error | lower (higher when a vector is swapped shows it is used) |
| Count, minutes | as stated | activity changes per day, error in minutes | count close to real; minutes lower |

Different units are different scales: do not compare an ARI with an R² or a Spearman ρ directly.

## How v3's checks relate to v1's

| | Count | Which |
| --- | ---: | --- |
| Carried over from v1 | 7 | S1, S2, S3, S6a, S6b, P1 (now ensembles), P4 |
| New in v3 | 9 | S4, S5a, S5b, P2, P3a, P3b, P5, R1, R2 |
| v1 checks dropped | 4 | swap test (A4), two "no loss vs no-pair-rule model" checks (B1c, B1d), split vs joint vector (C1) |

The new checks test what v3 changed (decoding and ensembles). The dropped ones compared training choices,
which v3 did not change.

## The story in three lines

1. The split still works on a year no model had seen (2025), with almost no drop from validation.
2. Budgeted decoding fixes the oversmoothing: realistic days, and short activities like meals come back.
3. Ensembles make groups steadier and better at separating days by rhythm, but hand-built features still give
   steadier groups.

Numbers worth remembering: **15 of 16 passed**, **12.8 vs 15.1 activity changes**, **ensemble ARI 0.36 vs 0.18
single vs 0.82 hand-built**.

Sources: `docs/v3_test_results.md`, `docs/v3_final_test.md`, `docs/preregistration_v3.md`.
