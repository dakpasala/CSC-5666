# Experiments and Results (draft for the final paper)

Aim: about 2 to 2.5 pages including tables. Three tables are suggested; cut Table 3 or move it to the appendix if
space is tight. All numbers below are copied from the repo documents named at the end.

## Protocol

We fixed every modeling choice on the 2022 validation split, then scored each held-out year once. Before each
test we wrote down the success criteria, had them approved, and stored checksums of the code, configuration
and trained models, so nothing could be adjusted after seeing the results. The first version of the model (v1)
was tested on 2023 (8,548 diaries), the structured alternative (v2) on 2024 (7,669 diaries), and the final
version (v3) on 2025 (6,146 diaries, no October data because of the federal shutdown). Each model was trained
with three or more random seeds, and we report the mean and standard deviation across seeds. Probes, scalers
and cluster centers are always fitted on 2022 training days only. All counts are unweighted, so they describe
the survey sample and not the U.S. population.

We compare against four baselines from the same pipeline: k-means on activity totals, k-means on hand-built
timing features, a VAE with one joint vector of 24 numbers (the standard design closest to ours), and our
model without the invariance term. We also report an untrained network of the same shape, to show what the
architecture gives without learning. We did not reimplement Li and Mandt (2018) directly, because their model
splits a sequence into fixed and per-frame parts, which does not match our question; the joint VAE plays the
role of the standard alternative.

## Experiment 1: Does the model separate amount from timing?

We measure separation in four ways. First, we rearrange held-out days without changing their totals and
measure how far each vector moves, relative to the typical distance between two random days (pair
sensitivity). Second, we use nearest-neighbor probes to check whether the activity totals can be read from the
amount vector, and whether clock-timing features can be read from each vector beyond what the totals already
tell us. Third, we take two days with nearly identical totals, give the decoder one day's amount vector and the
other day's timing vector, and check whose timing the rebuilt day follows (the swap test).

Table 1 shows the 2023 results for v1. The amount vector reads the totals well (R² 0.85, against 0.71 for an
untrained network) and carries almost no timing beyond them (0.008). Rearranging a day moves it by only 0.02
to 0.03, less than half as much as without the invariance term. The timing vector carries timing beyond the
totals (0.24, against 0.002 untrained), and in the swap test the rebuilt timing follows the day that supplied
the timing vector 79% of the time, against 67% for a model trained without guidance and 50% for chance. The
same pattern held on 2025 for v3, where the stronger tree probes give a totals R² of 0.92 for the amount vector,
0.015 timing leakage, and a timing gain of 0.62 for the timing vector. The decoder depends on both vectors:
replacing either one with another day's vector raises the reconstruction error to four to six times its
normal value (3.43 for the amount vector and 2.30 for the timing vector, against 0.57).

The split did not create extra timing information. The joint model's single vector holds about as much timing
as our timing vector (0.23 against 0.24), so this pre-registered criterion failed. What the split adds is an
organization of the information, with a part that tracks totals and ignores rearrangement, not more of it.

**Table 1.** Separation on 2023 (v1, mean ± SD over 3 seeds).

| Measure | Ours | No invariance | No guidance | Joint vector | Untrained |
| --- | ---: | ---: | ---: | ---: | ---: |
| Amount vector moves under shift (lower is better) | 0.019 ± 0.000 | 0.043 ± 0.002 | 0.293 ± 0.028 | | 0.012 |
| Amount vector moves under swap | 0.028 ± 0.002 | 0.060 ± 0.003 | 0.316 ± 0.046 | | 0.009 |
| Totals read from amount vector (R²) | 0.849 ± 0.001 | 0.829 ± 0.011 | 0.328 ± 0.066 | | 0.714 |
| Timing leaked into amount vector | 0.008 ± 0.001 | 0.014 ± 0.003 | 0.106 ± 0.011 | | 0.003 |
| Timing beyond totals in timing vector | 0.237 ± 0.014 | 0.242 ± 0.006 | 0.205 ± 0.009 | 0.229 ± 0.012 | 0.002 |
| Swap test: rebuilt timing follows timing vector | 0.789 ± 0.008 | 0.793 ± 0.002 | 0.672 ± 0.012 | | 0.499 |

*(The untrained amount vector barely moves because it barely encodes anything; it fails the totals row.)*

## Experiment 2: Ablations and sensitivity to the loss weights

Comparing the columns of Table 1 separates the two guidance terms. The amount prediction head does most of the
work: adding it alone raises the totals R² of the amount vector from 0.33 to 0.83 and cuts its movement under
rearrangement from about 0.3 to about 0.05. The invariance term then halves that movement again, at a cost of
about 2% in reconstruction error (cross-entropy 0.454 against 0.444).

We chose the invariance weight with a rule fixed in advance: the strongest weight whose validation
reconstruction error stays within 5% of the model without invariance. On the full 2022 data, with three seeds
each, λ = 1 raised the error by 11%, λ = 0.3 by 3.7% and λ = 0.1 by 3.9%, with average amount movement of
0.014, 0.024 and 0.033. The rule selected λ = 0.3. An earlier choice of λ = 1 on a 400-diary subset did not
hold at full scale, which is why we reran the sweep before testing.

During v3 development we also tested three changes on 2022 validation and rejected all of them: more realistic
local rearrangements instead of whole-day shifts, direct supervision of the timing vector with hand-built
timing features, and a stronger KL penalty on the timing vector. None improved group reliability enough to
justify its cost in information or reconstruction.

## Experiment 3: Rebuilding days

Rebuilt days with v1's slot-by-slot decoding are accurate per slot but too smooth. On 2023 the model labels 84%
of slots correctly (macro-F1 0.67), but a rebuilt day has about 7 activity changes against about 15 in a real
day, and short activities such as meals disappear. The budgeted decoding of v3 fixes most of this without
retraining. On 2025 it gives 12.8 activity changes per day against 15.1 in real days, and lowers the average
error in activity minutes from 25.6 to 10.3. The structured v2, which builds the totals into the decoder, was
the best model of individual days on 2024 (lowest cross-entropy, 0.409; 15.0 activity changes against 14.8;
duration error 9.8 minutes), but its timing vector was harder for probes to read (0.55 against 0.64 for v1), so
we kept v1 with budgeted decoding as the final model.

**[Insert Figure: original and rebuilt timelines for 4 to 6 held-out days, showing slot-by-slot and budgeted decoding]**

## Experiment 4: Do the timing vectors form stable groups?

We cluster days into six groups with k-means and measure reliability with the adjusted Rand index (ARI)
between groupings of the same days from models trained with different seeds. A value of 1 means identical
groups and 0 means chance agreement. We also check whether the groups split days with similar totals by timing,
using the ratio of the sleep-midpoint difference between days placed in different groups and days placed
together.

The learned groups are readable. They separate, for example, daytime sleepers, late starters and full daytime
work days, and their average minutes per activity are similar while their clock timing differs. They are not
stable, however (Table 2). Single v1 models agree at an ARI of only 0.26, while groups from hand-built timing
features agree at 0.76. Combining three models (v3) doubles the agreement to 0.36 on 2025 but stays well below
the hand-built features (0.82). The ensemble groups do separate days with similar totals by rhythm more
strongly than the hand-built groups: days they split differ about twice as much in sleep midpoint as days they
keep together (ratio 2.0), against 1.28 for the hand-built features.

**Table 2.** Group reliability (ARI between groupings of the same days).

| Representation | ARI | Year |
| --- | ---: | --- |
| v1 timing vector, single models | 0.257 ± 0.004 | 2023 |
| Joint vector, single models | 0.279 ± 0.020 | 2023 |
| v2 timing vector, single models | 0.225 | 2024 |
| v3 ensembles of 3 models (A vs B) | 0.360 | 2025 |
| Hand-built timing features | 0.761 ± 0.067 (2023), 0.818 (2025) | 2023, 2025 |

## Experiment 5: Why the groups are unstable (2022 validation only)

These follow-up checks were done after the test years had been used, so they come from 2022 validation data
only and are exploratory. We varied the number of groups from 2 to 10 and found no k with clearly better
reliability (about 0.3 to 0.4 throughout). A model that learns groups inside its prior (a Gaussian mixture on
the timing vector) agreed across seeds at only 0.16 to 0.19, and was more reproducible on simulated data with no
groups at all (0.31). This null check shows that moderate agreement alone is not evidence of real types.

The evidence points to timing varying along a continuum. When we instead give each day a score on the main
direction of variation in the timing vectors, aligned across nine models, the scores agree closely (Spearman
ρ = 0.90 between independent model sets, with 78% of days in the same third). This axis runs from
earlier-shifted to later-shifted days and is nearly unrelated to the activity totals (|ρ| ≤ 0.16). Grouping by
what a day is spent on, rather than when, gives five very stable types (bootstrap ARI 0.93): work, leisure,
home and chores, religious or volunteer, and school days. Specific patterns can also be read from the timing
vector: a probe separates night-work days with AUC 0.98, and naps with AUC 0.95. Day of week explains at most
1.1% of the rhythm axes, and survey weights barely change the rhythm results, though they do change the
activity amounts.

**[Insert Figure: optional, day scores on rhythm axis 1, or the six group profiles over the day]**

## Summary of findings

Across three held-out years, the separation between amount and timing worked as designed and transferred
without loss. The model rebuilds days well once short activities are given their expected time. The split does
not add timing information beyond a single joint vector, and hard groups from the timing vectors are less
reliable than groups from hand-built features. Follow-up analysis suggests the reason is that daily timing is a
continuum, which the timing vector describes well as a continuous scale but not as separate types.

---

*Sources: `docs/final_results_test.md` and `docs/checkpoint5_results.md` (2023), `docs/v2_test_results.md`
(2024), `docs/v3_test_results.md` (2025), `docs/checkpoint5_robustness_notes.md` (λ sweep),
`docs/v3_results.md`, `docs/v4_results.md`, `docs/continuum_axes.md`, `docs/two_level_types.md`,
`docs/patterns.md`, `docs/weekday_weekend.md`, `docs/weighted_sensitivity.md`.*
