# Discovering rare schedule patterns, a pattern dashboard, and a combined typology: plan

Written 2026-10-22, before running. 2022 data only (development); frozen v1 80-epoch models (seeds 42-50);
no training.

## Plain-language pattern labels (used for EVALUATION only, never for discovery)

Defined on the 288-slot day (window 04:00-04:00):

| Pattern | Definition |
| --- | --- |
| night work | ≥ 60 min of work, work centre between 21:00 and 06:00 |
| early riser | wake (end of the first sleep run, if the diary starts asleep) before 05:30 |
| late riser | wake after 10:00 |
| daytime nap | a sleep run of ≥ 20 min that starts between 11:00 and 19:00 and touches neither window edge |
| split work day | ≥ 2 work bouts separated by ≥ 2 h, after merging gaps < 60 min |
| late-night leisure | ≥ 30 min of leisure between 00:00 and 04:00 |
| fragmented day | ≥ 25 activity transitions |

## Part 1: unsupervised discovery (ICA)

Rare groups that spread along a direction create heavy-tailed directions, which ICA finds without labels.
- **Per model:** FastICA (16 components) on the timing posterior means, fitted on train rows. Components are
  ranked by |excess kurtosis| on train rows. Each is oriented so its heavy tail is positive. "Tail days" are
  the top 3% of train scores, threshold applied to validation.
- **Null:** the same procedure on Gaussian samples with the matched covariance, giving the kurtosis
  expected from no structure.
- **What each top component captures** (evaluation only): validation AUC of the component score for each
  pattern label; the pattern prevalence among tail days vs all days; and the composition-type mix of tail days.
- **Reproducibility:** for each of seed 42's top-6 components, the best-matching component (|Spearman ρ| on
  validation) in each other model; the share of models matching at |ρ| ≥ 0.7; and the Jaccard overlap of
  tail-day sets.
- **Counts as discovery:** a component that is clearly more non-Gaussian than the null, reproducible
  (≥ 7 of 9 models at |ρ| ≥ 0.7), and whose tail is strongly enriched for a pattern (AUC ≥ 0.9 or ≥ 5×
  enrichment) that was never used to find it.

## Part 2: pattern dashboard (supervised probes, labelled as such)

For each pattern, logistic-regression probes fitted on train and scored by validation AUC. Inputs: (a) the
timing latent; (b) activity shares (totals) only; (c) both. Mean ± SD over the 9 models. This shows what the
model knows in plain terms, and how much of it goes beyond totals.

## Part 3: combined typology

Per day: composition type (5 stable types, `docs/two_level_types.md`) × axis-1 rhythm band (earlier /
typical / later-shifted, 9-model consensus) × rare-pattern flags (validated ICA directions if any, otherwise
the plain-language labels). Report:
- the share of days per cell;
- pattern prevalence per composition type;
- agreement of the rhythm band across the three disjoint model sets (composition labels are deterministic
  given the features).
