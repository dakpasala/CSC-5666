# Two-level schedule types: plan (written before running)

Started 2026-10-22. Development result on 2022 data (no held-out year remains). Uses existing frozen models;
no training.

**Hypothesis.** Stable day types are defined mainly by *what* is done (composition). Timing is largely a
continuum within each type, except possibly for a few real discrete patterns such as shift work.

## Level 1: composition types
- **Features:** activity shares (13 categories), standardized on 2022 train rows. k-means with n_init = 50.
- **Choosing k:** for k = 3..8, bootstrap stability (10 resamples of train; ARI of validation assignments)
  and the smallest type share. Rule, fixed now: the largest k with mean bootstrap ARI ≥ 0.8 and smallest
  type ≥ 3%.
- **Description:** mean minutes per activity, weekend share, and labour-force share per type.

## Level 2: timing cores within each composition type (types with ≥ 200 diaries)
- **Representations:**
  - (a) VAE rhythm axes: 3 PCA axes of the v1 80-epoch timing latent, Procrustes-aligned to seed 42 on
    train rows, averaged within each of three disjoint model sets (A = 42-44, B = 45-47, C = 48-50);
  - (b) engineered timing features (standardized, PCA to 3 within the type) for comparison.
- **HDBSCAN** on all 2022 days of the type: min_cluster_size = max(20, 5% of the type's days),
  min_samples = 10. It labels dense cores and leaves other days as noise ("in-between").
- **Reproducibility (VAE):** across sets A/B/C on the type's validation days, ARI over all days (noise as
  its own label) and ARI on days that are in a core in both sets.
- **Null comparison:** HDBSCAN on a multivariate Gaussian with the same mean and covariance as the real
  representation (5 draws). Real cores count as evidence only if the real data yields clearly more
  clustered structure than the matched null (number of cores, share of days in cores).
- **Unimodality:** Hartigan dip test on each VAE axis within the type.

## Shift-work check
Within workday-type(s): the distribution of work-centre clock time (dip test). Do level-2 cores, if any,
separate day, evening and night work?

**Expected outcome if the hypothesis holds:** composition types are stable; within types, timing is
unimodal or only weakly structured, except for detectable shift-work cores.
