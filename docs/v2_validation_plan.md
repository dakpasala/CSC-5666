# Model v2 validation plan (written before any v2 training run)

2022 data only: train on the full train split, select on validation. 2023 is already spent on v1; 2024 is
reserved for the single v2 test. 2025 stays untouched.

## Stage 1: settings sweep (seed 42, 30 epochs each, otherwise `configs/vae_v2.yaml`)

| Run | Change from base | Question |
| --- | --- | --- |
| `sw_base` | none (predicted budgets, β = 1, standard prior, pairs with split/merge) | reference |
| `sw_observed` | `budget_source: observed` during training | do uninformative early budgets hurt? |
| `sw_beta0.3` | β_amount = β_timing = 0.3 | more detail in the latents (oversmoothing) |
| `sw_beta0.1` | β = 0.1 | same, stronger |
| `sw_condprior` | `conditional_prior: true` | does p(z_timing given z_amount) help? |
| `sw_nopairs` | `pairs.enabled: false` | is pair augmentation useful for v2? |

All runs are scored with **predicted** budgets (the fair, fully generative comparison with v1), whatever
budget source they trained with.

## Selection rule (fixed now)

1. **Eligible:** validation CE with predicted budgets within 5% of the best sweep run, **and** at least 4
   active timing dimensions (no collapse).
2. Among eligible runs, pick the highest **matched-totals mix-and-match** score: the share of
   same-total pairs where the decoded timing follows the timing latent's source. This is the most direct
   test of "the timing latent controls the timing".
3. Ties (within 0.01) go to the higher mean kNN timing gain (clock and fragmentation), then the lower CE.

## Sanity checks before final training

- Training curves are stable (no divergence; the amount head learns budgets).
- Exact-budget reconstructions of random validation days look plausible (not just the best ones).
- The timing latent responds to split/merge changes (fragmentation), not only to shifts.

## Stage 2: final v2 models

The chosen setting with seeds 42/43/44 and 80 epochs, on the same budget as the v1 final models. It is
compared on 2022 validation with the existing frozen v1 models (guided λ = 0.3, no_invariance, unguided,
joint), re-scored with the new metrics, and with the engineered baselines. Then a new pre-registration
and one evaluation on 2024.
