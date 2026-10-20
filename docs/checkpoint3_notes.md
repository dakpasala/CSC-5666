# Checkpoint 3 notes: end-to-end VAE

Run: `outputs/checkpoint3/vae_cp3-20261019-185748` (config `configs/vae_checkpoint3.yaml`, git-ignored).
This is an **engineering check on a 400-diary subset**, not a research finding. 2023 was not used.
The run used uncommitted code (`git_dirty: true` in its resolved config).

## Setup

- Data: 400 random diaries from the 2022 train split (seed 42). Model selection and evaluation use the
  full 2022 validation split (1,627 diaries).
- Model (61,786 parameters):
  - encoder: activity embedding (16) plus clock sin/cos, a bidirectional GRU (64 per direction) and
    mean pooling;
  - Gaussian heads: amount latent (8 dims) and timing latent (16 dims);
  - decoder: a per-position MLP (2x128) fed [z_amount, z_timing, 4 clock harmonics]. It never sees the
    input sequence, so it cannot copy the input and must reconstruct through the latents.
- Loss: reconstruction + KL_amount + KL_timing (betas 1, KL warmup 20 epochs). Reported in nats per
  position; KL is divided by the 288 positions, so the total is the negative ELBO / T. Invariance and
  amount-prediction terms exist in `src/models/losses.py` but have weight 0 here (Checkpoint 4).
- Adam, lr 1e-3, batch 32, 200 epochs on CPU (~6 min). `best.pt` = lowest validation loss after warmup
  (epoch 117); validation loss rises after that (overfitting the small subset).

## Results on validation diaries (posterior means, best.pt)

| Metric | VAE | Position-majority reference |
| --- | ---: | ---: |
| Per-slot accuracy | 0.728 | 0.530 |
| Macro-F1 (13 categories) | 0.433 | 0.122 |
| Cross-entropy (nats/slot) | 0.865 | — |
| Mean abs duration error per category (min) | 38.8 | 84.8 |
| Transitions per day, predicted vs true | 4.7 vs 14.8 | 3.0 vs 14.8 |

The position-majority reference predicts the most common training activity at each time of day.

- Per-category F1: sleep 0.95, work 0.83, leisure 0.73, education 0.56, household 0.49, other 0.46,
  caregiving 0.44, exercise 0.34, personal care 0.29, travel 0.22, shopping/services 0.20, eating 0.10,
  unknown 0.02.
- **Main weakness: oversmoothing.** Reconstructions keep the large blocks (sleep/wake times, work
  periods, evening leisure) but drop short episodes such as meals, travel legs and errands. This matches
  the transition gap (4.7 vs 14.8). Examples in `reconstructions_val.png` (random, not best-case).
- **Latents are in use.** All 8 amount and 16 timing dimensions are active (posterior-mean variance
  > 0.01); mean KL is 6.5 nats (amount) and 13.4 nats (timing) per diary.
- **Decoder reliance.** Replacing one latent with its validation mean lowers accuracy from 0.73 to 0.66
  (amount fixed) or 0.59 (timing fixed). This only shows the decoder uses both. Without separation
  guidance the names "amount" and "timing" are just labels.

## Verified

- 18 tests pass: shapes, finite losses, gradients reaching both heads, the decoder and the amount head,
  tiny-batch overfit (>90% accuracy), and checkpoint reload.
- After training, the in-memory model and `last.pt` reloaded from disk give identical posterior means
  (max diff 0).
- `scripts/evaluate_vae.py` run in a fresh process reproduces `eval_best.json` exactly.
- Posterior-mean embeddings for all 8,136 diaries in 2022 are exported to `embeddings_best.npz` (with
  split and subset flags).

## Implications for next steps

- Before Checkpoint 4 comparisons, oversmoothing should be tracked as a primary diagnostic. Options to
  test only if diagnostics justify them: more training data (the full 6.5k train split is the obvious
  first step), a lower beta, or more decoder capacity or harmonics. The per-position MLP decoder has no
  way to model within-day dependence except through the latents.
- The overfitting on 400 diaries is expected; Checkpoint 5 uses the full train split.
