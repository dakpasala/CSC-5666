# Model v2 proposal: building the amount/timing split into the architecture

Status: **proposal, not implemented.** Written 2026-10-20, after the v1 test on 2023. Any v2 development
uses 2022 train/validation only, followed by a fresh pre-registered test on ATUS 2024. 2025 stays in reserve.
The 2023 result stays in the record as the honest test of v1.

## Why v1 cannot fully separate the factors

v1 learns an "amount" latent from scratch and pushes it toward meaning amount with penalties (invariance to
shifts and swaps, plus an amount-prediction head). On 2023 it met the amount criteria, but:

- **Invariance and reconstruction trade off.** λ = 1 cost 11% validation CE at full data, so λ had to be
  lowered to 0.3.
- **Fragmentation is in the wrong place or nowhere.** Shifts and block swaps never change how many separate
  periods an activity has. kNN probes on the final models (2022 validation, seed 42) show that bout and
  transition information beyond totals is barely represented anywhere. What exists sits in the amount latent:

  | Model | Latent | Bouts | Clock centre | First onset | Last offset | Transitions |
  | --- | --- | ---: | ---: | ---: | ---: | ---: |
  | guided | amount | 0.027 | 0.008 | 0.017 | 0.009 | 0.040 |
  | guided | timing | −0.037 | 0.292 | 0.174 | 0.212 | 0.011 |
  | joint | single | 0.009 | 0.257 | 0.151 | 0.168 | 0.054 |

  Values are kNN R² gains over the true totals.
- **The timing latent duplicates totals.** Linear totals R² from the timing latent is 0.5-0.9.
- **Oversmoothing.** Each slot is predicted independently by argmax, which drops short activities:
  7.1 predicted vs 14.8 true transitions per day.

The root cause: activity totals are **observed**. They can be counted from the diary, so learning them
indirectly creates leaks and trade-offs with no benefit. Locatello et al. (2019) argue separation needs
explicit inductive biases or supervision. Here the factor we want to separate is directly available.

## Design

1. **Amount pathway sees totals only.** z_amount = Gaussian encoder over category proportions (an MLP on
   counts, permutation-invariant by construction). It cannot carry order, clock timing or fragmentation.
   An amount head reconstructs proportions from z_amount.
2. **Timing pathway is conditioned on totals.** q(z_timing | sequence, totals): the BiGRU encoder also
   receives the proportions. With the decoder also given the totals, the KL term charges z_timing only for
   information beyond totals. Variant to test: a learned conditional prior p(z_timing | z_amount), in the
   spirit of Hsu et al. (2017).
3. **Budget-constrained decoder.**
   - A network g(z_timing, position) scores every (slot, activity).
   - Sinkhorn normalisation (Cuturi, 2013) scales the scores so that each slot sums to 1 and each activity
     column sums to its budget in slots.
   - Training reconstruction is cross-entropy against the resulting soft assignment.
   - Decoding uses an exact-budget assignment, so a decoded day has exactly the budgeted minutes.
   - Budgets come from z_amount's head. A diagnostic mode uses the observed counts.
   - This makes "amount says how much, timing says when" structural, and short activities cannot disappear.
4. **Pairs become timing contrasts.** Add a **split/merge** transform: split one block into two pieces
   separated elsewhere, or merge two same-activity runs, always with identical counts. Use it for
   augmentation and as a diagnostic: the timing latent should respond to fragmentation changes.
5. **Prior and objective.** reconstruction + β_a·KL_amount + β_t·KL_timing (+ amount-head CE). With
   amount split out by construction, there is no invariance penalty and no λ to tune.

Unchanged: data, 5-minute bins, splits, compact sizes (amount 8, timing 16), and posterior means for export.
AGENTS.md constraints still hold: no per-frame dynamics latent, no autoregressive decoder, no teacher forcing.

## Evaluation changes

Amount separation is guaranteed by design, so the criteria move to the timing latent:

- **Totals leakage:** predicting totals from z_timing should do no better than predicting them from a
  totals-independent baseline.
- **Timing beyond totals:** kNN gain for clock features **and** fragmentation (bouts, transitions),
  compared with v1 guided, the joint model and engineered features.
- **Mix-and-match on any pair:** decode(budget of A, timing of B) must have A's exact totals, with timing
  closer to B.
- **Split/merge sensitivity:** z_timing should move on fragmentation-only changes.
- **Group stability:** cross-seed ARI and timing-split AUC, same definitions as v1.
- **Reconstruction:** CE, macro-F1, per-category F1 (eating, travel) and transitions.

## Framing

- v1 asked whether guidance alone can teach the split. Answer: partly. Amounts separate; there is no timing
  gain over a joint latent; groups are unstable.
- v2 asks: if the split is built in, is the timing representation a better description of daily
  organization than a joint latent or hand-made features?

Both are reported. v2 is a design change motivated by v1's test result and is labelled as such.

## Risks

- Sinkhorn soft assignments may blur during training. Mitigations: temperature annealing, and checking
  exact-assignment reconstruction on validation.
- If budgets come from an 8-dim z_amount, decoded totals are approximate. Report both budget sources.
- The timing latent might still not beat the joint model or engineered features. That would be a clear,
  reportable answer.

## References

- Cuturi, M. (2013). Sinkhorn Distances: Lightspeed Computation of Optimal Transport. NeurIPS 26.
- Hsu, W.-N., Zhang, Y., & Glass, J. (2017). Unsupervised Learning of Disentangled and Interpretable
  Representations from Sequential Data. NeurIPS 30.
- Locatello, F., et al. (2019). Challenging Common Assumptions in the Unsupervised Learning of Disentangled
  Representations. ICML.
- Locatello, F., et al. (2020). Weakly-Supervised Disentanglement Without Compromises. ICML.
