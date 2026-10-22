# Handoff: daily schedule representations (ATUS)

Written 2026-10-22 for teammates or anyone continuing this work. It summarizes what was done, what was found,
how to reproduce it, and what remains open. Day-by-day progress is in `docs/status.md`; the research plan is
in `AGENTS.md`.

## What the project does

A sequence VAE learns two vectors from one ATUS diary day (288 five-minute slots, 13 activity categories):
- an **amount** vector: how much time goes to each activity;
- a **timing** vector: when activities happen, in what order and how fragmented, beyond what totals explain.

We test whether the two separate cleanly, and whether the timing vector reveals recurring daily rhythms.

## Main findings (honest summary)

| Question | Answer | Evidence |
| --- | --- | --- |
| Do amount and timing separate? | **Yes**, and it holds on unseen years | v1/v3 on 2025: amount R² 0.92 with ~0 timing; timing carries clock timing beyond totals (0.62); decoder uses both; 15/16 pre-registered checks (`docs/v3_test_results.md`) |
| Does the model see rhythm differences among same-total days? | **Yes** | days split by the model differ ~2 h in sleep midpoint vs ~1 h kept together; beats engineered-feature groups on this test |
| Are there stable discrete schedule *types*? | **No**, the evidence points to a continuum | hard groups agree across model sets at ARI ≈ 0.25-0.41 for any k, longer training or ensembles; a learned mixture prior was *more* reproducible on null data with no groups (`docs/v4_mixture_plan.md`) |
| What should be used instead? | **Continuous rhythm axes** | axis 1 (earlier- ↔ later-shifted day) reproducible at ρ = 0.90, 78% same band (`docs/continuum_axes.md`) |
| Reconstruction quality | Decent | ~84% slot accuracy (80 epochs); self-budgeted decoding gives realistic days (~13 vs 15 transitions/day) |
| Robust to day type and survey weights? | Yes, for rhythm results | `docs/weekday_weekend.md`, `docs/weighted_sensitivity.md`; amounts must be weighted for population statements |

What did not work, kept as documented negative results:
- realistic local transforms;
- timing-feature supervision;
- stronger timing KL (small, costly gain);
- the k = 4 "best group count" (did not replicate);
- a mixture prior on the timing latent;
- stratifying groups by day type.

## Model versions and where they live

| Version | What it is | Final models (git-ignored `outputs/`) | Tested on |
| --- | --- | --- | --- |
| v1 | two-latent GRU VAE, shift/swap pairs, amount invariance (λ = 0.3) + amount prediction | `outputs/checkpoint5_final/` (80 ep, seeds 42-44) | 2023 (`docs/checkpoint5_results.md`) |
| v2 | split built into the architecture (counts-only amount encoder, Sinkhorn decoder) | `outputs/v2/` | 2024 (`docs/v2_test_results.md`) |
| v3 | v1 + self-budgeted decoding + 3-model ensembles | `outputs/checkpoint5_robustness/guided_l0.3_s42-44`, `outputs/v3/base_s45-47` (40 ep) | 2025 (`docs/v3_test_results.md`) |
| extra v1 seeds | 80-epoch v1, seeds 45-50 | `outputs/v3/base80_s*` | development only |
| v4 | v1 + mixture prior on z_timing | `outputs/v4/` | development only (not adopted) |

**Every ATUS test year (2023-2025) has been used once,** each after a written, approved pre-registration.
Checksums are in `docs/locks/`. New claims can only be checked on 2022 validation until ATUS 2026 is released.

## Reproducing from scratch

Environment: Python 3.13; `python3.13 -m venv .venv && .venv/bin/pip install -r requirements.txt`.
CPU is fine: one thread per run with 6 parallel runs worked best on an 8-core Mac. The Apple GPU was slower
for this small GRU at batch 32.

```bash
# Data (official BLS files; links resolved from the BLS year pages; checksums in data/raw/manifest.csv)
python scripts/download_atus.py --years 2022 2023 2024 2025
python scripts/download_atus.py --years 2022 --kinds replicate_weights
for y in 2022 2023 2024 2025; do python scripts/prepare_data.py --year $y; done

# Baselines (Checkpoint 2) and v1 development (Checkpoints 3-5): see README.md steps 4-7
# v2: README step 8
# v3 experiments and ensembles
./scripts/run_v3.sh                                  # E1-E3, seeds 42-44, 40 epochs
SEEDS="45 46 47" ./scripts/run_v3.sh base            # extra 40-epoch seeds for ensembles
SEEDS="45 46 47 48 49 50" ./scripts/run_v3.sh base80 # extra 80-epoch seeds
python scripts/evaluate_v3.py && python scripts/evaluate_ensembles.py
python scripts/k_sweep_ensembles.py
# v4 mixture prior (+ null data)
python scripts/make_null_data.py && ./scripts/run_v4.sh && python scripts/evaluate_v4.py
# Continuum, day type, weights
python scripts/continuum_axes.py && python scripts/weekday_weekend.py && python scripts/weighted_sensitivity.py
# Tests
python -m pytest
```

Long runs: launch detached so editor or terminal restarts do not kill them, e.g.
`nohup caffeinate -i ./scripts/run_v3.sh base80 > outputs/v3/logs/launcher.log 2>&1 < /dev/null & disown`.

## Pitfalls we hit (save yourself the time)

- **BLS downloads:** blocked without browser-like headers (`src/data/download.py` sets them).
- **Replicate weights:** the info file says `TUFWGT001-160`, but the data columns are `FINLWGT001-160`.
- **Laptop sleep:** closing the lid pauses runs; a restart of the editor or Claude session killed attached
  runs. Use `nohup` + `caffeinate`.
- **Launchers:** macOS bash 3.2 has no associative arrays. An empty settings string leaves a trailing blank,
  which `xargs -L` treats as a line continuation (it merged four jobs once).
- **zsh:** unmatched globs error out (`nomatch`); wrap checks in `bash -c` or use `find`.
- **IDs:** read case IDs as strings (`dtype={"case_id": str}`), otherwise joins silently fail.
- **Mixture prior init:** fitting the GMM to posterior *means* only made the KL jump >1000× at the switch.
  Add the average posterior variance.
- **Stability evidence:** always compare against a null. Stable-looking clusters appeared on data with no
  groups.

## Open questions and good next experiments

1. **Independent replication:** rerun from scratch with new seeds. The repo has configs, commands and
   checksums.
2. **Fresh test on ATUS 2026** (expected around mid-2027): the only remaining untouched year. Reuse the
   pre-registration templates.
3. **Multi-day data:** ATUS has one day per person, so we can only ask about *days*. Surveys with several
   diary days per person could test whether *people* have stable rhythms. Candidates to check for design and
   access terms: UK Time Use Survey, Multinational Time Use Study.
4. **Other model families:** transformer encoders or contrastive timing representations. Test whether any
   representation shows more group structure than the null.
5. **Health associations** (optional in `AGENTS.md`): only after the analysis is fixed, weighted, framed as
   associations, with missingness and confounding reported.
6. Remaining open decisions are listed at the end of `docs/status.md`.

## Data and ethics notes

- Raw respondent records and model checkpoints stay out of git.
- No BMI or health variables were used in any model or evaluation.
- Groups and axes describe **days, not people**.
- Unweighted results are not population estimates; see `docs/weighted_sensitivity.md`.
- 2025 has no October data (federal shutdown).
