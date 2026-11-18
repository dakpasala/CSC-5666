# Visuals

Made by `scripts/make_visuals.py` (`OMP_NUM_THREADS=1 python scripts/make_visuals.py`, or `--only 2 5` for a subset).
Everything is computed on **2022 validation days**; no test year is read. "Final model" = M_big2 (1.09M
parameters, trained on 2022 train + 2023 + 2024); "v3" = the 40-epoch v1-design model (seed 42 where one model is
shown). Colors: the eight most common activities use the reference categorical palette in fixed order, and the
remaining five categories (shopping, exercise, other, unknown, education) are folded into gray.

| File | What it shows | Notes |
| --- | --- | --- |
| `fig1_architecture.png` | The final model: input slots, embedding, GRU, averaging, the two heads, sampling, decoder, budgeted decoding, amount head, loss | Diagram, no data |
| `fig2_reconstructions.png` | Four validation days (two work days, two non-work days, picked at random) rebuilt by v3 (most likely activity, budgeted) and the final model (budgeted) | Days chosen by a fixed random seed, not by how well they rebuild |
| `fig3_pair_test.png` | How far each vector moves when a day is shifted or block-swapped, for v3 and the final model | Six-seed means from `docs/v3_final_val.md` and `docs/v3_mbig2_final_val.md` |
| `fig4_embeddings.png` | First two principal components of the timing vector (colored by sleep midpoint) and amount vector (colored by hours of work) | Final model, seed 42; PCA fitted on training days |
| `fig5_ensemble_groups.png` | The six groups found by each of two independent ensembles (A = seeds 42-44, B = 45-47): share of each activity by time of day | B's group numbers matched to A's for display only; ARI 0.36 |
| `fig5b_group_overlap.png` | Where each ensemble-A group's days land in ensemble B | A perfect match would be a diagonal of 100% |
| `fig6_rhythm_axis_agreement.png` | Early-to-late score from ensemble A vs ensemble B for every validation day | Same alignment method as `scripts/continuum_axes.py`; ρ = 0.78 for the final model (v1 models reached 0.90 in that analysis) |
| `fig6b_rhythm_bands.png` | Average day for the earliest, middle and latest third of days by the early-to-late score | Band cut points fixed on training days |
| `fig7_model_selection.png` | Validation reconstruction error against model size, plus the GRU decoder, 80-epoch and more-data variants | Three-seed means from the capacity and round-2 to round-5 results |

Caveats to carry into captions: results are unweighted; days, not people; 2022 validation numbers were used to
make modeling choices.
