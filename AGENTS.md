# AGENTS.md — Daily Schedule Representations

## 1. Purpose and status

This file guides development of a CSC 5666 machine learning research prototype. Read it before changing the repository. It describes the research goal and a proposed implementation, not completed experiments or proven results.

Build an end-to-end prototype that learns two representations of a person's recorded day: **how much time they spend on activities** and **how those activities are organized**. Then test whether this separation helps identify recurring daily schedule patterns.

The immediate goal is a working, understandable research pipeline. Start small, establish baselines, and expand only after the pipeline works. The architecture and numerical defaults below are starting points to test; they are not requirements from the course rubric or claims already supported by the literature.

## 2. The problem in plain language

Daily totals leave out the structure of a day. Two diaries can both contain eight hours of sleep, but one can contain a continuous sleep period and the other several shorter periods. The same work total can occur early, late, or across multiple blocks.

We want to compare daily schedules while distinguishing:

- **Activity amounts:** total minutes spent sleeping, working, eating, and doing other activities.
- **Temporal organization:** when activities happen, their order, and how they are divided into periods.

The research question is: **Can a sequence model, trained with guidance about which schedules have the same activity totals, learn a useful separation between amounts and temporal organization?**

The final research result should include evidence about the separation and descriptions of recurring schedule groups. A useful negative result is also possible: simpler timing features may work just as well.

Do not describe this project as recommending better schedules, predicting the best lifestyle, establishing health effects, or assigning permanent lifestyle types to people. Each diary records one day. Discovered groups are **daily schedule archetypes**, not enduring personality or lifestyle labels.

## 3. Inputs and outputs

| Stage | Input | Output |
| --- | --- | --- |
| Preparation | Recorded activity episodes and their times | A cleaned, ordered daily activity sequence and activity totals |
| Encoder | One full daily sequence | Two distributions: an amount distribution and a timing distribution |
| Latent sampling | Each distribution's mean and variance | One amount vector and one timing vector per diary |
| Decoder | Both vectors and positions within the day | Activity probabilities at every time step; a reconstructed schedule |
| Analysis | Learned timing vectors | Clusters, representative schedules, and descriptions of recurring patterns |
| Evaluation | Held-out diaries, transformed pairs, and baselines | Reconstruction, separation, stability, and usefulness measurements |

An amount vector is a learned numeric representation, not necessarily a literal list of minutes. Keep the actual activity-total vector separately as a measurable target. The model itself does not output human-readable trends; clustering and analysis produce those descriptions afterward.

Use posterior means rather than random samples when exporting embeddings for reproducible clustering and probes.

## 4. Dataset and official documentation

Use the **American Time Use Survey (ATUS)** from the U.S. Bureau of Labor Statistics. The main activity diaries are the core data. The Eating & Health Module supplies optional additional variables; it does not replace the main diary files.

Official entry points:

- Main data: https://www.bls.gov/tus/data.htm
- File structure and usage: https://www.bls.gov/tus/other-documentation/howto.htm
- Eating & Health Module: https://www.bls.gov/tus/modules/ehdatafiles.htm
- 2022 module files: https://www.bls.gov/tus/data/ehdatafiles-2022.htm
- 2023 module files: https://www.bls.gov/tus/data/ehdatafiles-2023.htm

The main activity file contains activity-level records. The respondent file contains respondent-level information. Documented identifiers include `TUCASEID` and the activity identifier `TUACTIVITY_N`. `TUACTDUR24` records activity minutes within the diary window. Tiered activity codes describe activity categories. Verify exact columns, definitions, missing-value codes, and file versions against each year's dictionary before implementation.

Download the 2022 and 2023 main activity and respondent files, their dictionaries, and the activity coding lexicon. Add module files only when needed. Resolve actual downloads from official pages; do not invent archive URLs or assume schemas are identical across years.

Record source URL, retrieval date, year, checksum, and local filename in a source manifest. Keep raw files unchanged. If downloads are blocked, support user-provided files in the raw directory and explain what is missing. Synthetic fixtures are for software checks only, never presented as ATUS results.

Do not assume a sample size. Report observed counts of respondents, episodes, valid diaries, exclusions, and module matches after ingestion.

### Data separation

- Use 2022 for model training and validation, initially an 80/20 respondent split with a fixed seed.
- Reserve 2023 for final cross-year testing. Inspect its schema and integrity as needed, but do not tune based on its model performance.
- Split before constructing transformed pairs. A diary and every transformation of it belong to the same split.
- Fit normalization, feature selection, PCA, cluster centers, and other learned preprocessing on training data only.
- Keep BMI and general health out of encoders, loss functions, hyperparameter selection, and cluster selection. Use them only for optional external exploratory associations after the analysis is fixed.
- Do not restrict the main modeling cohort to module respondents unless a documented research decision requires it.

## 5. Data preparation requirements

Create a canonical episode table with year, respondent ID, episode ID, diary-relative start/end, duration, and broad activity category. Store weights and demographic metadata separately from model inputs.

Preserve identifiers as strings and leading zeros in activity codes. Validate join cardinality: attaching respondent attributes should not multiply activity rows. Join module activity information with both respondent and activity identifiers when required by its dictionary. Do not merge roster records without selecting the appropriate person.

Verify the survey's diary boundaries and clock conventions before constructing sequences. Account explicitly for the overnight wrap. Use durations within the 24-hour window rather than extending episodes beyond it. Check that the prepared timeline spans 1,440 minutes without unexplained gaps or overlaps. Log and investigate invalid diaries; do not silently repair them with arbitrary activities.

### Activity categories

Create a versioned mapping from official codes into a manageable set, such as sleep, work, education, eating/drinking, household tasks, caregiving, travel, leisure, exercise, other personal care, and other/unknown.

The mapping must distinguish sleep from other personal care, eating from food preparation, and exercise from watching sports. Do not assume all codes in a broad top-level category mean the same activity. Keep an explicit unknown category and report its frequency. Save the mapping alongside each run.

### Sequence resolution

Start with five-minute bins: 288 steps per diary. Assign the activity with the largest overlap in each bin using a deterministic tie rule. Keep the original episode table for inspection and finer-resolution analysis.

This discretization can remove short episodes and change durations. Measure those errors against raw diary totals and report them. The model's amount targets should initially be derived from its discretized sequence so that its reconstruction and amount objectives are consistent.

Transformations must preserve activity counts exactly at the modeled resolution. Do not claim this makes the discretized representation identical to the original diary. If quantization error is substantial, evaluate a finer resolution before expanding model complexity.

Include clock-position features, such as sine/cosine of time of day, so the model can distinguish early from late activity. Record how diary-relative positions map to clock time.

### Boundary and missing-data care

Sleep at the start and end of a diary may be cut off by the observation window. Do not automatically interpret it as two separate sleep episodes or merge it into one real-world episode. Report within-window run counts and boundary censoring explicitly. Avoid calling fragmentation a measure of sleep quality.

Decode official missing-value sentinels before using numeric fields. Distinguish missing values from zeros. Never fill missing BMI or health with a meaningful-looking value.

## 6. Prototype model

Use Python and PyTorch for the model, with pandas or Polars for preparation and scikit-learn for baselines and probes. Prefer existing repository conventions if present. Record dependency versions when implementing rather than assuming current versions here.

Start with a compact sequence VAE:

1. Embed each activity category and add clock-position features.
2. Process the full day using a bidirectional GRU or LSTM.
3. Pool its sequence outputs into a whole-day representation.
4. Use separate heads to predict mean and log-variance for the amount and timing latent distributions.
5. Sample each latent vector using reparameterization during training.
6. Decode both vectors together with time-position features into categorical activity probabilities at all 288 positions.

A shared backbone with two heads is sufficient initially; two independent encoder networks are optional. Separate heads alone do not establish meaningful separation.

Suggested initial settings: activity embedding dimension 16, one recurrent layer with hidden dimension 64 per direction, amount latent dimension 8, timing latent dimension 16, batch size 32, Adam with learning rate 0.001, seed 42. Make all settings configurable. Treat these as modest engineering defaults, not optimized choices. Support CPU execution and optional GPU acceleration.

A small decoder MLP applied at each time position is enough for the first version: concatenate both day-level vectors and the position features, then output category logits. Improve decoder capacity only if reconstruction diagnostics justify it. Avoid teacher-forcing shortcuts that let the decoder ignore the latent variables.

### Relationship to Li and Mandt

Li and Mandt's model separates shared sequence content from time-varying dynamics. Our proposed split is different: **both amount and timing vectors describe an entire day**. We are adapting the principle of structured latent representations, not reproducing their video model unchanged.

Do not automatically introduce a per-frame dynamics latent or a learned autoregressive dynamics prior. Independent standard normal priors for the two day-level latents are a reasonable prototype starting point. The project does not need unconditional schedule generation to answer its research question.

## 7. Giving the model guidance

The model cannot understand names such as “amount” and “timing” on their own. Construct schedule pairs whose activity totals remain fixed while their arrangement changes.

Begin with two controlled transformation types:

- A bounded circular shift, testing a change in clock timing while retaining activity counts. Clearly mark its artificial boundary effects.
- A swap of selected activity blocks, testing changes in order and potentially within-window fragmentation.

Reject unchanged outputs. Assert identical per-category counts before and after every transformation. Record transformation type and parameters. Do not randomly shuffle individual time slots as the main augmentation: it produces implausible schedules and can make the task artificially easy.

Inspect examples visually. Synthetic transformations are training guidance and diagnostic cases, not evidence that the transformed day is realistic. Evaluate on untouched diaries too.

### Initial training objective

Implement and log each term separately:

`loss = reconstruction + beta_amount * KL_amount + beta_timing * KL_timing + lambda_invariance * amount_invariance + lambda_amount * amount_prediction`

- **Reconstruction:** mean categorical cross-entropy over positions, calculated for original and transformed schedules.
- **KL terms:** regularize each approximate posterior toward its prior. Use an explicit reduction convention so loss scale does not change accidentally with sequence length.
- **Amount invariance:** mean squared distance between the amount posterior means of the original and transformed day.
- **Amount prediction:** an auxiliary head predicts per-category activity proportions from the amount vector. This encourages the amount representation to retain useful information rather than becoming constant.

The auxiliary head is a proposed anti-collapse design choice, not a result established by the papers. Test its contribution separately when feasible. Start by verifying reconstruction, then add constraints with validation-selected weights. A KL warmup can be tested if needed.

Do not maximize arbitrary distance between timing vectors. A large distance alone is not meaningful timing information. Check whether timing vectors capture measurable temporal differences. Optional timing-target supervision should be clearly documented as an additional experiment.

A very strong invariance loss can erase useful information. A weak one can fail to separate the factors. Tune the balance on validation data and retain the unconstrained comparison.

## 8. Baselines and ablations

Implement simpler comparisons before making novelty claims:

1. **Activity totals:** standardized activity proportions, with optional PCA, followed by clustering.
2. **Engineered timing features:** activity presence, circular timing features, transition counts, and within-window bout summaries; compare both timing-only and totals-plus-timing variants when feasible.
3. **Joint sequence model:** a comparable autoencoder or VAE with one joint latent representation.
4. **Split model without invariance:** the same two-head model with the separation constraint removed.

For meaningful ablations, keep data splits, decoder, total latent capacity, optimization budget, and evaluation as comparable as possible. Record remaining differences. Separate the effect of amount prediction from invariance if both are used.

PCA and k-means serve different roles: PCA reduces dimensions; k-means groups representations. Do not describe either as a sequence reconstruction model. Compare each baseline only on metrics it can support.

## 9. Evaluation plan

### Reconstruction

Report cross-entropy, per-slot accuracy, macro-F1 across activity categories, and activity-duration error in minutes. Include transition or bout-count errors to expose oversmoothing. Plot original and reconstructed timelines for several held-out examples, not just the best example. Majority activities can inflate accuracy, so accuracy alone is insufficient.

### Separation and retained information

- Measure amount-vector changes on total-preserving pairs relative to normal between-diary variation; raw vector distance alone is scale-dependent.
- Train simple probes to predict activity totals from each latent component separately.
- Train simple probes to predict timing features from each component separately.
- Fit probes on training embeddings and select settings on validation data; evaluate final settings on held-out data.
- Report latent variance and per-component KL. A nearly constant representation is a warning sign.
- Test decoder sensitivity when one component is replaced or held fixed. Treat swaps as diagnostics, not proof of identifiable causal factors.

Timing features need clear definitions. Use circular features for clock time and masks for absent activities. Do not compare a nonexistent meal time as if it were midnight.

The desired pattern is useful amount information in the amount vector and additional temporal information in the timing vector. Perfect statistical independence is not assumed: real activity totals and timing can be related. Evaluate timing value within similar-total diaries or beyond a totals-only predictor rather than demanding zero cross-prediction.

Reconstruction alone does not prove separation. A constant amount vector plus a timing vector that stores everything can reconstruct well and satisfy invariance. The probes, auxiliary target, and latent diagnostics address this failure mode.

### Schedule groups and stability

Cluster timing posterior means using a reproducible baseline such as k-means. Fit scaling and cluster centers on training data. Choose cluster count using validation evidence, interpretability, and stability rather than the best-looking final-test plot.

Describe groups using representative actual diaries, distributions of timing features, and activity totals. Check whether temporal differences remain among days with similar totals. Avoid health-value labels such as “good” or “bad.”

For resampled training or different seeds, assign a common reference set of diaries to each model's clusters. Compare assignments with adjusted Rand index or a similar label-invariant measure. Do not calculate assignment agreement between unrelated respondents from different years.

After all choices are fixed, apply the frozen pipeline to 2023. Compare reconstruction, separation, cluster profiles, and occupancy. Cross-year prevalence changes may reflect sampling or population differences; they do not automatically mean the representation failed. Consider weekday/weekend composition when interpreting changes.

Use one seed for initial debugging and at least three seeds for final comparisons if resources permit. Report variability and actual compute limits.

### Optional external associations

Compare BMI or general-health distributions across fixed schedule groups only after primary analyses are complete. Report missingness, sample sizes, uncertainty, and possible confounding. These are associations, not effects of schedules on health.

An unweighted prototype is acceptable if labeled clearly. Population estimates require appropriate survey weights and attention to survey design; module analyses may require different weights. Do not report unweighted cluster frequencies as national prevalence. Add weighted sensitivity analyses when making population-facing conclusions.

### What counts as success

A successful research prototype runs end to end and provides honest evidence about whether the proposed separation adds value. Stronger evidence would combine useful reconstruction, informative amount vectors, temporal sensitivity, and stable interpretable groups beyond simpler baselines.

Set concrete comparison criteria after establishing validation baselines and before opening final 2023 results. Do not invent performance targets or guarantee improvement. A finding that engineered features perform equally well is a valid limitation to report.

## 10. Repository and reproducibility

Suggested layout, to adapt to any existing repository:

```text
configs/                 # smoke, prototype, and full experiment settings
data/raw/                # unchanged downloads; ignored by git
data/processed/          # reproducible intermediate tables; ignored by git
src/data/                # ingestion, mapping, validation, rasterization, pairs
src/models/              # encoders, decoder, losses
src/baselines/           # totals, engineered features, joint models
src/evaluation/          # reconstruction, probes, clustering, stability
scripts/                 # command-line entry points
tests/                   # important data and pipeline invariants
notebooks/               # exploratory analysis, not the sole pipeline
outputs/                 # versioned run directories; large artifacts ignored
docs/                    # data decisions, experiments, limitations
README.md
AGENTS.md
```

Provide commands for data preparation, baseline fitting, model training, embedding export, evaluation, and plots. Document actual implemented commands in the README; do not present planned interfaces as working software.

Every run should save its resolved configuration, seed, split IDs, data manifest, category mapping, checkpoint, preprocessing objects, metrics, and relevant plots. Include a code revision when available. Avoid overwriting prior runs. A saved checkpoint must be reloadable without refitting preprocessing.

Keep health variables and survey metadata out of model tensors by explicit column selection. Do not commit raw respondent records, large checkpoints, credentials, or generated caches. Use aggregate figures and appropriately handled examples in shared reports.

## 11. Minimum useful prototype and checkpoints

### Checkpoint 1 — Understand and prepare the data

Load real 2022 data, document schema, validate joins and durations, implement category mapping, and plot example days. Report exclusions and discretization error. Deliver a deterministic split and prepared sequences.

### Checkpoint 2 — Establish a baseline

Run totals and engineered-feature baselines. Save metrics and example clusters. Confirm that the evaluation can distinguish differences in temporal organization even when totals match.

### Checkpoint 3 — End-to-end model

Use a representative subset of roughly 100–500 valid diaries for a quick prototype. Train a small VAE, reload its checkpoint, reconstruct held-out examples, and export both embeddings. Subset results are engineering checks, not final findings.

### Checkpoint 4 — Add separation guidance

Implement controlled pairs, count-preservation checks, invariance, and amount prediction. Compare with the constraint removed. Show both reconstruction and collapse diagnostics. This is the core midpoint demonstration.

### Checkpoint 5 — Evaluate the research claim

Expand to the full intended 2022 cohort, tune only on validation, run probes and stability checks, then freeze the pipeline and evaluate on 2023. Separate final test results from earlier exploratory work.

### Checkpoint 6 — Report and presentation

Prepare a concise results table, model diagram, example timelines, limitations, and a reproducible run guide. Explain what the model outputs and what clustering adds. Verify current course deadlines and formatting against the actual assignment PDF; this file does not establish dates or page limits.

## 12. Essential software checks

Prioritize tests that catch meaningful failures:

- Each retained diary has a valid ordered timeline and the expected number of bins.
- Joins preserve episode counts and intended key uniqueness.
- Splits and transformed descendants do not overlap.
- Every transformation preserves category counts exactly.
- Model distributions, samples, logits, and losses have expected shapes and finite values.
- Gradients reach both encoder heads and the decoder.
- A tiny debugging batch can be overfit sufficiently to reveal major pipeline errors; this is not a generalization result.
- Checkpoint reload reproduces deterministic posterior-mean embeddings and evaluation outputs within numerical tolerance.

Avoid broad test suites that merely repeat implementation details. Do not launch large training runs before these checks pass.

## 13. Instructions for coding agents

First inspect the repository, existing instructions, dependencies, and current changes. Preserve teammates' work. If the repository is empty, scaffold the smallest pipeline needed for Checkpoint 1 and continue incrementally.

Explain scientific choices in plain language. Keep changes focused, make reasonable reversible decisions, and record them. Ask only when a missing decision materially changes scope or a required resource is unavailable. Do not treat this file as authorization to publish, message others, or use paid compute.

Use configurable paths and parameters. Prefer scripts that work without notebooks. Do not claim a command, experiment, dataset download, or test ran unless it actually did. Label synthetic demonstrations and incomplete analyses clearly.

Before ending a development task, report what changed, what was verified, any blocker, and the next useful step. Maintain a short project-status document with completed checkpoints and unresolved decisions. Do not rewrite the project's research claims to match convenient results.

## 14. Literature and how it informs this project

- **Li, Y., & Mandt, S. (2018). Disentangled Sequential Autoencoder.** ICML, PMLR 80, 5670–5679. https://proceedings.mlr.press/v80/yingzhen18a.html — Architectural inspiration for separating sequence information. Their static/dynamic split is not our amount/timing split.
- **Hsu, W.-N., Zhang, Y., & Glass, J. (2017). Unsupervised Learning of Disentangled and Interpretable Representations from Sequential Data.** NeurIPS 30, 1876–1887. — Related hierarchical separation of sequence-level and segment-level information. Use it to understand structural assumptions, not as evidence our factors will automatically separate.
- **Locatello et al. (2019). Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations.** ICML, PMLR 97, 4114–4124. https://proceedings.mlr.press/v97/locatello19a.html — Motivates stating the assumptions and guidance behind an intended separation.
- **Locatello et al. (2020). Weakly-Supervised Disentanglement Without Compromises.** ICML, PMLR 119, 6348–6359. https://proceedings.mlr.press/v119/locatello20a.html — Motivates learning from paired observations with shared factors. Our constructed schedule pairs are an adaptation; their results do not guarantee success for diaries.

Read the original papers before asserting detailed architectural or experimental claims. The intended contribution is a measured adaptation to daily activity amounts and temporal organization, not the invention of disentanglement, VAEs, or clustering.
