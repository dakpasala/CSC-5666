# Daily Schedule Representations (CSC 5666)

Research prototype that learns separate representations of **activity amounts** and **temporal organization**
in American Time Use Survey (ATUS) diaries. See `AGENTS.md` for the research plan and `docs/status.md`
for progress. Checkpoints 1-5 are complete: data preparation, engineered-feature baselines, an end-to-end VAE, separation
guidance with ablations, and a pre-registered final evaluation on 2023 (`docs/checkpoint5_results.md`).

## Setup

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Implemented commands

```bash
# 1. Download official ATUS files (2022 train/val, 2023 held-out test) into data/raw/<year>/
python scripts/download_atus.py --years 2022 2023
#    If BLS blocks the request, place files in data/raw/<year>/ by hand, then:
python scripts/download_atus.py --years 2022 --register

# 2. Build episode tables, validate diaries, bin to 5-minute sequences, split, and report
python scripts/prepare_data.py --year 2022
python scripts/prepare_data.py --year 2023      # integrity check only

# 3. Example timelines and time-of-day profile (written to outputs/checkpoint1/<year>/)
python scripts/plot_examples.py --year 2022

# 4. Checkpoint 2 baselines on 2022 train/val (new run dir in outputs/checkpoint2/<timestamp>/)
python scripts/run_baselines.py
#    -> docs/baselines_report.md (aggregate summary); interpretation in docs/checkpoint2_notes.md

# 5. Checkpoint 3 VAE (new run dir in outputs/checkpoint3/<name>-<timestamp>/)
python scripts/train_vae.py --config configs/vae_smoke.yaml        # seconds; pipeline check only
python scripts/train_vae.py --config configs/vae_checkpoint3.yaml  # ~6 min on CPU
python scripts/evaluate_vae.py --run outputs/checkpoint3/<run-dir> # re-evaluate from checkpoint alone

# 6. Checkpoint 4 variants (7 runs in parallel, ~35 min on 8 CPU cores) and comparison report
./scripts/run_checkpoint4.sh
python scripts/compare_checkpoint4.py     # -> docs/checkpoint4_report.md, outputs/checkpoint4/comparison.csv

# 7. Checkpoint 5: full-data robustness, lambda sweep, final models, final evaluation
./scripts/run_robustness.sh && ./scripts/run_lambda_sweep.sh && python scripts/compare_robustness.py
./scripts/run_final.sh                                   # 12 runs x 80 epochs (~2 h on 8 CPU cores)
python scripts/evaluate_final.py --target val            # 2022 validation reference
python scripts/evaluate_final.py --target test --confirm-preregistered   # 2023, only after approval

# Tests (synthetic fixtures + checks on prepared 2022 data when present)
python -m pytest
```

## Outputs of `prepare_data.py` (`data/processed/<year>/`, git-ignored)

| File | Contents |
| --- | --- |
| `episodes.parquet` | One row per episode: diary-relative start/end (minute 0 = 04:00), duration, six-digit code, category |
| `diaries.parquet` | One row per diary: validation checks, exclusion reason, split |
| `metadata.parquet` | Weights and demographics, kept out of model inputs |
| `sequences.npz` | Valid diaries: `case_ids`, `split`, `seq` (N×288 category indices), `counts` (N×13 bins per category), `clock_features` (288×2 sin/cos), `categories` |
| `split.csv` | Respondent split (2022: train/val, seed 42) |
| `category_mapping.yaml`, `source_manifest.csv`, `resolved_config.yaml`, `prep_report.json` | Provenance for the run |

An aggregate report is also written to `docs/data_report_<year>.md`.

## Layout

```text
configs/   data settings and versioned category mapping
src/data/  download, ingestion, mapping, validation, binning, split, reports, transforms
src/baselines/   totals and engineered timing features
src/models/      two-latent sequence VAE, loss terms, training and checkpoints
src/evaluation/  clustering, k selection, timing checks, reconstruction metrics, plotting
scripts/   command-line entry points
tests/     pipeline invariants
docs/      data decisions, data reports, project status
```
