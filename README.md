# Daily Schedule Representations (CSC 5666)

Research prototype that learns separate representations of **activity amounts** and **temporal organization**
in American Time Use Survey (ATUS) diaries. See `AGENTS.md` for the research plan and `docs/status.md`
for progress. Only Checkpoint 1 (data preparation) is implemented so far.

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
src/data/  download, ingestion, mapping, validation, binning, split, reports
scripts/   command-line entry points
tests/     pipeline invariants
docs/      data decisions, data reports, project status
```
