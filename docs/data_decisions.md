# Data decisions (Checkpoint 1)

Decisions made while preparing ATUS diaries, with the reasoning. Each is reversible; change the config
or mapping version rather than editing raw data.

## Sources

- Files come from the official year pages (`https://www.bls.gov/tus/data/datafiles-<year>.htm`). Links are
  resolved from the page, not hardcoded. `data/raw/manifest.csv` records URL, retrieval time (UTC),
  SHA-256 and size for every file.
- Downloaded for 2022 and 2023: activity, respondent, roster files; interview data dictionary; activity
  coding lexicon (PDF + XLS). Module files (Eating & Health) are not downloaded yet; not needed for
  Checkpoint 1.
- BLS returns HTTP 403 to non-browser clients. `src/data/download.py` sends browser-like headers. If that
  stops working, download manually into `data/raw/<year>/` and run `download_atus.py --register`.

## Schema facts verified against the 2022 interview data dictionary

| Fact | Source |
| --- | --- |
| Diary window is 4:00 a.m. to 4:00 a.m. next day; `TUACTDUR24` truncates the last activity at 4:00 a.m. | Dictionary entries for `TUACTDUR24`, `TUCUMDUR24` |
| `TUCUMDUR24` = running total of `TUACTDUR24` per `TUCASEID`, max 1440 | Dictionary |
| Six-digit activity code column is `TRCODE` (tiers in `TUTIER1CODE`..`TUTIER3CODE`) | Dictionary, file header |
| Respondent is always `TULINENO == 1` on the roster | Dictionary introduction |
| `-1` blank, `-2` don't know, `-3` refused | Dictionary |
| Final weight column is `TUFINLWGT` | Dictionary, respondent header |
| `TUDIARYDAY` 1 = Sunday ... 7 = Saturday | Dictionary |

The activity, respondent and roster headers are identical in 2022 and 2023 (checked column-by-column).
Files are comma-delimited with a header row; every column is read as a string.

## Timeline construction

- Diary-relative minute 0 = 04:00. Episode start/end come from cumulative `TUACTDUR24` in
  `TUACTIVITY_N` order. Clock time = (240 + minute) mod 1440.
- Checks: consecutive episode IDs, durations >= 1, our cumulative sum equals `TUCUMDUR24` (no
  gaps/overlaps), total 1440, respondent record exists. `TUSTARTTIM` is compared with the implied clock and
  logged but does not exclude.
- Result: all 8,136 (2022) and 8,548 (2023) diaries pass. BLS already removes diaries that fail its own
  quality rules, so zero exclusions here is plausible, not evidence that checks are vacuous (tests confirm
  that they fire on broken fixtures).

## Category mapping `v1` (`configs/category_mapping_v1.yaml`)

13 categories: sleep, personal_care, eating_drinking, household, caregiving, work, education,
shopping_services, leisure_social, exercise_sports, travel, other, unknown. Longest-prefix rules.

- `0101` (sleeping, **including 010102 sleeplessness**) -> sleep; rest of `01` -> personal_care.
  Sleeplessness is lexicon tier-2 "Sleeping"; a different choice is defensible.
- `0202` food & drink prep stays in household; `11` eating & drinking is its own category.
- `1301` participating in sports/exercise -> exercise_sports; `1302` attending events and the
  spectator-side waiting/security codes (`130302`, `130402`) -> leisure_social. TV sports (`1203`) is leisure.
- `050202`/`050203` (eating or sports "as part of job") stay work.
- `07`-`09` purchases and services -> shopping_services; `10` civic, `14` religious, `15` volunteer -> other.
- `16` telephone calls -> leisure_social.
- `18` travel -> travel regardless of purpose.
- `50` data codes (refused, gap/can't remember, insufficient detail, missing travel) -> unknown. Unknown is
  0.75% of minutes in 2022. No observed code matched no rule in either year.

## Discretization (5-minute bins, 288 per diary)

- Each bin takes the category with most minutes; ties go to the category appearing first in the bin.
- 93% of 2022 episode durations are multiples of 5, so the error is small: mean 0.83 minutes per diary
  reassigned (p99 7 min). But 1.9% of episodes (mostly short travel, household, caregiving segments)
  vanish entirely, which lowers within-day transition counts slightly. See `docs/data_report_2022.md`.
- Amount targets for the model should come from the binned sequence (`counts` in `sequences.npz`).
  A finer resolution is not justified by these errors at present.

## Boundaries

- 92.7% of 2022 diaries start in sleep and 95.6% end in sleep (90.3% both). Sleep crossing 04:00 is
  censored on both sides; within-window sleep-run counts (mostly 2) reflect the window, not sleep quality.

## Split

- 2022 valid diaries, 80/20 by respondent, seed 42: 6,509 train / 1,627 val. Each respondent has one diary.
- 2023 is labelled `test` and only integrity-checked. No model has been evaluated on it.

## Kept out of model inputs

Weights (`TUFINLWGT`), month, diary day, holiday flag, labour-force status, age and sex are in
`metadata.parquet`, separate from `sequences.npz`. No health variables are loaded.
