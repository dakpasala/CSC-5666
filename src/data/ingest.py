"""Read raw ATUS files and build the canonical episode table plus separate respondent metadata.

All raw columns are read as strings so identifiers and leading zeros in activity codes survive.
Numeric fields are converted explicitly, and official missing-value sentinels are decoded to NA.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .mapping import CategoryMapping

EPISODE_COLUMNS = [
    "TUCASEID", "TUACTIVITY_N", "TUACTDUR", "TUACTDUR24", "TUCUMDUR24",
    "TUSTARTTIM", "TUSTOPTIME", "TRCODE",
]


def _find_member(zf: zipfile.ZipFile, stem: str) -> str:
    names = [n for n in zf.namelist() if n.lower().endswith(".dat") and stem in n.lower()]
    if len(names) != 1:
        raise FileNotFoundError(f"expected one {stem}*.dat in {zf.filename}, found {names}")
    return names[0]


def read_atus_file(raw_dir: Path, year: int, kind: str) -> pd.DataFrame:
    """Read e.g. kind='act' from data/raw/<year>/atusact-<year>.zip (or an extracted .dat)."""
    ydir = Path(raw_dir) / str(year)
    stem = f"atus{kind}_{year}"
    dat = ydir / f"{stem}.dat"
    if dat.exists():
        return pd.read_csv(dat, dtype=str, keep_default_na=False)
    zpath = ydir / f"atus{kind}-{year}.zip"
    if not zpath.exists():
        raise FileNotFoundError(
            f"missing {zpath} (or extracted {dat}). Run scripts/download_atus.py --years {year} "
            "or place the official BLS file there."
        )
    with zipfile.ZipFile(zpath) as zf:
        with zf.open(_find_member(zf, stem)) as f:
            return pd.read_csv(f, dtype=str, keep_default_na=False)


def clock_to_minutes(s: pd.Series) -> pd.Series:
    """'HH:MM:SS' -> minutes after midnight (24:00:00 -> 1440)."""
    parts = s.str.split(":", expand=True).astype(int)
    return parts[0] * 60 + parts[1] + (parts[2] // 60)


def decode_missing(df: pd.DataFrame, cols: list[str], codes: list[int]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        v = pd.to_numeric(out[c], errors="raise")
        out[c] = v.mask(v.isin(codes))
    return out


def build_episodes(act: pd.DataFrame, year: int, mapping: CategoryMapping) -> pd.DataFrame:
    """Canonical episode table, ordered within diary, with diary-relative minutes [start, end)."""
    missing = [c for c in EPISODE_COLUMNS if c not in act.columns]
    if missing:
        raise KeyError(f"activity file lacks expected columns {missing}; check the {year} dictionary")
    ep = pd.DataFrame({
        "year": year,
        "case_id": act["TUCASEID"].astype(str),
        "episode_id": act["TUACTIVITY_N"].astype(int),
        "activity_code": act["TRCODE"].astype(str).str.zfill(6),
        "duration_min": act["TUACTDUR24"].astype(int),
        "duration_untruncated_min": act["TUACTDUR"].astype(int),
        "cum_end_min": act["TUCUMDUR24"].astype(int),
        "start_clock_min": clock_to_minutes(act["TUSTARTTIM"]),
    })
    ep = ep.sort_values(["case_id", "episode_id"], kind="stable").reset_index(drop=True)
    # Diary-relative position from durations within the 24h window (minute 0 = 04:00).
    ep["end_min"] = ep.groupby("case_id")["duration_min"].cumsum()
    ep["start_min"] = ep["end_min"] - ep["duration_min"]
    ep["category"], ep["code_matched"] = mapping.map_codes(ep["activity_code"])
    ep["category_idx"] = ep["category"].map(mapping.index).astype(np.int16)
    cols = ["year", "case_id", "episode_id", "start_min", "end_min", "duration_min",
            "duration_untruncated_min", "cum_end_min", "start_clock_min", "activity_code",
            "category", "category_idx", "code_matched"]
    return ep[cols]


def build_metadata(resp: pd.DataFrame, rost: pd.DataFrame | None, resp_cols: list[str],
                   rost_cols: list[str], missing_codes: list[int]) -> pd.DataFrame:
    """One row per respondent: survey metadata kept out of model inputs."""
    if resp["TUCASEID"].duplicated().any():
        raise ValueError("respondent file has duplicate TUCASEID")
    have = [c for c in resp_cols if c in resp.columns]
    meta = resp[["TUCASEID", *have]].rename(columns={"TUCASEID": "case_id"})
    weight_cols = [c for c in have if "WGT" in c]
    meta = decode_missing(meta, [c for c in have if c not in weight_cols], missing_codes)
    for c in weight_cols:
        meta[c] = pd.to_numeric(meta[c])
    if rost is not None and rost_cols:
        # The respondent is always roster line 1 (interview data dictionary); never merge all lines.
        own = rost[rost["TULINENO"].astype(int) == 1]
        if own["TUCASEID"].duplicated().any():
            raise ValueError("roster has multiple TULINENO==1 rows for a case")
        own = own[["TUCASEID", *[c for c in rost_cols if c in own.columns]]]
        own = decode_missing(own.rename(columns={"TUCASEID": "case_id"}),
                             [c for c in rost_cols if c in own.columns], missing_codes)
        n = len(meta)
        meta = meta.merge(own, on="case_id", how="left", validate="one_to_one")
        assert len(meta) == n
    return meta
