"""Per-diary integrity checks. Invalid diaries are logged with reasons, never silently repaired."""

from __future__ import annotations

import numpy as np
import pandas as pd


def validate_diaries(ep: pd.DataFrame, respondent_ids: set[str], diary_minutes: int,
                     diary_start_clock_minute: int, unknown_category: str,
                     max_unknown_minutes: int | None = None) -> pd.DataFrame:
    """Return one row per diary with check results, an `exclusion_reason`, and `valid`.

    Structural failures exclude a diary. Clock-time disagreements are recorded but do not exclude,
    because diary-relative positions are derived from TUACTDUR24, not from TUSTARTTIM.
    """
    g = ep.groupby("case_id", sort=True)
    d = pd.DataFrame({
        "n_episodes": g.size(),
        "total_min": g["duration_min"].sum(),
        "min_duration": g["duration_min"].min(),
        "max_episode_id": g["episode_id"].max(),
        "min_episode_id": g["episode_id"].min(),
    })
    # TUCUMDUR24 must equal our running sum of TUACTDUR24: otherwise there is a gap or overlap.
    cum_mismatch = (ep["cum_end_min"] != ep["end_min"]).groupby(ep["case_id"]).sum()
    d["n_cum_mismatch"] = cum_mismatch
    # Reported start clock vs. clock implied by diary-relative start.
    implied = (diary_start_clock_minute + ep["start_min"]) % 1440
    reported = ep["start_clock_min"] % 1440
    d["n_clock_mismatch"] = (implied != reported).groupby(ep["case_id"]).sum()
    unk = ep["category"] == unknown_category
    d["unknown_min"] = ep["duration_min"].where(unk, 0).groupby(ep["case_id"]).sum()
    d["n_unmatched_codes"] = (~ep["code_matched"]).groupby(ep["case_id"]).sum()
    d["last_truncated"] = g.apply(
        lambda x: bool(x["duration_untruncated_min"].iloc[-1] > x["duration_min"].iloc[-1]),
        include_groups=False,
    )

    reasons = pd.Series("", index=d.index, dtype=object)

    def flag(mask: pd.Series, why: str) -> None:
        reasons[mask] = np.where(reasons[mask] == "", why, reasons[mask] + ";" + why)

    flag(~d.index.isin(list(respondent_ids)), "no_respondent_record")
    flag((d["min_episode_id"] != 1) | (d["max_episode_id"] != d["n_episodes"]), "nonconsecutive_episode_ids")
    flag(d["min_duration"] < 1, "nonpositive_duration")
    flag(d["n_cum_mismatch"] > 0, "gap_or_overlap")
    flag(d["total_min"] != diary_minutes, "total_not_1440")
    if max_unknown_minutes is not None:
        flag(d["unknown_min"] > max_unknown_minutes, "too_much_unknown")

    d["exclusion_reason"] = reasons
    d["valid"] = reasons == ""
    return d.reset_index()


def check_episode_join(ep_before: int, ep_after: int) -> None:
    if ep_before != ep_after:
        raise AssertionError(f"join changed episode count: {ep_before} -> {ep_after}")
