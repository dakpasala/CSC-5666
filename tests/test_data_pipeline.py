"""Data-pipeline invariants. Synthetic diaries here are software fixtures, not ATUS results."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.ingest import build_episodes, build_metadata
from src.data.mapping import CategoryMapping
from src.data.rasterize import bin_timelines, category_counts, clock_features, minute_timelines
from src.data.split import respondent_split
from src.data.validate import validate_diaries

ROOT = Path(__file__).resolve().parents[1]
MAPPING = CategoryMapping.load(ROOT / "configs/category_mapping_v1.yaml")


def fake_activity(case_id: str, episodes: list[tuple[str, int]], cum_override: dict | None = None) -> pd.DataFrame:
    """Build string-typed rows shaped like the ATUS activity file from (code, minutes) pairs."""
    rows, cum, clock = [], 0, 240
    for i, (code, dur) in enumerate(episodes, start=1):
        cum += dur
        rows.append({
            "TUCASEID": case_id, "TUACTIVITY_N": str(i), "TUACTDUR": str(dur), "TUACTDUR24": str(dur),
            "TUCUMDUR24": str((cum_override or {}).get(i, cum)),
            "TUSTARTTIM": f"{clock // 60 % 24:02d}:{clock % 60:02d}:00",
            "TUSTOPTIME": "00:00:00", "TRCODE": code,
        })
        clock += dur
    return pd.DataFrame(rows)


GOOD_DAY = [("010101", 180), ("020201", 32), ("110101", 28), ("050101", 480), ("180501", 40),
            ("130124", 60), ("120303", 140), ("010101", 480)]  # sums to 1440


def validate(act: pd.DataFrame, resp_ids=None) -> pd.DataFrame:
    ep = build_episodes(act, 2022, MAPPING)
    ids = set(act["TUCASEID"]) if resp_ids is None else resp_ids
    return validate_diaries(ep, ids, 1440, 240, MAPPING.unknown)


def test_mapping_distinguishes_required_categories():
    m = MAPPING.map_code
    assert m("010101") == "sleep" and m("010201") == "personal_care"
    assert m("110101") == "eating_drinking" and m("020201") == "household"   # eating vs food prep
    assert m("130124") == "exercise_sports" and m("130213") == "leisure_social"  # running vs watching
    assert m("120303") == "leisure_social"        # TV, including sports on TV
    assert m("500106") == "unknown" and m("999999") == "unknown"
    assert m("10101") == "sleep"                  # leading zero restored


def test_valid_diary_passes_and_tiles_window():
    act = fake_activity("20220101000001", GOOD_DAY)
    d = validate(act)
    assert d["valid"].all() and d["n_clock_mismatch"].sum() == 0
    ep = build_episodes(act, 2022, MAPPING)
    assert len(ep) == len(act)
    assert ep["start_min"].iloc[0] == 0 and ep["end_min"].iloc[-1] == 1440
    assert (ep["start_min"].iloc[1:].to_numpy() == ep["end_min"].iloc[:-1].to_numpy()).all()
    assert ep["activity_code"].str.len().eq(6).all() and pd.api.types.is_string_dtype(ep["case_id"])


@pytest.mark.parametrize("act,reason", [
    (fake_activity("c1", GOOD_DAY[:-1]), "total_not_1440"),
    (fake_activity("c2", GOOD_DAY, cum_override={3: 999}), "gap_or_overlap"),
])
def test_invalid_diaries_are_flagged(act, reason):
    d = validate(act)
    assert not d["valid"].any() and reason in d["exclusion_reason"].iloc[0]


def test_missing_respondent_flagged():
    d = validate(fake_activity("c3", GOOD_DAY), resp_ids=set())
    assert "no_respondent_record" in d["exclusion_reason"].iloc[0]


def test_binning_majority_and_tie_rule():
    # 3 min of category 2 beat 2 min of category 1
    minutes = np.array([[2, 2, 2, 1, 1] + [0] * 1435], dtype=np.int16)
    assert bin_timelines(minutes, 5, 13)[0, 0] == 2
    # w=4 tie: 2 min cat 5 then 2 min cat 3 -> earliest (5) wins
    tie = np.array([[5, 5, 3, 3] + [0] * 1436], dtype=np.int16)
    assert bin_timelines(tie, 4, 13)[0, 0] == 5


def test_sequences_have_expected_shape_and_counts():
    act = pd.concat([fake_activity("a", GOOD_DAY), fake_activity("b", GOOD_DAY[::-1])])
    ep = build_episodes(act, 2022, MAPPING)
    mins = minute_timelines(ep, ["a", "b"])
    seq = bin_timelines(mins, 5, len(MAPPING.categories))
    assert seq.shape == (2, 288)
    counts = category_counts(seq, len(MAPPING.categories))
    assert (counts.sum(axis=1) == 288).all()
    assert clock_features(288, 5, 240).shape == (288, 2)


def test_split_deterministic_and_disjoint():
    ids = [f"{i:014d}" for i in range(1000)]
    a, b = respondent_split(ids, 0.2, 42), respondent_split(ids[::-1], 0.2, 42)
    pd.testing.assert_frame_equal(a, b)
    assert (a["split"] == "val").sum() == 200 and a["case_id"].is_unique


def test_metadata_uses_respondent_roster_line_and_decodes_missing():
    resp = pd.DataFrame({"TUCASEID": ["x", "y"], "TELFS": ["1", "-3"], "TUFINLWGT": ["10.5", "0"]})
    rost = pd.DataFrame({"TUCASEID": ["x", "x", "y"], "TULINENO": ["1", "2", "1"],
                         "TEAGE": ["40", "8", "-1"], "TESEX": ["2", "1", "1"]})
    meta = build_metadata(resp, rost, ["TELFS", "TUFINLWGT"], ["TEAGE", "TESEX"], [-1, -2, -3])
    assert len(meta) == 2
    assert meta.set_index("case_id").loc["x", "TEAGE"] == 40
    assert pd.isna(meta.set_index("case_id").loc["y", "TELFS"])
    assert pd.isna(meta.set_index("case_id").loc["y", "TEAGE"])
    assert meta.set_index("case_id").loc["y", "TUFINLWGT"] == 0   # weight 0 is a value, not missing
    with pytest.raises(ValueError):
        build_metadata(pd.concat([resp, resp]), None, ["TELFS"], [], [-1])


# --- Checks on real prepared data (skipped if scripts/prepare_data.py has not been run) --------

PROC = ROOT / "data/processed/2022"


@pytest.mark.skipif(not (PROC / "sequences.npz").exists(), reason="2022 data not prepared")
def test_prepared_2022_invariants():
    z = np.load(PROC / "sequences.npz")
    ep = pd.read_parquet(PROC / "episodes.parquet")
    diaries = pd.read_parquet(PROC / "diaries.parquet")
    assert z["seq"].shape[1] == 288 and len(z["case_ids"]) == diaries["valid"].sum()
    assert (z["counts"].sum(axis=1) == 288).all()
    assert pd.Index(z["case_ids"]).is_unique and diaries["case_id"].is_unique
    assert set(z["split"]) == {"train", "val"}
    tr, va = set(z["case_ids"][z["split"] == "train"]), set(z["case_ids"][z["split"] == "val"])
    assert not tr & va
    g = ep.groupby("case_id")
    assert (g["duration_min"].sum().loc[z["case_ids"]] == 1440).all()
    assert ep.duplicated(["case_id", "episode_id"]).sum() == 0
    assert ep["code_matched"].all()
