"""Checkpoint 1: build episode tables, validate diaries, bin sequences, split, and report.

Usage:
    python scripts/prepare_data.py --year 2022
    python scripts/prepare_data.py --year 2023   # integrity check only; not split, not tuned on

Writes data/processed/<year>/ (git-ignored) and an aggregate summary to docs/data_report_<year>.md.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.ingest import build_episodes, build_metadata, read_atus_file  # noqa: E402
from src.data.mapping import CategoryMapping  # noqa: E402
from src.data.rasterize import bin_timelines, category_counts, clock_features, minute_timelines  # noqa: E402
from src.data.report import boundary_report, discretization_report  # noqa: E402
from src.data.split import respondent_split  # noqa: E402
from src.data.validate import validate_diaries  # noqa: E402


def git_rev() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--config", type=Path, default=ROOT / "configs/data.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    year = args.year
    raw_dir = ROOT / cfg["raw_dir"]
    out = ROOT / cfg["processed_dir"] / str(year)
    out.mkdir(parents=True, exist_ok=True)
    mapping = CategoryMapping.load(ROOT / cfg["mapping"])
    D, w = cfg["diary_minutes"], cfg["bin_minutes"]

    # --- Ingest ---------------------------------------------------------------------------
    act = read_atus_file(raw_dir, year, "act")
    resp = read_atus_file(raw_dir, year, "resp")
    rost = read_atus_file(raw_dir, year, "rost")
    ep = build_episodes(act, year, mapping)
    assert len(ep) == len(act), "episode construction changed row count"
    meta = build_metadata(resp, rost, cfg["respondent_metadata_columns"], cfg["roster_metadata_columns"],
                          cfg["missing_value_codes"])

    # --- Validate ---------------------------------------------------------------------------
    diaries = validate_diaries(ep, set(meta["case_id"]), D, cfg["diary_start_clock_minute"],
                               mapping.unknown, cfg["max_unknown_minutes"])
    valid_ids = diaries.loc[diaries["valid"], "case_id"].tolist()  # sorted by case_id

    # --- Rasterize --------------------------------------------------------------------------
    minutes = minute_timelines(ep, valid_ids, D)
    seq = bin_timelines(minutes, w, len(mapping.categories))
    counts = category_counts(seq, len(mapping.categories))
    assert seq.shape == (len(valid_ids), D // w)
    assert (counts.sum(axis=1) == D // w).all()

    # --- Split ------------------------------------------------------------------------------
    if year == cfg["train_year"]:
        split = respondent_split(valid_ids, cfg["split"]["val_fraction"], cfg["split"]["seed"])
    else:
        split = pd.DataFrame({"case_id": valid_ids, "split": "test" if year == cfg["test_year"] else "unused"})
    diaries = diaries.merge(split, on="case_id", how="left", validate="one_to_one")
    split_arr = split.set_index("case_id").loc[valid_ids, "split"].to_numpy()

    # --- Reports ----------------------------------------------------------------------------
    disc = discretization_report(ep, valid_ids, minutes, seq, w, mapping.categories)
    bound = boundary_report(ep, valid_ids, seq, mapping.index["sleep"])
    unmatched = ep.loc[~ep["code_matched"], "activity_code"].value_counts().to_dict()
    report = {
        "year": year,
        "counts": {
            "activity_rows": int(len(act)),
            "respondents_in_respondent_file": int(len(meta)),
            "diaries_in_activity_file": int(len(diaries)),
            "valid_diaries": int(len(valid_ids)),
            "excluded_diaries": int((~diaries["valid"]).sum()),
            "respondents_without_activity_rows": int(len(set(meta["case_id"]) - set(ep["case_id"]))),
            "split": pd.Series(split_arr).value_counts().to_dict(),
        },
        "exclusions_by_reason": diaries.loc[~diaries["valid"], "exclusion_reason"].value_counts().to_dict(),
        "clock_mismatch": {
            "diaries_with_any": int((diaries["n_clock_mismatch"] > 0).sum()),
            "episodes": int(diaries["n_clock_mismatch"].sum()),
        },
        "category_minutes_share": {
            c: round(float((minutes == i).mean()), 4) for i, c in enumerate(mapping.categories)
        },
        "unknown": {
            "share_of_minutes": round(float((minutes == mapping.index[mapping.unknown]).mean()), 4),
            "diaries_with_any_unknown": int((counts[:, mapping.index[mapping.unknown]] > 0).sum()),
            "unknown_minutes_per_diary": {
                k: float(v) for k, v in diaries.loc[diaries["valid"], "unknown_min"].describe().round(2).items()
            },
            "codes_matching_no_rule": unmatched,
        },
        "metadata_missing": {c: int(meta[c].isna().sum()) for c in meta.columns if c != "case_id"},
        "discretization": disc,
        "boundary": bound,
        "mapping_version": mapping.version,
        "git_revision": git_rev(),
    }

    # --- Save -------------------------------------------------------------------------------
    ep.to_parquet(out / "episodes.parquet", index=False)
    diaries.to_parquet(out / "diaries.parquet", index=False)
    meta.to_parquet(out / "metadata.parquet", index=False)
    split.to_csv(out / "split.csv", index=False)
    np.savez_compressed(
        out / "sequences.npz",
        case_ids=np.array(valid_ids), split=split_arr.astype(str), seq=seq, counts=counts,
        clock_features=clock_features(D // w, w, cfg["diary_start_clock_minute"]),
        categories=np.array(mapping.categories),
        bin_minutes=w, diary_start_clock_minute=cfg["diary_start_clock_minute"],
    )
    shutil.copy(ROOT / cfg["mapping"], out / "category_mapping.yaml")
    shutil.copy(raw_dir / "manifest.csv", out / "source_manifest.csv")
    (out / "resolved_config.yaml").write_text(yaml.safe_dump({**cfg, "year": year}, sort_keys=False))
    (out / "prep_report.json").write_text(json.dumps(report, indent=2, default=int))
    write_markdown(report, ROOT / "docs" / f"data_report_{year}.md")

    c = report["counts"]
    print(f"[{year}] diaries={c['diaries_in_activity_file']} valid={c['valid_diaries']} "
          f"excluded={c['excluded_diaries']} split={c['split']}")
    print(f"[{year}] mean minutes reassigned by {w}-min binning: "
          f"{disc['reassigned_minutes_per_diary']['mean']} ; episodes lost: "
          f"{disc['episodes_lost_after_binning']}/{disc['episodes_total']}")
    print(f"[{year}] wrote {out}")


def write_markdown(r: dict, path: Path) -> None:
    c, d, b = r["counts"], r["discretization"], r["boundary"]
    lines = [
        f"# ATUS {r['year']} data preparation report",
        "",
        "Generated by `scripts/prepare_data.py`. Aggregate counts only; unweighted.",
        f"Mapping version: `{r['mapping_version']}`. Git revision: `{r['git_revision']}`.",
        "",
        "## Counts",
        "",
        "| Item | Count |", "| --- | ---: |",
        *[f"| {k.replace('_', ' ')} | {v} |" for k, v in c.items() if k != "split"],
        *[f"| split: {k} | {v} |" for k, v in c["split"].items()],
        "",
        "Exclusions by reason: " + (json.dumps(r["exclusions_by_reason"]) if r["exclusions_by_reason"] else "none"),
        "",
        f"Start-clock mismatches (recorded, not excluded): {r['clock_mismatch']['diaries_with_any']} diaries, "
        f"{r['clock_mismatch']['episodes']} episodes.",
        "",
        "## Category share of minutes (valid diaries)",
        "",
        "| Category | Share of minutes | Mean raw min/day | Mean abs binning error (min) | Mean signed error (min) |",
        "| --- | ---: | ---: | ---: | ---: |",
        *[f"| {k} | {v:.2%} | {d['per_category_raw_minutes_mean'][k]} | "
          f"{d['per_category_abs_error_minutes_mean'][k]} | {d['per_category_signed_error_minutes_mean'][k]} |"
          for k, v in r["category_minutes_share"].items()],
        "",
        f"Unknown/data-code minutes: {r['unknown']['share_of_minutes']:.2%} of all minutes; "
        f"{r['unknown']['diaries_with_any_unknown']} diaries contain some after binning. "
        f"Codes matching no mapping rule: {r['unknown']['codes_matching_no_rule'] or 'none'}.",
        "",
        f"## Discretization error ({d['bin_minutes']}-minute bins)",
        "",
        f"- Minutes reassigned to another category per diary: {d['reassigned_minutes_per_diary']}",
        f"- Mean share of the day reassigned: {d['share_of_day_reassigned_mean']:.2%}",
        f"- Episodes shorter than one bin: {d['episodes_shorter_than_bin']} of {d['episodes_total']}",
        f"- Episodes whose category vanishes from all overlapping bins: {d['episodes_lost_after_binning']} "
        f"({d['episodes_lost_after_binning'] / d['episodes_total']:.2%}); by category: {d['episodes_lost_by_category']}",
        f"- Diaries where a present category disappears entirely: {d['categories_dropped_entirely']}",
        f"- Runs per diary, minute level: {d['runs_per_diary_raw_minute']}",
        f"- Runs per diary, binned: {d['runs_per_diary_binned']}",
        "",
        "## Diary window boundaries (04:00 to 04:00)",
        "",
        f"- Starting in sleep: {b['share_starting_in_sleep']:.2%}; ending in sleep: {b['share_ending_in_sleep']:.2%}; "
        f"both: {b['share_sleep_at_both_boundaries']:.2%}",
        f"- Last episode truncated at window end: {b['share_last_episode_truncated_at_window_end']:.2%}",
        f"- Within-window sleep runs (binned) -> diaries: {b['within_window_sleep_runs_binned']}",
        f"- {b['note']}",
        "",
        "## Metadata missingness (after decoding -1/-2/-3)",
        "",
        ", ".join(f"{k}: {v}" for k, v in r["metadata_missing"].items()),
        "",
    ]
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
