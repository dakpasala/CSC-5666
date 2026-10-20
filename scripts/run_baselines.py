"""Checkpoint 2: fit totals / timing / totals+timing clustering baselines and run timing checks.

Usage:
    python scripts/run_baselines.py                       # configs/baselines.yaml
    python scripts/run_baselines.py --config <yaml>

Everything is fitted on the 2022 train split; k and diagnostics use the validation split.
Writes a new run directory outputs/checkpoint2/<timestamp>/ (never overwritten) and an aggregate
summary to docs/baselines_report.md.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.baselines.features import FeatureSet  # noqa: E402
from src.evaluation.clustering import ClusterPipeline, choose_k, select_k  # noqa: E402
from src.evaluation.plotting import INK, palette, plt, stacked_profile, timeline_rows  # noqa: E402
from src.evaluation.timing_checks import (  # noqa: E402
    matched_pairs_report, matched_totals_pairs, totals_to_timing_r2, transform_sensitivity,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clock_str(sin: float, cos: float) -> str:
    minute = (np.degrees(np.arctan2(sin, cos)) % 360) / 360 * 1440
    return f"{int(minute // 60):02d}:{int(minute % 60):02d}"


def profile_table(labels: np.ndarray, seq: np.ndarray, counts: np.ndarray, cats: list[str],
                  bin_minutes: int, clock: np.ndarray) -> pd.DataFrame:
    """Per-cluster descriptions from actual diaries: totals and when key activities happen."""
    rows = []
    for c in np.unique(labels):
        m = labels == c
        r = {"cluster": int(c), "n": int(m.sum()), "share": float(m.mean())}
        for k, name in enumerate(cats):
            r[f"min_{name}"] = round(float(counts[m, k].mean() * bin_minutes), 1)
        for name in ("sleep", "work", "eating_drinking", "leisure_social"):
            k = cats.index(name)
            oh = seq[m] == k
            present = oh.any(axis=1)
            r[f"pct_with_{name}"] = round(float(present.mean() * 100), 1)
            if present.any():
                s = (oh[present] * clock[None, :, 0]).sum() / oh[present].sum()
                co = (oh[present] * clock[None, :, 1]).sum() / oh[present].sum()
                r[f"center_{name}"] = clock_str(s, co)   # circular mean clock time among diaries with it
        rows.append(r)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "configs/baselines.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    dcfg = yaml.safe_load((ROOT / cfg["data_config"]).read_text())
    seed = cfg["seed"]

    proc = ROOT / dcfg["processed_dir"] / str(cfg["year"])
    npz = proc / "sequences.npz"
    z = np.load(npz)
    cats = z["categories"].tolist()
    clock = z["clock_features"]
    w, start_clock = int(z["bin_minutes"]), int(z["diary_start_clock_minute"])
    tr, va = z["split"] == "train", z["split"] == "val"
    assert not set(z["case_ids"][tr]) & set(z["case_ids"][va])
    seq_tr, seq_va = z["seq"][tr], z["seq"][va]
    cnt_tr, cnt_va = z["counts"][tr], z["counts"][va]
    ids_va = z["case_ids"][va]

    run = ROOT / cfg["output_dir"] / dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run.mkdir(parents=True, exist_ok=False)
    rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    (run / "resolved_config.yaml").write_text(yaml.safe_dump(
        {**cfg, "data": dcfg, "git_revision": rev, "sequences_sha256": sha256(npz),
         "n_train": int(tr.sum()), "n_val": int(va.sum())}, sort_keys=False))
    shutil.copy(proc / "category_mapping.yaml", run / "category_mapping.yaml")
    shutil.copy(proc / "split.csv", run / "split.csv")

    ks = cfg["k_selection"]
    fitted, k_tables, chosen, labels_va = {}, [], {}, {}
    for name, b in cfg["baselines"].items():
        fs = FeatureSet(name, b["use_totals"], b["use_timing"], cats, clock).fit(seq_tr)
        X_tr, X_va = fs.transform(seq_tr, cnt_tr), fs.transform(seq_va, cnt_va)
        kw = {"pca_variance": b["pca_variance"]}
        table = select_k(X_tr, X_va, ks["ks"], ks["n_bootstrap"], seed, ks["silhouette_sample"], **kw)
        table.insert(0, "baseline", name)
        k_tables.append(table)
        k = choose_k(table, ks["min_stability"], ks["min_cluster_share"])
        pipe = ClusterPipeline(k, seed=seed, **kw).fit(X_tr)
        fitted[name] = (fs, pipe)
        chosen[name] = k
        labels_va[name] = pipe.predict(X_va)
        kc = ks["comparison_k"]
        cmp_pipe = ClusterPipeline(kc, seed=seed, **kw).fit(X_tr)
        fitted[f"{name}@k{kc}"] = (fs, cmp_pipe)
        labels_va[f"{name}@k{kc}"] = cmp_pipe.predict(X_va)
        print(f"[{name}] {X_tr.shape[1]} features, chosen k={k}")
    k_table = pd.concat(k_tables, ignore_index=True)
    k_table.to_csv(run / "k_selection.csv", index=False)
    joblib.dump(fitted, run / "baselines.joblib")
    pd.DataFrame({"case_id": ids_va, **{f"cluster_{n}": l for n, l in labels_va.items()}}).to_parquet(
        run / "val_assignments.parquet", index=False)

    # --- Timing checks --------------------------------------------------------------------------
    tc = cfg["timing_checks"]
    sens = transform_sensitivity(fitted, seq_va, cnt_va, len(cats), tc["shifts_bins"], seed)
    sens.to_csv(run / "transform_sensitivity.csv", index=False)

    timing_fs, timing_pipe = fitted["timing_only"]
    Zt_va = timing_pipe.embed(timing_fs.transform(seq_va, cnt_va))
    matched = {}
    for thr in tc["matched_totals_max_reassigned_minutes"]:
        pairs = matched_totals_pairs(cnt_va, thr, w)
        matched[thr] = matched_pairs_report(pairs, labels_va, Zt_va, seed) if len(pairs) else {"n_pairs": 0}

    tot_fs = fitted["totals"][0]
    r2 = totals_to_timing_r2(tot_fs.transform(seq_tr, cnt_tr), timing_fs.transform(seq_tr, cnt_tr),
                             tot_fs.transform(seq_va, cnt_va), timing_fs.transform(seq_va, cnt_va),
                             timing_fs.names)
    r2.to_csv(run / "totals_to_timing_r2.csv", index=False)
    r2_summary = r2.groupby("group")["val_r2"].describe()[["count", "mean", "50%", "min", "max"]].round(3)

    # --- Cluster profiles and figures -----------------------------------------------------------
    cmap_handles = palette(cats)[1]
    profiles = {}
    for name, (fs, pipe) in fitted.items():
        lab = labels_va[name]
        prof = profile_table(lab, seq_va, cnt_va, cats, w, clock)
        prof.to_csv(run / f"profiles_{name}.csv", index=False)
        profiles[name] = prof
        k = pipe.k
        # Time-of-day profile per cluster (aggregate).
        fig, axes = plt.subplots(k, 1, figsize=(9, 1.5 * k + 1), sharex=True, squeeze=False)
        for c, ax in enumerate(axes[:, 0]):
            stacked_profile(ax, seq_va[lab == c], cats, start_clock)
            ax.set_ylabel(f"C{c}\n{(lab == c).mean():.0%}", rotation=0, ha="right", va="center", color=INK)
            ax.set_yticks([0, 1])
        fig.suptitle(f"{name}: validation clusters (k={k}), share of diaries in each activity", x=0.01,
                     ha="left", color=INK)
        fig.legend(handles=cmap_handles, loc="lower center", ncol=9, frameon=False, fontsize=7)
        fig.tight_layout(rect=(0, 0.1, 1, 0.97))
        fig.savefig(run / f"cluster_profiles_{name}.png", dpi=130)
        plt.close(fig)
        # Representative actual diaries: closest to each centroid (individual records; git-ignored).
        X_va = fs.transform(seq_va, cnt_va)
        dist = pipe.km_.transform(pipe.embed(X_va))
        n_rep = cfg["profiles"]["n_representatives"]
        rows, ylab = [], []
        for c in range(k):
            members = np.flatnonzero(lab == c)
            for j in members[np.argsort(dist[members, c])[:n_rep]]:
                rows.append(seq_va[j])
                ylab.append(f"C{c}")
        fig, ax = plt.subplots(figsize=(9, 0.22 * len(rows) + 1.2))
        timeline_rows(ax, np.array(rows), cats, start_clock, ylab)
        ax.set_title(f"{name}: {n_rep} validation diaries nearest each centroid", loc="left", color=INK)
        fig.legend(handles=cmap_handles, loc="lower center", ncol=9, frameon=False, fontsize=7)
        fig.tight_layout(rect=(0, 0.13, 1, 1))
        fig.savefig(run / f"representatives_{name}.png", dpi=130)
        plt.close(fig)

    metrics = {
        "chosen_k": chosen,
        "k_selection_rule": f"best val silhouette with bootstrap ARI >= {ks['min_stability']} and smallest "
                            f"val cluster >= {ks['min_cluster_share']:.0%}; else most stable k",
        "chosen_k_rows": k_table.merge(pd.DataFrame(chosen.items(), columns=["baseline", "k"])).to_dict("records"),
        "transform_sensitivity": sens.to_dict("records"),
        "matched_totals_pairs": matched,
        "totals_to_timing_r2_by_group": r2_summary.reset_index().to_dict("records"),
    }
    (run / "metrics.json").write_text(json.dumps(metrics, indent=2))
    write_report(run, metrics, k_table, sens, r2_summary, profiles, tc, int(tr.sum()), int(va.sum()))
    print(f"wrote {run}")


def write_report(run: Path, m: dict, k_table: pd.DataFrame, sens: pd.DataFrame, r2: pd.DataFrame,
                 profiles: dict, tc: dict, n_tr: int, n_va: int) -> None:
    def md(df: pd.DataFrame) -> str:
        return df.to_markdown(index=False, floatfmt=".3f")

    mp = m["matched_totals_pairs"]
    labels = next(r for r in mp.values() if r["n_pairs"])["same_cluster_rate"].keys()
    sens_wide = sens.pivot_table(index="transform", columns="baseline",
                                 values=["rel_distance_median", "cluster_reassignment_rate"]).round(3)
    sens_wide.columns = [f"{b} {v.replace('rel_distance_median', 'rel dist').replace('cluster_reassignment_rate', 'reassigned')}"
                         for v, b in sens_wide.columns]
    keep_cols = ["cluster", "n", "share", "min_sleep", "min_work", "min_leisure_social", "min_household",
                 "min_eating_drinking", "min_travel", "min_education", "min_shopping_services", "min_caregiving", "min_other", "pct_with_work", "center_sleep", "center_work",
                 "center_eating_drinking"]
    lines = [
        "# Checkpoint 2 baselines report",
        "",
        f"Run: `{run.relative_to(ROOT)}`. Data: ATUS 2022, train n={n_tr}, validation n={n_va}. Unweighted.",
        "All scaling, imputation, PCA and k-means fitted on train only; k and diagnostics on validation.",
        "2023 was not used. These are baseline results, not final findings.",
        "",
        "## k selection (validation)",
        "",
        f"Rule (fixed before inspecting profiles): {m['k_selection_rule']}. Chosen: {m['chosen_k']}. "
        "A fixed comparison k is also reported for every baseline (`<baseline>@k<k>`); it was added after "
        "seeing this table and is for like-for-like comparison, not a selected k.",
        "",
        md(k_table.round(3)),
        "",
        "## Can the evaluation see timing when totals match?",
        "",
        "### Total-preserving transforms of validation diaries",
        "",
        "`rel dist` = median distance between a diary and its transformed copy, in that baseline's scaled "
        "feature space, divided by the median distance between two random validation diaries. "
        "`reassigned` = share of diaries whose cluster changes. Shifts wrap across 04:00 (artificial "
        "boundary effect); swaps exchange two non-boundary runs.",
        "",
        sens_wide.reset_index().to_markdown(index=False),
        "",
        "### Real diaries with near-identical totals",
        "",
        "Each validation diary is paired with its nearest neighbour by category counts; pairs are kept "
        "if totals differ by at most the threshold (minutes reassigned between categories). `timing dist` "
        "is the median timing-feature distance relative to random pairs (1.0 = as different as random "
        "diaries). Same-cluster rates: matched pairs / random pairs.",
        "",
        "| Max totals diff (min) | Pairs | timing dist | " + " | ".join(labels) + " |",
        "| ---: | ---: | ---: | " + " | ".join("---:" for _ in labels) + " |",
        *[f"| {thr} | {r['n_pairs']} | {r['timing_distance_rel_to_random_pairs']['median']:.2f} | "
          + " | ".join(f"{r['same_cluster_rate'][b]:.2f} / {r['same_cluster_rate_random_pairs'][b]:.2f}" for b in labels)
          + " |" for thr, r in mp.items() if r["n_pairs"]],
        "",
        "### How much of each timing feature do totals explain? (ridge, validation R^2)",
        "",
        r2.reset_index().to_markdown(index=False),
        "",
        "## Cluster profiles (validation diaries)",
        "",
        "Minutes are mean minutes/day from binned sequences. `center_*` is the circular mean clock time "
        "of the activity among diaries that contain it. Descriptions, not value judgements.",
        "",
    ]
    for name, prof in profiles.items():
        lines += [f"### {name} (k={prof['cluster'].nunique()})", "", md(prof[[c for c in keep_cols if c in prof]]), ""]
    lines += ["Figures (git-ignored, in the run directory): `cluster_profiles_<baseline>.png`, "
              "`representatives_<baseline>.png`.", ""]
    (ROOT / "docs/baselines_report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
