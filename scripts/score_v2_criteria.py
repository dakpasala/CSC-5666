"""Score the pre-registered v2 criteria (docs/preregistration_v2.md) mechanically from evaluate_v2 output.

Usage:
    python scripts/score_v2_criteria.py --target test     # or val, to reproduce the validation column

Reads outputs/v2/eval_<target>/{per_run.csv,clusters.json}; writes docs/v2_criteria_<target>.md.
Written before the 2024 results were seen.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["val", "test"], required=True)
    args = ap.parse_args()
    d = ROOT / "outputs/v2" / f"eval_{args.target}"
    df = pd.read_csv(d / "per_run.csv")
    cl = json.loads((d / "clusters.json").read_text())
    m = df.groupby("model").mean(numeric_only=True)
    v1 = [x for x in m.index if x.startswith("v1_")]
    v2, g = m.loc["v2"], m.loc["v1_guided"]
    true_tr = df["transitions_true"].dropna().iloc[0]
    crit = [
        ("R1", "v2 CE < every v1 model", v2.CE, m.loc[v1, "CE"].min(), v2.CE < m.loc[v1, "CE"].min()),
        ("R2", "transitions within 10% of true", v2.transitions_pred, true_tr,
         abs(v2.transitions_pred - true_tr) <= 0.1 * true_tr),
        ("R3", "duration error < v1 guided", v2.dur_err_min, g.dur_err_min, v2.dur_err_min < g.dur_err_min),
        ("S1a", "amount sensitivity (shift) = 0", v2.amount_sens_shift, 0.0, abs(v2.amount_sens_shift) < 1e-9),
        ("S1b", "amount sensitivity (swap) = 0", v2.amount_sens_swap, 0.0, abs(v2.amount_sens_swap) < 1e-9),
        ("S2", "matched mix-and-match > v1 guided", v2.matched_mix, g.matched_mix, v2.matched_mix > g.matched_mix),
        ("S3", "split/merge sensitivity > v1 guided", v2.timing_sens_splitmerge, g.timing_sens_splitmerge,
         v2.timing_sens_splitmerge > g.timing_sens_splitmerge),
        ("S4", "totals leakage ≤ 0.5 × untrained v2", v2.leak_totalsR2_timing_knn,
         0.5 * m.loc["untrained_v2", "leak_totalsR2_timing_knn"],
         v2.leak_totalsR2_timing_knn <= 0.5 * m.loc["untrained_v2", "leak_totalsR2_timing_knn"]),
        ("S5", "tree fragmentation gain ≥ v1 guided − 0.02", v2.tree_frag_gain, g.tree_frag_gain - 0.02,
         v2.tree_frag_gain >= g.tree_frag_gain - 0.02),
        ("S6", "tree clock gain ≥ v1 guided − 0.02", v2.tree_clock_gain, g.tree_clock_gain - 0.02,
         v2.tree_clock_gain >= g.tree_clock_gain - 0.02),
        ("G1", "cross-seed ARI ≥ 0.5", cl["v2"]["cross_seed_ari_mean"], 0.5, cl["v2"]["cross_seed_ari_mean"] >= 0.5),
        ("G2", "timing-split AUC ≥ 0.60", cl["v2"]["timing_split_auc_mean"], 0.60, cl["v2"]["timing_split_auc_mean"] >= 0.60),
    ]
    lines = [f"# v2 pre-registered criteria — {'ATUS 2024 TEST' if args.target == 'test' else '2022 validation'}", "",
             "Scored mechanically by `scripts/score_v2_criteria.py` from `outputs/v2/eval_" + args.target + "/`.", "",
             "| # | Criterion | v2 value | Reference / threshold | Met |", "| --- | --- | ---: | ---: | :---: |",
             *[f"| {c} | {t} | {v:.3f} | {r:.3f} | {'yes' if ok else 'no'} |" for c, t, v, r, ok in crit],
             "", f"Met: {sum(ok for *_, ok in crit)} of {len(crit)}.", ""]
    (ROOT / f"docs/v2_criteria_{args.target}.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
