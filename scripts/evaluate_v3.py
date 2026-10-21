"""Evaluate v3 experiments against the v1 baseline, the joint model and engineered features (2022 validation).

Usage:
    python scripts/evaluate_v3.py                          # all groups found
    python scripts/evaluate_v3.py --groups v1_base E1_local

Groups are defined in GROUPS (run-directory glob per seed). Writes outputs/v3/eval/{per_seed.csv,cross_seed.json}
and docs/v3_results.md. 2022 only.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.baselines.features import FeatureSet, TimingFeaturizer, totals_features  # noqa: E402
from src.data.rasterize import category_counts  # noqa: E402
from src.evaluation import v3_metrics as M  # noqa: E402
from src.evaluation.timing_checks import matched_totals_pairs  # noqa: E402
from src.models.train import load_checkpoint, posterior_means  # noqa: E402

SEEDS = [42, 43, 44]
K = 6
GROUPS = {  # name -> glob pattern with {s} for the seed; latest match is used
    "v1_base": "outputs/checkpoint5_robustness/guided_l0.3_s{s}-*",
    "joint": "outputs/checkpoint5_robustness/joint_s{s}-*",
    "E1_local": "outputs/v3/E1_local_s{s}-*",
    "E2_betaT2": "outputs/v3/E2_betaT2_s{s}-*",
    "E3_timing_sup": "outputs/v3/E3_tsup_s{s}-*",
    "C1_combined": "outputs/v3/C1_combined_s{s}-*",
}


def find(pattern: str, s: int) -> Path | None:
    runs = sorted(p for p in glob.glob(str(ROOT / pattern.format(s=s))) if Path(p, "best.pt").exists()
                  and Path(p, "eval_best.json").exists())
    return Path(runs[-1]) if runs else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", nargs="*", default=list(GROUPS))
    args = ap.parse_args()
    out_dir = ROOT / "outputs/v3/eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    z = np.load(ROOT / "data/processed/2022/sequences.npz")
    cats = z["categories"].tolist()
    C, clock, seq = len(cats), z["clock_features"], z["seq"]
    tr, va = z["split"] == "train", z["split"] == "val"
    tot, _ = totals_features(category_counts(seq, C), cats)
    tf = TimingFeaturizer(cats, clock).fit(seq[tr])
    X, names = tf.transform(seq), tf.names_
    # Same probe targets as the v2 evaluation (clock: supervised in E3; fragmentation: never supervised).
    from src.evaluation.separation import PROBE_TARGETS
    clock_cols = [names.index(t) for t in PROBE_TARGETS["clock"]]
    frag_cols = [names.index(t) for t in PROBE_TARGETS["fragmentation"]]
    sv = seq[va]
    pairs = matched_totals_pairs(category_counts(sv, C), 120, 5)
    ccols = [i for i, n in enumerate(names) if n.split("_")[0] in ("center", "first", "last")]
    Tc = StandardScaler().fit(X[tr][:, ccols]).transform(X[va][:, ccols])
    pair_dist = np.linalg.norm(Tc[pairs[:, 0]] - Tc[pairs[:, 1]], axis=1)

    rows, cross = [], {}
    for g in args.groups:
        reps, labels = {}, {}
        for s in SEEDS:
            run = find(GROUPS[g], s)
            if run is None:
                print(f"missing {g} seed {s}")
                continue
            model, ck = load_checkpoint(run / "best.pt")
            emb = posterior_means(model, torch.from_numpy(seq).long())
            emb_va = {k: v[va] for k, v in emb.items()}
            r = {"group": g, "seed": s, "epoch": ck.get("epoch"), "run": str(run.relative_to(ROOT))}
            r.update(M.latent_summary(emb_va))
            r.update(M.decoder_dependence(model, sv))
            r.update(M.reconstruction(model, sv, cats))
            r.update(M.transform_sensitivity(model, sv, emb_va, C))
            for lat in ("amount", "timing"):
                Z = emb[f"{lat}_mu"]
                if Z.shape[1] == 0:
                    continue
                r[f"{lat}_totalsR2"] = M.totals_r2(Z, tot, tr, va, cats)
                r[f"{lat}_clock_gain"] = M.gain_beyond_totals(Z, tot, X, clock_cols, tr, va)
                r[f"{lat}_frag_gain"] = M.gain_beyond_totals(Z, tot, X, frag_cols, tr, va)
            Zt = emb["timing_mu"]
            labels[s] = M.kmeans_labels(Zt[tr], Zt[va], K, s, scale=False)
            init = [M.kmeans_labels(Zt[tr], Zt[va], K, 100 + i, scale=False) for i in range(3)]
            from sklearn.metrics import adjusted_rand_score
            r["kmeans_init_ARI"] = float(np.mean([adjusted_rand_score(init[a], init[b]) for a, b in ((0, 1), (0, 2), (1, 2))]))
            r["timing_split_AUC"] = M.timing_split_auc(labels[s], pairs, pair_dist)
            reps[s] = Zt[va]
            rows.append(r)
            print(f"{g} s{s}: CE {r['CE']:.3f} timing clock gain {r.get('timing_clock_gain', np.nan):.3f} "
                  f"frag gain {r.get('timing_frag_gain', np.nan):.3f}", flush=True)
        if len(reps) >= 2:
            cross[g] = M.cross_seed(reps, labels)
            print(f"  {g} cross-seed: {cross[g]}", flush=True)

    # Engineered timing features: deterministic, so only k-means initialisation varies.
    fs = FeatureSet("eng", False, True, cats, clock).fit(seq[tr])
    E = fs.transform(seq, category_counts(seq, C))
    E = StandardScaler().fit(E[tr]).transform(E)
    eng_labels = {s: M.kmeans_labels(E[tr], E[va], K, s, scale=False) for s in SEEDS}
    cross["engineered_timing"] = {**M.cross_seed({s: E[va] for s in SEEDS}, eng_labels),
                                  "timing_split_AUC": float(np.mean([M.timing_split_auc(l, pairs, pair_dist) for l in eng_labels.values()]))}
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "per_seed.csv", index=False)
    (out_dir / "cross_seed.json").write_text(json.dumps(cross, indent=2))
    write_report(df, cross)


def write_report(df: pd.DataFrame, cross: dict) -> None:
    sections = {
        "Information in each latent (tree probes; gains are R² beyond the true totals)": [
            "amount_totalsR2", "timing_totalsR2", "amount_clock_gain", "timing_clock_gain", "amount_frag_gain", "timing_frag_gain"],
        "Collapse and decoder dependence": ["amount_KL", "amount_active", "timing_KL", "timing_active", "CE_own",
                                            "CE_amount_from_other", "CE_timing_from_other"],
        "Responses to controlled transforms (relative to random-pair distance)": [
            "amount_global_shift", "timing_global_shift", "amount_far_swap", "timing_far_swap", "amount_local_shift",
            "timing_local_shift", "amount_split_merge", "timing_split_merge"],
        "Reconstruction (argmax vs self-budgeted decoding)": [
            "CE", "argmax_transitions", "selfbudget_transitions", "argmax_dur_err_min", "selfbudget_dur_err_min",
            "argmax_F1_eating_drinking", "selfbudget_F1_eating_drinking", "selfbudget_F1_eating_drinking_15min",
            "argmax_F1_travel", "selfbudget_F1_travel", "argmax_macroF1", "selfbudget_macroF1"],
        "Groups within one model (k=6, n_init=50)": ["kmeans_init_ARI", "timing_split_AUC"],
    }
    groups = list(dict.fromkeys(df["group"]))
    agg = df.groupby("group")
    lines = ["# v3 results (2022 validation)", "",
             "Generated by `scripts/evaluate_v3.py`. Mean ± SD over seeds 42/43/44; 40-epoch runs; 1,627 validation diaries.",
             "All probes, scalers and centroids fitted on 2022 train-split rows. See docs/v3_plan.md for definitions.",
             f"True transitions per day: {df['true_transitions'].iloc[0]:.1f}.", ""]
    for title, cols in sections.items():
        cols = [c for c in cols if c in df]
        lines += [f"## {title}", "", "| Metric | " + " | ".join(groups) + " |", "| --- | " + " | ".join("---:" for _ in groups) + " |"]
        for c in cols:
            cells = []
            for g in groups:
                v = agg.get_group(g)[c]
                cells.append("—" if v.isna().all() else f"{v.mean():.3f} ± {v.std():.3f}")
            lines.append(f"| {c} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["## Stability across model seeds (same validation diaries)", "",
              "kNN-20 overlap and linear CKA compare representations without clustering; k-means ARI uses n_init=50.",
              "Engineered features are deterministic (overlap 1); only their k-means initialisation varies.", "",
              "| Representation | kNN-20 overlap | Linear CKA | k-means ARI (k=6) | Timing-split AUC |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for g, c in cross.items():
        auc = c.get("timing_split_AUC", df[df["group"] == g]["timing_split_AUC"].mean() if g in groups else np.nan)
        lines.append(f"| {g} | {c['knn20_overlap']:.3f} | {c['linear_CKA']:.3f} | {c['kmeans_ARI']:.3f} | {auc:.3f} |")
    (ROOT / "docs/v3_results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
