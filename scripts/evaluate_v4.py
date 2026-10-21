"""Evaluate v4 mixture-prior models (docs/v4_mixture_plan.md). 2022 validation; null models on null validation.

Usage:
    python scripts/evaluate_v4.py

Groups = argmax of the model's own responsibilities q(c|x) at the posterior mean. For each family (real K=4,
real K=6, null K=6) reports cross-seed ARI (3 pairs), ARI on confident days (top half of max responsibility
in both models), component usage, same-total sleep-midpoint ratio and totals/timing variance explained
(real data), and the v3 separation/reconstruction metrics. v1 80-epoch references: single-model k-means on
the timing latent (same k) and the 3-ensemble results in docs/v3_k_sweep_ensembles.md.
Writes outputs/v4/eval/ and docs/v4_results.md.
"""

from __future__ import annotations

import glob
import importlib.util
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.baselines.features import TimingFeaturizer, totals_features  # noqa: E402
from src.data.rasterize import category_counts  # noqa: E402
from src.evaluation import v3_metrics as M  # noqa: E402
from src.evaluation.separation import PROBE_TARGETS  # noqa: E402
from src.evaluation.timing_checks import matched_totals_pairs  # noqa: E402
from src.models.train import load_checkpoint, posterior_means  # noqa: E402


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


v3f = _load("v3f", "scripts/evaluate_v3_final.py")
ks3 = _load("ks3", "scripts/k_sweep_v3.py")
SEEDS = (42, 43, 44)
FAMILIES = {"gm_k4": ("2022", 4), "gm_k6": ("2022", 6), "null_k6": ("null2022", 6)}


@torch.no_grad()
def groups(model, seq: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    emb = posterior_means(model, torch.from_numpy(seq).long())
    g = model.mixture_responsibilities(torch.from_numpy(emb["timing_mu"])).numpy()
    return g.argmax(1), g.max(1), emb


def main() -> None:
    out_dir = ROOT / "outputs/v4/eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    data = {}
    for key in ("2022", "null2022"):
        z = np.load(ROOT / f"data/processed/{key}/sequences.npz")
        data[key] = z
    rows, fam_summary = [], {}
    for fam, (key, K) in FAMILIES.items():
        z = data[key]
        cats = z["categories"].tolist()
        C, clock, seq = len(cats), z["clock_features"], z["seq"]
        tr, va = z["split"] == "train", z["split"] == "val"
        sv = seq[va]
        tot, _ = totals_features(category_counts(seq, C), cats)
        tf = TimingFeaturizer(cats, clock).fit(seq[tr])
        X, names = tf.transform(seq), tf.names_
        clock_cols = [names.index(t) for t in PROBE_TARGETS["clock"]]
        ccols = [i for i, n in enumerate(names) if n.split("_")[0] in ("center", "first", "last")]
        Tclock = StandardScaler().fit(X[tr][:, ccols]).transform(X[va][:, ccols])
        Ttot = StandardScaler().fit(tot[tr]).transform(tot[va])
        pairs = matched_totals_pairs(category_counts(sv, C), 60, 5)
        mid = v3f.sleep_midpoint_hours(sv, cats.index("sleep"))
        lab, conf = {}, {}
        for s in SEEDS:
            runs = sorted(p for p in glob.glob(str(ROOT / f"outputs/v4/{fam}_s{s}-*")) if Path(p, "eval_best.json").exists())
            if not runs:
                print(f"missing {fam} s{s}")
                continue
            model, ck = load_checkpoint(Path(runs[-1]) / "best.pt")
            l_all, c_all, emb = groups(model, seq)
            lab[s], conf[s] = l_all[va], c_all[va]
            e_va = {k: v[va] for k, v in emb.items()}
            r = {"family": fam, "seed": s, "epoch": ck.get("epoch"), "K": K,
                 "components_used": int((np.bincount(lab[s], minlength=K) > 0.01 * len(sv)).sum()),
                 "smallest_group": float(np.bincount(lab[s], minlength=K).min() / len(sv)),
                 "mean_max_resp": float(conf[s].mean()),
                 "eta2_timing": ks3.eta2(Tclock, lab[s]), "eta2_totals": ks3.eta2(Ttot, lab[s]),
                 "midpoint_ratio": v3f.midpoint_ratio(lab[s], pairs, mid)["ratio"]}
            r.update(M.latent_summary(e_va))
            r.update(M.decoder_dependence(model, sv))
            rec = M.reconstruction(model, sv, cats)
            r.update({k: rec[k] for k in ("CE", "argmax_acc", "selfbudget_transitions", "selfbudget_dur_err_min")})
            r.update(M.transform_sensitivity(model, sv, e_va, C))
            r["amount_totalsR2"] = M.totals_r2(emb["amount_mu"], tot, tr, va, cats)
            r["amount_clock_gain"] = M.gain_beyond_totals(emb["amount_mu"], tot, X, clock_cols, tr, va)
            r["timing_clock_gain"] = M.gain_beyond_totals(emb["timing_mu"], tot, X, clock_cols, tr, va)
            rows.append(r)
            print(f"{fam} s{s}: used {r['components_used']}/{K}, smallest {r['smallest_group']:.1%}, "
                  f"conf {r['mean_max_resp']:.2f}, CE {r['CE']:.3f}, midpoint {r['midpoint_ratio']:.2f}", flush=True)
        pairs_s = list(combinations(sorted(lab), 2))
        aris, cores = [], []
        for a, b in pairs_s:
            core = (conf[a] >= np.quantile(conf[a], 0.5)) & (conf[b] >= np.quantile(conf[b], 0.5))
            aris.append(adjusted_rand_score(lab[a], lab[b]))
            cores.append(adjusted_rand_score(lab[a][core], lab[b][core]))
        fam_summary[fam] = {"cross_seed_ARI_mean": float(np.mean(aris)), "cross_seed_ARI_sd": float(np.std(aris)),
                            "core_ARI_mean": float(np.mean(cores)), "n_pairs": len(aris)}
        print(fam, fam_summary[fam], flush=True)

    # v1 80-epoch reference: single-model k-means on the timing latent at the same k (2022 validation).
    z = data["2022"]
    tr, va = z["split"] == "train", z["split"] == "val"
    for K in (4, 6):
        L = {}
        for s in SEEDS:
            e = posterior_means(load_checkpoint(v3f.latest(f"outputs/checkpoint5_final/guided_s{s}-*") / "best.pt")[0],
                                torch.from_numpy(z["seq"]).long())["timing_mu"]
            L[s] = M.kmeans_labels(e[tr], e[va], K, s, scale=False)
        fam_summary[f"v1_80ep_kmeans_k{K}"] = {"cross_seed_ARI_mean": float(np.mean([adjusted_rand_score(L[a], L[b]) for a, b in combinations(L, 2)]))}
    ens = pd.read_csv(ROOT / "outputs/v3/k_sweep_ensembles/k_sweep_ensembles.csv")
    for K in (4, 6):
        row = ens[(ens["epochs"].astype(str) == "80") & (ens["k"] == K)].iloc[0]
        fam_summary[f"v1_80ep_3ensembles_k{K}"] = {"cross_seed_ARI_mean": float(row["ARI_mean"]), "cross_seed_ARI_sd": float(row["ARI_sd"]),
                                                   "core_ARI_mean": float(row["ARI_core_mean"])}

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "per_model.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(fam_summary, indent=2))
    cols = ["components_used", "smallest_group", "mean_max_resp", "midpoint_ratio", "eta2_timing", "eta2_totals",
            "CE", "argmax_acc", "selfbudget_transitions", "amount_totalsR2", "amount_clock_gain", "timing_clock_gain",
            "timing_KL", "timing_active", "CE_amount_from_other", "CE_timing_from_other", "amount_global_shift",
            "timing_global_shift"]
    agg = df.groupby("family", sort=False)[[c for c in cols if c in df]].agg(["mean", "std"])
    lines = ["# v4 mixture-prior results (development; 2022 validation, null models on null validation)", "",
             "Generated by `scripts/evaluate_v4.py`. Groups = the model's own argmax responsibility. 3 seeds per family.", "",
             "## Group reproducibility across seeds", "", "| Representation | ARI (mean ± SD over pairs) | ARI on confident days |",
             "| --- | ---: | ---: |"]
    for k, v in fam_summary.items():
        sd = f" ± {v['cross_seed_ARI_sd']:.3f}" if "cross_seed_ARI_sd" in v else ""
        core = f"{v['core_ARI_mean']:.3f}" if "core_ARI_mean" in v else "—"
        lines.append(f"| {k} | {v['cross_seed_ARI_mean']:.3f}{sd} | {core} |")
    lines += ["", "## Per-family metrics (mean ± SD over seeds)", "", "| Metric | " + " | ".join(agg.index) + " |",
              "| --- | " + " | ".join("---:" for _ in agg.index) + " |"]
    for c in [c for c in cols if c in df]:
        lines.append(f"| {c} | " + " | ".join(f"{agg.loc[f, (c, 'mean')]:.3f} ± {agg.loc[f, (c, 'std')]:.3f}" for f in agg.index) + " |")
    (ROOT / "docs/v4_results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
