"""Frozen v3 evaluation (docs/preregistration_v3.md): separation, pattern recognition, reconstruction.

Usage:
    python scripts/evaluate_v3_final.py --target val                              # 2022 validation references
    python scripts/evaluate_v3_final.py --target test --confirm-preregistered     # ATUS 2025, once, after approval

Frozen v3 = v1 two-latent VAE (λ_invariance = 0.3, 40 epochs) with seeds 42-47, self-budgeted decoding, and
two disjoint 3-model ensembles (A = 42-44, B = 45-47) for groups. Probes, scalers, timing-feature fill values
and k-means centroids are fitted on 2022 train-split rows only and applied unchanged to the target diaries.
Writes outputs/v3/final_<target>/ and docs/v3_final_<target>.md.
"""

from __future__ import annotations

import argparse
import glob
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

from src.baselines.features import FeatureSet, TimingFeaturizer, totals_features  # noqa: E402
from src.data.rasterize import category_counts  # noqa: E402
from src.evaluation import v3_metrics as M  # noqa: E402
from src.evaluation.separation import PROBE_TARGETS  # noqa: E402
from src.evaluation.timing_checks import matched_totals_pairs  # noqa: E402
from src.models.train import load_checkpoint, posterior_means  # noqa: E402

TEST_YEAR = 2025
K = 6
MODELS = {42: "outputs/checkpoint5_robustness/guided_l0.3_s42-*", 43: "outputs/checkpoint5_robustness/guided_l0.3_s43-*",
          44: "outputs/checkpoint5_robustness/guided_l0.3_s44-*", 45: "outputs/v3/base_s45-*",
          46: "outputs/v3/base_s46-*", 47: "outputs/v3/base_s47-*"}
JOINT = {s: f"outputs/checkpoint5_robustness/joint_s{s}-*" for s in (42, 43, 44)}
ENSEMBLES = {"A": (42, 43, 44), "B": (45, 46, 47)}


def latest(pat: str) -> Path:
    runs = sorted(glob.glob(str(ROOT / pat)))
    if not runs:
        raise FileNotFoundError(pat)
    return Path(runs[-1])


def sleep_midpoint_hours(seq: np.ndarray, sleep_idx: int, start_clock: int = 240, w: int = 5) -> np.ndarray:
    m = seq == sleep_idx
    ang = 2 * np.pi * ((start_clock + w * np.arange(seq.shape[1]) + w / 2) % 1440) / 1440
    return np.degrees(np.arctan2((m * np.sin(ang)).sum(1), (m * np.cos(ang)).sum(1))) % 360 / 360 * 24


def midpoint_ratio(labels: np.ndarray, pairs: np.ndarray, mid: np.ndarray) -> dict:
    """Among matched-totals pairs: mean sleep-midpoint difference (h) for split vs kept-together pairs."""
    d = np.abs(mid[pairs[:, 0]] - mid[pairs[:, 1]])
    d = np.minimum(d, 24 - d)
    split = labels[pairs[:, 0]] != labels[pairs[:, 1]]
    a, b = d[split].mean(), d[~split].mean()
    return {"split_h": float(a), "kept_h": float(b), "ratio": float(a / b), "share_split": float(split.mean())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["val", "test"], required=True)
    ap.add_argument("--confirm-preregistered", action="store_true")
    args = ap.parse_args()
    if args.target == "test" and not args.confirm_preregistered:
        sys.exit(f"Refusing to open {TEST_YEAR}: approve docs/preregistration_v3.md, then pass --confirm-preregistered.")

    z = np.load(ROOT / "data/processed/2022/sequences.npz")
    cats = z["categories"].tolist()
    C, clock = len(cats), z["clock_features"]
    trm = z["split"] == "train"
    if args.target == "val":
        tgt = z["seq"][z["split"] == "val"]
    else:
        zt = np.load(ROOT / f"data/processed/{TEST_YEAR}/sequences.npz")
        assert zt["categories"].tolist() == cats
        tgt = zt["seq"]
    seq = np.concatenate([z["seq"][trm], tgt])
    split = np.array(["train"] * int(trm.sum()) + ["val"] * len(tgt))
    tr, ev = split == "train", split == "val"
    sv = seq[ev]
    tot, _ = totals_features(category_counts(seq, C), cats)
    tf = TimingFeaturizer(cats, clock).fit(seq[tr])
    X, names = tf.transform(seq), tf.names_
    clock_cols = [names.index(t) for t in PROBE_TARGETS["clock"]]
    frag_cols = [names.index(t) for t in PROBE_TARGETS["fragmentation"]]
    ccols = [i for i, n in enumerate(names) if n.split("_")[0] in ("center", "first", "last")]
    Tc = StandardScaler().fit(X[tr][:, ccols]).transform(X[ev][:, ccols])
    cnt_ev = category_counts(sv, C)
    pairs120 = matched_totals_pairs(cnt_ev, 120, 5)
    pdist = np.linalg.norm(Tc[pairs120[:, 0]] - Tc[pairs120[:, 1]], axis=1)
    pairs60 = matched_totals_pairs(cnt_ev, 60, 5)
    mid = sleep_midpoint_hours(sv, cats.index("sleep"))
    out_dir = ROOT / f"outputs/v3/final_{args.target}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, emb_t, labels = [], {}, {}
    for s, pat in MODELS.items():
        run = latest(pat)
        model, ck = load_checkpoint(run / "best.pt")
        emb = posterior_means(model, torch.from_numpy(seq).long())
        e_ev = {k: v[ev] for k, v in emb.items()}
        r = {"seed": s, "run": str(run.relative_to(ROOT)), "epoch": ck.get("epoch")}
        r.update(M.latent_summary(e_ev))
        r.update(M.decoder_dependence(model, sv))
        r.update(M.reconstruction(model, sv, cats))
        r.update(M.transform_sensitivity(model, sv, e_ev, C))
        for lat in ("amount", "timing"):
            Z = emb[f"{lat}_mu"]
            r[f"{lat}_totalsR2"] = M.totals_r2(Z, tot, tr, ev, cats)
            r[f"{lat}_clock_gain"] = M.gain_beyond_totals(Z, tot, X, clock_cols, tr, ev)
            r[f"{lat}_frag_gain"] = M.gain_beyond_totals(Z, tot, X, frag_cols, tr, ev)
        emb_t[s] = emb["timing_mu"]
        labels[s] = M.kmeans_labels(emb["timing_mu"][tr], emb["timing_mu"][ev], K, s, scale=False)
        rows.append(r)
        print(f"seed {s}: CE {r['CE']:.3f} amount totals R² {r['amount_totalsR2']:.3f} timing clock gain {r['timing_clock_gain']:.3f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "per_model.csv", index=False)

    single_pairs = list(combinations(sorted(MODELS), 2))
    ens = {"single_ARI_mean": float(np.mean([adjusted_rand_score(labels[a], labels[b]) for a, b in single_pairs])),
           "single_knn20_mean": float(np.mean([M.knn_overlap(emb_t[a][ev], emb_t[b][ev]) for a, b in single_pairs]))}
    ens_lab, ens_rep = {}, {}
    for name, seeds in ENSEMBLES.items():
        R = np.hstack([emb_t[s] for s in seeds])
        ens_rep[name] = R
        ens_lab[name] = M.kmeans_labels(R[tr], R[ev], K, 0, scale=False)
        ens[f"{name}_timing_clock_gain"] = M.gain_beyond_totals(R, tot, X, clock_cols, tr, ev)
        ens[f"{name}_timing_split_AUC"] = M.timing_split_auc(ens_lab[name], pairs120, pdist)
        ens[f"{name}_midpoint"] = midpoint_ratio(ens_lab[name], pairs60, mid)
        ens[f"{name}_group_shares"] = (np.bincount(ens_lab[name], minlength=K) / len(sv)).round(3).tolist()
    ens["A_vs_B_ARI"] = float(adjusted_rand_score(ens_lab["A"], ens_lab["B"]))
    ens["A_vs_B_knn20"] = float(M.knn_overlap(ens_rep["A"][ev], ens_rep["B"][ev]))

    # References: engineered timing features and the joint latent (40 epochs, seeds 42-44).
    fs = FeatureSet("eng", False, True, cats, clock).fit(seq[tr])
    E = fs.transform(seq, category_counts(seq, C))
    E = StandardScaler().fit(E[tr]).transform(E)
    eng_lab = {s: M.kmeans_labels(E[tr], E[ev], K, s, scale=False) for s in (42, 43, 44)}
    ref = {"engineered_kmeans_ARI": float(np.mean([adjusted_rand_score(eng_lab[a], eng_lab[b]) for a, b in combinations(eng_lab, 2)])),
           "engineered_timing_split_AUC": float(np.mean([M.timing_split_auc(l, pairs120, pdist) for l in eng_lab.values()])),
           "engineered_midpoint_ratio": float(np.mean([midpoint_ratio(l, pairs60, mid)["ratio"] for l in eng_lab.values()])),
           "engineered_clockonly_totalsR2": M.totals_r2(X[:, ccols], tot, tr, ev, cats)}
    jl = {}
    for s, pat in JOINT.items():
        jm, _ = load_checkpoint(latest(pat) / "best.pt")
        je = posterior_means(jm, torch.from_numpy(seq).long())["timing_mu"]
        jl[s] = M.kmeans_labels(je[tr], je[ev], K, s, scale=False)
    ref["joint_single_ARI_mean"] = float(np.mean([adjusted_rand_score(jl[a], jl[b]) for a, b in combinations(jl, 2)]))
    ref["joint_timing_split_AUC"] = float(np.mean([M.timing_split_auc(l, pairs120, pdist) for l in jl.values()]))
    ref["n_pairs_120"], ref["n_pairs_60"] = int(len(pairs120)), int(len(pairs60))

    result = {"target": args.target, "n_diaries": int(len(sv)), "ensembles": ens, "references": ref,
              "per_model_mean": df.drop(columns=["run"]).mean(numeric_only=True).to_dict()}
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2))
    write_report(df, ens, ref, args.target, len(sv), out_dir)


def criteria(df: pd.DataFrame, ens: dict, ref: dict) -> list[tuple]:
    m = df.mean(numeric_only=True)
    return [
        ("S1", "Amount latent holds totals: mean amount totals R² ≥ 0.85", m["amount_totalsR2"], 0.85, m["amount_totalsR2"] >= 0.85),
        ("S2", "Amount latent excludes timing: mean amount clock gain ≤ 0.05", m["amount_clock_gain"], 0.05, m["amount_clock_gain"] <= 0.05),
        ("S3", "Timing latent holds timing beyond totals: mean timing clock gain ≥ 0.50", m["timing_clock_gain"], 0.50, m["timing_clock_gain"] >= 0.50),
        ("S4", "Timing latent does not duplicate amounts: timing totals R² ≤ 0.5 × engineered clock-only",
         m["timing_totalsR2"], 0.5 * ref["engineered_clockonly_totalsR2"], m["timing_totalsR2"] <= 0.5 * ref["engineered_clockonly_totalsR2"]),
        ("S5a", "Decoder uses amount: CE with another diary's amount ≥ 2 × own", m["CE_amount_from_other"], 2 * m["CE_own"], m["CE_amount_from_other"] >= 2 * m["CE_own"]),
        ("S5b", "Decoder uses timing: CE with another diary's timing ≥ 2 × own", m["CE_timing_from_other"], 2 * m["CE_own"], m["CE_timing_from_other"] >= 2 * m["CE_own"]),
        ("S6a", "Amount ignores global shift: amount ≤ 0.25 × timing sensitivity", m["amount_global_shift"], 0.25 * m["timing_global_shift"], m["amount_global_shift"] <= 0.25 * m["timing_global_shift"]),
        ("S6b", "Amount ignores block swap: amount ≤ 0.25 × timing sensitivity", m["amount_far_swap"], 0.25 * m["timing_far_swap"], m["amount_far_swap"] <= 0.25 * m["timing_far_swap"]),
        ("P1", "Ensemble groups reproducible: ensemble A vs B ARI ≥ 0.30", ens["A_vs_B_ARI"], 0.30, ens["A_vs_B_ARI"] >= 0.30),
        ("P2", "Ensembling improves reliability: A vs B ARI ≥ 1.5 × mean single-model ARI", ens["A_vs_B_ARI"], 1.5 * ens["single_ARI_mean"], ens["A_vs_B_ARI"] >= 1.5 * ens["single_ARI_mean"]),
        ("P3a", "Groups split same-total days by timing: sleep-midpoint ratio ≥ 1.5 (ensemble A)", ens["A_midpoint"]["ratio"], 1.5, ens["A_midpoint"]["ratio"] >= 1.5),
        ("P3b", "Same, ensemble B", ens["B_midpoint"]["ratio"], 1.5, ens["B_midpoint"]["ratio"] >= 1.5),
        ("P4", "Group splits track timing: mean ensemble timing-split AUC ≥ 0.58",
         (ens["A_timing_split_AUC"] + ens["B_timing_split_AUC"]) / 2, 0.58, (ens["A_timing_split_AUC"] + ens["B_timing_split_AUC"]) / 2 >= 0.58),
        ("P5", "As reliable as engineered features: A vs B ARI ≥ engineered k-means ARI", ens["A_vs_B_ARI"], ref["engineered_kmeans_ARI"], ens["A_vs_B_ARI"] >= ref["engineered_kmeans_ARI"]),
        ("R1", "Self-budgeted decoding realistic: transitions within 20% of true",
         m["selfbudget_transitions"], m["true_transitions"], abs(m["selfbudget_transitions"] - m["true_transitions"]) <= 0.2 * m["true_transitions"]),
        ("R2", "Self-budgeted duration error ≤ 0.6 × argmax", m["selfbudget_dur_err_min"], 0.6 * m["argmax_dur_err_min"], m["selfbudget_dur_err_min"] <= 0.6 * m["argmax_dur_err_min"]),
    ]


def write_report(df, ens, ref, target, n, out_dir) -> None:
    crit = criteria(df, ens, ref)
    title = "2022 validation (references)" if target == "val" else f"ATUS {TEST_YEAR} TEST (frozen v3)"
    lines = [f"# v3 final evaluation: {title}", "",
             f"Generated by `scripts/evaluate_v3_final.py --target {target}`. {n} diaries. Six frozen v1 models (seeds 42-47,",
             "40 epochs); ensembles A = 42-44, B = 45-47. All fitting on 2022 train-split rows. Unweighted.", "",
             "| # | Criterion | Value | Threshold | Met |", "| --- | --- | ---: | ---: | :---: |",
             *[f"| {c} | {t} | {v:.3f} | {th:.3f} | {'yes' if ok else 'no'} |" for c, t, v, th, ok in crit],
             "", f"Met: {sum(bool(ok) for *_, ok in crit)} of {len(crit)}.", "",
             "## Per-model means (± SD over 6 seeds)", "", "| Metric | Mean ± SD |", "| --- | ---: |"]
    for c in ["amount_totalsR2", "timing_totalsR2", "amount_clock_gain", "timing_clock_gain", "amount_frag_gain",
              "timing_frag_gain", "amount_KL", "timing_KL", "CE_own", "CE_amount_from_other", "CE_timing_from_other",
              "amount_global_shift", "timing_global_shift", "amount_far_swap", "timing_far_swap", "amount_split_merge",
              "timing_split_merge", "argmax_transitions", "selfbudget_transitions", "true_transitions",
              "argmax_dur_err_min", "selfbudget_dur_err_min", "selfbudget_F1_eating_drinking_15min"]:
        lines.append(f"| {c} | {df[c].mean():.3f} ± {df[c].std():.3f} |")
    lines += ["", "## Groups", "", f"```\n{json.dumps({'ensembles': ens, 'references': ref}, indent=1)}\n```", ""]
    (ROOT / f"docs/v3_final_{target}.md").write_text("\n".join(lines))
    print("\n".join(lines[:9 + len(crit) + 2]))


if __name__ == "__main__":
    main()
