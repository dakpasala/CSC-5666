"""Final evaluation of frozen models against the pre-registered criteria (docs/preregistration.md).

Usage:
    python scripts/evaluate_final.py --target val                     # 2022 validation (references)
    python scripts/evaluate_final.py --target test --confirm-preregistered   # 2023, once, after approval

Everything is fitted on 2022 TRAIN-split data only (probe models, scalers, k-means centroids, timing-feature
fill values) and applied unchanged to the target diaries. Models are the latest runs named <variant>_s<seed>
in --runs-dir. Writes outputs/checkpoint5_final/eval_<target>/ and docs/final_results_<target>.md.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import adjusted_rand_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.baselines.features import FeatureSet, TimingFeaturizer  # noqa: E402
from src.evaluation.clustering import ClusterPipeline  # noqa: E402
from src.evaluation.plotting import INK, palette, plt, stacked_profile  # noqa: E402
from src.evaluation.reconstruction import latent_diagnostics, reconstruction_metrics  # noqa: E402
from src.evaluation.separation import evaluate_separation  # noqa: E402
from src.evaluation.timing_checks import matched_totals_pairs  # noqa: E402
from src.models.train import load_checkpoint, posterior_means, reconstruct, set_seed  # noqa: E402
from src.models.vae import ScheduleVAE, VAEConfig  # noqa: E402

_spec = importlib.util.spec_from_file_location("cmp4", ROOT / "scripts/compare_checkpoint4.py")
cmp4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cmp4)
_spec2 = importlib.util.spec_from_file_location("rb", ROOT / "scripts/run_baselines.py")
rb = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(rb)

VARIANTS = ["guided", "no_invariance", "unguided", "joint"]
SEEDS = [42, 43, 44]
K = 6
MATCH_MINUTES = 120


def load_year(year: int) -> np.lib.npyio.NpzFile:
    return np.load(ROOT / "data/processed" / str(year) / "sequences.npz")


def combined(z22, ztarget, target: str) -> tuple[np.ndarray, np.ndarray]:
    """Rows = 2022 train + target diaries, labelled train / val so separation code fits on train only."""
    tr = z22["split"] == "train"
    tgt_seq = z22["seq"][z22["split"] == "val"] if target == "val" else ztarget["seq"]
    seq = np.concatenate([z22["seq"][tr], tgt_seq])
    split = np.array(["train"] * int(tr.sum()) + ["val"] * len(tgt_seq))
    return seq, split


def cluster_stats(labels_by_seed: dict[int, np.ndarray], pairs: np.ndarray, pair_timing_dist: np.ndarray) -> dict:
    """Cross-seed agreement, plus whether splits of same-total pairs track real timing differences.

    timing-split AUC: among matched-totals pairs, how well the engineered clock-timing distance separates
    pairs placed in different groups from pairs kept together (0.5 = splits unrelated to timing).
    """
    aris = [adjusted_rand_score(labels_by_seed[a], labels_by_seed[b]) for a, b in combinations(labels_by_seed, 2)]
    split_share, aucs = [], []
    for lab in labels_by_seed.values():
        split = lab[pairs[:, 0]] != lab[pairs[:, 1]]
        split_share.append(float(split.mean()))
        if 0 < split.sum() < len(split):
            aucs.append(float(roc_auc_score(split, pair_timing_dist)))
    return {"cross_seed_ari_mean": float(np.mean(aris)), "cross_seed_ari_sd": float(np.std(aris)),
            "matched_pairs_split_share_mean": float(np.mean(split_share)),
            "matched_pairs_split_share_sd": float(np.std(split_share)),
            "timing_split_auc_mean": float(np.mean(aucs)) if aucs else float("nan"),
            "timing_split_auc_sd": float(np.std(aucs)) if aucs else float("nan"), "n_matched_pairs": int(len(pairs))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["val", "test"], required=True)
    ap.add_argument("--confirm-preregistered", action="store_true")
    ap.add_argument("--runs-dir", type=Path, default=ROOT / "outputs/checkpoint5_final")
    args = ap.parse_args()
    args.runs_dir = args.runs_dir.resolve()
    if args.target == "test" and not args.confirm_preregistered:
        sys.exit("Refusing to open 2023: approve docs/preregistration.md, then pass --confirm-preregistered.")

    z22 = load_year(2022)
    ztg = load_year(2023) if args.target == "test" else None
    cats = z22["categories"].tolist()
    C, w, clock = len(cats), int(z22["bin_minutes"]), z22["clock_features"]
    seq, split = combined(z22, ztg, args.target)
    tr, ev = split == "train", split == "val"
    seq_tr, seq_ev = seq[tr], seq[ev]
    cnt_ev = (seq_ev[..., None] == np.arange(C)).sum(1)
    pairs = matched_totals_pairs(cnt_ev, MATCH_MINUTES, w)
    tfz = TimingFeaturizer(cats, clock).fit(seq_tr)
    clock_cols = np.array([n.split("_")[0] in ("center", "first", "last") for n in tfz.names_])
    sc = StandardScaler().fit(tfz.transform(seq_tr)[:, clock_cols])
    T_ev = sc.transform(tfz.transform(seq_ev)[:, clock_cols])
    pair_timing_dist = np.linalg.norm(T_ev[pairs[:, 0]] - T_ev[pairs[:, 1]], axis=1)
    out_dir = args.runs_dir / f"eval_{args.target}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, cluster_labels, model_stats = [], {}, {}
    for v in VARIANTS:
        for s in SEEDS:
            run = cmp4.latest_run(args.runs_dir, f"{v}_s{s}")
            if run is None:
                print(f"missing {v}_s{s}")
                continue
            model, ck = load_checkpoint(run / "best.pt")
            pred, logp = reconstruct(model, torch.from_numpy(seq_ev).long())
            emb = posterior_means(model, torch.from_numpy(seq).long())
            m = {"val": reconstruction_metrics(seq_ev, pred, C, w, logp), "checkpoint_epoch": ck.get("epoch"),
                 "separation": evaluate_separation(model, seq, split, cats, clock, emb),
                 "latents_val": {
                     "amount": latent_diagnostics(emb["amount_mu"][ev], emb["amount_logvar"][ev])
                     if emb["amount_mu"].shape[1] else None,
                     "timing": latent_diagnostics(emb["timing_mu"][ev], emb["timing_logvar"][ev])},
                 "decoder_reliance_val_accuracy": {"amount_fixed_to_mean": np.nan, "timing_fixed_to_mean": np.nan}}
            r = cmp4.row(v, m, yaml.safe_load((run / "resolved_config.yaml").read_text()))
            # Cluster the timing latent (joint: the single latent) at k=6, fitted on 2022 train embeddings.
            # Unstandardized: the N(0,1) prior already sets latent scale; standardizing would inflate
            # near-inactive dimensions (decided on 2022 validation, before 2023).
            pipe = ClusterPipeline(K, seed=s, scale=False).fit(emb["timing_mu"][tr])
            cluster_labels.setdefault(v, {})[s] = pipe.predict(emb["timing_mu"][ev])
            rows.append({**r, "seed": s, "run": str(run.relative_to(ROOT))})
            print(f"{v}_s{s}: CE {r['val_CE']:.3f} acc {r['val_acc']:.3f}")
    for s in SEEDS:
        set_seed(s)
        cfg = yaml.safe_load((ROOT / "configs/vae_final.yaml").read_text())
        model = ScheduleVAE(VAEConfig(n_categories=C, **cfg["model"]), torch.from_numpy(clock)).eval()
        emb = posterior_means(model, torch.from_numpy(seq).long())
        r = cmp4.row("untrained", {"separation": evaluate_separation(model, seq, split, cats, clock, emb)}, None)
        rows.append({**r, "seed": s})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "per_run.csv", index=False)

    # Engineered baselines at k=6 (features deterministic; seeds vary k-means initialisation only).
    for name, (ut, um) in {"eng_totals": (True, False), "eng_timing_only": (False, True),
                           "eng_totals_plus_timing": (True, True)}.items():
        fs = FeatureSet(name, ut, um, cats, clock).fit(seq_tr)
        X_tr = fs.transform(seq_tr, (seq_tr[..., None] == np.arange(C)).sum(1))
        X_ev = fs.transform(seq_ev, cnt_ev)
        cluster_labels[name] = {s: ClusterPipeline(K, seed=s).fit(X_tr).predict(X_ev) for s in SEEDS}
    clusters = {name: cluster_stats(lbl, pairs, pair_timing_dist) for name, lbl in cluster_labels.items()}
    (out_dir / "clusters.json").write_text(json.dumps(clusters, indent=2))

    # Profiles of the primary model's groups (guided, seed 42) on the target diaries.
    profile_md = ""
    if "guided" in cluster_labels and 42 in cluster_labels["guided"]:
        lab = cluster_labels["guided"][42]
        prof = rb.profile_table(lab, seq_ev, cnt_ev, cats, w, clock)
        meta_year = 2022 if args.target == "val" else 2023
        meta = pd.read_parquet(ROOT / "data/processed" / str(meta_year) / "metadata.parquet").set_index("case_id")
        ids = z22["case_ids"][z22["split"] == "val"] if args.target == "val" else ztg["case_ids"]
        weekend = meta.loc[ids, "TUDIARYDAY"].isin([1, 7]).to_numpy()
        prof["pct_weekend"] = [round(100 * weekend[lab == c].mean(), 1) for c in prof["cluster"]]
        prof["share"] = (prof["share"] * 100).round(1)   # percent of target diaries
        prof.to_csv(out_dir / "profiles_guided_s42.csv", index=False)
        keep = ["cluster", "n", "share", "pct_weekend", "min_sleep", "min_work", "min_leisure_social", "min_household",
                "min_travel", "min_education", "min_other", "center_sleep", "center_work", "center_eating_drinking"]
        profile_md = prof[[c for c in keep if c in prof]].to_markdown(index=False, floatfmt=".1f")
        fig, axes = plt.subplots(K, 1, figsize=(9, 1.5 * K + 1), sharex=True, squeeze=False)
        for c, ax in enumerate(axes[:, 0]):
            stacked_profile(ax, seq_ev[lab == c], cats, int(z22["diary_start_clock_minute"]))
            ax.set_ylabel(f"C{c}\n{(lab == c).mean():.0%}", rotation=0, ha="right", va="center", color=INK)
            ax.set_yticks([0, 1])
        fig.suptitle(f"Guided timing-latent groups (seed 42, k={K}) on {args.target} diaries", x=0.01, ha="left", color=INK)
        fig.legend(handles=palette(cats)[1], loc="lower center", ncol=9, frameon=False, fontsize=7)
        fig.tight_layout(rect=(0, 0.1, 1, 0.97))
        fig.savefig(out_dir / "group_profiles_guided_s42.png", dpi=130)
        plt.close(fig)

    write_report(df, clusters, profile_md, args.target, len(seq_ev))


def write_report(df: pd.DataFrame, clusters: dict, profile_md: str, target: str, n_eval: int) -> None:
    metrics = ["val_CE", "val_acc", "val_macroF1", "transitions_pred", "amount_sens_shift", "amount_sens_swap",
               "knn_totalsR2_from_amount", "knn_clock_gain_amount", "knn_clock_gain_timing",
               "matched_mix_timing_follow_timing", "mix_totals_follow_amount"]
    agg = df.groupby("variant", sort=False)[[c for c in metrics if c in df]].agg(["mean", "std"])

    def g(v, c):
        try:
            return float(agg.loc[v, (c, "mean")])
        except KeyError:
            return float("nan")

    crit = [
        ("A1", "guided kNN totals R² (amount) ≥ untrained + 0.05", g("guided", "knn_totalsR2_from_amount"),
         g("untrained", "knn_totalsR2_from_amount") + 0.05, ">="),
        ("A2", "guided kNN timing gain from amount ≤ 0.05", g("guided", "knn_clock_gain_amount"), 0.05, "<="),
        ("A3", "guided kNN timing gain from timing ≥ untrained + 0.10", g("guided", "knn_clock_gain_timing"),
         g("untrained", "knn_clock_gain_timing") + 0.10, ">="),
        ("A4", "matched mix-and-match ≥ unguided + 0.05", g("guided", "matched_mix_timing_follow_timing"),
         g("unguided", "matched_mix_timing_follow_timing") + 0.05, ">="),
        ("B1a", "amount sensitivity (shift) ≤ 0.7 × no_invariance", g("guided", "amount_sens_shift"),
         0.7 * g("no_invariance", "amount_sens_shift"), "<="),
        ("B1b", "amount sensitivity (swap) ≤ 0.7 × no_invariance", g("guided", "amount_sens_swap"),
         0.7 * g("no_invariance", "amount_sens_swap"), "<="),
        ("B1c", "kNN timing gain ≥ no_invariance − 0.02", g("guided", "knn_clock_gain_timing"),
         g("no_invariance", "knn_clock_gain_timing") - 0.02, ">="),
        ("B1d", "matched mix-and-match ≥ no_invariance − 0.02", g("guided", "matched_mix_timing_follow_timing"),
         g("no_invariance", "matched_mix_timing_follow_timing") - 0.02, ">="),
        ("C1", "guided timing gain ≥ joint + 0.02", g("guided", "knn_clock_gain_timing"),
         g("joint", "knn_clock_gain_timing") + 0.02, ">="),
        ("D1", "guided cross-seed ARI (k=6) ≥ 0.5", clusters.get("guided", {}).get("cross_seed_ari_mean", np.nan), 0.5, ">="),
        ("D2", "guided timing-split AUC on matched-totals pairs ≥ 0.60",
         clusters.get("guided", {}).get("timing_split_auc_mean", np.nan), 0.60, ">="),
    ]
    order = [v for v in [*VARIANTS, "untrained"] if v in agg.index]
    label = "2022 validation (reference, not a test result)" if target == "val" else "2023 TEST (frozen pipeline)"
    lines = [f"# Final evaluation: {label}", "",
             f"Generated by `scripts/evaluate_final.py --target {target}`. {n_eval} diaries. Mean ± SD over seeds 42/43/44.",
             "All fitting (probes, scalers, centroids) on 2022 train-split data only. Unweighted.", "",
             "## Pre-registered criteria", "",
             "| # | Criterion | Value | Threshold | Met |", "| --- | --- | ---: | ---: | :---: |"]
    for cid, text, val, thr, op in crit:
        met = "—" if np.isnan(val) or np.isnan(thr) else ("yes" if (val >= thr if op == ">=" else val <= thr) else "no")
        lines.append(f"| {cid} | {text} | {val:.3f} | {thr:.3f} | {met} |")
    lines += ["", "## Metrics by model", "", "| Metric | " + " | ".join(order) + " |",
              "| --- | " + " | ".join("---:" for _ in order) + " |"]
    for c in metrics:
        if c not in df:
            continue
        cells = []
        for v in order:
            m, sd = agg.loc[v, (c, "mean")], agg.loc[v, (c, "std")]
            cells.append("—" if pd.isna(m) else f"{m:.3f} ± {sd:.3f}")
        lines.append(f"| {c} | " + " | ".join(cells) + " |")
    lines += ["", f"## Groups (k={K})", "",
              "Cross-seed ARI: agreement of cluster assignments for the same diaries across seeds (engineered features:",
              "across k-means seeds only). Split share: matched-totals pairs (≤ 120 min) placed in different groups.", "",
              "Timing-split AUC: do split pairs differ more in engineered clock timing than pairs kept together (0.5 = no).",
              "Engineered timing features score high on that AUC partly by construction (same features).", "",
              "| Representation | Cross-seed ARI | Matched-pair split share | Timing-split AUC | Pairs |",
              "| --- | ---: | ---: | ---: | ---: |",
              *[f"| {n} | {c['cross_seed_ari_mean']:.3f} ± {c['cross_seed_ari_sd']:.3f} | "
                f"{c['matched_pairs_split_share_mean']:.3f} ± {c['matched_pairs_split_share_sd']:.3f} | "
                f"{c['timing_split_auc_mean']:.3f} ± {c['timing_split_auc_sd']:.3f} | {c['n_matched_pairs']} |"
                for n, c in clusters.items()],
              "", "## Guided timing-latent groups (seed 42): descriptions of days, not people", "", profile_md, ""]
    (ROOT / f"docs/final_results_{target}.md").write_text("\n".join(lines))
    print("\n".join(lines[:22]))


if __name__ == "__main__":
    main()
