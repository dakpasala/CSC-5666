"""Evaluate model v2 against the frozen v1 models (docs/v2_validation_plan.md).

Usage:
    python scripts/evaluate_v2.py --target val                               # 2022 validation
    python scripts/evaluate_v2.py --target test --confirm-preregistered      # ATUS 2024, once, after approval

All fitting (probes, scalers, k-means centroids, timing-feature fill values) uses 2022 TRAIN-split data only.
v2 reconstruction uses budgets predicted from its own amount latent (exact-budget decoding; soft CE with
predicted budgets). v1 uses per-slot argmax and softmax CE. Writes outputs/v2/eval_<target>/ and
docs/v2_results_<target>.md. The frozen v1 evaluation (scripts/evaluate_final.py) is left untouched.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.baselines.features import FeatureSet, TimingFeaturizer  # noqa: E402
from src.evaluation.clustering import ClusterPipeline  # noqa: E402
from src.evaluation.plotting import INK, palette, plt, stacked_profile  # noqa: E402
from src.evaluation.reconstruction import latent_diagnostics, reconstruction_metrics  # noqa: E402
from src.evaluation.separation import evaluate_separation, tree_probe_gains  # noqa: E402
from src.evaluation.timing_checks import matched_totals_pairs  # noqa: E402
from src.models.train import build_model, load_checkpoint, posterior_means, reconstruct, set_seed  # noqa: E402

_spec = importlib.util.spec_from_file_location("ef", ROOT / "scripts/evaluate_final.py")
ef = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ef)

SEEDS = [42, 43, 44]
K = 6
MODELS = {  # label -> (run directory, run-name prefix)
    "v2": ("outputs/v2", "v2"),
    "v1_guided": ("outputs/checkpoint5_final", "guided"),
    "v1_no_invariance": ("outputs/checkpoint5_final", "no_invariance"),
    "v1_unguided": ("outputs/checkpoint5_final", "unguided"),
    "v1_joint": ("outputs/checkpoint5_final", "joint"),
}
TEST_YEAR = 2024


def latest(base: Path, name: str) -> Path | None:
    runs = sorted(p for p in base.glob(f"{name}-*") if (p / "eval_best.json").exists())
    return runs[-1] if runs else None


@torch.no_grad()
def reconstruction(model, seq_ev: np.ndarray, C: int, w: int) -> dict:
    t = torch.from_numpy(seq_ev).long()
    if hasattr(model, "decode_exact"):
        q = model.encode(t)
        log_p = model.decode(q["amount_mu"], q["timing_mu"]).numpy()
        pred = model.decode_exact(q["amount_mu"], q["timing_mu"])
    else:
        pred, log_p = reconstruct(model, t)
    m = reconstruction_metrics(seq_ev, pred, C, w, log_p)
    m["pred"] = pred
    return m


def evaluate_model(model, seq, split, cats, clock, w, with_trees: bool) -> tuple[dict, dict]:
    tr, ev = split == "train", split == "val"
    C = len(cats)
    rec = reconstruction(model, seq[ev], C, w)
    emb = posterior_means(model, torch.from_numpy(seq).long())
    sep = evaluate_separation(model, seq, split, cats, clock, emb)
    ps, tim = sep["pair_sensitivity_val"], sep["probes"]["inputs"]["timing"]
    mt = sep["mix_and_match_matched_totals_val"] or {}
    lt = latent_diagnostics(emb["timing_mu"][ev], emb["timing_logvar"][ev])
    r = {
        "CE": rec["cross_entropy"], "acc": rec["accuracy"], "macroF1": rec["macro_f1"],
        "transitions_pred": rec["transitions_pred_mean"], "transitions_true": rec["transitions_true_mean"],
        "F1_eating": rec["per_category_f1"][cats.index("eating_drinking")],
        "F1_travel": rec["per_category_f1"][cats.index("travel")],
        "dur_err_min": rec["duration_abs_error_min_mean"],
        "amount_sens_shift": (ps["shift"]["amount"] or {}).get("median", np.nan),
        "amount_sens_swap": (ps["swap"]["amount"] or {}).get("median", np.nan),
        "timing_sens_shift": ps["shift"]["timing"]["median"], "timing_sens_swap": ps["swap"]["timing"]["median"],
        "timing_sens_splitmerge": ps["split_merge"]["timing"]["median"],
        "leak_totalsR2_timing_knn": tim["knn_totals_r2"],
        "matched_mix": mt.get("timing_closer_to_timing_source", np.nan),
        "timing_active": lt["active_dims"],
    }
    if with_trees:
        tp = tree_probe_gains(emb["timing_mu"], seq, split, cats, clock)
        r["tree_clock_gain"] = tp["clock"]["mean_gain"]
        r["tree_frag_gain"] = tp["fragmentation"]["mean_gain"]
    return r, {"timing_mu": emb["timing_mu"], "pred": rec["pred"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["val", "test"], required=True)
    ap.add_argument("--confirm-preregistered", action="store_true")
    ap.add_argument("--no-trees", action="store_true", help="skip gradient-boosted probes (faster)")
    args = ap.parse_args()
    if args.target == "test" and not args.confirm_preregistered:
        sys.exit(f"Refusing to open {TEST_YEAR}: approve docs/preregistration_v2.md, then pass --confirm-preregistered.")

    z22 = np.load(ROOT / "data/processed/2022/sequences.npz")
    cats = z22["categories"].tolist()
    C, w, clock = len(cats), int(z22["bin_minutes"]), z22["clock_features"]
    trm = z22["split"] == "train"
    if args.target == "val":
        tgt_seq, tgt_ids, meta_year = z22["seq"][z22["split"] == "val"], z22["case_ids"][z22["split"] == "val"], 2022
    else:
        zt = np.load(ROOT / f"data/processed/{TEST_YEAR}/sequences.npz")
        assert zt["categories"].tolist() == cats
        tgt_seq, tgt_ids, meta_year = zt["seq"], zt["case_ids"], TEST_YEAR
    seq = np.concatenate([z22["seq"][trm], tgt_seq])
    split = np.array(["train"] * int(trm.sum()) + ["val"] * len(tgt_seq))
    tr, ev = split == "train", split == "val"
    seq_tr, seq_ev = seq[tr], seq[ev]
    cnt_ev = (seq_ev[..., None] == np.arange(C)).sum(1)
    pairs = matched_totals_pairs(cnt_ev, ef.MATCH_MINUTES, w)
    tfz = TimingFeaturizer(cats, clock).fit(seq_tr)
    clock_cols = np.array([n.split("_")[0] in ("center", "first", "last") for n in tfz.names_])
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(tfz.transform(seq_tr)[:, clock_cols])
    T_ev = sc.transform(tfz.transform(seq_ev)[:, clock_cols])
    pair_dist = np.linalg.norm(T_ev[pairs[:, 0]] - T_ev[pairs[:, 1]], axis=1)
    out_dir = ROOT / "outputs/v2" / f"eval_{args.target}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, labels, preds = [], {}, {}
    for label, (base, prefix) in MODELS.items():
        for s in SEEDS:
            run = latest(ROOT / base, f"{prefix}_s{s}")
            if run is None:
                print(f"missing {label} seed {s}")
                continue
            model, ck = load_checkpoint(run / "best.pt")
            r, extra = evaluate_model(model, seq, split, cats, clock, w, not args.no_trees)
            pipe = ClusterPipeline(K, seed=s, scale=False).fit(posterior_means(model, torch.from_numpy(seq_tr).long())["timing_mu"])
            labels.setdefault(label, {})[s] = pipe.predict(extra["timing_mu"][ev])
            preds[(label, s)] = extra["pred"]
            rows.append({"model": label, "seed": s, "epoch": ck.get("epoch"), "run": str(run.relative_to(ROOT)), **r})
            print(f"{label} s{s}: CE {r['CE']:.3f} acc {r['acc']:.3f} transitions {r['transitions_pred']:.1f}", flush=True)
    for kind in ("v1", "v2"):
        for s in SEEDS:
            set_seed(s)
            mcfg = {"n_categories": C, **({"model_type": "v2", "z_amount": 8, "z_timing": 16} if kind == "v2" else {})}
            model = build_model(mcfg, torch.from_numpy(clock)).eval()
            r, extra = evaluate_model(model, seq, split, cats, clock, w, not args.no_trees)
            rows.append({"model": f"untrained_{kind}", "seed": s, **{k: v for k, v in r.items()
                         if k not in ("CE", "acc", "macroF1", "transitions_pred", "F1_eating", "F1_travel", "dur_err_min")}})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "per_run.csv", index=False)

    for name, (ut, um) in {"eng_timing_only": (False, True), "eng_totals_plus_timing": (True, True)}.items():
        fs = FeatureSet(name, ut, um, cats, clock).fit(seq_tr)
        X_tr = fs.transform(seq_tr, (seq_tr[..., None] == np.arange(C)).sum(1))
        X_ev = fs.transform(seq_ev, cnt_ev)
        labels[name] = {s: ClusterPipeline(K, seed=s).fit(X_tr).predict(X_ev) for s in SEEDS}
    clusters = {name: ef.cluster_stats(lbl, pairs, pair_dist) for name, lbl in labels.items()}
    (out_dir / "clusters.json").write_text(json.dumps(clusters, indent=2))

    # v2 group profiles (seed 42) on the target diaries.
    profile_md = ""
    if "v2" in labels and 42 in labels["v2"]:
        lab = labels["v2"][42]
        prof = ef.rb.profile_table(lab, seq_ev, cnt_ev, cats, w, clock)
        meta = pd.read_parquet(ROOT / f"data/processed/{meta_year}/metadata.parquet").set_index("case_id")
        weekend = meta.loc[tgt_ids, "TUDIARYDAY"].isin([1, 7]).to_numpy()
        prof["share"] = (prof["share"] * 100).round(1)
        prof["pct_weekend"] = [round(100 * weekend[lab == c].mean(), 1) for c in prof["cluster"]]
        prof.to_csv(out_dir / "profiles_v2_s42.csv", index=False)
        keep = ["cluster", "n", "share", "pct_weekend", "min_sleep", "min_work", "min_leisure_social", "min_household",
                "min_travel", "center_sleep", "center_work", "center_eating_drinking"]
        profile_md = prof[[c for c in keep if c in prof]].to_markdown(index=False, floatfmt=".1f")
        fig, axes = plt.subplots(K, 1, figsize=(9, 1.5 * K + 1), sharex=True, squeeze=False)
        for c, ax in enumerate(axes[:, 0]):
            stacked_profile(ax, seq_ev[lab == c], cats, int(z22["diary_start_clock_minute"]))
            ax.set_ylabel(f"C{c}\n{(lab == c).mean():.0%}", rotation=0, ha="right", va="center", color=INK)
            ax.set_yticks([0, 1])
        fig.suptitle(f"v2 timing-latent groups (seed 42, k={K}) on {args.target} diaries", x=0.01, ha="left", color=INK)
        fig.legend(handles=palette(cats)[1], loc="lower center", ncol=9, frameon=False, fontsize=7)
        fig.tight_layout(rect=(0, 0.1, 1, 0.97))
        fig.savefig(out_dir / "group_profiles_v2_s42.png", dpi=130)
        plt.close(fig)

    write_report(df, clusters, profile_md, args.target, len(seq_ev), out_dir)


def write_report(df: pd.DataFrame, clusters: dict, profile_md: str, target: str, n: int, out_dir: Path) -> None:
    metrics = ["CE", "acc", "macroF1", "transitions_pred", "F1_eating", "F1_travel", "dur_err_min",
               "amount_sens_shift", "amount_sens_swap", "timing_sens_shift", "timing_sens_swap",
               "timing_sens_splitmerge", "leak_totalsR2_timing_knn", "tree_clock_gain", "tree_frag_gain",
               "matched_mix", "timing_active"]
    metrics = [m for m in metrics if m in df]
    agg = df.groupby("model", sort=False)[metrics].agg(["mean", "std"])
    order = list(agg.index)

    def cell(v, c):
        m, sd = agg.loc[v, (c, "mean")], agg.loc[v, (c, "std")]
        return "—" if pd.isna(m) else f"{m:.3f} ± {sd:.3f}"

    title = "2022 validation (development reference)" if target == "val" else f"ATUS {TEST_YEAR} TEST (frozen v2 pipeline)"
    lines = [f"# Model v2 evaluation: {title}", "",
             f"Generated by `scripts/evaluate_v2.py --target {target}`. {n} diaries; mean ± SD over seeds 42/43/44.",
             "All fitting on 2022 train-split data only. Unweighted. v2 reconstruction uses budgets predicted from its",
             "own amount latent (exact-budget decoding); v1 uses per-slot argmax.", "",
             "| Metric | " + " | ".join(order) + " |", "| --- | " + " | ".join("---:" for _ in order) + " |",
             *[f"| {c} | " + " | ".join(cell(v, c) for v in order) + " |" for c in metrics],
             f"", f"True transitions per day: {df['transitions_true'].dropna().iloc[0]:.1f}.", "",
             f"## Groups (k={K}, timing latent unstandardized; engineered features standardized)", "",
             "| Representation | Cross-seed ARI | Matched-pair split share | Timing-split AUC | Pairs |",
             "| --- | ---: | ---: | ---: | ---: |",
             *[f"| {k} | {c['cross_seed_ari_mean']:.3f} ± {c['cross_seed_ari_sd']:.3f} | "
               f"{c['matched_pairs_split_share_mean']:.3f} | {c['timing_split_auc_mean']:.3f} ± {c['timing_split_auc_sd']:.3f} | "
               f"{c['n_matched_pairs']} |" for k, c in clusters.items()],
             "", "## v2 timing-latent groups (seed 42): descriptions of days, not people", "", profile_md, ""]
    (ROOT / f"docs/v2_results_{target}.md").write_text("\n".join(lines))
    print("\n".join(lines[:9 + len(metrics) + 12]))


if __name__ == "__main__":
    main()
