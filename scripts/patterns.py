"""Rare-pattern discovery (ICA), pattern dashboard (probes), combined typology (docs/patterns_plan.md).

Usage:
    python scripts/patterns.py

2022 data only (development). Nine frozen 80-epoch v1 models; no training. Pattern labels are used for
evaluation only, never to find the ICA directions. Writes outputs/patterns/ and docs/patterns.md.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")   # avoid the macOS libomp clash between torch and sklearn

import glob  # noqa: E402
import importlib.util  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from scipy.stats import kurtosis, skew, spearmanr  # noqa: E402
from sklearn.decomposition import FastICA  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.rasterize import category_counts  # noqa: E402
from src.models.train import load_checkpoint, posterior_means  # noqa: E402


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ca = _load("ca", "scripts/continuum_axes.py")
TAIL = 0.03
TOP = 6
N_NULL = 5
START, W = 240, 5


def runs(row: np.ndarray, c: int) -> list[tuple[int, int]]:
    m = np.concatenate([[False], row == c, [False]])
    d = np.diff(m.astype(int))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def pattern_labels(seq: np.ndarray, cats: list[str]) -> pd.DataFrame:
    rm = ca.rhythm_measures(seq, cats)
    sl, wk, le = cats.index("sleep"), cats.index("work"), cats.index("leisure_social")
    T = seq.shape[1]
    hour = ((START + W * np.arange(T)) % 1440) / 60
    wc = rm["work_centre_h"].to_numpy()
    nap, split = np.zeros(len(seq), bool), np.zeros(len(seq), bool)
    for i, row in enumerate(seq):
        for s, e in runs(row, sl):
            if s > 0 and e < T and e - s >= 4 and 11 <= hour[s] < 19:
                nap[i] = True
                break
        bouts = []
        for s, e in runs(row, wk):
            if bouts and s - bouts[-1][1] < 12:          # merge gaps < 60 min
                bouts[-1] = (bouts[-1][0], e)
            else:
                bouts.append((s, e))
        split[i] = any(b[0] - a[1] >= 24 for a, b in zip(bouts, bouts[1:]))
    late_night = ((seq == le) & (hour < 4)[None, :]).sum(1) >= 6
    return pd.DataFrame({
        "night_work": (rm["work_min"] >= 60) & ((wc >= 21) | (wc < 6)),
        "early_riser": rm["wake_h"] < 5.5,
        "late_riser": rm["wake_h"] > 10,
        "daytime_nap": nap,
        "split_work_day": split,
        "late_night_leisure": late_night,
        "fragmented_day": rm["transitions"] >= 25,
    })


def fit_ica(e_tr: np.ndarray, seed: int = 0) -> FastICA:
    return FastICA(n_components=e_tr.shape[1], whiten="unit-variance", random_state=seed, max_iter=2000, tol=1e-4).fit(e_tr)


def oriented_scores(ica: FastICA, e: np.ndarray, tr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    S = ica.transform(e)
    S = S * np.sign(skew(S[tr], axis=0))[None, :]        # heavy tail -> positive
    return S, kurtosis(S[tr], axis=0)                     # excess kurtosis


def main() -> None:
    out = ROOT / "outputs/patterns"
    out.mkdir(parents=True, exist_ok=True)
    z = np.load(ROOT / "data/processed/2022/sequences.npz")
    cats = z["categories"].tolist()
    seq, ids = z["seq"], z["case_ids"]
    tr, va = z["split"] == "train", z["split"] == "val"
    L = pattern_labels(seq, cats)
    comp = pd.read_csv(ROOT / "outputs/two_level/composition_types.csv", dtype={"case_id": str})["composition_type"].to_numpy()
    comp_names = {0: "religious/volunteer", 1: "home & chores", 2: "work", 3: "leisure", 4: "school"}
    cnt = category_counts(seq, len(cats))
    props = cnt / cnt.sum(1, keepdims=True)
    prevalence = L[va].mean()
    print("pattern prevalence (val):", prevalence.round(3).to_dict(), flush=True)

    emb = {s: posterior_means(load_checkpoint(Path(sorted(glob.glob(str(ROOT / p)))[-1]) / "best.pt")[0],
                              torch.from_numpy(seq).long())["timing_mu"] for s, p in ca.MODELS.items()}

    # ---------------- Part 1: ICA discovery ----------------
    scores, kurt = {}, {}
    for s, e in emb.items():
        ica = fit_ica(e[tr])
        scores[s], kurt[s] = oriented_scores(ica, e, tr)
    rng = np.random.default_rng(0)
    e42 = emb[42]
    null_max = []
    for _ in range(N_NULL):
        g = rng.multivariate_normal(e42[tr].mean(0), np.cov(e42[tr].T), size=int(tr.sum()))
        null_max.append(np.abs(kurtosis(fit_ica(g).transform(g), axis=0)).max())
    null_thr = float(np.max(null_max))
    print(f"null max |excess kurtosis| over {N_NULL} Gaussian draws: {null_thr:.3f}", flush=True)

    S42, k42 = scores[42], kurt[42]
    top = np.argsort(-np.abs(k42))[:TOP]
    comp_rows = []
    for rank, j in enumerate(top, 1):
        thr = np.quantile(S42[tr, j], 1 - TAIL)
        tail = va & (S42[:, j] >= thr)
        aucs = {p: roc_auc_score(L[p][va], S42[va, j]) for p in L.columns if L[p][va].nunique() > 1}
        best_p = max(aucs, key=lambda p: aucs[p])
        enrich = {p: (L[p][tail].mean() / max(L[p][va].mean(), 1e-9)) for p in L.columns}
        # reproducibility across the other 8 models
        rhos, jacs = [], []
        tail_set = set(np.flatnonzero(tail))
        for s in ca.MODELS:
            if s == 42:
                continue
            r = np.array([spearmanr(S42[va, j], scores[s][va, c])[0] for c in range(scores[s].shape[1])])
            c = int(np.argmax(np.abs(r)))
            rhos.append(abs(r[c]))
            sc = scores[s][:, c] * np.sign(r[c])
            t2 = va & (sc >= np.quantile(sc[tr], 1 - TAIL))
            other = set(np.flatnonzero(t2))
            jacs.append(len(tail_set & other) / max(len(tail_set | other), 1))
        mix = pd.Series(comp[tail]).map(comp_names).value_counts(normalize=True).round(2).to_dict()
        comp_rows.append({"rank": rank, "component": int(j), "excess_kurtosis": float(k42[j]),
                          "best_pattern": best_p, "best_AUC": float(aucs[best_p]),
                          "enrichment_best": float(enrich[best_p]),
                          "tail_days_val": int(tail.sum()),
                          "models_matching_rho_0.7": f"{int(np.sum(np.array(rhos) >= 0.7))}/8",
                          "match_rho_mean": float(np.mean(rhos)), "tail_jaccard_mean": float(np.mean(jacs)),
                          "tail_composition_mix": mix,
                          **{f"AUC_{p}": float(v) for p, v in aucs.items()}})
        print(comp_rows[-1], flush=True)
    comps = pd.DataFrame(comp_rows)

    # Does every model have an unsupervised direction that finds night work, and how highly is it ranked?
    nw_rows = []
    for s in ca.MODELS:
        S, k = scores[s], kurt[s]
        a = np.array([roc_auc_score(L["night_work"][va], S[va, c]) for c in range(S.shape[1])])
        best = int(np.argmax(a))
        rank_by_kurt = int(np.where(np.argsort(-np.abs(k)) == best)[0][0]) + 1
        nw_rows.append({"seed": s, "best_component_AUC": float(a[best]), "its_kurtosis_rank_of_16": rank_by_kurt,
                        "its_excess_kurtosis": float(k[best]), "above_null": bool(abs(k[best]) > null_thr)})
    nw = pd.DataFrame(nw_rows)

    # ---------------- Part 2: dashboard ----------------
    dash = []
    for p in L.columns:
        y = L[p].to_numpy()
        if y[tr].sum() < 10 or y[va].sum() < 5:
            continue

        def auc(X):
            m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(X[tr], y[tr])
            return roc_auc_score(y[va], m.predict_proba(X[va])[:, 1])
        a_tot = auc(props)
        a_tim = [auc(emb[s]) for s in ca.MODELS]
        a_both = [auc(np.hstack([props, emb[s]])) for s in ca.MODELS]
        dash.append({"pattern": p, "prevalence_val": float(y[va].mean()), "AUC_totals_only": a_tot,
                     "AUC_timing_latent": f"{np.mean(a_tim):.3f} ± {np.std(a_tim):.3f}",
                     "AUC_totals_plus_timing": f"{np.mean(a_both):.3f} ± {np.std(a_both):.3f}"})
        print(dash[-1], flush=True)
    dash = pd.DataFrame(dash)

    # ---------------- Part 3: combined typology ----------------
    axes = pd.read_csv(ROOT / "outputs/continuum/axes_val_2022.csv", dtype={"case_id": str}).set_index("case_id")
    vid = ids[va]
    band = axes.loc[vid, "band1"].map({0: "earlier-shifted", 1: "typical", 2: "later-shifted"}).to_numpy()
    cv = pd.Series(comp[va]).map(comp_names).to_numpy()
    cells = pd.crosstab(cv, band, normalize=True).mul(100).round(1)
    cells = cells[["earlier-shifted", "typical", "later-shifted"]]
    prev_by_type = L[va].groupby(cv).mean().mul(100).round(1)
    prev_by_type.insert(0, "share_of_days_%", pd.Series(cv).value_counts(normalize=True).mul(100).round(1))
    examples = pd.DataFrame({"composition": cv, "rhythm_band": band,
                             "patterns": [", ".join(p for p in L.columns if L[p].to_numpy()[va][i]) or "—" for i in range(va.sum())]})
    combo = (examples["composition"] + " · " + examples["rhythm_band"] + " · " + examples["patterns"]).value_counts().head(12)
    examples.assign(case_id=vid).to_csv(out / "typology_val_2022.csv", index=False)

    md = lambda d, i=False: d.to_markdown(index=i, floatfmt=".3f")  # noqa: E731
    lines = ["# Rare-pattern discovery, pattern dashboard and combined typology (2022 validation; development)", "",
             "Generated by `scripts/patterns.py`; plan in `docs/patterns_plan.md`. Pattern labels are hand-defined and",
             "used only for evaluation (part 1) or as supervised probe targets (part 2).", "",
             "Pattern prevalence on validation days: " + ", ".join(f"{p} {v:.1%}" for p, v in prevalence.items()) + ".", "",
             "## Part 1: unsupervised discovery with ICA", "",
             f"Null (matched Gaussian): max |excess kurtosis| over {N_NULL} draws = {null_thr:.3f}. Seed 42's six most",
             "non-Gaussian components; AUC/enrichment computed afterwards against the pattern labels; reproducibility",
             "= best-matching component in each of the other 8 models.", "",
             md(comps[["rank", "component", "excess_kurtosis", "best_pattern", "best_AUC", "enrichment_best", "tail_days_val",
                       "models_matching_rho_0.7", "match_rho_mean", "tail_jaccard_mean", "tail_composition_mix"]]), "",
             "AUC of each top component for every pattern:", "",
             md(comps[["rank"] + [c for c in comps.columns if c.startswith("AUC_")]]), "",
             "### Is night work found without labels in every model?", "",
             "For each model: the single ICA component that best separates night-work days, its AUC, and where it ranks",
             "among that model's 16 components by non-Gaussianity (the unsupervised criterion).", "",
             md(nw), "",
             "## Part 2: pattern dashboard (supervised probes)", "",
             "Logistic-regression probes fitted on train, AUC on validation; timing-latent columns are mean ± SD over 9 models.", "",
             md(dash), "",
             "## Part 3: combined typology (validation days)", "",
             "Share of days (%) by composition type × axis-1 rhythm band (9-model consensus). Rhythm bands agree across",
             "disjoint model sets for 78% of days (`docs/continuum_axes.md`); composition types are stable (ARI 0.93).", "",
             md(cells, True), "", "Pattern prevalence (%) within each composition type:", "", md(prev_by_type, True), "",
             "Most common combined descriptions:", "", md(combo.rename("days").to_frame(), True), ""]
    (ROOT / "docs/patterns.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
