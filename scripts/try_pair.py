"""Try a trained model on one diary and a rearranged copy of it (a hands-on demo, not an evaluation).

Picks one real diary, makes a total-preserving copy (circular shift or block swap), runs both through a trained
v1 checkpoint and prints:
- that the totals are identical;
- how far the amount and timing vectors move, relative to the typical distance between two random diaries
  (the same scale as the pair-sensitivity metric);
- the true vs rebuilt timelines as text (one character per 30 minutes);
- optionally, a swap test with a second diary: A's amount vector + B's timing vector.

Examples:
    python scripts/try_pair.py                                  # random 2022 validation diary, 1-hour shift
    python scripts/try_pair.py --transform swap --seed 7
    python scripts/try_pair.py --case-id 20220101220000 --shift 24
    python scripts/try_pair.py --mix-with random                # also run the swap test with a second diary
    python scripts/try_pair.py --plot outputs/try_pair.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.transforms import assert_counts_preserved, block_swap, circular_shift  # noqa: E402
from src.models.train import load_checkpoint  # noqa: E402

DEFAULT_CKPT = "outputs/checkpoint5_final/guided_s42-20261019-212407/best.pt"
LETTERS = {"sleep": "S", "personal_care": "p", "eating_drinking": "E", "household": "h", "caregiving": "c",
           "work": "W", "education": "e", "shopping_services": "$", "leisure_social": "L",
           "exercise_sports": "X", "travel": "t", "other": "o", "unknown": "?"}


def timeline(seq: np.ndarray, cats: list[str], bins_per_char: int = 6) -> str:
    """One character per 30 minutes: the most common activity in that half hour."""
    chars = []
    for i in range(0, len(seq), bins_per_char):
        chunk = seq[i:i + bins_per_char]
        chars.append(LETTERS[cats[np.bincount(chunk).argmax()]])
    return "".join(chars)


def clock_ruler(start_minute: int, n_chars: int, minutes_per_char: int = 30) -> str:
    out = [" "] * n_chars
    for c in range(0, n_chars, 8):                     # a label every 4 hours
        label = f"{((start_minute + c * minutes_per_char) // 60) % 24:02d}"
        out[c:c + 2] = label
    return "".join(out[:n_chars])


def transitions(seq: np.ndarray) -> int:
    return int((np.diff(seq) != 0).sum())


@torch.no_grad()
def run(model, seq: np.ndarray) -> dict:
    x = torch.as_tensor(seq[None], dtype=torch.long)
    out = model(x, sample=False)                        # posterior means, no random noise
    return {"amount": out["amount_mu"][0].numpy(), "timing": out["timing_mu"][0].numpy(),
            "rebuilt": out["logits"][0].argmax(-1).numpy(),
            "shares_pred": torch.softmax(out["amount_logits"][0], -1).numpy()}


def typical_distance(model, seqs: np.ndarray, rng: np.random.Generator, n: int = 500) -> dict:
    """Median distance between random diaries, per vector, so movements are scale-free."""
    idx = rng.choice(len(seqs), size=min(n, len(seqs)), replace=False)
    with torch.no_grad():
        q = model.encode(torch.as_tensor(seqs[idx], dtype=torch.long))
    out = {}
    for key, name in (("amount_mu", "amount"), ("timing_mu", "timing")):
        z = q[key].numpy()
        i, j = rng.integers(0, len(z), 2000), rng.integers(0, len(z), 2000)
        keep = i != j
        out[name] = float(np.median(np.linalg.norm(z[i[keep]] - z[j[keep]], axis=1)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--year", type=int, default=2022)
    ap.add_argument("--split", default="val", help="val/train for 2022; 2023+ diaries are all 'test'")
    ap.add_argument("--case-id", default=None, help="diary to use (default: random from the split)")
    ap.add_argument("--transform", choices=["shift", "swap"], default="shift")
    ap.add_argument("--shift", type=int, default=12, help="shift in 5-minute bins (12 = 1 hour; negative = earlier)")
    ap.add_argument("--mix-with", default=None, help="case ID or 'random': swap test with a second diary")
    ap.add_argument("--plot", default=None, help="optional PNG path for true/rebuilt timelines")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    d = np.load(f"data/processed/{args.year}/sequences.npz", allow_pickle=True)
    cats = [str(c) for c in d["categories"]]
    C, start = len(cats), int(d["diary_start_clock_minute"])
    ids, split, seqs = d["case_ids"].astype(str), d["split"].astype(str), d["seq"].astype(np.int64)
    pool = np.flatnonzero(split == args.split)
    if len(pool) == 0:
        sys.exit(f"no diaries with split '{args.split}' in {args.year} (available: {sorted(set(split))})")

    model, _ = load_checkpoint(Path(args.checkpoint))

    def pick(case_id: str | None) -> int:
        if case_id in (None, "random"):
            return int(rng.choice(pool))
        hit = np.flatnonzero(ids == case_id)
        if len(hit) == 0:
            sys.exit(f"case ID {case_id} not found in {args.year}")
        return int(hit[0])

    a = pick(args.case_id)
    seq = seqs[a]
    if args.transform == "shift":
        copy, params = circular_shift(seq, args.shift, C)
    else:
        copy, params = block_swap(seq, rng, C)
    if copy is None:
        sys.exit(f"transform rejected ({params.get('rejected')}); try another --seed or --shift")
    assert_counts_preserved(seq, copy, C)

    scale = typical_distance(model, seqs[pool], rng)
    r0, r1 = run(model, seq), run(model, copy)

    print(f"Diary {ids[a]} ({args.year} {args.split}), checkpoint {args.checkpoint}")
    print(f"Transform: {params}")
    print("Totals identical:", np.array_equal(np.bincount(seq, minlength=C), np.bincount(copy, minlength=C)))
    print()
    ruler = clock_ruler(start, len(seq) // 6)
    print("            " + ruler)
    print("original    " + timeline(seq, cats))
    print("  rebuilt   " + timeline(r0["rebuilt"], cats))
    print("copy        " + timeline(copy, cats))
    print("  rebuilt   " + timeline(r1["rebuilt"], cats))
    print("Key: " + "  ".join(f"{v}={k}" for k, v in LETTERS.items()))
    print()
    for name, r, s in (("original", r0, seq), ("copy", r1, copy)):
        acc = (r["rebuilt"] == s).mean()
        print(f"{name:9s} rebuilt: {acc:.0%} of slots correct, {transitions(r['rebuilt'])} activity changes "
              f"(real: {transitions(s)})")
    print()
    print("How far each vector moved (1.0 = typical gap between two random diaries):")
    for name in ("amount", "timing"):
        moved = np.linalg.norm(r0[name] - r1[name]) / scale[name]
        print(f"  {name:6s} vector: {moved:.3f}")
    print("  Expect: amount small (the pair rule), timing clearly larger.")
    print()
    true_share = np.bincount(seq, minlength=C) / len(seq)
    print("Amount head, minutes per day (true vs predicted from the amount vector):")
    for c in np.argsort(-true_share)[:6]:
        print(f"  {cats[c]:18s} {true_share[c] * 1440:6.0f}  vs {r0['shares_pred'][c] * 1440:6.0f}")

    rows = [("original", seq), ("rebuilt", r0["rebuilt"]), ("copy", copy), ("copy rebuilt", r1["rebuilt"])]

    if args.mix_with:
        b = pick(args.mix_with)
        rb = run(model, seqs[b])
        with torch.no_grad():
            logits = model.decode(torch.as_tensor(r0["amount"][None]), torch.as_tensor(rb["timing"][None]))
        mixed = logits[0].argmax(-1).numpy()
        print()
        print(f"Swap test: amount vector from {ids[a]} (A) + timing vector from {ids[b]} (B)")
        print("            " + ruler)
        print("A           " + timeline(seq, cats))
        print("B           " + timeline(seqs[b], cats))
        print("A amt+B tim " + timeline(mixed, cats))
        print(f"  slots matching A: {(mixed == seq).mean():.0%}, matching B: {(mixed == seqs[b]).mean():.0%}")
        print("  (works best when A and B have similar totals; very different days give mixed results)")
        rows += [("B", seqs[b]), ("A amount + B timing", mixed)]

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        cmap = ListedColormap(plt.get_cmap("tab20").colors[:C])
        fig, ax = plt.subplots(figsize=(11, 0.6 * len(rows) + 1.2))
        ax.imshow(np.stack([r for _, r in rows]), aspect="auto", cmap=cmap, vmin=-0.5, vmax=C - 0.5,
                  interpolation="nearest")
        ax.set_yticks(range(len(rows)), [n for n, _ in rows])
        ticks = np.arange(0, len(seq) + 1, 48)
        ax.set_xticks(ticks - 0.5, [f"{((start + t * 5) // 60) % 24:02d}:00" for t in ticks])
        handles = [plt.Rectangle((0, 0), 1, 1, color=cmap(i)) for i in range(C)]
        ax.legend(handles, cats, ncol=7, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.25))
        fig.tight_layout()
        Path(args.plot).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.plot, dpi=150)
        print(f"\nSaved plot to {args.plot}")


if __name__ == "__main__":
    main()
