"""Re-evaluate a saved VAE run from its checkpoint alone (no retraining, no refitted preprocessing).

Usage:
    python scripts/evaluate_vae.py --run outputs/checkpoint3/<run-dir> [--checkpoint best.pt|last.pt]
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.vae_report import evaluate_run  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--checkpoint", default="best.pt")
    args = ap.parse_args()
    m = evaluate_run(args.run.resolve(), args.checkpoint)
    print(json.dumps({k: m[k] for k in ("checkpoint_epoch", "decoder_reliance_val_accuracy")}, indent=2))
    print({k: round(v, 4) for k, v in m["val"].items() if isinstance(v, float)})


if __name__ == "__main__":
    main()
