"""Re-score a trained run: rebuild the model from config.json, load best_model.pt,
and evaluate on the run's held-out fold and/or an external split.

Usage:
    python -m src.eval \
        --run_dir outputs/uni2h/nsclc/roam/fold0 \
        --feature_root /data/UNI2-h-features \
        --external_manifest splits/uni2h_cptac_nsclc.csv

Results go to <out_dir>/eval_{val,external}_metrics.json and
eval_{val,external}_preds.csv (out_dir defaults to run_dir); the training
outputs (summary.json, val_preds.csv, ...) are not touched.

By default each bag is capped at the run's --max_patches (first N patches in file
order), i.e. the same protocol as the scores written during training. Pass
--max_patches 0 to score full bags.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from src.train import (
    build_criterion, build_model, build_parser, evaluate, make_loader,
    save_preds, scalar_metrics,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", type=str, required=True,
                   help="Training output directory with config.json and best_model.pt")
    p.add_argument("--manifest", type=str, default=None,
                   help="Split whose held-out fold is scored (default: from config.json)")
    p.add_argument("--external_manifest", type=str, default=None,
                   help="External split to score (default: from config.json)")
    p.add_argument("--no_val", action="store_true", help="Skip the held-out fold")
    p.add_argument("--no_external", action="store_true", help="Skip the external split")
    p.add_argument("--feature_root", type=str, default=None,
                   help="Override feature_root from config.json")
    p.add_argument("--max_patches", type=int, default=None,
                   help="Override the patch cap (default: config.json value; 0 = full bags)")
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--num_workers", type=int, default=4)
    return p.parse_args()


def load_run_args(run_dir):
    """Parser defaults overlaid with the run's config.json (unknown keys ignored)."""
    defaults = build_parser().parse_args(["--manifest", "-", "--output_dir", "-"])
    with open(Path(run_dir) / "config.json") as f:
        cfg = json.load(f)
    for k, v in cfg.items():
        if hasattr(defaults, k):
            setattr(defaults, k, v)
    return defaults


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    run = load_run_args(run_dir)
    run.num_workers = args.num_workers
    if args.feature_root is not None:
        run.feature_root = args.feature_root
    if args.max_patches is not None:
        run.max_patches = args.max_patches

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(run)
    model.load_state_dict(torch.load(run_dir / "best_model.pt", weights_only=True,
                                     map_location=device))
    model = model.to(device).eval()
    criterion = build_criterion(run.task)
    print(f"Fold: {run.fold}, task: {run.task}, "
          f"max_patches: {run.max_patches or 'full bag'}")

    targets = []
    if not args.no_val:
        df = pd.read_csv(args.manifest or run.manifest)
        targets.append(("val", df[df["fold"] == run.fold].reset_index(drop=True)))
    ext = args.external_manifest or run.external_manifest
    if ext and not args.no_external:
        targets.append(("external", pd.read_csv(ext)))

    for name, df in targets:
        loader = make_loader(df, run, training=False)
        metrics, labels, preds, slide_ids = evaluate(
            model, loader, criterion, device,
            task=run.task, n_classes=run.n_classes,
        )
        metrics = scalar_metrics(metrics)
        print(f"{name} ({len(df)} slides): {metrics}")
        save_preds(labels, preds, slide_ids, run.task, run.n_classes,
                   out_dir / f"eval_{name}_preds.csv")
        with open(out_dir / f"eval_{name}_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
