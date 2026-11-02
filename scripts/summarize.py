"""Summarise cross-validation runs: mean ± std (population std, ddof=0) over folds.

    python -m scripts.summarize --out outputs

Reads <out>/<encoder>/<task>/<run>/fold*/summary.json (internal held-out fold) and
external_metrics.json (CPTAC, when present). Binary tasks report AUC, PANDA QWK.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def mean_std(values):
    return f"{np.mean(values):.3f} ± {np.std(values):.3f}" if values else "-"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("outputs"))
    a = p.parse_args()

    rows = []
    for run in sorted(d for d in a.out.glob("*/*/*") if d.is_dir()):
        folds = sorted(run.glob("fold*/summary.json"))
        if not folds:
            continue
        internal, external = [], []
        for s in folds:
            summary = json.load(open(s))
            metric = "qwk" if summary["task"] == "multiclass" else "auc"
            internal.append(summary["best_val_metrics"][metric])
            ext = s.parent / "external_metrics.json"
            if ext.exists():
                external.append(json.load(open(ext))["auc"])
        enc, task, name = run.relative_to(a.out).parts
        rows.append((enc, task, name, metric, len(folds), mean_std(internal), mean_std(external)))

    header = ("encoder", "task", "run", "metric", "folds", "internal", "external (CPTAC)")
    widths = [max(len(str(r[i])) for r in rows + [header]) for i in range(len(header))]
    for r in [header] + rows:
        print("  ".join(str(c).ljust(w) for c, w in zip(r, widths)))


if __name__ == "__main__":
    main()
