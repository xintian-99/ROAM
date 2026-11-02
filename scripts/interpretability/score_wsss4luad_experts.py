# ROAM: Region-Graph Optimal Transport Routing for Mixture-of-Experts WSI Classification
# Xin Tian et al., IEEE BIBM 2026 | https://arxiv.org/abs/2604.07298 | https://github.com/xintian-99/ROAM
"""Per-expert tissue AUROC of ROAM's routing on WSSS4LUAD (paper Sec. IV-B).

ROAM is run on each ROI written by extract_wsss4luad_uni2h.py (full bag, no patch cap),
and every tile takes the sparse dispatch weights gamma (Eq. 3) of its region. Pooled over
all ROIs, each expert's weight gamma_{.,e} is scored as a one-vs-rest ranker of each tissue
class (AUROC; tied weights, e.g. the many zeros of top-k dispatch, count as half).
Experts are numbered e0 .. e{E-1} as in the paper. Expert identities are specific to one
trained model; the paper uses the UNI2-h NSCLC model of fold 0.

Usage:
    python -m scripts.interpretability.score_wsss4luad_experts \
        --run_dir outputs/uni2h/nsclc/roam/fold0 \
        --feat_dir /data/wsss4luad_uni2h --out wsss4luad_expert_auc.csv
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from src.eval import load_run_args
from src.train import build_model

TISSUES = ["tumour", "stroma", "normal"]      # WSSS4LUAD mask values 0, 1, 2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True, help="Run with config.json + best_model.pt")
    p.add_argument("--feat_dir", required=True, help="<roi>.h5 files from extract_wsss4luad_uni2h.py")
    p.add_argument("--out", required=True, help="Output CSV (experts x tissues AUROC)")
    a = p.parse_args()

    run = load_run_args(a.run_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(run)
    model.load_state_dict(torch.load(Path(a.run_dir) / "best_model.pt", weights_only=True,
                                     map_location=device))
    model = model.to(device).eval()

    rois = sorted(Path(a.feat_dir).glob("*.h5"))
    gammas, labels = [], []
    for path in rois:
        with h5py.File(path, "r") as f:
            feats = torch.from_numpy(f["features"][:]).float()
            coords = torch.from_numpy(f["coords"][:]).long()
            labels.append(f["labels"][:])
        with torch.no_grad():
            model(feats.to(device), coords.to(device))
        gammas.append(model._diag["Alpha"][model._diag["region_id"]].cpu().numpy())   # (N, E)
    gamma, y = np.concatenate(gammas), np.concatenate(labels)
    print(f"{len(rois)} ROIs, {len(y)} tiles: "
          + ", ".join(f"{t} {int((y == c).sum())}" for c, t in enumerate(TISSUES)))

    experts = range(gamma.shape[1])
    table = pd.DataFrame({t: [roc_auc_score(y == c, gamma[:, e]) for e in experts]
                          for c, t in enumerate(TISSUES)},
                         index=pd.Index([f"e{e}" for e in experts], name="expert"))
    print("One-vs-rest AUROC of each expert's dispatch weight:")
    print(table.round(3).to_string())
    table.to_csv(a.out)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
