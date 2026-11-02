"""Export ROAM's per-patch routing for selected slides (paper Fig. 2).

For every patch: coordinates, region index, dominant routed expert (argmax of the
sparse dispatch gamma of its region) and the dispatch weight of each expert, plus
the slide prediction (binary: P(label 1); multiclass: predicted class). Draw the
map by colouring each patch by `expert` at (x, y) over a slide thumbnail.

    python -m scripts.export_routing --run_dir outputs/uni2h/nsclc/roam/fold0 \
        --manifest splits/uni2h_cptac_nsclc.csv --feature_root $UNI2H_ROOT \
        --slides <slide_id> [<slide_id> ...] --out routing.csv
"""

import argparse
from pathlib import Path

import pandas as pd
import torch

from src.data.dataset import MILFeatureDataset
from src.eval import load_run_args
from src.train import build_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True, help="Run with config.json + best_model.pt")
    p.add_argument("--manifest", required=True)
    p.add_argument("--feature_root", default=None)
    p.add_argument("--slides", nargs="+", required=True)
    p.add_argument("--max_patches", type=int, default=0,
                   help="Patch cap (default 0 = full bag)")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    run = load_run_args(a.run_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(run)
    model.load_state_dict(torch.load(Path(a.run_dir) / "best_model.pt", weights_only=True,
                                     map_location=device))
    model = model.to(device).eval()

    df = pd.read_csv(a.manifest)
    df = df[df["slide_id"].isin(a.slides)].reset_index(drop=True)
    missing = set(a.slides) - set(df["slide_id"])
    if missing:
        raise SystemExit(f"slides not in {a.manifest}: {sorted(missing)}")
    ds = MILFeatureDataset(df, max_patches=a.max_patches, feature_root=a.feature_root)

    tables = []
    for i in range(len(ds)):
        feats, coords, label, slide_id = ds[i]
        with torch.no_grad():
            logits, _ = model(feats.to(device), coords.to(device))
        # binary: P(label 1); multiclass: predicted class
        pred = round(torch.sigmoid(logits).item(), 4) if logits.numel() == 1 else int(logits.argmax())
        gamma = model._diag["Alpha"][model._diag["region_id"]].cpu()      # (N, E)
        t = pd.DataFrame(gamma.numpy(), columns=[f"gamma_e{e}" for e in range(gamma.shape[1])])
        t.insert(0, "expert", gamma.argmax(dim=1).numpy())
        t.insert(0, "region", model._diag["region_id"].cpu().numpy())
        t.insert(0, "y", coords[:, 1].numpy())
        t.insert(0, "x", coords[:, 0].numpy())
        t.insert(0, "label", int(label))
        t.insert(0, "pred", pred)
        t.insert(0, "slide_id", slide_id)
        tables.append(t)
        print(f"{slide_id}: {len(t)} patches, {t['region'].nunique()} regions, "
              f"experts used {sorted(t['expert'].unique().tolist())}")
    pd.concat(tables).to_csv(a.out, index=False)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
