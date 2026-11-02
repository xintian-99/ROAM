# ROAM: Region-Graph Optimal Transport Routing for Mixture-of-Experts WSI Classification
# Xin Tian et al., IEEE BIBM 2026 | https://arxiv.org/abs/2604.07298 | https://github.com/xintian-99/ROAM
"""Parameters, GFLOPs and latency of ROAM at a fixed bag size N
(paper Sec. IV-A, "Computational cost").

Protocol: real slides with >= N patches, truncated to their first N patches (real
coordinates); GFLOPs from torch FlopCounterMode on one forward pass (FLOPs =
2 x MACs); latency = mean of `--iters` eval-mode forward passes (batch size 1,
host-to-device copy included) after `--warmup` passes, cycling over the slides.

    python -m scripts.benchmark_cost --manifest splits/uni2h_tcga_nsclc.csv \
        --feature_root /data/UNI2-h-features --n 10000
"""

import argparse
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.flop_counter import FlopCounterMode

from src.data.dataset import MILFeatureDataset
from src.train import build_model, build_parser


def load_slides(manifest, feature_root, n, n_slides):
    ds = MILFeatureDataset(pd.read_csv(manifest), max_patches=n, feature_root=feature_root)
    slides = []
    for i in range(len(ds)):
        feats, coords, _, _ = ds[i]
        if feats.shape[0] == n:
            slides.append((feats, coords))
            if len(slides) == n_slides:
                break
    return slides


@torch.no_grad()
def forward(model, feats, coords, device):
    return model(feats.to(device), coords)[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--feature_root", default=None)
    p.add_argument("--n", type=int, default=10000)
    p.add_argument("--n_slides", type=int, default=10)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    a = p.parse_args()

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    slides = load_slides(a.manifest, a.feature_root, a.n, a.n_slides)
    if not slides:
        raise SystemExit(f"no slide with >= {a.n} patches")
    print(f"device={device}  N={a.n}  slides={len(slides)}")

    model = build_model(build_parser().parse_args(["--manifest", "-", "--output_dir", "-"]))
    model = model.to(device).eval()
    n_params = sum(t.numel() for t in model.parameters())

    fcm = FlopCounterMode(display=False)
    with fcm:
        forward(model, *slides[0], device)
    gflops = fcm.get_total_flops() / 1e9

    latency = "n/a (CPU)"
    if device.type == "cuda":
        for i in range(a.warmup):
            forward(model, *slides[i % len(slides)], device)
        times = []
        for i in range(a.iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            forward(model, *slides[i % len(slides)], device)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
        latency = f"{np.mean(times):.2f} ± {np.std(times):.2f} ms"
    print(f"params {n_params / 1e6:.2f} M | {gflops:.1f} GFLOPs | latency {latency}")


if __name__ == "__main__":
    main()
