# ROAM: Region-Graph Optimal Transport Routing for Mixture-of-Experts WSI Classification
# Xin Tian et al., IEEE BIBM 2026 | https://arxiv.org/abs/2604.07298 | https://github.com/xintian-99/ROAM
"""Smoke tests: ROAM trains (forward + backward) and evaluates on random bags, and
train.py / eval.py run end to end on a tiny synthetic dataset.

    pytest -q tests/
"""

import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import torch

from src.train import build_criterion, build_model, build_parser, forward_slide

D_IN = 1536
REPO = Path(__file__).resolve().parents[1]


def make_args(task="binary", **overrides):
    argv = ["--manifest", "-", "--output_dir", "-", "--task", task]
    if task == "multiclass":
        argv += ["--n_classes", "6"]
    args = build_parser().parse_args(argv)
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


def random_bag(n, grid=64):
    feats = torch.randn(n, D_IN)
    coords = torch.randint(0, grid, (n, 2)) * 256
    return feats, coords


def step(model, args, feats, coords, label):
    criterion = build_criterion(args.task)
    return forward_slide(model, feats, coords, label, criterion, "cpu", task=args.task)


@pytest.mark.parametrize("task", ["binary", "multiclass"])
def test_train_and_eval_step(task):
    torch.manual_seed(0)
    args = make_args(task)
    model = build_model(args)
    feats, coords = random_bag(600)
    label = torch.tensor(1)

    model.train()
    loss, _ = step(model, args, feats, coords, label)
    assert torch.isfinite(loss)
    loss.backward()
    n_grad = sum(p.grad is not None and torch.isfinite(p.grad).all().item()
                 for p in model.parameters())
    assert n_grad > 0

    model.eval()
    with torch.no_grad():
        loss, pred = step(model, args, feats, coords, label)
    assert torch.isfinite(loss)
    if task == "binary":
        assert 0.0 <= pred <= 1.0
    else:
        assert pred.shape == (6,) and abs(pred.sum() - 1.0) < 1e-4


@pytest.mark.parametrize("n", [1, 5, 50, 256, 257, 5000])
def test_roam_bag_sizes(n):
    torch.manual_seed(0)
    args = make_args()
    model = build_model(args)
    model.eval()
    feats, coords = random_bag(n)
    with torch.no_grad():
        logits, saliency = model(feats, coords)
    assert logits.shape == (1,) and torch.isfinite(logits).all()
    assert saliency.shape == (n,)
    alpha = model._diag["Alpha"]
    assert alpha.shape[0] <= 256
    assert ((alpha > 0).sum(dim=1) <= args.topk).all()


def test_roam_transport_marginals():
    """Dense plan: rows sum to the supply r, columns to 1/E (up to Sinkhorn error)."""
    torch.manual_seed(0)
    model = build_model(make_args())
    model.eval()
    feats, coords = random_bag(4096)
    with torch.no_grad():
        model(feats, coords)
    Pi = model._diag["Pi"]
    assert abs(Pi.sum().item() - 1.0) < 1e-3
    assert torch.allclose(Pi.sum(dim=0), torch.full((8,), 1 / 8), atol=2e-2)


@pytest.mark.parametrize("flags", [
    {"no_routing_gnn": True},
    {"smooth_iters": 0},
    {"routing": "softmax"},
    {"no_ot_modulation": True},
    {"ot_mod_detach": True},
])
def test_roam_ablations(flags):
    torch.manual_seed(0)
    args = make_args(**flags)
    model = build_model(args)
    model.train()
    feats, coords = random_bag(1000)
    loss, _ = step(model, args, feats, coords, torch.tensor(0))
    loss.backward()
    assert torch.isfinite(loss)


def _write_dataset(root, n_slides=12):
    rows = []
    rng = np.random.default_rng(0)
    for i in range(n_slides):
        n = int(rng.integers(300, 900))
        label = i % 2
        feats = rng.normal(size=(1, n, D_IN)).astype("float32") + 0.5 * label
        coords = rng.integers(0, 80, size=(1, n, 2)) * 256
        rel = f"slides/slide{i}.h5"
        (root / "slides").mkdir(parents=True, exist_ok=True)
        with h5py.File(root / rel, "w") as f:
            f["features"] = feats
            f["coords"] = coords
        rows.append(dict(slide_id=f"slide{i}", case_id=f"case{i}", label=label,
                         label_str=str(label), fold=i % 3, feature_path=rel))
    df = pd.DataFrame(rows)
    df.to_csv(root / "split.csv", index=False)
    ext = df.copy()
    ext["fold"] = -1
    ext.to_csv(root / "external.csv", index=False)


def test_train_and_eval_cli(tmp_path):
    _write_dataset(tmp_path)
    out = tmp_path / "run"
    subprocess.run([
        sys.executable, "-m", "src.train",
        "--manifest", str(tmp_path / "split.csv"),
        "--external_manifest", str(tmp_path / "external.csv"),
        "--feature_root", str(tmp_path),
        "--fold", "0", "--max_epochs", "2",
        "--num_workers", "0", "--output_dir", str(out),
    ], check=True, cwd=REPO)
    for f in ["config.json", "history.csv", "best_model.pt", "summary.json",
              "val_preds.csv", "external_preds.csv", "external_metrics.json"]:
        assert (out / f).exists(), f

    subprocess.run([
        sys.executable, "-m", "src.eval", "--run_dir", str(out), "--num_workers", "0",
    ], check=True, cwd=REPO)
    trained = json.load(open(out / "summary.json"))["best_val_metrics"]["auc"]
    rescored = json.load(open(out / "eval_val_metrics.json"))["auc"]
    assert abs(trained - rescored) < 1e-6
    ext_trained = json.load(open(out / "external_metrics.json"))["auc"]
    ext_rescored = json.load(open(out / "eval_external_metrics.json"))["auc"]
    assert abs(ext_trained - ext_rescored) < 1e-6
