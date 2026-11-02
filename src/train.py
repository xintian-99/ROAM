"""Train ROAM on one cross-validation fold of a split file.

Usage (ROAM, paper configuration, NSCLC fold 0 with CPTAC as external test):
    python -m src.train \
        --manifest splits/uni2h_tcga_nsclc.csv \
        --external_manifest splits/uni2h_cptac_nsclc.csv \
        --feature_root /data/UNI2-h-features \
        --fold 0 --output_dir outputs/uni2h/nsclc/roam/fold0

Slides with `fold == --fold` are held out; all other folds are used for training.
The held-out fold drives early stopping and checkpoint selection, and its metrics
are written to summary.json. The external set (fold == -1) is scored once with
the selected checkpoint.

Outputs in --output_dir: config.json, history.csv, best_model.pt, summary.json,
val_preds.csv and, with --external_manifest, external_metrics.json and
external_preds.csv.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from src.data.dataset import MILFeatureDataset, mil_collate_fn
from src.utils.metrics import compute_binary_metrics, compute_multiclass_metrics


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=str, required=True,
                   help="Split CSV with feature_path, label, fold columns")
    p.add_argument("--external_manifest", type=str, default=None,
                   help="Optional external test split (e.g. CPTAC)")
    p.add_argument("--feature_root", type=str, default=None,
                   help="Directory that relative feature_path/coords_path entries are resolved against")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--task", type=str, default="binary",
                   choices=["binary", "multiclass"])
    p.add_argument("--n_classes", type=int, default=2,
                   help="Number of classes (2 for binary, 6 for PANDA)")
    p.add_argument("--output_dir", type=str, required=True)

    # Architecture (defaults = paper configuration)
    p.add_argument("--d_in", type=int, default=1536)
    p.add_argument("--d_proj", type=int, default=512)
    p.add_argument("--d_att", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.25)
    p.add_argument("--n_tokens", type=int, default=256,
                   help="Max region tokens M (grid side = sqrt(n_tokens))")
    p.add_argument("--num_experts", type=int, default=8,
                   help="Experts E")
    p.add_argument("--gnn_layers", type=int, default=2, help="GATv2 layers")
    p.add_argument("--gnn_heads", type=int, default=4, help="GATv2 heads")
    p.add_argument("--gnn_dropout", type=float, default=0.1)
    p.add_argument("--k_neighbors", type=int, default=8, help="Region kNN graph degree")
    p.add_argument("--ot_eps", type=float, default=0.1,
                   help="Entropic regularisation")
    p.add_argument("--sinkhorn_iters", type=int, default=20,
                   help="Sinkhorn updates T")
    p.add_argument("--expert_heads", type=int, default=2,
                   help="Attention heads per expert")
    p.add_argument("--d_comp", type=int, default=64,
                   help="Routing-profile embedding dim (0 disables it)")
    p.add_argument("--topk", type=int, default=2,
                   help="Top-k experts per region for sparse dispatch")
    p.add_argument("--lam_size", type=float, default=0.1,
                   help="Row-constant log-size term in the OT cost (absorbed by Sinkhorn)")
    p.add_argument("--lam_rare", type=float, default=0.2,
                   help="Rarity boost in the OT supply marginal")
    p.add_argument("--lam_smooth", type=float, default=0.1,
                   help="Graph-regularisation strength lambda_s")
    p.add_argument("--smooth_iters", type=int, default=3,
                   help="Graph-regularisation blocks inside Sinkhorn (0 = plain Sinkhorn)")
    # ROAM ablations (paper Table II)
    p.add_argument("--no_routing_gnn", action="store_true",
                   help="Ablation: remove the routing GNN (Z = H0); graph regularisation kept")
    p.add_argument("--routing", type=str, default="ot", choices=["ot", "softmax"],
                   help="Ablation: 'softmax' replaces graph-regularised capacity-constrained OT "
                        "with a row-softmax gate (no capacity constraint, no graph term; "
                        "dispatch rows sum to 1 instead of the region supply)")
    p.add_argument("--no_ot_modulation", action="store_true",
                   help="Ablation: drop the log(gamma) term in expert attention; each expert "
                        "then attends over all regions, not only its dispatched support")
    p.add_argument("--ot_mod_detach", action="store_true",
                   help="Ablation: stop-gradient on gamma inside expert attention")

    # Training
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4, help="AdamW weight decay")
    p.add_argument("--max_epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--max_patches", type=int, default=4096,
                   help="Patch cap per bag: random subset in training, first N in file "
                        "order at evaluation (0 = no cap)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=4)
    return p


def parse_args():
    return build_parser().parse_args()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _worker_init_fn(worker_id):
    """Seed each DataLoader worker for reproducible subsampling."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed + worker_id)


def build_model(args):
    """Build ROAM from parsed arguments."""
    from src.models.roam import ROAM
    if args.task not in ("binary", "multiclass"):
        raise ValueError(f"Unknown task: {args.task}")
    # binary: one logit (BCE); multiclass: one logit per class (CE)
    n_classes = 1 if args.task == "binary" else args.n_classes
    return ROAM(
        d_in=args.d_in, d_proj=args.d_proj, d_att=args.d_att,
        n_classes=n_classes, dropout=args.dropout,
        use_gnn=not args.no_routing_gnn,
        gnn_layers=args.gnn_layers, gnn_heads=args.gnn_heads,
        gnn_dropout=args.gnn_dropout, k_neighbors=args.k_neighbors,
        n_tokens=args.n_tokens, n_experts=args.num_experts,
        ot_eps=args.ot_eps, sinkhorn_iters=args.sinkhorn_iters,
        expert_heads=args.expert_heads, d_comp=args.d_comp,
        topk=args.topk, lam_size=args.lam_size,
        lam_rare=args.lam_rare, lam_smooth=args.lam_smooth,
        smooth_iters=args.smooth_iters,
        routing=args.routing,
        use_ot_modulation=not args.no_ot_modulation,
        ot_mod_detach=args.ot_mod_detach,
    )


# ── Forward pass ──────────────────────────────────────────────────────

def forward_slide(model, features, coords, label, criterion, device, task="binary"):
    """One slide -> (loss, prediction); binary: P(label 1), multiclass: class probabilities."""
    features = features.to(device)
    logits, _ = model(features, coords)
    if task == "multiclass":
        label_t = label.to(device).long()
        logits = logits.squeeze(0)
        loss = criterion(logits.unsqueeze(0), label_t.unsqueeze(0))
        return loss, torch.softmax(logits, dim=-1).detach().cpu().numpy()
    label_t = label.to(device).float()
    logits = logits.squeeze()
    loss = criterion(logits, label_t)
    return loss, torch.sigmoid(logits).detach().cpu().item()


# ── Training & evaluation ─────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip,
                    task="binary", n_classes=2):
    model.train()
    total_loss = 0.0
    all_labels, all_preds = [], []

    for features, coords, label, slide_id in loader:
        optimizer.zero_grad()

        loss, pred = forward_slide(model, features, coords, label, criterion,
                                   device, task=task)

        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        all_labels.append(label.cpu().item())
        all_preds.append(pred)

    avg_loss = total_loss / len(loader)

    if task == "multiclass":
        metrics = compute_multiclass_metrics(all_labels, all_preds, n_classes)
    else:
        metrics = compute_binary_metrics(all_labels, all_preds)
    metrics["loss"] = avg_loss
    return metrics


@torch.no_grad()
def evaluate(model, loader, criterion, device, task="binary", n_classes=2):
    model.eval()
    total_loss = 0.0
    all_labels, all_preds, all_slide_ids = [], [], []

    for features, coords, label, slide_id in loader:
        loss, pred = forward_slide(model, features, coords, label, criterion,
                                   device, task=task)

        total_loss += loss.item()
        all_labels.append(label.cpu().item())
        all_preds.append(pred)
        all_slide_ids.append(slide_id)

    avg_loss = total_loss / len(loader)

    if task == "multiclass":
        metrics = compute_multiclass_metrics(all_labels, all_preds, n_classes)
    else:
        metrics = compute_binary_metrics(all_labels, all_preds)
    metrics["loss"] = avg_loss
    return metrics, all_labels, all_preds, all_slide_ids


@torch.no_grad()
def measure_inference_time(model, loader, device, n_warmup=5, n_measure=50):
    """Measure per-slide inference time (seconds)."""
    model.eval()
    times = []
    slide_count = 0

    for features, coords, label, slide_id in loader:
        features = features.to(device)

        if slide_count < n_warmup:
            model(features, coords)
            slide_count += 1
            continue

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        model(features, coords)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        times.append(t1 - t0)
        slide_count += 1
        if len(times) >= n_measure:
            break

    if len(times) == 0:
        return 0.0, 0.0
    return float(np.mean(times)), float(np.std(times))


# ── Task-aware helpers ────────────────────────────────────────────────

def primary_metric(task):
    """Early-stopping / model-selection metric (higher is better)."""
    return {"binary": "auc", "multiclass": "qwk"}[task]


def build_criterion(task):
    """Cross-entropy: 1-logit BCE for binary tasks, CE for multiclass."""
    if task == "multiclass":
        return nn.CrossEntropyLoss()
    return nn.BCEWithLogitsLoss()


def _make_record(epoch, lr, elapsed, train_metrics, val_metrics, task):
    """Build a history record appropriate for the task."""
    record = {"epoch": epoch, "lr": lr, "elapsed": elapsed}

    if task == "binary":
        record.update({
            "train_loss": train_metrics["loss"],
            "train_auc": train_metrics["auc"],
            "val_loss": val_metrics["loss"],
            "val_auc": val_metrics["auc"],
            "val_bacc": val_metrics["balanced_accuracy"],
            "val_f1": val_metrics["f1"],
            "val_sens": val_metrics["sensitivity"],
            "val_spec": val_metrics["specificity"],
        })
    else:
        record.update({
            "train_loss": train_metrics["loss"],
            "train_qwk": train_metrics["qwk"],
            "val_loss": val_metrics["loss"],
            "val_qwk": val_metrics["qwk"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_accuracy": val_metrics["accuracy"],
            "val_auc_ovr": val_metrics.get("auc_ovr", 0.0),
        })
    return record


def _print_epoch(epoch, max_epochs, lr, elapsed, train_metrics, val_metrics,
                 task, improved):
    """Print epoch summary line."""
    if task == "binary":
        print(
            f"Epoch {epoch:3d}/{max_epochs} | "
            f"lr={lr:.2e} | "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_auc={train_metrics['auc']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_auc={val_metrics['auc']:.4f} "
            f"val_bacc={val_metrics['balanced_accuracy']:.4f}"
            f"{improved} | {elapsed:.1f}s"
        )
    else:
        print(
            f"Epoch {epoch:3d}/{max_epochs} | "
            f"lr={lr:.2e} | "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_qwk={train_metrics['qwk']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_qwk={val_metrics['qwk']:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}"
            f"{improved} | {elapsed:.1f}s"
        )


def save_preds(labels, preds, slide_ids, task, n_classes, path):
    """Save per-slide predictions in task-appropriate format."""
    if task == "multiclass":
        df = pd.DataFrame({
            "slide_id": slide_ids,
            "label": labels,
            "pred": [int(np.argmax(p)) for p in preds],
        })
        for c in range(n_classes):
            df[f"prob_class_{c}"] = [float(p[c]) for p in preds]
    else:  # binary
        df = pd.DataFrame({"slide_id": slide_ids, "label": labels, "prob": preds})
    df.to_csv(path, index=False)


def scalar_metrics(metrics):
    """Drop nested (non-JSON-scalar) entries such as per-class accuracy."""
    return {k: v for k, v in metrics.items() if not isinstance(v, dict)}


def make_loader(df, args, training):
    ds = MILFeatureDataset(df, max_patches=args.max_patches, training=training,
                           feature_root=args.feature_root)
    return DataLoader(
        ds, batch_size=1, shuffle=training,
        num_workers=args.num_workers, collate_fn=mil_collate_fn, pin_memory=True,
        worker_init_fn=_worker_init_fn,
    )


def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Task: {args.task}, n_classes: {args.n_classes}")

    # ── Load manifest and split ────────────────────────────────────────
    manifest = pd.read_csv(args.manifest)
    train_df = manifest[manifest["fold"] != args.fold].reset_index(drop=True)
    val_df = manifest[manifest["fold"] == args.fold].reset_index(drop=True)

    print(f"Fold {args.fold}: train={len(train_df)}, val={len(val_df)}")
    print(f"Train labels: {train_df['label'].value_counts().to_dict()}")
    print(f"Val labels:   {val_df['label'].value_counts().to_dict()}")

    train_loader = make_loader(train_df, args, training=True)
    val_loader = make_loader(val_df, args, training=False)

    # ── Model, optimizer, scheduler ────────────────────────────────────
    model = build_model(args).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"ROAM parameters: {n_params:,}")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Per-epoch schedule: linear warmup from 0.1*lr, then cosine decay
    warmup_scheduler = LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=args.warmup_epochs,
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=max(args.max_epochs - args.warmup_epochs, 1),
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[args.warmup_epochs],
    )

    criterion = build_criterion(args.task)

    # ── Early stopping setup ───────────────────────────────────────────
    es_metric = primary_metric(args.task)
    best_val_score = -float("inf")
    patience_counter = 0
    history = []

    print(f"\nStarting training: max_epochs={args.max_epochs}, patience={args.patience}")
    print(f"Early stopping metric: {es_metric} (higher is better)")
    print("-" * 80)

    for epoch in range(1, args.max_epochs + 1):
        t0 = time.time()
        lr = optimizer.param_groups[0]["lr"]

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device, args.grad_clip,
            task=args.task, n_classes=args.n_classes,
        )
        val_metrics, _, _, _ = evaluate(
            model, val_loader, criterion, device,
            task=args.task, n_classes=args.n_classes,
        )
        scheduler.step()

        elapsed = time.time() - t0

        history.append(_make_record(epoch, lr, elapsed, train_metrics, val_metrics,
                                    args.task))

        current_score = val_metrics[es_metric]
        improved = ""
        if current_score > best_val_score:
            best_val_score = current_score
            patience_counter = 0
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            improved = " *"
        else:
            patience_counter += 1

        _print_epoch(epoch, args.max_epochs, lr, elapsed, train_metrics,
                     val_metrics, args.task, improved)

        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch} (patience={args.patience})")
            break

    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)

    # ── Final evaluation on the held-out fold with the best model ──────
    print(f"\n{'='*80}")
    print(f"Best val {es_metric}: {best_val_score:.4f}")
    model.load_state_dict(torch.load(output_dir / "best_model.pt", weights_only=True))
    val_metrics, val_labels, val_preds, val_slide_ids = evaluate(
        model, val_loader, criterion, device,
        task=args.task, n_classes=args.n_classes,
    )
    print(f"Best model val metrics: {val_metrics}")
    save_preds(val_labels, val_preds, val_slide_ids, args.task, args.n_classes,
               output_dir / "val_preds.csv")

    print("\nMeasuring inference time...")
    time_mean, time_std = measure_inference_time(
        model, val_loader, device,
    )
    print(f"Inference time per slide: {time_mean*1000:.1f} +/- {time_std*1000:.1f} ms")

    # ── External test evaluation ───────────────────────────────────────
    if args.external_manifest:
        print(f"\nEvaluating on external test set: {args.external_manifest}")
        ext_df = pd.read_csv(args.external_manifest)
        ext_loader = make_loader(ext_df, args, training=False)
        ext_metrics, ext_labels, ext_preds, ext_slide_ids = evaluate(
            model, ext_loader, criterion, device,
            task=args.task, n_classes=args.n_classes,
        )
        print(f"External test metrics: {ext_metrics}")
        save_preds(ext_labels, ext_preds, ext_slide_ids, args.task, args.n_classes,
                   output_dir / "external_preds.csv")
        with open(output_dir / "external_metrics.json", "w") as f:
            json.dump(scalar_metrics(ext_metrics), f, indent=2)

    summary = {
        "task": args.task,
        "n_classes": args.n_classes,
        "fold": args.fold,
        "n_params": n_params,
        f"best_val_{es_metric}": best_val_score,
        "best_val_metrics": scalar_metrics(val_metrics),
        "total_epochs": len(history),
        "inference_time_ms_mean": round(time_mean * 1000, 2),
        "inference_time_ms_std": round(time_std * 1000, 2),
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
