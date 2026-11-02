# ROAM: Region-Graph Optimal Transport Routing for Mixture-of-Experts WSI Classification
# Xin Tian et al., IEEE BIBM 2026 | https://arxiv.org/abs/2604.07298 | https://github.com/xintian-99/ROAM
"""Extract UNI2-h features for the WSSS4LUAD regions of interest (paper Sec. IV-B).

Every image of the WSSS4LUAD validation split (<split>/img/*.png, pixel masks in
<split>/mask/*.png with values 0 tumour, 1 stroma, 2 normal, 3 background) is cut into a
non-overlapping grid of 80 x 80 px tiles; incomplete tiles at the right/bottom edges are
dropped. A tile is kept if at least half of its pixels are annotated (mask != 3), and its
label is the majority class among those pixels (ties go to the lower class index). Kept
tiles are resized to 224 x 224 (bilinear), normalised with ImageNet statistics and
embedded with UNI2-h. ROIs with fewer than 12 kept tiles are skipped. With these defaults
(the paper setting) 22 of the 40 validation ROIs are kept, with 7,885 tiles (tumour 4,901,
stroma 2,781, normal 203). WSSS4LUAD images are released at 10x, so a tile covers ~80 um.

Output: one <roi>.h5 per ROI with datasets
    features (N, 1536) float32
    coords   (N, 2)    int64, tile centre (x, y) in image pixels
    labels   (N,)      int8, tissue class (0 tumour, 1 stroma, 2 normal)

UNI2-h is gated: accept its terms at https://huggingface.co/MahmoodLab/UNI2-h and log in
with `huggingface-cli login` (or set HF_TOKEN). Needs `timm` (>= 0.9.8, not in
requirements.txt); a GPU is recommended.

Usage:
    python -m scripts.interpretability.extract_wsss4luad_uni2h \
        --wsss_root /data/WSSS4LUAD --out_dir /data/wsss4luad_uni2h
"""

import argparse
import glob
import os
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image

BG = 3
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_uni2h(device):
    """Load UNI2-h.

    This loading code (timm arguments) is taken from the UNI2-h model card by the
    Mahmood Lab: https://huggingface.co/MahmoodLab/UNI2-h. The ImageNet mean/std used
    in embed() follow the same model card.
    """
    import timm
    timm_kwargs = dict(img_size=224, patch_size=14, depth=24, num_heads=24, init_values=1e-5,
                       embed_dim=1536, mlp_ratio=2.66667 * 2, num_classes=0, no_embed_class=True,
                       mlp_layer=timm.layers.SwiGLUPacked, act_layer=torch.nn.SiLU, reg_tokens=8,
                       dynamic_img_size=True)
    model = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
    return model.to(device).eval()


@torch.no_grad()
def embed(model, tiles, device):
    """tiles (n, 224, 224, 3) uint8 -> (n, 1536) features."""
    x = torch.from_numpy(tiles).float().permute(0, 3, 1, 2) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    out = []
    for i in range(0, len(x), 32):
        out.append(model(x[i:i + 32].to(device)).cpu())
    return torch.cat(out).numpy()


def tile_roi(img, mask, tile, stride, resize, min_tissue):
    """Cut one ROI into labelled tiles. Returns (tiles, coords, labels) or None."""
    H, W = mask.shape
    tiles, coords, labels = [], [], []
    for y in range(0, H - tile + 1, stride):
        for x in range(0, W - tile + 1, stride):
            tm = mask[y:y + tile, x:x + tile]
            nonbg = tm != BG
            if nonbg.mean() < min_tissue:
                continue
            cls = np.bincount(tm[nonbg], minlength=3)[:3].argmax()
            ti = Image.fromarray(img[y:y + tile, x:x + tile]).resize((resize, resize), Image.BILINEAR)
            tiles.append(np.asarray(ti))
            coords.append([x + tile // 2, y + tile // 2])
            labels.append(int(cls))
    if not tiles:
        return None
    return np.stack(tiles), np.array(coords, np.int64), np.array(labels, np.int8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wsss_root", required=True, help="Unpacked WSSS4LUAD directory")
    p.add_argument("--split", default="2.validation", help="Split folder with img/ and mask/")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--tile", type=int, default=80, help="Tile size in image pixels")
    p.add_argument("--stride", type=int, default=80)
    p.add_argument("--resize", type=int, default=224, help="Encoder input size")
    p.add_argument("--min_tissue", type=float, default=0.5,
                   help="Minimum fraction of annotated (non-background) pixels per tile")
    p.add_argument("--min_tiles", type=int, default=12, help="Skip ROIs with fewer kept tiles")
    p.add_argument("--device", default=None)
    a = p.parse_args()

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_uni2h(device)

    n_rois, counts = 0, np.zeros(3, int)
    for ip in sorted(glob.glob(f"{a.wsss_root}/{a.split}/img/*.png")):
        stem = os.path.basename(ip).split(".")[0]
        mp = f"{a.wsss_root}/{a.split}/mask/{stem}.png"
        if not os.path.exists(mp):
            continue
        img = np.array(Image.open(ip).convert("RGB"))
        mask = np.array(Image.open(mp))
        r = tile_roi(img, mask, a.tile, a.stride, a.resize, a.min_tissue)
        if r is None or len(r[0]) < a.min_tiles:
            continue
        tiles, coords, labels = r
        feats = embed(model, tiles, device)
        with h5py.File(out / f"{stem}.h5", "w") as f:
            f["features"] = feats.astype(np.float32)
            f["coords"] = coords
            f["labels"] = labels
        cc = np.bincount(labels, minlength=3)
        counts += cc
        n_rois += 1
        print(f"{stem}: {len(labels)} tiles (tumour {cc[0]}, stroma {cc[1]}, normal {cc[2]})")
    print(f"kept {n_rois} ROIs, {counts.sum()} tiles (tumour {counts[0]}, stroma {counts[1]}, "
          f"normal {counts[2]}) -> {out}")


if __name__ == "__main__":
    main()
