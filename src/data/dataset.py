"""Slide-level dataset over pre-extracted patch features listed in a split CSV.

Two on-disk formats are supported:
  * .h5 (UNI2-h features, MahmoodLab/UNI2-h-features): datasets `features`
    (1, N, D) and `coords` (1, N, 2); (N, D) / (N, 2) is also accepted.
  * .pt (CPathPatchFeature, e.g. GigaPath): a (N, D) tensor, with patch
    coordinates in a separate .h5 file (`coords`, (N, 2)) given by `coords_path`.

Relative `feature_path` / `coords_path` entries are resolved against `feature_root`.
"""

import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class MILFeatureDataset(Dataset):
    """Each item is (features (N, D) float32, coords (N, 2) int64, label, slide_id)."""

    def __init__(self, manifest_df, max_patches=None, training=False, feature_root=None):
        """
        Args:
            manifest_df: DataFrame with columns [slide_id, feature_path, label]
                         (+ coords_path for .pt features).
            max_patches: Cap on patches per bag (None/0 = no cap). Training draws a
                         random subset every epoch; evaluation takes the first
                         max_patches patches in file order.
            training:    Random (True) or deterministic (False) subsampling.
            feature_root: Prefix for relative paths in the manifest.
        """
        self.manifest = manifest_df.reset_index(drop=True)
        self.max_patches = max_patches
        self.training = training
        self.feature_root = feature_root

    def __len__(self):
        return len(self.manifest)

    def _resolve(self, path):
        path = str(path)
        if self.feature_root and not os.path.isabs(path):
            return os.path.join(self.feature_root, path)
        return path

    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]
        path = self._resolve(row["feature_path"])

        if path.endswith(".pt"):
            feats = torch.load(path, map_location="cpu", weights_only=True).numpy()  # (N, D)
            with h5py.File(self._resolve(row["coords_path"]), "r") as f:
                coords = f["coords"][:]  # (N, 2)
        else:
            with h5py.File(path, "r") as f:
                feats = f["features"][:]  # (1, N, D) or (N, D)
                coords = f["coords"][:]   # (1, N, 2) or (N, 2)
            if feats.ndim == 3:
                feats, coords = feats[0], coords[0]

        n_patches = feats.shape[0]
        if self.max_patches and n_patches > self.max_patches:
            if self.training:
                idx_sub = np.random.choice(n_patches, self.max_patches, replace=False)
                idx_sub.sort()
            else:
                idx_sub = np.arange(self.max_patches)
            feats = feats[idx_sub]
            coords = coords[idx_sub]

        features = torch.from_numpy(feats).float()
        coords = torch.from_numpy(coords).long()
        label = torch.tensor(int(row["label"]), dtype=torch.long)
        return features, coords, label, row["slide_id"]


def mil_collate_fn(batch):
    """Batch size 1: unpack the single slide."""
    assert len(batch) == 1, "MIL training uses one slide per batch"
    return batch[0]
