# ROAM: Region-Graph Optimal Transport Routing for Mixture-of-Experts WSI Classification
# Xin Tian et al., IEEE BIBM 2026 | https://arxiv.org/abs/2604.07298 | https://github.com/xintian-99/ROAM
"""Download the pre-extracted patch features used in the paper from Hugging Face.

UNI2-h  -> MahmoodLab/UNI2-h-features (gated: accept the terms on the dataset page
           and run `huggingface-cli login` first). One .tar.gz per cohort; each is
           extracted into <out>/<TCGA|CPTAC|PANDA>/<archive name>/, which is the
           layout the splits/uni2h_*.csv paths expect.
GigaPath -> Dearcat/CPathPatchFeature. Files are downloaded as-is; the layout
           <out>/<dataset>/gigap/pt_files/*.pt and <out>/<dataset>/patches/*.h5 is
           what splits/gigapath_*.csv expects.

Examples:
    python scripts/download_features.py --encoder uni2h --tasks nsclc --out /data/UNI2-h-features
    python scripts/download_features.py --encoder gigapath --tasks nsclc brca panda --out /data/CPathPatchFeature

Approximate download sizes (UNI2-h): NSCLC 71 GB + CPTAC external 100 GB,
BRCA 64 GB, CRC 22 GB, PANDA 32 GB.
"""

import argparse
import tarfile
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

UNI2H_REPO = "MahmoodLab/UNI2-h-features"
UNI2H_ARCHIVES = {
    "nsclc": ["TCGA/TCGA-LUAD.tar.gz", "TCGA/TCGA-LUSC.tar.gz"],
    "cptac": ["CPTAC/cptac_luad_part1.tar.gz", "CPTAC/cptac_luad_part2.tar.gz",
              "CPTAC/cptac_lscc.tar.gz"],
    "brca": ["TCGA/TCGA-BRCA_IDC.tar.gz", "TCGA/TCGA-BRCA_OTHERS.tar.gz"],
    "crc": ["TCGA/TCGA-COAD.tar.gz", "TCGA/TCGA-READ.tar.gz"],
    "panda": ["PANDA/panda.tar.gz"],
}

CPATH_REPO = "Dearcat/CPathPatchFeature"
CPATH_DATASETS = {"nsclc": "nsclc", "brca": "brca", "panda": "panda"}


def download_uni2h(tasks, out, keep_archives):
    for task in tasks:
        for archive in UNI2H_ARCHIVES[task]:
            target = out / archive.replace(".tar.gz", "")
            if target.is_dir() and any(target.iterdir()):
                print(f"[skip] {target} already populated")
                continue
            print(f"[download] {UNI2H_REPO}/{archive}")
            path = Path(hf_hub_download(UNI2H_REPO, archive, repo_type="dataset",
                                        local_dir=out / "_archives"))
            print(f"[extract] {path.name} -> {target}")
            target.mkdir(parents=True, exist_ok=True)
            with tarfile.open(path) as tar:
                for m in tar:
                    if m.isfile() and m.name.endswith(".h5"):
                        m.name = Path(m.name).name    # flatten: <slide_id>.h5
                        tar.extract(m, target)
            if not keep_archives:
                path.unlink()


def download_gigapath(tasks, out):
    patterns = []
    for task in tasks:
        if task not in CPATH_DATASETS:
            raise SystemExit(f"No GigaPath features for '{task}' in {CPATH_REPO}; "
                             "extract them yourself (see README).")
        d = CPATH_DATASETS[task]
        patterns += [f"{d}/gigap/pt_files/*", f"{d}/patches/*"]
    print(f"[download] {CPATH_REPO} {patterns}")
    snapshot_download(CPATH_REPO, repo_type="dataset", local_dir=out,
                      allow_patterns=patterns, max_workers=8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--encoder", choices=["uni2h", "gigapath"], required=True)
    p.add_argument("--tasks", nargs="+", required=True,
                   choices=["nsclc", "cptac", "brca", "crc", "panda"],
                   help="'cptac' = the external NSCLC test cohort (UNI2-h only)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--keep_archives", action="store_true",
                   help="Keep the downloaded .tar.gz files after extraction")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.encoder == "uni2h":
        download_uni2h(a.tasks, a.out, a.keep_archives)
    else:
        download_gigapath(a.tasks, a.out)


if __name__ == "__main__":
    main()
