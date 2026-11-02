# ROAM: Region-Graph Optimal Transport Routing for Mixture-of-Experts Whole-Slide Image Classification

Code for the IEEE BIBM 2026 paper

> Xin Tian<sup>1</sup>, Jiuliu Lu<sup>2</sup>, Ephraim Tsalik<sup>3</sup>, Bart Wanders<sup>2</sup>, Colleen Knoth<sup>3</sup>, Julian Knight<sup>1</sup>.
> **Region-Graph Optimal Transport Routing for Mixture-of-Experts Whole-Slide Image Classification.**
> IEEE International Conference on Bioinformatics and Biomedicine (BIBM), 2026.
>
> <sup>1</sup>Centre for Human Genetics, University of Oxford, Oxford, UK;
> <sup>2</sup>Beckman Coulter Diagnostics, Brea, CA, USA;
> <sup>3</sup>Danaher Corporation, Washington, DC, USA

**Paper:** [arXiv:2604.07298](https://arxiv.org/abs/2604.07298) · **Project page:** [`index.html`](index.html)

ROAM (Region-graph OptimAl-transport Mixture-of-experts) is a multiple-instance-learning (MIL)
aggregator for whole-slide images (WSIs). It works on frozen patch embeddings and their slide
coordinates:

1. **Region tokenisation.** Patches are binned on a 16×16 grid (each coordinate axis scaled to
   [0, 1]), giving up to 256 region tokens (mean feature, size, centroid).
2. **Routing GNN.** A 2-layer GATv2 on a kNN graph (k = 8) of region centroids produces routing
   embeddings. These are used only for routing.
3. **Capacity-constrained OT routing.** Region-to-expert cost is 1 − cos(routing embedding, expert
   prototype). Region supply is proportional to region size, with a boost for locally distinct
   regions; each expert has capacity 1/E. Entropic OT is solved by Sinkhorn (T = 20 updates).
4. **Graph regularisation.** Three times during Sinkhorn, each region's log transport plan is
   smoothed towards its spatial neighbours (λ_s = 0.1), followed by re-projection.
5. **Sparse dispatch and expert pooling.** Each region keeps its top-2 experts. Each of the E = 8
   experts is a gated-attention pooler whose attention is biased by log(dispatch weight).
6. **Fusion.** Gated attention over the experts, concatenated with an embedding of each expert's
   dispatched load and entropy, followed by an MLP classifier.

The model is in [`src/models/roam.py`](src/models/roam.py).

## Repository layout

```
src/
  models/roam.py               ROAM
  data/dataset.py              reads .h5 (UNI2-h) and .pt + coords .h5 (CPathPatchFeature) features
  train.py                     train ROAM on one fold (+ optional external test set)
  eval.py                      re-score a trained run
  utils/metrics.py             AUC / balanced accuracy / F1, QWK
splits/                        slide lists with labels and 5-fold assignments (paths relative to a feature root)
scripts/
  download_features.py         download the pre-extracted features from Hugging Face
  run_cv.sh                    5-fold cross-validation for one task / encoder
  run_ablations.sh             ablations (Table II)
  summarize.py                 mean ± std over folds
  benchmark_cost.py            parameters, GFLOPs and latency at N = 10,000 patches
  export_routing.py            per-patch expert routing of selected slides (Fig. 2)
  interpretability/            WSSS4LUAD tissue analysis of the experts (Sec. IV-B)
tests/test_models.py           smoke tests (training step, edge cases, ablations, train/eval CLI)
index.html, assets/            project page
```

## Installation

```bash
git clone https://github.com/xintian-99/ROAM.git && cd ROAM
conda create -n roam python=3.10 -y && conda activate roam
pip install torch --index-url https://download.pytorch.org/whl/cu118   # pick the build for your CUDA
pip install -r requirements.txt
pytest -q                                                               # optional, a few minutes on CPU
```

Tested with Python 3.10, PyTorch 2.7.1 and CUDA 11.8. ROAM needs no graph library; the GATv2
layer is written in plain PyTorch.

## Data

All experiments use pre-extracted, frozen patch features released by other groups. We did not
re-extract features from the slides. The table below links the download page of every raw
dataset and feature set; `scripts/download_features.py` (see **Download**) fetches the features.
The raw WSIs are only needed to extract features for new cohorts or to visualise routing maps.

| Task (paper) | Cohort / label | Slides (patients) | Raw WSIs | UNI2-h features | GigaPath features | Split file(s) |
|---|---|---|---|---|---|---|
| NSCLC | TCGA-LUAD (0) vs TCGA-LUSC (1) | 1,043 (946) | GDC [TCGA-LUAD](https://portal.gdc.cancer.gov/projects/TCGA-LUAD), [TCGA-LUSC](https://portal.gdc.cancer.gov/projects/TCGA-LUSC) | [UNI2-h-features](https://huggingface.co/datasets/MahmoodLab/UNI2-h-features): `TCGA/TCGA-LUAD`, `TCGA/TCGA-LUSC` | [CPathPatchFeature](https://huggingface.co/datasets/Dearcat/CPathPatchFeature): `nsclc` (1,016 slides) | `uni2h_tcga_nsclc.csv`, `gigapath_tcga_nsclc.csv` |
| NSCLC external test | CPTAC-LUAD (0) vs CPTAC-LSCC (1) | 2,206 (453) | TCIA [CPTAC-LUAD](https://www.cancerimagingarchive.net/collection/cptac-luad/), [CPTAC-LSCC](https://www.cancerimagingarchive.net/collection/cptac-lscc/) | [UNI2-h-features](https://huggingface.co/datasets/MahmoodLab/UNI2-h-features): `CPTAC/cptac_luad_part1`, `_part2`, `CPTAC/cptac_lscc` | not provided | `uni2h_cptac_nsclc.csv` |
| BRCA | TCGA-BRCA, IDC (0) vs other (1) | 1,126 (1,055) | GDC [TCGA-BRCA](https://portal.gdc.cancer.gov/projects/TCGA-BRCA) | [UNI2-h-features](https://huggingface.co/datasets/MahmoodLab/UNI2-h-features): `TCGA/TCGA-BRCA_IDC`, `TCGA/TCGA-BRCA_OTHERS` | [CPathPatchFeature](https://huggingface.co/datasets/Dearcat/CPathPatchFeature): `brca` (1,114 slides) | `uni2h_tcga_brca.csv`, `gigapath_tcga_brca.csv` |
| CRC | TCGA-COAD (0) vs TCGA-READ (1) | 600 (591) | GDC [TCGA-COAD](https://portal.gdc.cancer.gov/projects/TCGA-COAD), [TCGA-READ](https://portal.gdc.cancer.gov/projects/TCGA-READ) | [UNI2-h-features](https://huggingface.co/datasets/MahmoodLab/UNI2-h-features): `TCGA/TCGA-COAD`, `TCGA/TCGA-READ` | not provided | `uni2h_tcga_crc.csv` |
| PANDA | ISUP grade 0–5 (prostate biopsies) | 10,202 | [Kaggle PANDA challenge](https://www.kaggle.com/competitions/prostate-cancer-grade-assessment) | [UNI2-h-features](https://huggingface.co/datasets/MahmoodLab/UNI2-h-features): `PANDA/panda` | [CPathPatchFeature](https://huggingface.co/datasets/Dearcat/CPathPatchFeature): `panda` (9,055 slides) | `uni2h_panda.csv`, `gigapath_panda.csv` |
| Interpretability (Sec. IV-B) | WSSS4LUAD tissue masks | 22 regions of interest | [WSSS4LUAD challenge](https://wsss4luad.grand-challenge.org/) | – | – | – |

**Feature sources**

* **UNI2-h**: [MahmoodLab/UNI2-h-features](https://huggingface.co/datasets/MahmoodLab/UNI2-h-features)
  (gated: accept the terms on the dataset page, then `huggingface-cli login`). Encoder:
  [MahmoodLab/UNI2-h](https://huggingface.co/MahmoodLab/UNI2-h). One `.h5` per slide with
  `features` (1, N, 1536) and `coords` (1, N, 2).
* **GigaPath**: [Dearcat/CPathPatchFeature](https://huggingface.co/datasets/Dearcat/CPathPatchFeature)
  (Tang et al.; code at [E2E-WSI-ABMILX](https://github.com/DearCaat/E2E-WSI-ABMILX)). Encoder:
  [prov-gigapath](https://huggingface.co/prov-gigapath/prov-gigapath). Features are stored as
  `<dataset>/gigap/pt_files/<slide>.pt` (N, 1536), with coordinates in `<dataset>/patches/<slide>.h5`.
  This release covers TCGA-NSCLC, TCGA-BRCA and PANDA. The split files contain only slides that
  have both a feature file and a coordinate file.
* **Other cohorts or encoders**: extract features with
  [TRIDENT](https://github.com/mahmoodlab/TRIDENT) (or [CLAM](https://github.com/mahmoodlab/CLAM))
  and save one `.h5` per slide with `features` and `coords` datasets, shaped (N, D) or (1, N, D).
  Then point a split file at them: copy one of ours and replace `feature_path`. Set `--d_in` if
  D ≠ 1536.

**Download**

```bash
# UNI2-h (needs Hugging Face access to the gated dataset)
python scripts/download_features.py --encoder uni2h --tasks nsclc cptac brca crc panda --out /data/UNI2-h-features
# GigaPath
python scripts/download_features.py --encoder gigapath --tasks nsclc brca panda --out /data/CPathPatchFeature
```

The UNI2-h archives are large: about 71 GB for TCGA-NSCLC, 100 GB for CPTAC, 64 GB for BRCA,
22 GB for CRC and 32 GB for PANDA. The script extracts each archive to
`<out>/<TCGA|CPTAC|PANDA>/<archive name>/<slide>.h5`, which is the layout the split files expect.

**Split files** (`splits/*.csv`). Columns: `slide_id, case_id, label, label_str, fold,
feature_path[, coords_path]`. Paths are relative to the feature root you pass with
`--feature_root`.

* TCGA folds come from `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)`,
  stratified by label and grouped by patient (`case_id`), so all slides of a patient fall in
  the same fold.
* PANDA folds are stratified by ISUP grade (one slide per case).
* CPTAC is external only (`fold = -1`).
* PANDA uses the curated 10,202-slide label table from
  [E2E-WSI-ABMILX](https://github.com/DearCaat/E2E-WSI-ABMILX). The official training set has
  10,616 slides.
* The GigaPath split files keep each slide's fold from the UNI2-h split.

## Training and evaluation

Commands are run from the repository root. Point `UNI2H_ROOT` and `GIGAPATH_ROOT` at the feature
roots.

**One fold of ROAM** (NSCLC, UNI2-h, CPTAC external test):

```bash
python -m src.train \
    --manifest splits/uni2h_tcga_nsclc.csv --external_manifest splits/uni2h_cptac_nsclc.csv \
    --feature_root $UNI2H_ROOT --fold 0 --output_dir outputs/uni2h/nsclc/roam/fold0
```

The defaults in `src/train.py` are the paper configuration.

**5-fold cross-validation** for any task and encoder:

```bash
export UNI2H_ROOT=/data/UNI2-h-features GIGAPATH_ROOT=/data/CPathPatchFeature
bash scripts/run_cv.sh nsclc uni2h                # also scores CPTAC with every fold model
bash scripts/run_cv.sh brca uni2h
bash scripts/run_cv.sh panda gigapath
bash scripts/run_ablations.sh                     # Table II variants (NSCLC, UNI2-h)
python -m scripts.summarize --out outputs         # mean ± std over folds, internal + CPTAC
```

Each fold writes `config.json`, `history.csv`, `best_model.pt`, `summary.json` and
`val_preds.csv` to `outputs/<encoder>/<task>/roam/fold<k>/`. With an external set it also
writes `external_metrics.json` and `external_preds.csv`. Folds run one after another; on a
cluster, submit one job per fold with `FOLDS=<k>`.

**Re-scoring a trained run.** `src.eval` writes `eval_*` files and does not touch the training
outputs.

```bash
python -m src.eval --run_dir outputs/uni2h/nsclc/roam/fold0 --feature_root $UNI2H_ROOT
python -m src.eval --run_dir outputs/uni2h/nsclc/roam/fold0 --feature_root $UNI2H_ROOT --max_patches 0   # full bags
```

**Computational cost** (Sec. IV-A; parameters, GFLOPs and latency at N = 10,000 patches):

```bash
python -m scripts.benchmark_cost --manifest splits/uni2h_tcga_nsclc.csv --feature_root $UNI2H_ROOT --n 10000
```

The paper reports 3.31 M parameters, 16.4 GFLOPs and 8.9 ms per slide for ROAM, measured on
one NVIDIA A100 40 GB GPU (aggregator only, N = 10,000 patches, latency = mean of 20 forward
passes at batch size 1). Latency depends on the GPU.

**Interpretability** (Sec. IV-B). Per-patch routing of selected slides, for maps like Fig. 2
(colour each patch by `expert` at its `x, y`):

```bash
python -m scripts.export_routing --run_dir outputs/uni2h/nsclc/roam/fold0 \
    --manifest splits/uni2h_cptac_nsclc.csv --feature_root $UNI2H_ROOT \
    --slides <slide_id> [<slide_id> ...] --out routing.csv
```

Tissue association of the experts on WSSS4LUAD: download the challenge data, embed its annotated
validation regions with UNI2-h (needs `timm` and access to the gated UNI2-h weights), then score
each expert's dispatch weight against the tissue masks:

```bash
python -m scripts.interpretability.extract_wsss4luad_uni2h --wsss_root /data/WSSS4LUAD --out_dir /data/wsss4luad_uni2h
python -m scripts.interpretability.score_wsss4luad_experts --run_dir outputs/uni2h/nsclc/roam/fold0 \
    --feat_dir /data/wsss4luad_uni2h --out wsss4luad_expert_auc.csv
```

### ROAM configuration

| | flag | value |
|---|---|---|
| working dim d | `--d_proj` | 512 |
| region tokens M (grid) | `--n_tokens` | 256 (16×16) |
| region kNN graph | `--k_neighbors` | 8 |
| routing GNN | `--gnn_layers / --gnn_heads / --gnn_dropout` | 2 GATv2 layers, 4 heads, 0.1 |
| experts E, top-k | `--num_experts / --topk` | 8, 2 |
| OT ε, Sinkhorn updates T | `--ot_eps / --sinkhorn_iters` | 0.1, 20 |
| graph regularisation λ_s, steps | `--lam_smooth / --smooth_iters` | 0.1, 3 |
| supply rarity boost | `--lam_rare` | 0.2 |
| expert heads, expert attention dim | `--expert_heads`, `--d_att` | 2, 64 (= d_att / 4) |
| routing-profile embedding | `--d_comp` | 64 |
| optimiser | `--lr / --weight_decay` | AdamW, 5e-4, 1e-4 |

Ablations (Table II), each changing one setting of the full model:

| Variant | flag |
|---|---|
| I. w/o routing GNN (Z = region features; graph regularisation kept) | `--no_routing_gnn` |
| II. w/o graph regularisation (plain Sinkhorn, 20 updates) | `--smooth_iters 0` |
| III. softmax routing (no capacity constraint) | `--routing softmax` |
| IV. w/o OT-guided pooling (no log γ term in expert attention) | `--no_ot_modulation` |
| V. OT-guided pooling with detached routing weights | `--ot_mod_detach` |

Softmax routing also removes the graph term (which lives inside Sinkhorn), and its dispatch rows
sum to 1 rather than to the region supply. Without the log γ term, each expert attends over all
regions rather than only the regions dispatched to it.

### Protocol details

* **Training.** AdamW; linear warm-up from 0.1 × lr over 5 epochs, then cosine decay over the
  remaining epochs (stepped once per epoch). Batch size 1 slide, dropout 0.25, gradient clipping
  at norm 1.0, at most 200 epochs, early stopping with patience 20.
* **Loss.** Binary tasks use one logit with `BCEWithLogitsLoss`; PANDA uses six-way
  cross-entropy.
* **Model selection.** Early stopping and checkpoint selection use the primary metric (AUC, or
  QWK for PANDA) on the held-out fold. That same fold's score is reported as the internal
  result. For NSCLC, the selected checkpoint of every fold is then scored on all 2,206 CPTAC
  slides.
* **Bag size.** Bags larger than `--max_patches` (default 4,096) are randomly subsampled every
  epoch during training. Evaluation, both in `src.train` and by default in `src.eval`, keeps the
  first 4,096 patches in file order. `src.eval --max_patches 0` scores full bags.
* **Metrics.** Slide-level AUC for binary tasks, quadratic weighted kappa for PANDA. The
  summaries report mean ± population std over the five folds.
* **Reproducibility.** Seeds are fixed (`--seed 0`). Some CUDA kernels (scatter-add, index
  backward) are non-deterministic, so GPU training reruns can differ slightly from run to run.
  Re-scoring a checkpoint on GPU reproduces its stored predictions; on CPU, floating-point
  differences change the predictions of a small fraction of slides, and AUC can move in the
  fourth decimal.

## Citation

```bibtex
@inproceedings{tian2026roam,
  title     = {Region-Graph Optimal Transport Routing for Mixture-of-Experts
               Whole-Slide Image Classification},
  author    = {Tian, Xin and Lu, Jiuliu and Tsalik, Ephraim and
               Wanders, Bart and Knoth, Colleen and Knight, Julian},
  booktitle = {Proc. IEEE Int. Conf. Bioinformatics and Biomedicine (BIBM)},
  year      = {2026}
}
```

Please also cite the feature and data sources listed above (UNI2-h, GigaPath, CPathPatchFeature,
TCGA, CPTAC, PANDA).

## Contact

Xin Tian (xin.tian@well.ox.ac.uk), Centre for Human Genetics, University of Oxford.

## Compared methods

Table I of the paper compares ROAM with the methods below, on the same features and splits and
with the same training loop (AdamW, lr 2e-4, weight decay 1e-5; MambaMIL with at most 512
patches per bag).

| Method | Paper | Code |
|---|---|---|
| MeanPool / MaxPool | – | – |
| ABMIL | Ilse et al., [Attention-based deep multiple instance learning](https://proceedings.mlr.press/v80/ilse18a.html), ICML 2018 | [AMLab-Amsterdam/AttentionDeepMIL](https://github.com/AMLab-Amsterdam/AttentionDeepMIL) |
| CLAM-SB / CLAM-MB | Lu et al., [Data-efficient and weakly supervised computational pathology on whole-slide images](https://doi.org/10.1038/s41551-020-00682-w), Nat. Biomed. Eng. 2021 | [mahmoodlab/CLAM](https://github.com/mahmoodlab/CLAM) |
| DSMIL | Li et al., [Dual-stream multiple instance learning network for whole slide image classification with self-supervised contrastive learning](https://arxiv.org/abs/2011.08939), CVPR 2021 | [binli123/dsmil-wsi](https://github.com/binli123/dsmil-wsi) |
| DTFD-MIL | Zhang et al., [DTFD-MIL: Double-tier feature distillation multiple instance learning for histopathology whole slide image classification](https://arxiv.org/abs/2203.12081), CVPR 2022 | [hrzhang1123/DTFD-MIL](https://github.com/hrzhang1123/DTFD-MIL) |
| ACMIL | Zhang et al., [Attention-challenging multiple instance learning for whole slide image classification](https://arxiv.org/abs/2311.07125), ECCV 2024 | [dazhangyu123/ACMIL](https://github.com/dazhangyu123/ACMIL) |
| TransMIL | Shao et al., [TransMIL: Transformer based correlated multiple instance learning for whole slide image classification](https://arxiv.org/abs/2106.00908), NeurIPS 2021 | [szc19990412/TransMIL](https://github.com/szc19990412/TransMIL) |
| ILRA | Xiang et al., [Exploring low-rank property in multiple instance learning for whole slide image classification](https://openreview.net/forum?id=01KmhBsEPFO), ICLR 2023 | [jinxixiang/low_rank_wsi](https://github.com/jinxixiang/low_rank_wsi) |
| RRT-MIL | Tang et al., [Feature re-embedding: Towards foundation model-level performance in computational pathology](https://arxiv.org/abs/2402.17228), CVPR 2024 | [DearCaat/RRT-MIL](https://github.com/DearCaat/RRT-MIL) |
| MambaMIL | Yang et al., [MambaMIL: Enhancing long sequence modeling with sequence reordering in computational pathology](https://arxiv.org/abs/2403.06800), MICCAI 2024 | [isyangshu/MambaMIL](https://github.com/isyangshu/MambaMIL) |
| TAD-Graph | Wang et al., [TAD-Graph: Enhancing whole slide image analysis via task-aware subgraph disentanglement](https://doi.org/10.1109/TMI.2025.3545680), IEEE TMI 2025 | [HKU-MedAI/TAD-Graph](https://github.com/HKU-MedAI/TAD-Graph) |
| MAMMOTH-ABMIL / MAMMOTH-TransMIL | Shao et al., [Mixture of mini experts: Overcoming the linear layer bottleneck in multiple instance learning](https://openreview.net/forum?id=S5Io33pc78), ICLR 2026 | [mahmoodlab/MAMMOTH](https://github.com/mahmoodlab/MAMMOTH) |
| PAMoE | Wu et al., [Learning heterogeneous tissues with mixture of experts for gigapixel whole slide images](https://openaccess.thecvf.com/content/CVPR2025/html/Wu_Learning_Heterogeneous_Tissues_with_Mixture_of_Experts_for_Gigapixel_Whole_CVPR_2025_paper.html), CVPR 2025 | [wjx-error/PAMoE](https://github.com/wjx-error/PAMoE) |
| OTSurv (used as a slide classifier) | Ren et al., [OTSurv: A novel multiple instance learning framework for survival prediction with heterogeneity-aware optimal transport](https://arxiv.org/abs/2506.20741), MICCAI 2025 | [Y-Research-SBU/OTSurv](https://github.com/Y-Research-SBU/OTSurv) |
