#!/usr/bin/env bash
# ROAM: Region-Graph Optimal Transport Routing for Mixture-of-Experts WSI Classification
# Xin Tian et al., IEEE BIBM 2026 | https://arxiv.org/abs/2604.07298 | https://github.com/xintian-99/ROAM
# ROAM ablations (paper Table II): NSCLC, UNI2-h, CPTAC external test.
# Each variant changes one setting of the full model.
#
#   UNI2H_ROOT=/data/UNI2-h-features bash scripts/run_ablations.sh
set -euo pipefail

declare -A VARIANTS=(
  [_no_routing_gnn]="--no_routing_gnn"        # I   w/o routing GNN (Z = H0)
  [_no_graph_reg]="--smooth_iters 0"           # II  w/o graph regularisation
  [_softmax_routing]="--routing softmax"       # III softmax routing, no capacity constraint
  [_no_ot_pooling]="--no_ot_modulation"        # IV  w/o OT-guided pooling
  [_detach_routing]="--ot_mod_detach"          # V   OT-guided pooling, detached weights
)

for tag in "${!VARIANTS[@]}"; do
  # shellcheck disable=SC2086
  TAG=$tag bash scripts/run_cv.sh nsclc uni2h ${VARIANTS[$tag]}
done
