#!/usr/bin/env bash
# ROAM: Region-Graph Optimal Transport Routing for Mixture-of-Experts WSI Classification
# Xin Tian et al., IEEE BIBM 2026 | https://arxiv.org/abs/2604.07298 | https://github.com/xintian-99/ROAM
# 5-fold cross-validation of ROAM on one task/encoder (folds run sequentially).
#
#   bash scripts/run_cv.sh <task> <encoder> [extra src.train args ...]
#
#   task    : nsclc | brca | crc | panda      (nsclc + uni2h also scores CPTAC)
#   encoder : uni2h | gigapath
#
# Environment:
#   UNI2H_ROOT     root of the extracted MahmoodLab/UNI2-h-features archives
#   GIGAPATH_ROOT  root of the Dearcat/CPathPatchFeature download
#   OUT            output root (default: outputs)
#   FOLDS          folds to run (default: "0 1 2 3 4")
#   TAG            suffix for the run name, e.g. _no_routing_gnn (default: none)
#
# Runs land in $OUT/<encoder>/<task>/roam$TAG/fold<k>/.
set -euo pipefail

TASK=${1:?task}; ENC=${2:?encoder}; shift 2
OUT=${OUT:-outputs}
FOLDS=${FOLDS:-"0 1 2 3 4"}
TAG=${TAG:-}

case $ENC in
  uni2h)    ROOT=${UNI2H_ROOT:?set UNI2H_ROOT}; PREFIX=uni2h ;;
  gigapath) ROOT=${GIGAPATH_ROOT:?set GIGAPATH_ROOT}; PREFIX=gigapath ;;
  *) echo "unknown encoder $ENC" >&2; exit 1 ;;
esac

TASK_ARGS=()
case $TASK in
  nsclc) SPLIT=splits/${PREFIX}_tcga_nsclc.csv
         [[ $ENC == uni2h ]] && TASK_ARGS+=(--external_manifest splits/uni2h_cptac_nsclc.csv) ;;
  brca)  SPLIT=splits/${PREFIX}_tcga_brca.csv ;;
  crc)   SPLIT=splits/${PREFIX}_tcga_crc.csv ;;
  panda) SPLIT=splits/${PREFIX}_panda.csv
         TASK_ARGS+=(--task multiclass --n_classes 6) ;;
  *) echo "unknown task $TASK" >&2; exit 1 ;;
esac
[[ -f $SPLIT ]] || { echo "no split file $SPLIT (no $ENC features for $TASK)" >&2; exit 1; }

RUN=$OUT/$ENC/$TASK/roam$TAG
for k in $FOLDS; do
  echo "=== roam$TAG | $ENC | $TASK | fold $k ==="
  python -m src.train --manifest "$SPLIT" --feature_root "$ROOT" \
      --fold "$k" --output_dir "$RUN/fold$k" \
      ${TASK_ARGS[@]+"${TASK_ARGS[@]}"} "$@"
done
