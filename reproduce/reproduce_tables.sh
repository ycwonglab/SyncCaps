#!/usr/bin/env bash
# Retrain and re-score the arms behind a results table.
#
# REQUIRES: UCF101 (and/or UCF-11) on disk, and a GPU. This is the expensive
# path; to CHECK a number rather than regenerate it, read
# results/seed_summaries/seed_level_results.csv or run reproduce_statistics.sh.
#
# Usage:
#   UCF101_ROOT=/data/UCF101_full bash reproduce/reproduce_tables.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.

: "${UCF101_ROOT:?set UCF101_ROOT to your UCF101 directory}"

# The paper's PRIMARY comparisons are on the frozen ResNet-18 layer3 stem, not
# on CLIP. The CLIP cell below is a rung of the backbone ladder (Section 5.1),
# not the headline. Both are given.
#
# Configs:
#   configs/resnet18/ucf101_resnet_ptfz_official1_noval_fc__B4_syncnorm.json
#   configs/resnet18/ucf101_resnet_ptfz_official1_noval_fc__B4_gram.json
#   configs/resnet18/ucf101_resnet_ptfz_official1_noval_fc__CB_tsketch.json
#   configs/clip/ucf101_clip_b32_ptfz_official1_noval_fc__B4_syncnorm.json
export SYNC_SPLIT=official1
export SYNC_SPLIT_NOVAL=1
# SYNC_FROZEN_BN_TRAIN is deliberately UNSET: leaving it unset keeps frozen
# stems' BatchNorm in eval mode and arms the assertion that catches the
# 2026-08-21 defect. Set it to 1 only to reproduce a superseded run.

# --- headline: frozen ResNet-18 layer3 -------------------------------------
# exact-vs-Tensor-Sketch is the nine-seed comparison; it was run as two batches
# (see the seed_batches array in each config).
python src/training/synccaps_followup_experiment.py \
  --dataset ucf101 --stem resnet --mode full \
  --arms B0_linear,B1_sync,B4_syncnorm,B4_gram,CB_tsketch \
  --seeds 42,1337,7 --nsynch 2048 --lr 1e-3 \
  --pretrained --freeze-stem --feat-cache

python src/training/synccaps_followup_experiment.py \
  --dataset ucf101 --stem resnet --mode full \
  --arms B4_gram,CB_tsketch \
  --seeds 5,11,19,23,101,2026 --nsynch 2048 --lr 1e-3 \
  --pretrained --freeze-stem --feat-cache

# --- backbone ladder rung: frozen CLIP ViT-B/32 ----------------------------
python src/training/synccaps_followup_experiment.py \
  --dataset ucf101 --stem clip_b32 --mode full \
  --arms B0_linear,B1_sync,B4_syncnorm \
  --seeds 42,1337,7 --nsynch 2048 --lr 1e-3 \
  --pretrained --freeze-stem --feat-cache

echo
echo "Results land in gating_results/. Compare against the committed copies in"
echo "results/seed_summaries/raw_results_json/ ."
