#!/usr/bin/env bash
# Regenerate the manuscript figures from the released checkpoint.
#
# REQUIRES: UCF101 on disk, plus the figure checkpoint
#   synccaps_ucf101_resnet_ptfz_official1_noval_fc_B4_syncnorm_seed42.pt
# from the GitHub Release, placed in checkpoints/.
#
# Verify it first — the figures are only meaningful on this exact file:
#   grep B4_syncnorm_seed42 checkpoints/CHECKSUMS.sha256 | sha256sum -c -
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.

CKPT=checkpoints/synccaps_ucf101_resnet_ptfz_official1_noval_fc_B4_syncnorm_seed42.pt
[ -f "$CKPT" ] || { echo "missing $CKPT (download from the Release)"; exit 1; }

mkdir -p figures/rendered
python figures/scripts/make_tick_strategy_figure.py figures/rendered
python figures/scripts/make_neuron_dynamics_figure.py
python figures/scripts/make_activity_web_grid_peaktick.py

echo
echo "Expected header values (see figures/FIGURE_MANIFEST.json):"
echo "  accuracy 71.82 %   mean exit tick 11.10   median e^-rho 0.8093"
