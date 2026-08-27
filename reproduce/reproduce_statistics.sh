#!/usr/bin/env bash
# Re-derive every statistical result from the released prediction dumps.
#
# This is the cheapest check in the bundle: it needs NO dataset and NO
# checkpoints. The dumps in results/predictions/ carry clip ids and
# source-video group ids, so both the clip-level and the group-clustered
# bootstrap run directly off them.
#
# Runtime: a few minutes (100,000 bootstrap resamples per comparison).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.

OUT=results/statistics
mkdir -p "$OUT"

for POLICY in certain hybrid final; do
  echo "=== policy: $POLICY ==="
  python src/evaluation/synccaps_mcnemar.py \
    --policy "$POLICY" \
    --cluster-groups \
    --perclip results/predictions \
    --frozen-tag ucf101_resnet_ptfz_official1_noval_fc \
    --ft-tag    ucf101_resnet_pt_official1_noval \
    | tee "$OUT/mcnemar_${POLICY}_frozenBNfix.txt"
done

echo
echo "Outputs written to $OUT/. These are the files quoted in the manuscript's"
echo "inference section: seed-level paired intervals, clip-level McNemar,"
echo "clip-level and source-video-clustered bootstrap, both one-sided TOST"
echo "statistics, and the Bonferroni family summary."
