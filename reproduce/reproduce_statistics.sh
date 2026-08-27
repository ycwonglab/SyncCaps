#!/usr/bin/env bash
# Re-derive the statistical results from released artifacts.
#
# Needs NO dataset, NO GPU and NO checkpoints. Three analyses run here:
#
#   1. synccaps_repair_report.py  -- every SEED-LEVEL contrast, from the
#      committed result JSONs: exact-vs-sketch (nine-seed, fresh-seed, CLIP,
#      and the backbone x operator interaction), the routing operational band
#      with TOST, the zero-decay control, pair composition, and the val-carved
#      replication.
#   2. synccaps_mcnemar.py        -- CLIP-LEVEL and SOURCE-VIDEO-CLUSTERED
#      inference from the per-clip prediction dumps.
#   3. synccaps_claim_analyses.py -- frame-permutation disagreement and the
#      head-parameter / latency ratios.
#
# Together these cover every row of provenance/claim_evidence_map.csv. They do
# NOT retrain anything and do NOT regenerate figures; see the other two scripts.
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

echo "=== seed-level contrasts (single view and three-clip) ==="
python src/evaluation/synccaps_repair_report.py \
  | tee "$OUT/repair_report_singleview.txt"
python src/evaluation/synccaps_repair_report.py --multiclip \
  | tee "$OUT/repair_report_multiclip.txt"

echo "=== frame-permutation disagreement and efficiency ratios ==="
python src/evaluation/synccaps_claim_analyses.py \
  | tee "$OUT/claim_analyses.txt"

echo
echo "Outputs written to $OUT/. Together with"
echo "provenance/claim_evidence_map.csv these cover every interval quoted in"
echo "the manuscript: seed-level paired intervals, clip-level McNemar,"
echo "clip-level and source-video-clustered bootstrap, both one-sided TOST"
echo "statistics, the Bonferroni family summary, exact-vs-sketch and its"
echo "backbone interaction, the routing operational band, pair composition,"
echo "the val-carved replication, and the per-clip disagreement result."
