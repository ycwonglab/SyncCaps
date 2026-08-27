"""synccaps_matrix_table.py — assemble the same-stem readout matrix table.

Reads the per-arm result JSONs for ONE cell (backbone x dataset x split) and
prints the table in the form the paper needs: mean +- SD over seeds, the paired
per-seed delta against a reference arm, the win count, and the trainable /
head-width columns that make the cost comparison honest.

Two rules this script enforces so the table cannot quietly lie:

  * ONE metric family. Every number is `test_acc_certain`, single view. The
    project's own ledger warns that test_acc_certain (non-causal, argmax over
    all T certainties) and test_acc_hybrid (causal, prefix-mean + theta exit)
    are different measurements, and that an exit-tick figure must never be
    paired with a certain-tick accuracy. Mixing them across a row would be the
    easiest possible way to manufacture a result.
  * PAIRED deltas. Seeds are matched arm-to-arm before differencing, so the
    reported delta is the mean of per-seed differences, not the difference of
    means, and the win count is over seeds rather than over runs.

    python synccaps_matrix_table.py
"""
import glob
import json
import os
import statistics as st

CELL = 'ucf101_clip_b32_ptfz_official1_noval_fc'
REF = 'B0_linear'
# Display order: first-order control, then the sync family, then the external
# second-order baselines, then routing.
ORDER = ['B0_linear', 'B1_sync', 'B4_syncnorm', 'B4_gram',
         'CB_tsketch', 'LR_bilinear', 'R3_route']
WIDTH = {'B0_linear': 2304, 'B1_sync': 2048, 'B4_syncnorm': 2048,
         'B4_gram': 2048, 'CB_tsketch': 2048, 'LR_bilinear': 2080,
         'R3_route': None}
NOTE = {
    'B0_linear':   'per-frame linear probe (first order)',
    'B1_sync':     'sampled pairs, learned decay, no normalisation',
    'B4_syncnorm': 'sampled pairs + signed-sqrt/L2 (headline)',
    'B4_gram':     'as B4, decay pinned rho = 0 (sampled Gram)',
    'CB_tsketch':  'compact bilinear / Tensor Sketch [Gao+ 16]',
    'LR_bilinear': 'full Gram of a learned rank-64 projection',
    'R3_route':    'dynamic routing r = 3 [Sabour+ 17]',
}


def load(cell):
    """arm -> ({seed: acc}, lr), merged over every result file for this cell.

    The learning rate is carried out of each file's `_config` and DISPLAYED,
    because the arms in this cell do not all share one. R3_route was tuned on a
    held-out validation split and selected 5e-4 while the sync arms use the
    project default 1e-3; a table that silently averaged over that would hide
    the single most attackable asymmetry in the comparison.
    """
    out, lrs = {}, {}
    for f in glob.glob(f'gating_results/synccaps_{cell}_*.json'):
        blob = json.load(open(f))
        lr = (blob.get('_config') or {}).get('lr')
        for arm, runs in blob.items():
            if arm == '_config' or not isinstance(runs, list):
                continue
            key = arm if lr in (None, 1e-3) else f'{arm}@lr{lr:g}'
            out.setdefault(key, {}).update(
                {r['seed']: r['test_acc_certain'] for r in runs})
            lrs[key] = lr
    return out, lrs


def fmt(vals):
    m = st.mean(vals)
    s = st.stdev(vals) if len(vals) > 1 else 0.0
    return f'{m:.2f} ± {s:.2f}'


if __name__ == '__main__':
    acc, lrs = load(CELL)
    if REF not in acc:
        raise SystemExit(f'reference arm {REF} not found in gating_results '
                         f'for cell {CELL}; found {sorted(acc)}')
    ref = acc[REF]
    hdr = (f'{"Readout":22s} {"lr":>7s} {"width":>6s} {"top-1 (%)":>16s} '
           f'{"vs B0":>16s}  note')
    print(f'\n=== same-stem readout matrix: {CELL} ===')
    print(hdr); print('-' * len(hdr))
    # An arm may appear only under an lr-suffixed key (e.g. R3_route@lr0.0005).
    # Expand each ORDER entry to whatever variants exist so a tuned arm is not
    # reported as "not yet run" while its results sit one key away.
    ordered = []
    for arm in ORDER:
        variants = [k for k in sorted(acc) if k == arm or k.startswith(arm + '@')]
        ordered += variants or [arm]
    ordered += [a for a in sorted(acc) if a not in ordered]
    for arm in ordered:
        if arm not in acc:
            print(f'{arm:22s} {"":>7s} {"":>6s} {"(not yet run)":>16s}')
            continue
        seeds = sorted(acc[arm])
        vals = [acc[arm][s] for s in seeds]
        shared = [s for s in seeds if s in ref]
        if arm == REF or not shared:
            delta = ''
        else:
            d = [acc[arm][s] - ref[s] for s in shared]
            delta = f'{st.mean(d):+.2f} ({sum(x > 0 for x in d)}/{len(d)})'
        w = WIDTH.get(arm.split('@')[0])
        lr = lrs.get(arm) or 1e-3
        print(f'{arm:22s} {lr:>7.0e} {str(w or "-"):>6s} {fmt(vals):>16s} '
              f'{delta:>16s}  {NOTE.get(arm.split("@")[0], "")}')
        if len(seeds) < 3:
            print(f'{"":22s} {"":>7s} {"":>6s} INCOMPLETE: seeds {seeds}')
    print('\nsingle view, test_acc_certain, 3 seeds (42/1337/7); deltas are '
          'paired per seed.')
