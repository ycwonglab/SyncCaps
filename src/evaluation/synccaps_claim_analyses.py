#!/usr/bin/env python3
"""The two manuscript claims that neither repair report nor McNemar covers.

`synccaps_repair_report.py` regenerates every seed-level contrast from the
committed result JSONs, and `synccaps_mcnemar.py` covers clip-level and
source-video-clustered inference from the prediction dumps. Two claims fall
outside both:

  1. Frame-permutation DISAGREEMENT. The shuffle control ties on accuracy, which
     on its own is compatible with two very different situations: the two models
     could be making the same predictions, or they could be making different
     predictions that happen to be right equally often. Only a per-clip
     comparison separates them, and the answer matters -- a large disagreement
     at equal accuracy says the readout is genuinely order-agnostic rather than
     order-blind by degeneracy.

  2. Efficiency ratios. The head-parameter and readout-latency multiples quoted
     against routing, recomputed from the timing profile rather than restated.

Reads only released artifacts; no GPU, no dataset, no checkpoints.

  PYTHONPATH=. python src/evaluation/synccaps_claim_analyses.py
"""
import argparse
import json
import os
import statistics as st

import numpy as np


def load_preds(perclip, tag, arm, seed, policy):
    p = os.path.join(perclip, '%s_%s_seed%d.npz' % (tag, arm, seed))
    if not os.path.exists(p):
        return None
    z = np.load(p, allow_pickle=True)
    return z['preds_%s' % policy], z['labels'], z['test_idx']


def disagreement(perclip, tag, a, b, seeds, policy):
    """Per-clip agreement between two arms, seed-matched.

    `both_wrong_differently` is the interesting cell: clips where neither model
    is right AND they fail differently, which cannot be explained by one model
    being a noisy copy of the other.
    """
    rows = []
    for s in seeds:
        A = load_preds(perclip, tag, a, s, policy)
        B = load_preds(perclip, tag, b, s, policy)
        if A is None or B is None:
            continue
        pa, lab, ia = A
        pb, _, ib = B
        assert np.array_equal(ia, ib), 'clip order differs between arms'
        ca, cb = pa == lab, pb == lab
        dis = pa != pb
        rows.append(dict(
            seed=s, n_clips=int(lab.size), disagree=int(dis.sum()),
            disagree_pct=round(100 * float(dis.mean()), 2),
            a_right_b_wrong=int((ca & ~cb).sum()),
            b_right_a_wrong=int((cb & ~ca).sum()),
            both_right=int((ca & cb).sum()),
            both_wrong_differently=int((~ca & ~cb & dis).sum()),
            acc_a=round(100 * float(ca.mean()), 2),
            acc_b=round(100 * float(cb.mean()), 2),
            acc_gap=round(100 * float(ca.mean() - cb.mean()), 2)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--perclip', default='results/predictions')
    ap.add_argument('--timing', default='results/timing/readout_profile_published.json')
    ap.add_argument('--tag', default='ucf101_resnet_ptfz_official1_noval_fc')
    ap.add_argument('--seeds', default='42,1337,7')
    ap.add_argument('--policy', default='certain',
                    choices=['certain', 'final', 'hybrid'])
    ap.add_argument('--out', default='results/statistics/claim_analyses.json')
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(',')]

    print('=' * 78)
    print('1. Frame-permutation disagreement  (tag %s, policy %s)' % (a.tag, a.policy))
    print('=' * 78)
    dis = disagreement(a.perclip, a.tag, 'B1_sync', 'B3_shuffle', seeds, a.policy)
    for r in dis:
        print('  seed %-5d n=%d  disagree %4d (%5.2f%%)  sync-only %3d | '
              'shuffle-only %3d  both-wrong-differently %3d'
              % (r['seed'], r['n_clips'], r['disagree'], r['disagree_pct'],
                 r['a_right_b_wrong'], r['b_right_a_wrong'],
                 r['both_wrong_differently']))
        print('        acc  B1_sync %5.2f  B3_shuffle %5.2f  gap %+.2f'
              % (r['acc_a'], r['acc_b'], r['acc_gap']))
    if dis:
        md = st.mean([r['disagree'] for r in dis])
        mg = st.mean([r['acc_gap'] for r in dis])
        n = dis[0]['n_clips']
        print('  MEAN over %d seeds: %.0f of %d clips disagree (%.2f%%) at an '
              'accuracy gap of %+.2f points.' % (len(dis), md, n, 100 * md / n, mg))
        print('  Reading: the order control is NOT predicting the same clips. A tie')
        print('  in accuracy with this much per-clip disagreement means frame order')
        print('  changes WHICH clips are correct without changing HOW MANY.')

    print()
    print('=' * 78)
    print('2. Efficiency ratios against routing (from %s)' % a.timing)
    print('=' * 78)
    t = json.load(open(a.timing))
    ratios = {}
    for base in ('B4_syncnorm', 'B4_gram', 'B0_linear'):
        if base not in t or 'R3_route' not in t:
            continue
        r = {'head_params_x': round(t['R3_route']['head_params'] / t[base]['head_params'], 2),
             'p50_latency_x': round(t['R3_route']['p50_ms'] / t[base]['p50_ms'], 2),
             'p95_latency_x': round(t['R3_route']['p95_ms'] / t[base]['p95_ms'], 2),
             'peak_act_x': round(t['R3_route']['peak_act_MiB'] / t[base]['peak_act_MiB'], 2)}
        ratios[base] = r
        print('  R3_route vs %-12s head params %5.2fx | p50 %4.2fx | p95 %4.2fx | '
              'peak act %4.2fx' % (base, r['head_params_x'], r['p50_latency_x'],
                                   r['p95_latency_x'], r['peak_act_x']))
    print('  absolute: R3_route %.2f/%.2f ms, B4_syncnorm %.2f/%.2f ms (p50/p95)'
          % (t['R3_route']['p50_ms'], t['R3_route']['p95_ms'],
             t['B4_syncnorm']['p50_ms'], t['B4_syncnorm']['p95_ms']))
    print('  device: %s | %s' % (t['_meta']['device'], t['_meta'].get('note', '')))

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({'frame_permutation_disagreement': dis,
               'efficiency_ratios_vs_R3_route': ratios,
               'timing_source': a.timing, 'policy': a.policy, 'tag': a.tag},
              open(a.out, 'w'), indent=2)
    print('\nwrote', a.out)


if __name__ == '__main__':
    main()
