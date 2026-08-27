"""synccaps_route_probe.py — tune the R3_route baseline BEFORE the matrix runs.

Why this exists. The same-stem readout matrix (docs/paper/
SyncCaps_v5_novelty_experiments.md #2) is only worth reporting if every
baseline was given a fair budget: "do not use default routing hyperparameters
without validation". Two specific hazards make routing the arm that needs it:

  1. Vote-tensor init. src/models/sync_caps.py:DynamicRoutingCaps notes that the
     init std must be re-tuned per (caps_grid, num_classes). At 101 classes an
     under-scaled init leaves every class capsule at the same small norm, the
     softmax over classes stays uniform, and training parks at ln(101) = 4.615
     with dead capsules -- a silent strawman that looks like a real result.
  2. Learning rate. Every sync arm uses 1e-3. Routing carries 3.72 M vote
     parameters against the sync head's 0.21 M, so the matched LR is not
     obviously the right LR for it.

SELECTION IS ON VALIDATION, NEVER TEST. This probe therefore runs the
official-split-1 protocol WITH val groups (8, 9, 10) held out, whereas the
final matrix runs use SYNC_SPLIT_NOVAL=1 and train on all 9537 clips as
published methods do. test_acc_* is computed by the shared runner and is
recorded here for the archive, but the selection rule is best_val and only
best_val.

    python synccaps_route_probe.py --stem clip_b32
"""
import argparse
import json
import os

from exp_base import UCF11VideoDataset, FrozenFeatureDataset
from synccaps_followup_experiment import DATASETS, make_arms
from synccaps_probe_experiment import run_experiment_sync

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=list(DATASETS), default='ucf101')
    ap.add_argument('--stem', default='clip_b32')
    ap.add_argument('--arm', default='R3_route')
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--nsynch', type=int, default=2048)
    ap.add_argument('--scales', default='20,60,100',
                    help='w_init_scale grid (std = 0.01 x scale)')
    ap.add_argument('--lrs', default='1e-3,5e-4')
    args = ap.parse_args()

    # Force the WITH-val protocol: selecting on a split the final runs also
    # train on would be test-adjacent leakage dressed up as tuning.
    os.environ['SYNC_SPLIT'] = 'official1'
    os.environ.pop('SYNC_SPLIT_NOVAL', None)

    path, ncls = DATASETS[args.dataset]
    ds = UCF11VideoDataset(path, sequence_length=16, sample_fps=5.0,
                           augment=False, cache_dir='.cache')
    ds = FrozenFeatureDataset(ds, args.stem,
                              os.environ.get('FEAT_DIR', '.featcache'))
    missing = sum(1 for s in ds.samples[:64] if not ds.feat_path(s[0]).exists())
    if missing:
        raise SystemExit(f'--feat-cache: {missing}/64 clips lack a cached '
                         f'{args.stem} feature; run synccaps_precompute_stem.py')

    # Defined here, above its first use: the output path depends on it.
    is_routing = args.arm == 'R3_route'
    out = (f'gating_results/synccaps_routeprobe_{args.dataset}_{args.stem}.json'
           if is_routing else
           f'gating_results/synccaps_lrprobe_{args.dataset}_{args.stem}_{args.arm}.json')
    os.makedirs('gating_results', exist_ok=True)
    probe = json.load(open(out)) if os.path.exists(out) else {}
    # w_init_scale is a ROUTING knob; make_arms passes it only to R3_route. For
    # any other arm the scale loop would retrain identical models under
    # different keys, so collapse it to one and keep it out of the key. The LR
    # grid stays the same across arms -- that shared grid is what makes the
    # tuning budget comparable, which is the whole point of probing the sync
    # arms at all (2026-08-20: routing was tuned and the sync arms were not,
    # and the asymmetry only became visible once the baseline won).
    scales = [float(x) for x in args.scales.split(',')] if is_routing else [0.0]
    for scale in scales:
        for lr in [float(x) for x in args.lrs.split(',')]:
            key = (f'{args.arm}_ws{scale:g}_lr{lr:g}' if is_routing
                   else f'{args.arm}_lr{lr:g}')
            if key in probe:
                print(key, 'already done -> skip', flush=True)
                continue
            arms = make_arms(args.stem, n_synch=args.nsynch, pretrained=True,
                             freeze_stem=True, feat_cache=True,
                             route_w_scale=scale if is_routing else 20.0)
            print('==', key, flush=True)
            probe[key] = run_experiment_sync(ds, ncls, args.arm, seed=args.seed,
                                             epochs=args.epochs, lr=lr, arms=arms)
            probe[key]['w_init_scale'] = scale
            probe[key]['lr'] = lr
            with open(out, 'w') as f:
                json.dump(probe, f, indent=1)
            print('  ', {k: round(v, 3) for k, v in probe[key].items()
                         if isinstance(v, float)}, flush=True)

    best = max(probe.items(), key=lambda kv: kv[1]['best_val'])
    print(f'\nSELECTED ON VAL: {best[0]}  best_val={best[1]["best_val"]:.3f}',
          flush=True)
    print('probe done ->', out, flush=True)
