"""synccaps_multiclip_eval.py — re-score saved checkpoints with multi-clip eval.

Why a separate driver: multi-clip changes only how a TRAINED model is scored, so
it needs no retraining. Every headline checkpoint is already on disk, and
re-scoring them costs minutes of GPU plus a one-off decode of the extra temporal
windows -- against ~8 GPU-hours to retrain the cell.

The gap this closes: at 5 fps a 16-frame clip spans ~3.2 s while UCF101 videos
average ~7 s, so every number in the ledger scored the model on less than half of
each test video. Published UCF101 results are multi-clip (usually multi-crop
too), which is why the ledger attaches a single-view caveat to every comparison
table. This driver measures the size of that caveat instead of asserting it.

BOTH metrics are written. Single-view remains the metric comparable to this
project's own history; multi-clip is the one comparable to the literature. They
must never be mixed in a single table.

    python synccaps_multiclip_eval.py --dataset ucf101 --stem resnet \
        --pretrained --freeze-stem --nsynch 2048 --split official1 --noval \
        --arms B0_linear,B1_sync,B4_syncnorm --seeds 42,1337,7 --views 3
"""
import argparse
import json
import os

import torch
from torch.utils.data import DataLoader, Subset

from src.training.exp_base import (UCF11VideoDataset, FrozenFeatureDataset,
                      make_stratified_splits, make_official_split1, DEVICE)
from src.training.synccaps_followup_experiment import DATASETS, make_arms
from src.training.synccaps_probe_experiment import evaluate_tr, evaluate_multiclip, set_seed

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=list(DATASETS), default='ucf101')
    ap.add_argument('--stem', default='resnet')
    ap.add_argument('--arms', default='B0_linear,B1_sync,B4_syncnorm')
    ap.add_argument('--seeds', default='42,1337,7')
    ap.add_argument('--nsynch', type=int, default=2048)
    ap.add_argument('--views', type=int, default=3)
    ap.add_argument('--eval-seed', type=int, default=None,
                    help='reseed immediately before scoring. Arms with '
                         'shuffle_frames=True draw a fresh torch.randperm INSIDE '
                         'forward(), so their evaluation is STOCHASTIC and two '
                         'scorings of one checkpoint disagree (measured: 1 clip '
                         'in 3783). Pass this to make a dump reproducible. '
                         'Default None leaves every pre-2026-08-21 number '
                         'untouched, since deterministic arms ignore it.')
    ap.add_argument('--bs', type=int, default=4)
    ap.add_argument('--split', choices=['seeded', 'official1'], default='official1')
    ap.add_argument('--noval', action='store_true')
    ap.add_argument('--pretrained', action='store_true')
    ap.add_argument('--freeze-stem', action='store_true')
    ap.add_argument('--feat-cache', action='store_true',
                    help='score through the precomputed frozen-stem features; '
                         'REQUIRED for checkpoints trained that way, since '
                         'their stem is only the 1x1 projection')
    ap.add_argument('--out-ch', type=int, default=256,
                    help='projection width. MUST reach the tag (trap #1): an '
                         'oc512 checkpoint lives at a different path, and '
                         'without this the driver silently scores the oc256 '
                         'checkpoints and writes them to the oc256 output.')
    args = ap.parse_args()

    path, ncls = DATASETS[args.dataset]
    arms_wanted = [a.strip() for a in args.arms.split(',')]
    seeds = [int(s) for s in args.seeds.split(',')]
    pt_tag = ('_ptfz' if args.freeze_stem else '_pt') if args.pretrained else ''
    if args.split == 'official1':
        split_tag = '_official1_noval' if args.noval else '_official1'
    else:
        split_tag = ''
    # `_fc` must reach the tag: it selects both the checkpoint filenames and the
    # output file, and a feature-cached run is a separate experiment on disk.
    fc_tag = '_fc' if args.feat_cache else ''
    oc_tag = '' if args.out_ch == 256 else f'_oc{args.out_ch}'
    tag = f'{args.dataset}_{args.stem}{pt_tag}{split_tag}{fc_tag}{oc_tag}'

    ds = UCF11VideoDataset(path, sequence_length=16, sample_fps=5.0,
                           augment=False, cache_dir='.cache')
    frame_ds = ds
    if args.feat_cache:
        ds = FrozenFeatureDataset(ds, args.stem,
                                  os.environ.get('FEAT_DIR', '.featcache'))
    nw = int(os.environ.get('GATING_NW', '0'))
    pin = nw > 0
    arms = make_arms(args.stem, n_synch=args.nsynch, pretrained=args.pretrained,
                     freeze_stem=args.freeze_stem, feat_cache=args.feat_cache,
                     out_ch=args.out_ch)
    out = f'gating_results/multiclip_{tag}.json'
    os.makedirs('gating_results', exist_ok=True)
    results = json.load(open(out)) if os.path.exists(out) else {}

    for arm in arms_wanted:
        rows = results.setdefault(arm, [])
        done = {r['seed'] for r in rows}
        for seed in seeds:
            if seed in done:
                print(arm, 'seed', seed, 'already scored -> skip', flush=True)
                continue
            ckpt = f'checkpoints/synccaps_{tag}_{arm}_seed{seed}.pt'
            if not os.path.exists(ckpt):
                print(f'{arm} seed {seed}: MISSING {ckpt} -> skip', flush=True)
                continue
            # The split must be rebuilt EXACTLY as at training time or the test
            # partition differs and the re-scored number is meaningless. The
            # seeded protocol REDRAWS its partition from the seed, so set_seed
            # first; official1 is fixed and ignores it.
            set_seed(seed)
            if args.split == 'official1':
                _, _, te = make_official_split1(
                    ds, val_groups=() if args.noval else (8, 9, 10))
            else:
                _, _, te = make_stratified_splits(ds, seed=seed)

            blob = torch.load(ckpt, map_location='cpu', weights_only=False)
            assert blob['arm'] == arm, f"{ckpt} holds arm {blob['arm']}, not {arm}"
            assert blob['seed'] == seed, f"{ckpt} holds seed {blob['seed']}"
            model = arms[arm](ncls).to(DEVICE)
            model.load_state_dict(blob['state_dict'])

            el = DataLoader(Subset(ds, te), args.bs, shuffle=False,
                            num_workers=nw, pin_memory=pin)
            sv = evaluate_tr(model, el)
            # Re-scoring MUST reproduce the committed single-view number. If it
            # does not, the split, the arm config or the checkpoint is wrong --
            # and then the multi-clip number is wrong too. Report the drift
            # rather than quietly publishing a second, different single-view
            # figure alongside the ledger's.
            was = blob['results']['test_acc_certain']
            drift = abs(sv['acc_certain'] - was)
            flag = 'OK' if drift < 1e-6 else f'!! DRIFT {drift:.4f}'
            print(f'{arm} seed {seed}: single-view {sv["acc_certain"]:.3f} '
                  f'(committed {was:.3f}) {flag}', flush=True)

            # Seed IMMEDIATELY before the multi-clip pass, not before the
            # single-view one: `evaluate_tr` above itself draws a randperm per
            # forward for shuffle_frames arms (~946 draws over 3783 clips), so
            # seeding earlier leaves the two scripts on different RNG positions.
            # `synccaps_perclip_dump.py` seeds at exactly this point, so the
            # committed number and the dump now come from the same draw.
            if args.eval_seed is not None:
                set_seed(args.eval_seed)
            mv = evaluate_multiclip(model, ds, te, args.views, args.bs, nw, pin)
            print(f'  multi-clip x{args.views}: certain {mv["acc_certain"]:.3f} '
                  f'({mv["acc_certain"] - sv["acc_certain"]:+.3f}) | '
                  f'final {mv["acc_final"]:.3f} | hybrid {mv["acc_hybrid"]:.3f}',
                  flush=True)

            rows.append(dict(
                seed=seed, n_views=args.views,
                single_view_drift=float(drift),
                test_acc_certain=float(sv['acc_certain']),
                test_acc_final=float(sv['acc_final']),
                test_acc_hybrid=float(sv['acc_hybrid']),
                mean_exit_tick=float(sv['mean_exit']),
                test_acc_certain_mv=float(mv['acc_certain']),
                test_acc_final_mv=float(mv['acc_final']),
                test_acc_hybrid_mv=float(mv['acc_hybrid']),
                mean_exit_tick_mv=float(mv['mean_exit'])))
            with open(out, 'w') as f:
                json.dump(results, f, indent=1)
    print('multi-clip scoring done ->', out, flush=True)
