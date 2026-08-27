"""synccaps_perclip_dump.py — dump PER-CLIP predictions for paired significance tests.

Why this exists: with 3 seeds on a 3783-clip test set the pipeline resolves about
+/-1 point, so the +1.82 frozen-readout-vs-fine-tuning result (Phase 11b) sits
inside the noise floor of a 3-seed t-test and does not survive a Bonferroni
correction over the four planned comparisons. A paired test over the test CLIPS
has vastly more power, because both arms are scored on the same 3783 items.

It reuses `evaluate_multiclip` -> `_readout_metrics` unchanged, which already
returns `preds_certain / preds_final / preds_hybrid / labels`. Nothing is
recomputed by hand, so a dumped prediction vector is by construction the same one
that produced the committed accuracy. The committed multi-clip accuracy is
re-derived from the dumped predictions and asserted against the results JSON:
if they disagree, the dump is wrong and the run aborts rather than writing a file
that would silently poison the statistics.

Resumable: one .npz per (tag, arm, seed); existing files are skipped.
"""
import argparse
import json
import os

import numpy as np
import torch

from exp_base import (UCF11VideoDataset, FrozenFeatureDataset,
                      make_official_split1, make_stratified_splits, DEVICE)
from synccaps_followup_experiment import DATASETS, make_arms
from synccaps_probe_experiment import evaluate_multiclip, set_seed

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=list(DATASETS), default='ucf101')
    ap.add_argument('--stem', default='resnet')
    ap.add_argument('--arms', default='B0_linear,B4_syncnorm')
    ap.add_argument('--seeds', default='42,1337,7')
    ap.add_argument('--nsynch', type=int, default=2048)
    ap.add_argument('--views', type=int, default=3)
    ap.add_argument('--bs', type=int, default=4)
    ap.add_argument('--split', choices=['seeded', 'official1'], default='official1')
    ap.add_argument('--noval', action='store_true')
    ap.add_argument('--pretrained', action='store_true')
    ap.add_argument('--freeze-stem', action='store_true')
    ap.add_argument('--feat-cache', action='store_true')
    ap.add_argument('--outdir', default='perclip')
    ap.add_argument('--eval-seed', type=int, default=None,
                    help='reseed immediately before scoring. Arms with '
                         'shuffle_frames=True draw a fresh torch.randperm INSIDE '
                         'forward(), so their evaluation is STOCHASTIC and two '
                         'scorings of one checkpoint disagree (measured: 1 clip '
                         'in 3783). Pass this to make a dump reproducible. '
                         'Default None leaves every pre-2026-08-21 number '
                         'untouched, since deterministic arms ignore it.')
    args = ap.parse_args()

    path, ncls = DATASETS[args.dataset]
    pt_tag = ('_ptfz' if args.freeze_stem else '_pt') if args.pretrained else ''
    split_tag = ('_official1_noval' if args.noval else '_official1') if args.split == 'official1' else ''
    fc_tag = '_fc' if args.feat_cache else ''
    tag = f'{args.dataset}_{args.stem}{pt_tag}{split_tag}{fc_tag}'

    ds = UCF11VideoDataset(path, sequence_length=16, sample_fps=5.0,
                           augment=False, cache_dir='.cache')
    if args.feat_cache:
        # 2026-08-21: --feat-cache reached the filename tag and make_arms (so the
        # MODEL was built for pooled [C,3,3] features) but the DATASET was never
        # wrapped, so raw frames were fed to a cached-stem model. It failed loudly
        # -- LayerNorm([256,3,3]) got [B*T,3,3,3] -- rather than silently, but only
        # because the shapes happened to disagree. The flag was simply never
        # exercised here: every per-clip dump before today was on uncached runs.
        ds = FrozenFeatureDataset(ds, args.stem,
                                  os.environ.get('FEAT_DIR', '.featcache'))
        missing = sum(1 for smp in ds.samples[:64]
                      if not ds.feat_path(smp[0]).exists())
        if missing:
            raise SystemExit(
                f'--feat-cache: {missing}/64 sampled clips have no cached '
                f'{args.stem} feature. Run:\n  python '
                f'synccaps_precompute_stem.py --dataset {args.dataset} '
                f'--stem {args.stem}')
    nw = int(os.environ.get('GATING_NW', '0'))
    arms = make_arms(args.stem, n_synch=args.nsynch, pretrained=args.pretrained,
                     freeze_stem=args.freeze_stem, feat_cache=args.feat_cache)
    os.makedirs(args.outdir, exist_ok=True)

    # Committed multi-clip numbers, used as the correctness assertion below.
    mc_path = f'gating_results/multiclip_{tag}.json'
    committed = json.load(open(mc_path)) if os.path.exists(mc_path) else {}

    for arm in [a.strip() for a in args.arms.split(',')]:
        for seed in [int(s) for s in args.seeds.split(',')]:
            out = f'{args.outdir}/{tag}_{arm}_seed{seed}.npz'
            if os.path.exists(out):
                print(f'{arm} seed {seed}: exists -> skip', flush=True)
                continue
            ckpt = f'checkpoints/synccaps_{tag}_{arm}_seed{seed}.pt'
            if not os.path.exists(ckpt):
                print(f'{arm} seed {seed}: MISSING {ckpt} -> skip', flush=True)
                continue
            set_seed(seed)
            if args.split == 'official1':
                _, _, te = make_official_split1(ds, val_groups=() if args.noval else (8, 9, 10))
            else:
                _, _, te = make_stratified_splits(ds, seed=seed)

            blob = torch.load(ckpt, map_location='cpu', weights_only=False)
            assert blob['arm'] == arm and blob['seed'] == seed, f'{ckpt} identity mismatch'
            model = arms[arm](ncls).to(DEVICE)
            model.load_state_dict(blob['state_dict'])

            if args.eval_seed is not None:
                set_seed(args.eval_seed)
            r = evaluate_multiclip(model, ds, te, args.views, args.bs, nw, nw > 0)

            # Re-derive the accuracy FROM the dumped vectors and check it against
            # the committed one. This is the whole safety net: it proves the
            # predictions correspond to the published number.
            acc = float((r['preds_certain'] == r['labels']).mean() * 100)
            assert abs(acc - r['acc_certain']) < 1e-9, 'internal readout mismatch'
            note = ''
            for row in committed.get(arm, []):
                if row['seed'] == seed:
                    drift = abs(acc - row['test_acc_certain_mv'])
                    assert drift < 1e-6, (f'{arm} seed {seed}: dumped {acc:.6f} vs '
                                          f'committed {row["test_acc_certain_mv"]:.6f}')
                    note = ' [matches committed]'
            np.savez_compressed(out, preds_certain=r['preds_certain'],
                                preds_final=r['preds_final'], preds_hybrid=r['preds_hybrid'],
                                labels=r['labels'], test_idx=np.asarray(te),
                                acc_certain=acc, acc_hybrid=r['acc_hybrid'])
            print(f'{arm} seed {seed}: certain_mv {acc:.3f}{note} -> {out}', flush=True)
    print('per-clip dump done', flush=True)
