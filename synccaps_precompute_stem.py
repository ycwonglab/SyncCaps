"""synccaps_precompute_stem.py — cache a FROZEN stem's pooled output per clip.

With a frozen stem the features never change, yet a 12-epoch run re-derives them
12 times: on UCF101 that streams ~540 GB off disk and pins the run to the stem's
cost rather than the head's. Caching the stem output makes a run head-bound.

WHAT IS CACHED, AND WHY IT IS EXACT
    The model computes  pre_caps_norm(adaptive_avg_pool2d(proj(body(x)), 3)).
    `proj` (a 1x1 conv) is linear over channels at each position; the pool is
    linear over positions within each channel; the two commute. So caching
    pool(body(x), 3) and training `proj` on the pooled tensor is arithmetically
    identical to the full path, at 221 KB/clip instead of 4.8 MB.
    Valid ONLY while the stem's last layer is 1x1 -- see FrozenFeatureDataset.

Views: --views 3 also caches the temporal windows multi-clip eval needs. View 0
is the historical window and reuses the existing frame cache; the others decode
once here rather than once per evaluated checkpoint.

    python synccaps_precompute_stem.py --dataset ucf101 --stem clip_b32
    python synccaps_precompute_stem.py --dataset ucf101 --stem clip_b32 --views 3
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from exp_base import UCF11VideoDataset, FrozenFeatureDataset, DEVICE
from synccaps_followup_experiment import DATASETS
from src.models.sync_caps import SyncCapsNet

FEAT_DIR = os.environ.get('FEAT_DIR', '.featcache')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=list(DATASETS), default='ucf101')
    ap.add_argument('--stem', default='clip_b32')
    ap.add_argument('--views', type=int, default=1,
                    help='temporal windows to cache (multi-clip eval needs 3)')
    ap.add_argument('--bs', type=int, default=8, help='clips per forward')
    ap.add_argument('--caps-grid', type=int, default=3)
    ap.add_argument('--only', choices=['all', 'official1_test'], default='all',
                    help="'official1_test' restricts to UCF101 split-1 test "
                         'clips — the only ones multi-clip eval needs, so the '
                         'extra views cost 3783 clips instead of 13320 (and '
                         'those frames are already decoded)')
    args = ap.parse_args()

    path, ncls = DATASETS[args.dataset]
    ds = UCF11VideoDataset(path, sequence_length=16, sample_fps=5.0,
                           augment=False, cache_dir='.cache')
    if args.only == 'official1_test':
        from exp_base import make_official_split1
        _, _, _te = make_official_split1(ds, val_groups=())
        subset = set(_te)
    else:
        subset = None

    # Build the real model, then keep only the FROZEN body -- everything before
    # the trainable 1x1 tail. Taking the body off a real SyncCapsNet, rather
    # than reconstructing it here, is what stops this driver drifting out of
    # sync with the model: a change to the stem is inherited, not mirrored.
    model = SyncCapsNet(ncls, caps_grid=args.caps_grid, stem=args.stem,
                        pretrained=True, freeze_stem=True)
    if getattr(model.conv, 'trainable_tail', False):
        body = model.conv[:-1]
    else:
        body = model.conv                     # conv4/resnet: no projection tail
    body = body.to(DEVICE).eval()
    for p in body.parameters():
        p.requires_grad_(False)

    fracs = [0.0] if args.views == 1 else [k / (args.views - 1)
                                           for k in range(args.views)]
    os.makedirs(FEAT_DIR, exist_ok=True)
    print(f'stem={args.stem} dataset={args.dataset} clips={len(ds)} '
          f'views={fracs} -> {FEAT_DIR}', flush=True)

    for frac in fracs:
        fds = FrozenFeatureDataset(ds, args.stem, FEAT_DIR, clip_start=frac)
        todo = [i for i in range(len(ds))
                if (subset is None or i in subset)
                and not fds.feat_path(ds.samples[i][0]).exists()]
        print(f'[view {frac:.3f}] {len(todo)} of {len(ds)} clips to compute',
              flush=True)
        if not todo:
            continue
        # Frames come from the frame dataset AT THIS VIEW; features are written
        # under the feature key for the same view.
        src = ds.view(frac)
        loader = DataLoader(Subset(src, todo), args.bs, shuffle=False,
                            num_workers=int(os.environ.get('GATING_NW', '0')))
        t0, done, shape = time.time(), 0, None
        with torch.no_grad():
            for bi, (x, _) in enumerate(loader):
                B, T = x.shape[0], x.shape[1]
                f = body(x.reshape(B * T, *x.shape[2:]).to(DEVICE))
                f = F.adaptive_avg_pool2d(f, args.caps_grid)     # POOL, then stop
                f = f.reshape(B, T, *f.shape[1:]).cpu().numpy().astype(np.float16)
                shape = f.shape[1:]
                for j in range(B):
                    idx = todo[bi * args.bs + j]
                    np.save(fds.feat_path(ds.samples[idx][0]), f[j])
                done += B
                if bi % 100 == 0:
                    el = time.time() - t0
                    rate = done / max(el, 1e-6)
                    print(f'  {done}/{len(todo)}  {rate:.1f} clips/s  '
                          f'eta {(len(todo)-done)/max(rate,1e-6)/60:.1f} min',
                          flush=True)
        print(f'[view {frac:.3f}] done in {(time.time()-t0)/60:.1f} min '
              f'(feature shape {shape})', flush=True)
    print('precompute done ->', FEAT_DIR, flush=True)
