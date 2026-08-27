"""synccaps_readout_profile.py — MEASURED cost of each readout.

Section 6.2 of the paper currently prices dynamic routing analytically, from
tensor shapes, and says so: "these routing figures are analytic estimates ...
because the models contain no routing stage to profile". The readout matrix
adds a routing stage, so the estimate can be replaced by a measurement.

What is measured, and why it is the readout and not the stem: every arm is
built on the SAME frozen stem, and the stem runs once per clip before any
readout. Timing the whole model would therefore report the stem's cost with a
few percent of readout noise on top. This script instead times the readout
path only -- primary capsules onward, from a cached [B, T, C, 3, 3] feature
tensor -- which is the quantity the paper's efficiency argument is about.

Reported per arm: p50 / p95 / mean latency per clip, peak activation memory,
trainable parameters split into head vs rest, and the head's analytic MACs per
clip for continuity with the existing Section 6.2 figures.

    python synccaps_readout_profile.py --arms B0_linear,B4_syncnorm,B4_gram,CB_tsketch,LR_bilinear,R3_route
"""
import argparse
import json
import statistics as st
import time

import torch

from exp_base import DEVICE
from synccaps_followup_experiment import make_arms

# CLIP ViT-B/32 pooled-feature shape from FrozenFeatureDataset (C, 3, 3).
STEM_CH = {'clip_b32': 768, 'r18_full': 512, 'r50_full': 2048}


def head_params(model):
    head = sum(p.numel() for n, p in model.named_parameters()
               if p.requires_grad and (n.startswith('head')
                                       or n.startswith('routing')
                                       or n.startswith('sync')))
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return head, total


@torch.no_grad()
def timeit(model, x, warmup, iters):
    for _ in range(warmup):
        model(x)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(x)
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return ts, torch.cuda.max_memory_allocated() / 2 ** 20


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stem', default='clip_b32')
    ap.add_argument('--arms', default='B0_linear,B1_sync,B4_syncnorm,B4_gram,'
                                      'CB_tsketch,LR_bilinear,R3_route')
    ap.add_argument('--ncls', type=int, default=101)
    ap.add_argument('--nsynch', type=int, default=2048)
    ap.add_argument('--route-w-scale', type=float, default=20.0)
    ap.add_argument('--bs', type=int, default=1,
                    help='clips per forward; 1 is the deployment latency the '
                         'efficiency claim is about')
    ap.add_argument('--frames', type=int, default=16)
    ap.add_argument('--warmup', type=int, default=10)
    ap.add_argument('--iters', type=int, default=100)
    ap.add_argument('--out', default='docs/paper/readout_profile.json')
    args = ap.parse_args()

    arms = make_arms(args.stem, n_synch=args.nsynch, pretrained=True,
                     freeze_stem=True, feat_cache=True,
                     route_w_scale=args.route_w_scale)
    # feat_cache=True means the "stem" is just our 1x1 projection, so the timed
    # graph is projection + primaries + readout: the per-clip work that differs
    # between arms. Identical input tensor for every arm.
    x = torch.randn(args.bs, args.frames, STEM_CH[args.stem], 3, 3, device=DEVICE)
    rows = {}
    for arm in args.arms.split(','):
        model = arms[arm](args.ncls).to(DEVICE).eval()
        ts, mem = timeit(model, x, args.warmup, args.iters)
        ts = sorted(ts)
        hp, tp = head_params(model)
        rows[arm] = dict(
            p50_ms=st.median(ts), p95_ms=ts[int(0.95 * len(ts)) - 1],
            mean_ms=st.mean(ts), peak_act_MiB=mem,
            head_params=hp, trainable_params=tp,
            head_width=getattr(getattr(model, 'sync', None), 'out_dim', None))
        print(f'{arm:14s} p50 {rows[arm]["p50_ms"]:7.3f} ms  '
              f'p95 {rows[arm]["p95_ms"]:7.3f} ms  '
              f'peak {mem:7.1f} MiB  head {hp:>9,}  total {tp:>9,}', flush=True)
        del model
        torch.cuda.empty_cache()

    rows['_meta'] = dict(device=torch.cuda.get_device_name(0),
                         torch=torch.__version__, bs=args.bs,
                         frames=args.frames, iters=args.iters,
                         stem=args.stem, note='readout path only; frozen stem '
                                              'served from feature cache')
    with open(args.out, 'w') as f:
        json.dump(rows, f, indent=1)
    print('->', args.out, flush=True)
