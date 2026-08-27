"""synccaps_probe_experiment.py — Sync-Caps Task 5/6 headless runner.

Plan: docs/plans/2026-07-06-sync-caps-implementation.md (Tasks 5-6)
Headless equivalent of notebook §7L SYNC-PROBE / SYNC-RUN: same code as the
notebook cells (train/eval utils copied verbatim from §7K cell; the sync
models imported from the unit-tested src classes instead of the NB mirrors).

  python synccaps_probe_experiment.py                 # probe: B1/A1 x lr {1e-3, 5e-4}, 3 epochs
  python synccaps_probe_experiment.py --mode full --lr 1e-3   # full: 4 arms x 2 seeds, 12 epochs
"""
import argparse
import json
import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from exp_base import (UCF11VideoDataset, make_stratified_splits,
                      make_official_split1, DEVICE)
from src.models.sync_caps import SyncCapsNet, SyncTRCapsNet
from src.models.temporal_routing import (normalized_entropy, ctm_tick_loss,
                                         hybrid_readout)

DATASET_PATH = 'UCF11_updated_mpg'
SYNC_A1_W_SCALE = 20.0   # revisit from probe (memory: init re-tunes per config)


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def per_class_recall(preds, labels, n):
    return {c: (float((preds[labels==c]==c).mean()*100) if (labels==c).any() else float('nan'))
            for c in range(n)}


def train_one_epoch_tr(model, loader, opt, loss_mode='ctm', grad_clip=5.0):
    model.train(); tot = correct = n = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        out = model(x)
        if loss_mode == 'final':
            loss = F.cross_entropy(out['logits'][:, :, -1], y)
            pred = out['logits'][:, :, -1].argmax(1)
        else:
            loss, idx = ctm_tick_loss(out['logits'], out['certainties'], y)
            pred = out['logits'][torch.arange(y.size(0), device=y.device),
                                 :, idx].argmax(1)
        loss.backward()
        if grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        tot += loss.item(); correct += (pred == y).sum().item(); n += y.size(0)
    return tot / len(loader), correct / n * 100


@torch.no_grad()
def _collect_logits(model, loader):
    """Per-clip tick logits [N, C, T] and labels [N], in loader order.

    Split out of evaluate_tr so multi-clip eval can average the logits of
    several temporal views BEFORE any readout policy runs. Every readout here
    (certainty argmax, hybrid cumsum, per-tick counts) is per-sample, so
    collecting first and reading out once is arithmetically identical to the
    old per-batch loop. Cost is trivial: UCF101's 3783-clip test set is
    3783 x 101 x 16 fp32 = 24 MB.
    """
    model.eval()
    lg, ys = [], []
    for x, y in loader:
        lg.append(model(x.to(DEVICE))['logits'].cpu())     # [B, C, T]
        ys.append(y)
    return torch.cat(lg), torch.cat(ys)


def _readout_metrics(lg, y, theta=0.5):
    """The three readout policies over collected logits [N, C, T]."""
    n = y.size(0)
    idx = (1 - normalized_entropy(lg.transpose(1, 2))).argmax(-1)
    b = torch.arange(n)
    p_cert = lg[b, :, idx].argmax(1).numpy()
    p_final = lg[..., -1].argmax(1).numpy()
    ph, ih, _ = hybrid_readout(lg, theta=theta)
    p_hyb, exits = ph.numpy(), (ih + 1).numpy()
    tick_correct = (lg.argmax(1) == y.unsqueeze(-1)).float().sum(0)
    labels = y.numpy()
    return dict(acc_certain=(p_cert == labels).mean() * 100,
                acc_final=(p_final == labels).mean() * 100,
                acc_hybrid=(p_hyb == labels).mean() * 100,
                mean_exit=float(np.mean(exits)),
                per_tick=(tick_correct / n * 100).tolist(),
                preds_certain=p_cert, preds_final=p_final,
                preds_hybrid=p_hyb, labels=labels)


def evaluate_tr(model, loader, theta=0.5):
    lg, y = _collect_logits(model, loader)
    return _readout_metrics(lg, y, theta=theta)


def evaluate_multiclip(model, dataset, idx, n_views, bs, nw, pin, theta=0.5):
    """Average tick logits over `n_views` evenly-spaced temporal windows.

    Why this exists: at 5 fps a 16-frame clip spans ~3.2 s while UCF101 videos
    average ~7 s, so single-view eval scores a model on less than half of each
    test video. Every published UCF101 number this project compares against is
    multi-clip (usually multi-crop too), so the single-view caveat had to be
    attached to every comparison table.

    Views sit at fractions 0, 1/(V-1), ..., 1 of the leftover frames, so view 0
    IS the historical window and reuses the existing 67 GB frame cache -- only
    the V-1 new windows decode. Videos too short to hold more than one window
    collapse all views onto the same frames, which is correct, and is why the
    gain is dataset-dependent rather than free.

    Logits are averaged BEFORE the readout so all three policies (certain,
    final, hybrid) stay defined and the exit-tick statistic keeps its meaning.
    The average is over raw per-tick logits -- the TSN-style pre-softmax
    consensus. Eval only: training is untouched, so no trained model changes.
    """
    fracs = [k / (n_views - 1) for k in range(n_views)]
    acc_lg, labels = None, None
    for f in fracs:
        loader = DataLoader(Subset(dataset.view(f), idx), bs, shuffle=False,
                            num_workers=nw, pin_memory=pin)
        lg, y = _collect_logits(model, loader)
        acc_lg = lg if acc_lg is None else acc_lg + lg
        if labels is None:
            labels = y
        else:
            # Views share `samples` by reference and shuffle=False, so item i is
            # the same source clip in every view. If that ever breaks, averaging
            # would silently mix clips -- assert rather than trust it.
            assert torch.equal(labels, y), 'view misalignment: labels differ'
    return _readout_metrics(acc_lg / n_views, labels, theta=theta)


def run_experiment_sync(dataset, ncls, arm, seed=42, epochs=12, bs=4, lr=1e-3,
                        readout='certain', verbose=True, arms=None,
                        save_path=None):
    """arm: a key of `arms` (defaults to SYNC_ARMS). Reuses the TR utils."""
    arms = arms if arms is not None else SYNC_ARMS
    set_seed(seed)
    # SYNC_SPLIT=official1 -> UCF101's published partition (test = groups 1-7),
    # for numbers directly comparable to the literature. Unset (default) keeps
    # the seeded StratifiedGroupKFold used by every controlled arm comparison,
    # so existing runs stay byte-identical. Both protocols are leak-free but are
    # DIFFERENT partitions; never mix their accuracies in one table.
    no_val = bool(os.environ.get('SYNC_SPLIT_NOVAL'))
    if os.environ.get('SYNC_SPLIT') == 'official1':
        # SYNC_SPLIT_NOVAL=1 -> train on ALL 9537 official train clips, as
        # published methods do (the official protocol defines no val set). The
        # val list is then EMPTY, never evaluated, and the FINAL epoch is kept
        # instead of a best-val checkpoint. Test is never touched either way.
        tr, va, te = make_official_split1(dataset,
                                          val_groups=() if no_val else (8, 9, 10))
    else:
        tr, va, te = make_stratified_splits(dataset, seed=seed)
    nw = int(os.environ.get('GATING_NW', '0')); pin = nw > 0
    tl = DataLoader(Subset(dataset, tr), bs, shuffle=True,  num_workers=nw, pin_memory=pin)
    vl = DataLoader(Subset(dataset, va), bs, shuffle=False, num_workers=nw, pin_memory=pin)
    el = DataLoader(Subset(dataset, te), bs, shuffle=False, num_workers=nw, pin_memory=pin)
    model = arms[arm](ncls).to(DEVICE)
    # Differential LR for dir-B fine-tuning (SYNC_DIFF_LR=1): a pretrained
    # backbone must train at a much lower LR than the fresh head or ImageNet
    # features get wiped. Env-gated so every from-scratch run stays identical.
    if os.environ.get('SYNC_DIFF_LR'):
        conv_p = [p for p in model.conv.parameters() if p.requires_grad]
        head_p = [p for n, p in model.named_parameters()
                  if p.requires_grad and not n.startswith('conv.')]
        ft_lr = float(os.environ.get('SYNC_BACKBONE_LR', str(lr * 0.1)))
        groups = ([{'params': conv_p, 'lr': ft_lr}] if conv_p else []) + \
                 [{'params': head_p, 'lr': lr}]
        opt = torch.optim.Adam(groups, weight_decay=1e-4)
        print(f'    [diff-lr] backbone {len(conv_p)} params @ {ft_lr:g}, '
              f'head {len(head_p)} @ {lr:g}', flush=True)
    else:
        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                               lr=lr, weight_decay=1e-4)
    warmup = 3
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda e: (e + 1) / warmup if e < warmup
        else 0.5 * (1 + math.cos(math.pi * (e - warmup) / max(epochs - warmup, 1))))
    best_val, best_state = 0.0, None
    for ep in range(epochs):
        loss, tra = train_one_epoch_tr(model, tl, opt, 'ctm')
        # no_val: nothing is held out, so there is nothing to select on — keep
        # the final epoch. best_state stays None and the load below is skipped.
        val = float('nan') if no_val else evaluate_tr(model, vl)['acc_' + readout]
        sched.step()
        if not no_val and val > best_val:
            best_val = val
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if verbose:
            print(f'  ep{ep+1:02d}/{epochs} loss {loss:.3f} train {tra:.3f} val {val:.3f}',
                  flush=True)
    if best_state is not None:          # None under no_val -> keep final epoch
        model.load_state_dict(best_state)
    ev = evaluate_tr(model, el)
    # SYNC_EVAL_VIEWS=V (V>1) ADDS multi-clip test metrics alongside the
    # single-view ones; it never replaces them. Both must be reported: every
    # prior run in the ledger is single-view, so that is the only metric
    # comparable across this project's own history, while the multi-clip number
    # is the one comparable to published UCF101 results. Quoting one where the
    # other belongs is exactly the metric-mixing the ledger already warns about.
    n_views = int(os.environ.get('SYNC_EVAL_VIEWS', '1'))
    mv = None
    if n_views > 1:
        mv = evaluate_multiclip(model, dataset, te, n_views, bs, nw, pin)
        if verbose:
            print(f'  [multi-clip x{n_views}] certain {mv["acc_certain"]:.3f} '
                  f'(single-view {ev["acc_certain"]:.3f})', flush=True)
    sync = getattr(model, 'sync', None)      # None for the B0 linear control
    decay_hist = (torch.histc(sync.rho.detach().float().cpu(),
                              bins=10, min=0, max=15).tolist()
                  if sync is not None else [])
    res = dict(seed=seed,
               test_acc_certain=float(ev['acc_certain']),
               test_acc_final=float(ev['acc_final']),
               test_acc_hybrid=float(ev['acc_hybrid']),
               mean_exit_tick=float(ev['mean_exit']),
               per_tick_acc=ev['per_tick'], best_val=float(best_val),
               decay_hist=decay_hist,
               per_class=per_class_recall(ev['preds_' + readout],
                                          ev['labels'], ncls))
    if mv is not None:
        res.update(n_eval_views=n_views,
                   test_acc_certain_mv=float(mv['acc_certain']),
                   test_acc_final_mv=float(mv['acc_final']),
                   test_acc_hybrid_mv=float(mv['acc_hybrid']),
                   mean_exit_tick_mv=float(mv['mean_exit']))
    if save_path:
        # best-val weights are already loaded into `model` at this point
        torch.save(dict(state_dict=model.state_dict(), arm=arm, seed=seed,
                        results=res), save_path)
        if verbose:
            print(f'  checkpoint -> {save_path}', flush=True)
    return res


def _b2(ncls):
    m = SyncCapsNet(ncls)
    m.sync.rho.requires_grad_(False)   # frozen at 0 => r = 1 (Gram control)
    return m


SYNC_ARMS = {
    'B1_sync':    lambda ncls: SyncCapsNet(ncls),
    'B2_gram':    _b2,
    'B3_shuffle': lambda ncls: SyncCapsNet(ncls, shuffle_frames=True),
    'A1_tr_sync': lambda ncls: SyncTRCapsNet(ncls, w_init_scale=SYNC_A1_W_SCALE),
}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['probe', 'full'], default='probe')
    ap.add_argument('--lr', type=float, default=1e-3,
                    help='full mode only; probe sweeps {1e-3, 5e-4}')
    args = ap.parse_args()

    ds11 = UCF11VideoDataset(DATASET_PATH, sequence_length=16, sample_fps=5.0,
                             augment=False, cache_dir='.cache')
    print(f'dataset: {len(ds11)} clips, mode={args.mode}', flush=True)
    os.makedirs('gating_results', exist_ok=True)

    if args.mode == 'probe':
        probe = {}
        for arm in ['B1_sync', 'A1_tr_sync']:
            for lr in [1e-3, 5e-4]:
                key = f'{arm}_lr{lr}'
                print(key, flush=True)
                probe[key] = run_experiment_sync(ds11, 11, arm, epochs=3, lr=lr)
                print(' ', {k: round(v, 3) for k, v in probe[key].items()
                            if isinstance(v, float)}, flush=True)
                with open('gating_results/synccaps_probe.json', 'w') as f:
                    json.dump(probe, f, indent=1)
        print('probe done -> gating_results/synccaps_probe.json', flush=True)
    else:
        # Probe verdict (gating_results/synccaps_probe.json, 2026-07-07):
        # B arms learn fastest at 1e-3; A1 diverges at 1e-3, stable at 5e-4.
        # B1 probe <= 78.8 -> controls drop to single seed (design doc gate).
        arm_lr = {'B1_sync': 1e-3, 'B2_gram': 1e-3,
                  'B3_shuffle': 1e-3, 'A1_tr_sync': 5e-4}
        arm_seeds = {'B1_sync': [42, 1337], 'B2_gram': [42],
                     'B3_shuffle': [42], 'A1_tr_sync': [42]}
        results = {}
        for arm in SYNC_ARMS:
            results[arm] = []
            for seed in arm_seeds[arm]:
                print(arm, 'seed', seed, 'lr', arm_lr[arm], flush=True)
                r = run_experiment_sync(ds11, 11, arm, seed=seed, lr=arm_lr[arm])
                results[arm].append(r)
                print(' ', {k: round(v, 3) for k, v in r.items()
                            if isinstance(v, float)}, flush=True)
                with open('gating_results/synccaps_ucf11.json', 'w') as f:
                    json.dump(results, f, indent=1)
        print('full runs done -> gating_results/synccaps_ucf11.json', flush=True)
