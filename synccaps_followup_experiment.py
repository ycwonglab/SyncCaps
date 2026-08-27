"""synccaps_followup_experiment.py — two sync-caps follow-ups (prepared, run manually).

Flow: docs/plans/2026-07-08-sync-caps-followups.md
Reuses the train/eval/seed/per-class utilities and run loop from
synccaps_probe_experiment.py (importing that module is side-effect-light: it
only defines functions and prints the device). Adds a stem-aware arm factory
and a UCF-11 / UCF101 dataset switch.

Baselines to compare against (UCF-11): 4-conv sync B1 = 87.8 (2 seeds,
gating_results/synccaps_ucf11.json); 4-conv norm-readout 78.8; V1 ResNet
norm-readout 88.8.

Flow 1 — does sync stack with depth? (UCF-11, ResNet stem)
    python synccaps_followup_experiment.py --dataset ucf11 --stem resnet --mode full
    # B1 sync on the V1 ResNet body. > 88.8 => sync stacks with depth;
    # ~= 88.8 => depth and sync are redundant; ~= 87.8 => stem didn't matter.

Flow 2 — does B1 survive when data isn't starved? (UCF101-full, 4-conv stem)
    python synccaps_followup_experiment.py --dataset ucf101 --stem conv4 --mode probe
    # -> pick LR (and n_synch) from gating_results/synccaps_probe_ucf101_conv4.json
    python synccaps_followup_experiment.py --dataset ucf101 --stem conv4 --mode full --lr <picked>
    # Compare to UCF101-full S3 baselines in memory (V1 ResNet 74.7, V2 85.1).

NOTE on the "W-init probe": B1 (SyncCapsNet) has NO routing vote tensor W — its
only trainable readout params are PairwiseSync.rho (init 0) and a Linear. So the
UCF101 probe is an LR x n_synch sweep, not a capsule-W sweep; loss stuck at
ln(101) = 4.615 means the sync+linear readout isn't learning, not dead capsules.
The routing-W-scale caution (capsnet-init memory) only applies if you add the A1
arm (TemporalRoutingCaps.W) on UCF101 — then sweep w_init_scale in {1,20,100}.
"""
import argparse
import json
import os

from exp_base import UCF11VideoDataset, FrozenFeatureDataset
from src.models.sync_caps import (SyncCapsNet, SyncTRCapsNet, RoutingCapsNet,
                                  _MOBILENET_STEMS, _VIT_STEMS, _RESNET_STEMS)
from synccaps_probe_experiment import run_experiment_sync

DATASETS = {
    'ucf11':  ('UCF11_updated_mpg', 11),
    'ucf101': ('UCF101_full', 101),
}
A1_W_SCALE = 20.0   # only used by the optional A1 arm (has routing W); probe it at 101 cls


def make_arms(stem, n_synch=1024, n_self=64, pretrained=False,
              freeze_stem=False, feat_cache=False, out_ch=256,
              route_w_scale=20.0, lr_rank=64, pair_seed=0,
              exclude_self=False):
    """Stem-aware B-arms (+ optional A1). A1 is always 4-conv (routing head).

    pretrained/freeze_stem (direction-B): the B0/B1/B3 arms share an
    ImageNet-pretrained stem so B1_sync - B0_linear isolates the sync
    readout's value on a STRONG representation, honestly (group-aware).
    """
    # pair_seed reaches BOTH dictionaries that a pair-set replicate must move:
    # PairwiseSync's sampled indices and CompactBilinear's Count-Sketch hash and
    # sign vectors. RoutingCapsNet forwards **stem_kw to a SyncCapsNet prototype,
    # so every arm below accepts these without a per-arm edit.
    pk = dict(pretrained=pretrained, freeze_stem=freeze_stem,
              feat_cache=feat_cache, out_ch=out_ch, pair_seed=pair_seed,
              exclude_self=exclude_self)
    def _b2(ncls):
        m = SyncCapsNet(ncls, n_synch=n_synch, n_self=n_self, stem=stem, **pk)
        m.sync.rho.requires_grad_(False)          # frozen r=1 (Gram control)
        return m

    def _b4g(ncls):
        # B4_gram = B4_syncnorm minus the learned decay. rho initialises to
        # zeros, so freezing it pins r = exp(0) = 1 and the statistic becomes the
        # count-normalised, ORDER-AGNOSTIC Gram entry, signed-sqrt/L2 normalised.
        #
        # Why it exists (2026-08-18, Phase 12): regenerating the figures showed
        # the headline model's learned memory is e^-rho median 0.780 / min 0.507,
        # i.e. the decay is substantially ENGAGED -- not the ~0.998 "dormant"
        # value the retired UCF-11 checkpoint showed. The shuffle controls still
        # tie, so the prediction is that pinning rho=0 costs nothing: the decay
        # is spending capacity on a weighting that buys no accuracy. B2_gram
        # cannot test this because it lacks sync_norm, and sync_norm is the arm
        # the claim is about.
        m = SyncCapsNet(ncls, n_synch=n_synch, n_self=n_self, stem=stem,
                        sync_norm=True, **pk)
        m.sync.rho.requires_grad_(False)
        return m
    return {
        'B1_sync':    lambda ncls: SyncCapsNet(ncls, n_synch=n_synch, n_self=n_self, stem=stem, **pk),
        'B2_gram':    _b2,
        'B4_gram':    _b4g,
        'B3_shuffle': lambda ncls: SyncCapsNet(ncls, n_synch=n_synch, n_self=n_self,
                                               stem=stem, shuffle_frames=True, **pk),
        # B0: same stem/primaries/tick machinery, plain per-frame Linear readout
        # (no sync). B1 - B0 = the sync statistic's contribution over the stem.
        'B0_linear':  lambda ncls: SyncCapsNet(ncls, stem=stem, readout='linear', **pk),
        # B0 + LayerNorm: closes the B5_concat confound (B5's linear branch saw
        # LayerNorm(z), B0's saw raw z). B0_linear_ln - B0_linear = that effect.
        'B0_linear_ln': lambda ncls: SyncCapsNet(ncls, stem=stem, readout='linear_ln', **pk),
        # 2026-08-15 readout-quality arms. Motivation: at the learned r->1 the
        # sync statistic IS a subsampled Gram matrix (order-free by
        # construction), so accuracy has to come from better second-order
        # POOLING, not from the temporal machinery.
        # B4 = improved-B-CNN normalisation (signed-sqrt + L2) on sync.
        # B5 = first+second order fusion; nests B0 and B1, so it answers
        #      "does sync add anything over a linear read of the same features?"
        'B4_syncnorm': lambda ncls: SyncCapsNet(ncls, n_synch=n_synch, n_self=n_self,
                                                stem=stem, sync_norm=True, **pk),
        'B5_concat':   lambda ncls: SyncCapsNet(ncls, n_synch=n_synch, n_self=n_self,
                                                stem=stem, readout='sync_linear',
                                                sync_norm=True, **pk),
        # ------------------------------------------------------------------
        # 2026-08-20 same-stem readout matrix (docs/paper/
        # SyncCaps_v5_novelty_experiments.md #2). The paper's Section 2.4
        # concedes that PairwiseSync is a subsampled Gram statistic under
        # improved-B-CNN normalisation and names compact bilinear pooling as
        # the established alternative; Section 6.2 prices routing analytically
        # because no routing stage existed to profile. These three arms close
        # both gaps ON THE SAME STEM, at the same head width, under the same
        # tick loop -- so what they measure is the readout and nothing else.
        # All three pin the temporal decay to r = 1, matching B4_gram, which is
        # the operating point Section 5.4's controls already establish.
        'CB_tsketch':  lambda ncls: SyncCapsNet(ncls, n_synch=n_synch, stem=stem,
                                                readout='cbp', **pk),
        'LR_bilinear': lambda ncls: SyncCapsNet(ncls, stem=stem, readout='lrbp',
                                                lr_rank=lr_rank, **pk),
        # NOT parameter-matched, and that is the finding rather than a flaw:
        # the vote tensor alone is 3.72 M against the sync head's 0.21 M.
        'R3_route':    lambda ncls: RoutingCapsNet(ncls, r=3, stem=stem,
                                                   w_init_scale=route_w_scale,
                                                   **pk),
        # Pose-aware sync arms: pair the 288 capsules and couple their 8-D pose
        # vectors. A_dot = raw <u_i,u_j> (self-pairs -> presence); A_cos =
        # direction-only cosine; A_dot_shuffle = order control for A_dot.
        'A_dot':         lambda ncls: SyncCapsNet(ncls, n_synch=n_synch, n_self=n_self,
                                                  stem=stem, pose_coupling='dot'),
        'A_cos':         lambda ncls: SyncCapsNet(ncls, n_synch=n_synch, n_self=n_self,
                                                  stem=stem, pose_coupling='cosine'),
        'A_dot_shuffle': lambda ncls: SyncCapsNet(ncls, n_synch=n_synch, n_self=n_self,
                                                  stem=stem, pose_coupling='dot',
                                                  shuffle_frames=True),
        # V-arms (docs/paper §6.2): each repairs one failure mode of A_dot.
        # V1 learned rank-1 pose metric per pair (init == a scalar cross-caps
        #    pair; +16.4k params). Separates "pose is uninformative" from
        #    "the identity metric was the wrong metric".
        # V2 learned per-type 8x8 pose frames before the dot (+2k params) —
        #    votes without routing; agreement in a learned common frame.
        # V3 full 64-D outer-product trace at n_synch/8 pairs (head width
        #    == 8*n_synch); the only arm that can see pose co-rotation over
        #    ticks — run where temporal binding exists, i.e. not UCF-11.
        'V1_bilinear': lambda ncls: SyncCapsNet(ncls, n_synch=n_synch, n_self=n_self,
                                                stem=stem, pose_coupling='bilinear'),
        'V2_aligned':  lambda ncls: SyncCapsNet(ncls, n_synch=n_synch, n_self=n_self,
                                                stem=stem, pose_coupling='aligned'),
        'V3_outer':    lambda ncls: SyncCapsNet(ncls, n_synch=max(n_synch // 8, 8),
                                                n_self=max(n_self // 8, 1),
                                                stem=stem, pose_coupling='outer'),
        'A1_tr_sync': lambda ncls: SyncTRCapsNet(ncls, w_init_scale=A1_W_SCALE),
    }


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=list(DATASETS), default='ucf11')
    # Keep in sync with SyncCapsNet's stem dispatch. _MOBILENET_STEMS/_VIT_STEMS
    # are the single source of truth for those families, so adding a stem there
    # makes it callable from the CLI automatically and the list cannot drift
    # (trap #6: the first MobileNet sweep died on a stale `choices` list while
    # the pre-flight, which called make_arms() directly, bypassed argparse).
    ap.add_argument('--stem', default='conv4',
                    choices=['conv4', 'resnet'] + sorted(_MOBILENET_STEMS)
                            + sorted(_VIT_STEMS) + sorted(_RESNET_STEMS))
    ap.add_argument('--mode', choices=['probe', 'full'], default='probe')
    ap.add_argument('--arms', default='B1_sync',
                    help='comma-separated arm keys (default B1_sync)')
    ap.add_argument('--lr', type=float, default=1e-3, help='full mode LR')
    ap.add_argument('--seeds', default='42,1337',
                    help='full mode seeds, comma-separated')
    ap.add_argument('--nsynch', type=int, default=1024)
    ap.add_argument('--n-self', type=int, default=64,
                    help='how many of the n_synch pairs are SELF-pairs (i == j, '
                         'a squared activation). 64 is the historical default. '
                         'n_self = n_synch is the self-only arm (pure '
                         'activation energy); n_self = 0 with --exclude-self is '
                         'the cross-only arm (pure co-activation between '
                         'distinct units).')
    ap.add_argument('--exclude-self', action='store_true',
                    help='reject i == j in the CROSS block. The cross pairs are '
                         'drawn independently, so ~n_cross/d_model of them are '
                         'self-pairs by accident (0.9 of 2048 at d=2304) -- '
                         'negligible for accuracy, but a cross-ONLY arm must '
                         'contain none. Off by default so every pre-2026-08-21 '
                         'pair dictionary is reproduced bit-identically.')
    ap.add_argument('--pair-seed', type=int, default=0,
                    help='structural seed for the sampled-pair dictionary AND '
                         'the Tensor-Sketch hash/sign vectors. Crossing this '
                         'with --seeds separates pair-set randomness from '
                         'optimizer randomness.')
    ap.add_argument('--pretrained', action='store_true',
                    help='dir-B: ImageNet-pretrained resnet stem (use with --stem resnet)')
    ap.add_argument('--freeze-stem', action='store_true',
                    help='dir-B: freeze the pretrained stem (linear-probe mode)')
    ap.add_argument('--feat-cache', action='store_true',
                    help='train on precomputed frozen-stem features '
                         '(synccaps_precompute_stem.py); frozen stems only')
    ap.add_argument('--route-w-scale', type=float, default=20.0,
                    help='R3_route vote-tensor init scale (std = 0.01 x this). '
                         'MUST be probed per (grid, num_classes): at 101 '
                         'classes an under-scaled init parks the loss at '
                         'ln(101) = 4.615 with dead class capsules.')
    ap.add_argument('--lr-rank', type=int, default=64,
                    help='LR_bilinear projection rank; head width is the '
                         'DERIVED rank(rank+1)/2 (64 -> 2080 vs n_synch 2048)')
    ap.add_argument('--out-ch', type=int, default=256,
                    help='projection width: stem channels are compressed to '
                         'this before PrimaryCaps. d_model is unaffected, so '
                         'this is a single-variable test of whether compression '
                         'before the pairwise products suppresses sync. The '
                         'feature cache stores BODY output, so no re-extraction '
                         'is needed when sweeping this.')
    args = ap.parse_args()
    if args.feat_cache and not args.freeze_stem:
        # Precomputed features are a fixed function of the input, so a stem that
        # is supposed to be LEARNING would silently train against stale features
        # and its gradients would never reach the body. Refuse instead.
        raise SystemExit('--feat-cache requires --freeze-stem (a trainable stem '
                         'cannot be served from a fixed feature cache)')

    path, ncls = DATASETS[args.dataset]
    arms_wanted = [a.strip() for a in args.arms.split(',')]
    seeds = [int(s) for s in args.seeds.split(',')]
    # suffix keeps dir-B outputs from colliding with the from-scratch runs
    pt_tag = ('_ptfz' if args.freeze_stem else '_pt') if args.pretrained else ''
    # SYNC_SPLIT must reach the filename: official-split and seeded-split runs of
    # the same arms would otherwise share an output path, and the resume logic
    # below would silently skip the new protocol's seeds as "already done".
    # EVERY variable that changes what is trained must reach the filename, or the
    # resume logic below silently skips the new config's seeds as "already done"
    # and reports exit 0 having trained nothing. SYNC_SPLIT_NOVAL was omitted
    # once (2026-08-16) and cost a whole queued stage.
    if os.environ.get('SYNC_SPLIT') == 'official1':
        split_tag = '_official1_noval' if os.environ.get('SYNC_SPLIT_NOVAL') else '_official1'
    else:
        split_tag = ''
    # `_fc` in the tag keeps feature-cached runs in their own file. The cached
    # path is arithmetically identical to the full one (1x1 conv and avg-pool
    # commute) but is produced from fp16 pooled features, so the two are kept
    # separable on disk rather than silently interleaved in one 3-seed mean.
    fc_tag = '_fc' if args.feat_cache else ''
    # out_ch reaches the filename (trap #1): a 512-wide projection is a different
    # experiment from a 256-wide one, and sharing a path would let the resume
    # logic skip it as "already done".
    oc_tag = '' if args.out_ch == 256 else f'_oc{args.out_ch}'
    # lr reaches the filename (closes the second half of trap #1, which the
    # comment below still lists as open). The config guard already REFUSES a
    # path whose stored lr differs, so an lr change was never silently skipped
    # -- but it did mean two legitimate LRs could not coexist on disk, which is
    # exactly what a comparable-tuning-budget comparison needs them to do.
    # 1e-3 maps to the empty string so every pre-2026-08-20 path is unchanged.
    lr_tag = '' if args.lr == 1e-3 else f'_lr{args.lr:g}'
    # n_self / exclude_self / pair_seed change WHICH PRODUCTS the head sees, so
    # by trap #1 they must reach the path. All three map to the empty string at
    # their historical defaults, leaving every pre-2026-08-21 filename untouched.
    nself_tag = '' if args.n_self == 64 else f'_nself{args.n_self}'
    xs_tag = '_xself' if args.exclude_self else ''
    ps_tag = '' if args.pair_seed == 0 else f'_pair{args.pair_seed}'
    tag = (f'{args.dataset}_{args.stem}{pt_tag}{split_tag}{fc_tag}{oc_tag}'
           f'{lr_tag}{nself_tag}{xs_tag}{ps_tag}')

    ds = UCF11VideoDataset(path, sequence_length=16, sample_fps=5.0,
                           augment=False, cache_dir='.cache')
    if args.feat_cache:
        ds = FrozenFeatureDataset(ds, args.stem,
                                  os.environ.get('FEAT_DIR', '.featcache'))
        missing = sum(1 for s in ds.samples[:64] if not ds.feat_path(s[0]).exists())
        if missing:
            raise SystemExit(
                f'--feat-cache: {missing}/64 sampled clips have no cached '
                f'{args.stem} feature. Run:\n  python '
                f'synccaps_precompute_stem.py --dataset {args.dataset} '
                f'--stem {args.stem}')
    print(f'dataset={args.dataset} ({len(ds)} clips, {ncls} cls) stem={args.stem} '
          f'mode={args.mode} arms={arms_wanted}', flush=True)
    os.makedirs('gating_results', exist_ok=True)

    if args.mode == 'probe':
        # LR x n_synch sweep on the requested arms (3 epochs, seed 42).
        probe = {}
        out = f'gating_results/synccaps_probe_{tag}.json'
        for arm in arms_wanted:
            for lr in [1e-3, 5e-4]:
                for ns in sorted({args.nsynch, 512, 2048}):
                    arms = make_arms(args.stem, n_synch=ns,
                                     pretrained=args.pretrained,
                                     freeze_stem=args.freeze_stem,
                                     feat_cache=args.feat_cache)
                    key = f'{arm}_lr{lr}_ns{ns}'
                    print(key, flush=True)
                    probe[key] = run_experiment_sync(ds, ncls, arm, epochs=3,
                                                     lr=lr, arms=arms)
                    print('  ', {k: round(v, 3) for k, v in probe[key].items()
                                 if isinstance(v, float)}, flush=True)
                    with open(out, 'w') as f:
                        json.dump(probe, f, indent=1)
        print('probe done ->', out, flush=True)
    else:
        arms = make_arms(args.stem, n_synch=args.nsynch, n_self=args.n_self,
                         pretrained=args.pretrained, freeze_stem=args.freeze_stem,
                         feat_cache=args.feat_cache, out_ch=args.out_ch,
                         route_w_scale=args.route_w_scale, lr_rank=args.lr_rank,
                         pair_seed=args.pair_seed,
                         exclude_self=args.exclude_self)
        # arm(s) in the filename so a control run (e.g. B0_linear) never
        # overwrites another arm's committed result (e.g. B1_sync).
        out = f'gating_results/synccaps_{tag}_{"-".join(arms_wanted)}.json'
        os.makedirs('checkpoints', exist_ok=True)
        # Resume: reload results already on disk so a crashed/interrupted sweep
        # doesn't re-train completed runs.
        #
        # TRAP #1 (the worst failure mode in this pipeline): the filename does
        # NOT encode every variable that changes what is trained -- `n_synch` and
        # `lr` are still absent -- so a differently-configured sweep can land on
        # an existing path and be skipped wholesale, printing `skip` and exiting 0
        # having trained NOTHING. Two fixes here, both rule-level:
        #   (a) the full training config is stored INSIDE the json and re-running
        #       the same path under a different config is refused, not skipped;
        #   (b) resume is keyed on the seed RECORDED IN EACH RESULT, not on list
        #       position, so `--seeds 7` after a 42,1337 sweep trains seed 7
        #       instead of skipping it as "entry 0 already exists".
        # `seeds` is deliberately excluded from the guard: extending the seed list
        # is the intended way to grow a sweep.
        # Env vars that change what is trained MUST reach the stamp, or the
        # guard silently blesses two different experiments sharing one path.
        # Phase 1's fine-tuned runs are unusable as a controlled baseline
        # precisely because their lr and diff-lr setting were never recorded.
        cfg = dict(dataset=args.dataset, stem=args.stem, nsynch=args.nsynch,
                   lr=args.lr, pretrained=args.pretrained,
                   freeze_stem=args.freeze_stem, split=split_tag or 'seeded',
                   feat_cache=args.feat_cache,
                   diff_lr=bool(os.environ.get('SYNC_DIFF_LR')),
                   backbone_lr=os.environ.get('SYNC_BACKBONE_LR'),
                   out_ch=args.out_ch)
        # Stamped ONLY when the arm that consumes them is in play. Adding these
        # keys unconditionally would change the stamp of every pre-2026-08-20
        # sweep and make the guard below refuse to resume files whose training
        # never depended on them -- a false positive that would cost a rerun.
        # Stamped only when non-default, for the same reason the two keys below
        # are arm-gated: adding them unconditionally would change the stamp of
        # every pre-2026-08-21 sweep and make the guard refuse to resume files
        # whose training never depended on them.
        if args.n_self != 64:
            cfg['n_self'] = args.n_self
        if args.exclude_self:
            cfg['exclude_self'] = True
        if args.pair_seed != 0:
            cfg['pair_seed'] = args.pair_seed
        if 'R3_route' in arms_wanted:
            cfg['route_w_scale'] = args.route_w_scale
        if 'LR_bilinear' in arms_wanted:
            cfg['lr_rank'] = args.lr_rank
        results = json.load(open(out)) if os.path.exists(out) else {}
        prev_cfg = results.get('_config')
        if prev_cfg is not None and prev_cfg != cfg:
            diff = {k: (prev_cfg.get(k), cfg[k]) for k in cfg
                    if prev_cfg.get(k) != cfg[k]}
            raise SystemExit(
                f'REFUSING TO RESUME {out}\n'
                f'  on disk : {prev_cfg}\n'
                f'  request : {cfg}\n'
                f'  differs : {diff}\n'
                'These are different experiments sharing one output path. Rename '
                'the existing file or add the differing variable to `tag`.')
        if prev_cfg is None and any(k != '_config' for k in results):
            print(f'  [warn] {out} predates config stamping - cannot verify it '
                  'was trained with this config', flush=True)
        results['_config'] = cfg
        for arm in arms_wanted:
            done = results.setdefault(arm, [])
            done_seeds = {d['seed'] for d in done}
            for seed in seeds:
                if seed in done_seeds:
                    print(arm, 'seed', seed, 'already done -> skip', flush=True)
                    continue
                print(arm, 'seed', seed, 'lr', args.lr, flush=True)
                ckpt = f'checkpoints/synccaps_{tag}_{arm}_seed{seed}.pt'
                r = run_experiment_sync(ds, ncls, arm, seed=seed, lr=args.lr,
                                        arms=arms, save_path=ckpt)
                done.append(r)
                done_seeds.add(seed)
                print('  ', {k: round(v, 3) for k, v in r.items()
                             if isinstance(v, float)}, flush=True)
                with open(out, 'w') as f:
                    json.dump(results, f, indent=1)
        print('full runs done ->', out, flush=True)
