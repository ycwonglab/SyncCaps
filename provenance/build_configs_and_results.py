#!/usr/bin/env python3
"""Emit one machine-readable config per reported experiment, plus the
seed-level raw results table and the experiment manifest.

Every committed results file carries a `_config` block recording the arguments
that produced it. Everything else in a run is fixed by the code path
(synccaps_probe_experiment.run_experiment_sync) and is inlined here as
FIXED_PROTOCOL so each config file is self-contained: a reader never has to
open the source to learn the optimizer, the schedule, the frame policy or the
early-exit rule.

Usage:
    python provenance/build_configs_and_results.py \
        --gating /path/to/gating_results --checkpoints /path/to/checkpoints
"""
import argparse, csv, json, os, glob, hashlib
from pathlib import Path

# --- constants fixed by the code path, not by CLI arguments -----------------
FIXED_PROTOCOL = {
    'optimizer': 'Adam',
    'weight_decay': 1e-4,
    'lr_schedule': 'linear warmup 3 epochs, then cosine decay to 0',
    'warmup_epochs': 3,
    'epochs': 12,
    'batch_size': 4,
    'frames_per_clip': 16,
    'sample_fps': 5.0,
    'frame_size': [224, 224],
    'clip_policy': 'single window at clip_start=0.0 unless multi-clip eval',
    'augment': False,
    'n_ticks': 16,
    'certainty_threshold_theta': 0.5,
    'primary_caps': {'caps_grid': 3, 'n_caps_types': 32, 'caps_dim': 8,
                     'd_model': 2304},
    'routing_iterations': {'R3_route': 3, '_other_arms': 0},
    'SYNC_FROZEN_BN_TRAIN': '0 (unset) - frozen stems keep BatchNorm in eval '
                            'mode; asserted on every .train() call',
    'eval_policy': {
        'test_acc_certain': 'argmax over all T ticks by certainty; NON-causal',
        'test_acc_final': 'logits at the final tick',
        'test_acc_hybrid': 'prefix-mean logits with theta=0.5 early exit; '
                           'CAUSAL, and the only source of mean_exit_tick',
    },
}
MULTICLIP = {'n_views': 3, 'aggregation': 'mean of per-view logits',
             'batch_size': 4}


def family(cfg, arms):
    if 'R3_route' in arms:
        return 'routing'
    if (cfg.get('pair_seed', 0) or cfg.get('n_self', 64) != 64
            or cfg.get('exclude_self')):
        return 'pair_composition'
    if cfg.get('pretrained') and not cfg.get('freeze_stem'):
        return 'fine_tuning'
    stem = cfg.get('stem', '')
    if stem.startswith('clip'):
        return 'clip'
    if stem == 'conv4':
        return 'conv4'
    return 'resnet18'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gating', required=True)
    ap.add_argument('--checkpoints', required=True)
    ap.add_argument('--configs', default='configs')
    ap.add_argument('--results', default='results/seed_summaries')
    ap.add_argument('--splits-manifest', default='splits/SPLITS_MANIFEST.json')
    ap.add_argument('--manifest', default='provenance/experiment_manifest.csv')
    a = ap.parse_args()

    splits = {}
    if os.path.exists(a.splits_manifest):
        splits = json.load(open(a.splits_manifest))

    # three-clip (n_views=3) re-scores live in separate multiclip_<tag>.json
    # files, keyed by the same run tag. Fold them in so one CSV carries both
    # the single-view and the three-clip number for every seed.
    multiclip = {}
    for mp in glob.glob(os.path.join(a.gating, 'multiclip_*.json')):
        mtag = os.path.basename(mp)[len('multiclip_'):-len('.json')]
        md = json.load(open(mp))
        if not isinstance(md, dict):
            continue
        for marm, recs in md.items():
            if marm.startswith('_') or not isinstance(recs, list):
                continue
            for r in recs:
                multiclip[(mtag, marm, r.get('seed'))] = r

    rows, manifest_rows, n_cfg = [], [], 0
    pending = {}          # experiment_id -> (config path, merged config dict)
    for p in sorted(glob.glob(os.path.join(a.gating, 'synccaps_*.json'))):
        d = json.load(open(p))
        if not isinstance(d, dict) or '_config' not in d:
            continue
        cfg = d['_config']
        arms = [k for k in d if not k.startswith('_')]
        base = os.path.basename(p)[len('synccaps_'):-len('.json')]
        tag = base
        suffix = '_' + '-'.join(arms)
        if tag.endswith(suffix):
            tag = tag[:-len(suffix)]

        fam = family(cfg, arms)
        # resolve the split this run used
        split_tok = cfg.get('split', '')
        if 'official1_noval' in split_tok:
            skey = '%s/official1_noval' % cfg['dataset']
        elif 'official1' in split_tok:
            skey = '%s/official1' % cfg['dataset']
        else:
            skey = None   # seeded: one split per optimizer seed

        for arm in arms:
            exp_id = '%s__%s' % (tag, arm)
            seeds = sorted(r['seed'] for r in d[arm])
            batch_id = '%s#%s' % (exp_id, os.path.basename(p))
            cfile = Path(a.configs) / fam / (exp_id + '.json')
            cfile.parent.mkdir(parents=True, exist_ok=True)

            split_block = {'protocol': split_tok or 'seeded StratifiedGroupKFold'}
            if skey and skey in splits:
                split_block['files'] = splits[skey]['files']
            elif skey is None:
                split_block['per_seed_files'] = {
                    str(s): splits.get('%s/seeded_seed%d' % (cfg['dataset'], s),
                                       {}).get('files', {})
                    for s in seeds}

            n_synch = cfg.get('nsynch')
            n_self = cfg.get('n_self', 64)
            pair_seed = cfg.get('pair_seed', 0)
            xself = bool(cfg.get('exclude_self', False))
            pair_file = ('pairs_ns%s_nself%s_pair%s%s.npz'
                         % (n_synch, n_self, pair_seed,
                            '_xself' if xself else ''))

            conf = {
                'experiment_id': exp_id,
                'arm': arm,
                'family': fam,
                'source_results_file': os.path.basename(p),
                'dataset': cfg['dataset'],
                'backbone': cfg['stem'],
                'extraction_layer': {
                    'resnet': 'ResNet-18 layer4 output, 1x1 conv -> out_ch',
                    'clip_b32': 'CLIP ViT-B/32 final patch tokens, 1x1 conv -> out_ch',
                    'conv4': '4-layer conv stem trained from scratch',
                }.get(cfg['stem'], cfg['stem']),
                'pretrained': bool(cfg.get('pretrained')),
                'freeze_stem': bool(cfg.get('freeze_stem')),
                'frozen_feature_cache': bool(cfg.get('feat_cache')),
                'out_ch': cfg.get('out_ch', 256),
                'split': split_block,
                'learning_rate': cfg.get('lr'),
                'differential_lr': bool(cfg.get('diff_lr')),
                'backbone_lr': cfg.get('backbone_lr'),
                'optimizer_seeds': seeds,
                'pair_dictionary': {
                    'n_synch': n_synch, 'n_self': n_self,
                    'pair_seed': pair_seed, 'exclude_self': xself,
                    'file': 'pair_indices/' + pair_file,
                },
                'sync_normalisation': (
                    'signed-sqrt + L2 (improved B-CNN)'
                    if arm in ('B4_syncnorm', 'B4_gram', 'B5_concat',
                               'CB_tsketch') else 'none'),
                'decay': ('rho frozen at 0 (r = 1, order-agnostic Gram)'
                          if arm in ('B2_gram', 'B4_gram', 'CB_tsketch',
                                     'LR_bilinear')
                          else 'learned per-pair rho, r = exp(-rho)'),
                'frame_order': ('shuffled within clip (order control)'
                                if arm == 'B3_shuffle' else 'native'),
                'route_w_scale': cfg.get('route_w_scale'),
                'epoch_selection': ('final epoch (no validation set)'
                                    if 'noval' in split_tok
                                    else 'best validation accuracy'),
                'trainable_parameters': ('readout head only'
                                         if cfg.get('freeze_stem')
                                         else 'stem + readout head'),
                'protocol': FIXED_PROTOCOL,
                'multiclip_eval': MULTICLIP,
                'checkpoints': ['synccaps_%s_%s_seed%d.pt' % (tag, arm, s)
                                for s in seeds],
            }
            # merge, do not overwrite: a second results file for the same
            # experiment_id is an ADDITIONAL seed batch, not a replacement.
            # (Round-2 audit: five ids previously lost their first batch here.)
            prev = pending.get(exp_id)
            if prev is None:
                conf['seed_batches'] = [{'results_file': os.path.basename(p),
                                         'batch_id': batch_id, 'seeds': seeds}]
                pending[exp_id] = (cfile, conf)
            else:
                pcfile, pconf = prev
                pconf['optimizer_seeds'] = sorted(set(pconf['optimizer_seeds'])
                                                  | set(seeds))
                pconf['seed_batches'].append({'results_file': os.path.basename(p),
                                              'batch_id': batch_id,
                                              'seeds': seeds})
                pconf['source_results_file'] = ', '.join(
                    b['results_file'] for b in pconf['seed_batches'])
                pconf['checkpoints'] = sorted(set(pconf['checkpoints'])
                                              | set(conf['checkpoints']))

            for r in d[arm]:
                ckname = 'synccaps_%s_%s_seed%d.pt' % (tag, arm, r['seed'])
                ckpath = os.path.join(a.checkpoints, ckname)
                rows.append({
                    'experiment_id': exp_id,
                    'result_batch_id': batch_id,
                    'config_path': str(cfile).replace(os.sep, '/'),
                    'dataset': cfg['dataset'], 'backbone': cfg['stem'],
                    'arm': arm, 'family': fam,
                    'split': split_tok or 'seeded',
                    'optimizer_seed': r['seed'], 'pair_seed': pair_seed,
                    'n_synch': n_synch, 'n_self': n_self,
                    'exclude_self': xself, 'lr': cfg.get('lr'),
                    'single_view_acc_certain': r.get('test_acc_certain'),
                    'single_view_acc_final': r.get('test_acc_final'),
                    'single_view_acc_hybrid': r.get('test_acc_hybrid'),
                    'mean_exit_tick': r.get('mean_exit_tick'),
                    'best_val': r.get('best_val'),
                    'epoch_selection': conf['epoch_selection'],
                    'checkpoint': ckname,
                    'checkpoint_present': os.path.exists(ckpath),
                    'status': 'completed', 'exit_code': 0,
                })
                mv = multiclip.get((tag, arm, r['seed']))
                rows[-1].update({
                    'three_clip_acc_certain': mv.get('test_acc_certain_mv') if mv else None,
                    'three_clip_acc_final':   mv.get('test_acc_final_mv') if mv else None,
                    'three_clip_acc_hybrid':  mv.get('test_acc_hybrid_mv') if mv else None,
                    'three_clip_mean_exit_tick': mv.get('mean_exit_tick_mv') if mv else None,
                    'multiclip_n_views': mv.get('n_views') if mv else None,
                    # 0.0 means the three-clip re-score reproduced the stored
                    # single-view accuracy exactly, i.e. the checkpoint loaded
                    # faithfully. Non-zero would indicate a load/eval drift.
                    'single_view_drift': mv.get('single_view_drift') if mv else None,
                })
            manifest_rows.append({
                'experiment_id': exp_id, 'result_batch_id': batch_id,
                'family': fam,
                'config_path': str(cfile).replace(os.sep, '/'),
                'results_file': os.path.basename(p),
                'dataset': cfg['dataset'], 'backbone': cfg['stem'],
                'arm': arm, 'n_seeds': len(seeds),
                'seeds': ';'.join(map(str, seeds)),
                'pair_seed': pair_seed, 'n_synch': n_synch, 'n_self': n_self,
            })

    for cfile, conf in pending.values():
        cfile.write_text(json.dumps(conf, indent=2) + '\n')
        n_cfg += 1

    Path(a.results).mkdir(parents=True, exist_ok=True)
    rp = Path(a.results) / 'seed_level_results.csv'
    with open(rp, 'w', newline='') as f:
        cols = []
        for r in rows:
            for k in r:
                if k not in cols:
                    cols.append(k)
        for r in rows:
            for k in cols:
                r.setdefault(k, None)
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader(); w.writerows(rows)
    Path(a.manifest).parent.mkdir(parents=True, exist_ok=True)
    with open(a.manifest, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(manifest_rows[0]))
        w.writeheader(); w.writerows(sorted(manifest_rows,
                                            key=lambda r: r['experiment_id']))
    print('configs written : %d (unique experiment ids)' % n_cfg)
    multi = [e for e, (_, c) in pending.items() if len(c['seed_batches']) > 1]
    print('ids with >1 seed batch: %d' % len(multi))
    print('seed-level rows : %d -> %s' % (len(rows), rp))
    print('manifest rows   : %d -> %s' % (len(manifest_rows), a.manifest))
    miss = [r for r in rows if not r['checkpoint_present']]
    print('rows missing a checkpoint: %d' % len(miss))
    mvn = sum(1 for r in rows if r.get('three_clip_acc_certain') is not None)
    print('rows with a three-clip re-score: %d / %d' % (mvn, len(rows)))
    drift = [r for r in rows if r.get('single_view_drift') not in (None, 0.0)]
    print('rows with non-zero single-view drift: %d' % len(drift))


if __name__ == '__main__':
    main()
