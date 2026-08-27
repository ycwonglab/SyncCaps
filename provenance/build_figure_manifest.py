#!/usr/bin/env python3
"""Record what the manuscript figures were drawn from, and check their headers.

push2github.txt asks for a manifest confirming the figure header values. Rather
than transcribe them, this recomputes each one from the primary artifact -- the
seed-42 checkpoint and the seed-level results table -- so the manifest is a
CHECK, not a restatement.

The exit-tick distribution is the one value that cannot be read off either: it
needs the per-tick logits, which the prediction dumps do not store. It is
therefore parsed from the figure script's own stdout when a regeneration log is
supplied, and left explicitly unverified otherwise.
"""
import argparse, csv, hashlib, json, os, re
from pathlib import Path

import numpy as np
import torch

FIG_CKPT = 'synccaps_ucf101_resnet_ptfz_official1_noval_fc_B4_syncnorm_seed42.pt'


def sha256(p, chunk=1 << 20):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoints', required=True)
    ap.add_argument('--results', default='results/seed_summaries/seed_level_results.csv')
    ap.add_argument('--figures', default='figures')
    ap.add_argument('--regen-log', default=None,
                    help='stdout of make_tick_strategy_figure.py, which prints '
                         'the exit-tick distribution')
    a = ap.parse_args()

    ck = os.path.join(a.checkpoints, FIG_CKPT)
    sd = torch.load(ck, map_location='cpu', weights_only=False)
    for k in ('model', 'state_dict'):
        if isinstance(sd, dict) and k in sd:
            sd = sd[k]
    mem = torch.exp(-sd['sync.rho']).numpy()

    row = next(r for r in csv.DictReader(open(a.results))
               if r['checkpoint'] == FIG_CKPT)

    checks = {
        'accuracy_header_pct': {
            'expected': 71.8, 'measured': round(float(row['single_view_acc_certain']), 4),
            'source': 'results/seed_summaries/seed_level_results.csv '
                      '(single_view_acc_certain)',
        },
        'mean_exit_tick': {
            'expected': 11.1, 'measured': round(float(row['mean_exit_tick']), 4),
            'source': 'results/seed_summaries/seed_level_results.csv '
                      '(mean_exit_tick, hybrid policy)',
        },
        'median_memory_exp_neg_rho': {
            'expected': 0.809, 'measured': round(float(np.median(mem)), 4),
            'source': 'exp(-sync.rho) of the figure checkpoint',
        },
    }
    for c in checks.values():
        c['agrees'] = abs(c['measured'] - c['expected']) <= 0.05

    dist = {'expected_pct_exit_at_tick_1': 26, 'expected_pct_run_full_budget': 64,
            'measured_pct_exit_at_tick_1': None,
            'measured_pct_run_full_budget': None,
            'source': 'printed by figures/scripts/make_tick_strategy_figure.py; '
                      'needs per-tick logits, which the prediction dumps do not '
                      'store',
            'agrees': None}
    if a.regen_log and os.path.exists(a.regen_log):
        txt = Path(a.regen_log).read_text()
        m = re.search(r'BIMODAL:\s*([\d.]+)%\s*at tick 1,\s*([\d.]+)%\s*at tick', txt)
        if m:
            dist['measured_pct_exit_at_tick_1'] = float(m.group(1))
            dist['measured_pct_run_full_budget'] = float(m.group(2))
            dist['agrees'] = (abs(float(m.group(1)) - 26) <= 1.5
                              and abs(float(m.group(2)) - 64) <= 1.5)
    checks['exit_tick_distribution'] = dist

    manifest = {
        'figure_checkpoint': {
            'filename': FIG_CKPT, 'sha256': sha256(ck),
            'bytes': os.path.getsize(ck),
            'experiment_id': row['experiment_id'],
            'config': row['config_path'], 'optimizer_seed': int(row['optimizer_seed']),
        },
        'exact_command': (
            'PYTHONPATH=. python figures/scripts/make_tick_strategy_figure.py '
            'figures/rendered'),
        'other_figure_commands': [
            'PYTHONPATH=. python figures/scripts/make_neuron_dynamics_figure.py',
            'PYTHONPATH=. python figures/scripts/make_activity_web_grid_peaktick.py',
            'SYNCCAPS_FIG_CFG=legacy PYTHONPATH=. python '
            'figures/scripts/make_neuron_dynamics_figure.py   # superseded config',
        ],
        'header_checks': checks,
        'inputs': sorted(os.path.basename(p) for p in
                         Path(a.figures, 'inputs').glob('*')),
        'rendered': sorted(os.path.basename(p) for p in
                           Path(a.figures, 'rendered').glob('*')),
        'note': 'Figures were regenerated on the frozen-BatchNorm-corrected '
                'checkpoint. Artifacts drawn on the pre-correction checkpoint '
                '(accuracy 70.10 rather than 71.82) are superseded.',
    }
    outp = Path(a.figures) / 'FIGURE_MANIFEST.json'
    outp.write_text(json.dumps(manifest, indent=2) + '\n')
    for k, c in checks.items():
        print('%-32s expected %-7s measured %-9s %s' % (
            k, c.get('expected'), c.get('measured'),
            {True: 'OK', False: 'MISMATCH', None: 'UNVERIFIED'}[c['agrees']]))
    print('wrote', outp)


if __name__ == '__main__':
    main()
