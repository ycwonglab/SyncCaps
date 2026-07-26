#!/usr/bin/env python3
"""Recompute every number the SyncCaps manuscript reports, from the archived logs.

    python verify.py

No dependencies beyond the standard library, no dataset, no GPU. Each check
recomputes a published value from data/*.json and compares it against the value
printed in the paper. Anything that does not match is reported as FAIL.

This verifies the *statistics*, not the training runs. Reproducing the training
itself needs the two public datasets and roughly an hour per arm on one GPU; see
README.md for the dataset links and the exact configuration.
"""
import json
import os
import sys

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
TOL = 0.05          # published values are quoted to 1 or 2 decimals


def load(name):
    with open(os.path.join(DATA, name), encoding='utf-8') as f:
        return json.load(f)


def seeds(arm_runs):
    """[(seed, certain, hybrid, exit_tick), ...] sorted by seed."""
    return sorted((r['seed'], r['test_acc_certain'], r['test_acc_hybrid'],
                   r['mean_exit_tick']) for r in arm_runs)


def mean(xs):
    return sum(xs) / len(xs)


class Report:
    def __init__(self):
        self.rows = []

    def check(self, label, computed, published, tol=TOL):
        ok = abs(computed - published) <= tol
        self.rows.append((ok, label, computed, published))
        return ok

    def note(self, label, computed):
        self.rows.append((None, label, computed, None))

    def render(self):
        width = max(len(row[1]) for row in self.rows)
        failures = 0
        for ok, label, computed, published in self.rows:
            if ok is None:
                print('  ---- {:<{w}}  {:>10.2f}'.format(label, computed, w=width))
                continue
            if not ok:
                failures += 1
            print('  {} {:<{w}}  {:>10.2f}   paper {:>9.2f}'
                  .format('ok  ' if ok else 'FAIL', label, computed, published, w=width))
        return failures


def main():
    r = Report()

    # ---- Table 1: same-stem readout control ------------------------------
    u11_b1 = seeds(load('synccaps_ucf11.json')['B1_sync'])
    u11_b0 = seeds(load('synccaps_ucf11_B0_linear.json')['B0_linear'])
    u101_b1 = seeds(load('synccaps_ucf101_conv4.json')['B1_sync'])
    u101_b0 = seeds(load('synccaps_ucf101_conv4_B0_linear.json')['B0_linear'])

    b1_11, b0_11 = mean([x[1] for x in u11_b1]), mean([x[1] for x in u11_b0])
    b1_101, b0_101 = mean([x[1] for x in u101_b1]), mean([x[1] for x in u101_b0])

    r.check('T1  UCF-11   B0_linear mean', b0_11, 74.9)
    r.check('T1  UCF-11   B1_sync   mean', b1_11, 87.8)
    r.check('T1  UCF-11   delta', b1_11 - b0_11, 12.9, tol=0.1)
    r.check('T1  UCF101   B0_linear mean', b0_101, 76.2)
    r.check('T1  UCF101   B1_sync   mean', b1_101, 86.6)
    # the paper computes this gap from the ROUNDED means (86.6 - 76.2)
    r.check('T1  UCF101   delta (from rounded means)',
            round(b1_101, 1) - round(b0_101, 1), 10.4, tol=0.01)
    r.note('T1  UCF101   delta (unrounded, for reference)', b1_101 - b0_101)

    # ---- Section 7: is the gain the learned decay? -----------------------
    gram = seeds(load('synccaps_ucf11.json')['B2_gram'])
    g_mean = mean([x[1] for x in gram])
    r.check('S7  B2_gram  mean (rho frozen at 0)', g_mean, 86.5)
    r.check('S7  gap  sync - gram', b1_11 - g_mean, 1.2, tol=0.1)
    r.check('S7  B1_sync hybrid mean', mean([x[2] for x in u11_b1]), 86.10, tol=0.01)
    r.check('S7  B2_gram hybrid mean', mean([x[2] for x in gram]), 86.10, tol=0.01)
    r.note('S7  B2_gram seed spread',
           max(x[1] for x in gram) - min(x[1] for x in gram))
    r.note('S7  B1_sync seed spread',
           max(x[1] for x in u11_b1) - min(x[1] for x in u11_b1))

    # ---- Table 3: depth does not stack -----------------------------------
    resnet = seeds(load('synccaps_ucf11_resnet.json')['B1_sync'])
    r.check('T3  UCF-11   ResNet stem mean', mean([x[1] for x in resnet]), 84.4)

    # ---- Table 4: pose-coupling ablation ---------------------------------
    pose = load('synccaps_ucf11_conv4_B1_sync-A_dot-A_cos-A_dot_shuffle.json')
    for arm, published in [('B1_sync', 88.6), ('A_dot', 87.8),
                           ('A_cos', 86.3), ('A_dot_shuffle', 87.6)]:
        r.check('T4  {:<13} mean'.format(arm),
                mean([x[1] for x in seeds(pose[arm])]), published)

    # ---- Table 7: vector-synchronisation ladder --------------------------
    v = load('synccaps_ucf101_conv4_V1_bilinear-V2_aligned-V3_outer.json')
    shuf = load('synccaps_ucf101_conv4_B3_shuffle.json')
    r.check('T7  V1_bilinear mean', mean([x[1] for x in seeds(v['V1_bilinear'])]), 86.0)
    r.check('T7  V3_outer    mean', mean([x[1] for x in seeds(v['V3_outer'])]), 85.1)
    r.check('T7  V2_aligned  mean', mean([x[1] for x in seeds(v['V2_aligned'])]), 84.8)
    r.check('T7  B3_shuffle  mean', mean([x[1] for x in seeds(shuf['B3_shuffle'])]), 86.8)

    # ---- Table 6: measured cost ------------------------------------------
    eff = load('efficiency_numbers.json')['models']
    for key, params, gmacs in [('B1_ucf11', 1708043, 17.89), ('B0_ucf11', 1721099, 17.89),
                               ('B1_ucf101', 1904741, 17.90), ('B0_ucf101', 1928549, 17.90)]:
        m = eff[key]
        r.check('T6  {:<10} params'.format(key), m['params']['total'], params, tol=0)
        r.check('T6  {:<10} GMACs/clip'.format(key), m['thop_macs'] / 1e9, gmacs, tol=0.01)
    r.check('T6  UCF-11 sync   read-out params',
            eff['B1_ucf11']['params']['readout_head'], 12299, tol=0)
    r.check('T6  UCF-11 linear read-out params',
            eff['B0_ucf11']['params']['readout_head'], 25355, tol=0)

    print('\nRecomputed from data/*.json:\n')
    failures = r.render()

    # a qualitative claim that a tolerance cannot express
    best_gram = max(x[1] for x in gram)
    best_sync = max(x[1] for x in u11_b1)
    print('\nSection 7 claim: the frozen-decay arm\'s best seed exceeds both '
          'learned-decay seeds')
    print('  {} {:.2f} (B2_gram) vs {:.2f} (best B1_sync)'
          .format('ok  ' if best_gram > best_sync else 'FAIL', best_gram, best_sync))
    if best_gram <= best_sync:
        failures += 1

    print()
    if failures:
        print('{} check(s) FAILED'.format(failures))
        return 1
    print('All checks passed — every published value matches the archived logs.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
