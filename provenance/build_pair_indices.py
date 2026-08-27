#!/usr/bin/env python3
"""Freeze every pair dictionary and Count-Sketch hash used in the manuscript.

PairwiseSync draws its pair dictionary from a seeded torch.Generator at
construction time; CompactBilinear draws two independent Count-Sketch hash/sign
vectors the same way. Neither was ever written to disk. This script rebuilds
each one from the SAME code path the experiments used and writes it out, so a
reviewer can check the exact indices behind every reported number -- including
the self/cross composition that Section 5.4's controls turn on.

Every dictionary is emitted as a .npz (compressed) plus a JSON sidecar carrying
pair_seed, counts and the SHA-256 of the .npz.

Usage:
    PYTHONPATH=. python provenance/build_pair_indices.py --out pair_indices
"""
import argparse, hashlib, json
from pathlib import Path

import numpy as np
import torch

from src.models.sync_caps import PairwiseSync, CompactBilinear

# d_model = 32 caps-types x caps_grid^2 x caps_dim = 32*3*3*8 at the S3 grid
# used by every reported run. pose_coupling='scalar' pairs over the flat
# d_model neurons, so bound == d_model.
D_MODEL = 32 * 3 * 3 * 8          # 2304

# (n_synch, n_self, pair_seed, exclude_self, role)
# Enumerated from the _config blocks of every committed results file.
DICTS = [
    (2048,   64, 0, False, 'main mixed dictionary (headline; 64 self + 1984 cross)'),
    (2048,   64, 1, False, 'mixed replicate, pair_seed=1'),
    (2048,   64, 2, False, 'mixed replicate, pair_seed=2'),
    (2048,   64, 3, False, 'mixed replicate, pair_seed=3'),
    (2048,    0, 0, True,  'cross-only (exclude_self), pair_seed=0'),
    (2048,    0, 1, True,  'cross-only (exclude_self), pair_seed=1'),
    (2048,    0, 2, True,  'cross-only (exclude_self), pair_seed=2'),
    (2048,    0, 3, True,  'cross-only (exclude_self), pair_seed=3'),
    (2048, 2048, 0, False, 'self-only (all pairs i == j), pair_seed=0'),
    (2048, 2048, 1, False, 'self-only (all pairs i == j), pair_seed=1'),
    (2048, 2048, 2, False, 'self-only (all pairs i == j), pair_seed=2'),
    (2048, 2048, 3, False, 'self-only (all pairs i == j), pair_seed=3'),
    (1024,   64, 0, False, 'mixed dictionary at n_synch=1024 (scaling probe)'),
]

# CompactBilinear (CB_tsketch) is the Tensor-Sketch counterpart of the exact
# pair dictionary: same head width, same stem, projection scheme is the only
# difference. Its hashes move with the SAME pair_seed.
SKETCH_SEEDS = [0, 1, 2, 3]
SKETCH_OUT_DIM = 2048


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='pair_indices')
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    manifest = {}

    for n_synch, n_self, seed, xself, role in DICTS:
        ps = PairwiseSync(D_MODEL, n_synch=n_synch, n_self=n_self, seed=seed,
                          exclude_self=xself, pose_coupling='scalar')
        i = ps.left.numpy().astype(np.int32)
        j = ps.right.numpy().astype(np.int32)
        name = 'pairs_ns%d_nself%d_pair%d%s' % (
            n_synch, n_self, seed, '_xself' if xself else '')
        fn = out / (name + '.npz')
        np.savez_compressed(fn, i_indices=i, j_indices=j)
        n_selfpairs = int((i == j).sum())
        # unordered uniqueness: {i,j} treated as a set
        uniq = len({tuple(sorted(p)) for p in zip(i.tolist(), j.tolist())})
        entry = {
            'file': fn.name, 'role': role, 'pair_seed': seed,
            'n_synch': int(n_synch), 'n_self_requested': int(n_self),
            'exclude_self': bool(xself), 'd_model': D_MODEL,
            'bound': D_MODEL, 'pose_coupling': 'scalar',
            'n_self_pairs_actual': n_selfpairs,
            'n_unique_unordered_pairs': uniq,
            'sha256': sha256(fn),
        }
        manifest[name] = entry
        print('%-34s self=%5d unique=%5d %s' % (name, n_selfpairs, uniq,
                                                entry['sha256'][:12]))

    for seed in SKETCH_SEEDS:
        cb = CompactBilinear(D_MODEL, out_dim=SKETCH_OUT_DIM, seed=seed)
        name = 'tsketch_out%d_pair%d' % (SKETCH_OUT_DIM, seed)
        fn = out / (name + '.npz')
        np.savez_compressed(
            fn,
            h1=cb.h1.numpy().astype(np.int32), s1=cb.s1.numpy().astype(np.int8),
            h2=cb.h2.numpy().astype(np.int32), s2=cb.s2.numpy().astype(np.int8))
        entry = {'file': fn.name,
                 'role': 'Count-Sketch hash/sign pair for CB_tsketch '
                         '(exact-vs-Tensor-Sketch contrast)',
                 'pair_seed': seed, 'out_dim': SKETCH_OUT_DIM,
                 'd_model': D_MODEL, 'sha256': sha256(fn)}
        manifest[name] = entry
        print('%-34s %s' % (name, entry['sha256'][:12]))

    (out / 'PAIR_INDICES_MANIFEST.json').write_text(
        json.dumps(manifest, indent=2) + '\n')
    print('wrote', out / 'PAIR_INDICES_MANIFEST.json')


if __name__ == '__main__':
    main()
