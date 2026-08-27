#!/usr/bin/env python3
"""Materialise every train/val/test partition used in the manuscript.

The experiment code computes splits on the fly (exp_base.make_official_split1
for the published UCF101 protocol, exp_base.make_stratified_splits for the
seeded group-disjoint protocol). Nothing was ever written to disk, so there was
no "split file" to checksum. This script freezes each partition as a sorted
list of clip paths relative to the dataset root, so a reviewer can verify the
partition WITHOUT rerunning the pipeline, and so every config file can name a
concrete SHA-256.

Usage:
    PYTHONPATH=. python provenance/build_splits.py \
        --ucf101 /path/to/UCF101_full --ucf11 /path/to/UCF11_updated_mpg
"""
import argparse, hashlib, json, os
from pathlib import Path

from src.training.exp_base import (UCF11VideoDataset, make_stratified_splits,
                                   make_official_split1)

SEEDS = (42, 1337, 7, 5, 11, 23)


def _rel(ds, i, root):
    p = ds.samples[i]
    p = p[0] if isinstance(p, (list, tuple)) else p
    return os.path.relpath(str(p), root).replace(os.sep, '/')


def dump(ds, root, idx, path):
    """Write one partition as newline-delimited relative clip paths."""
    lines = sorted(_rel(ds, i, root) for i in idx)
    Path(path).write_text('\n'.join(lines) + '\n')
    return len(lines)


def sha256(path):
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ucf101', required=True)
    ap.add_argument('--ucf11', required=True)
    ap.add_argument('--out', default='splits')
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    manifest = {}

    for name, root, ncls in (('ucf101', a.ucf101, 101), ('ucf11', a.ucf11, 11)):
        ds = UCF11VideoDataset(root, sequence_length=16, sample_fps=5.0,
                               augment=False)
        protocols = {}
        if name == 'ucf101':
            # Published UCF101 split-1: test = source groups 1-7.
            protocols['official1'] = make_official_split1(ds, val_groups=(8, 9, 10))
            # SYNC_SPLIT_NOVAL=1: train on all 9537 official train clips, no val,
            # keep the FINAL epoch. This is the protocol behind the headline.
            protocols['official1_noval'] = make_official_split1(ds, val_groups=())
        for s in SEEDS:
            protocols['seeded_seed%d' % s] = make_stratified_splits(ds, seed=s)

        for pname, (tr, va, te) in protocols.items():
            entry = {'dataset': name, 'protocol': pname, 'n_classes': ncls,
                     'n_clips_total': len(ds.samples), 'files': {}}
            for part, idx in (('train', tr), ('val', va), ('test', te)):
                fn = out / ('%s_%s_%s.txt' % (name, pname, part))
                n = dump(ds, root, idx, fn)
                entry['files'][part] = {'file': fn.name, 'n_clips': n,
                                        'sha256': sha256(fn)}
            manifest['%s/%s' % (name, pname)] = entry
            print('%-28s train=%5d val=%5d test=%5d' % (
                '%s/%s' % (name, pname), entry['files']['train']['n_clips'],
                entry['files']['val']['n_clips'], entry['files']['test']['n_clips']))

    (out / 'SPLITS_MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('wrote', out / 'SPLITS_MANIFEST.json')


if __name__ == '__main__':
    main()
