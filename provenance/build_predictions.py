#!/usr/bin/env python3
"""Enrich the per-clip prediction dumps with clip and source-video identifiers.

synccaps_perclip_dump.py stores predictions against `test_idx`, an index into
the dataset's sample list. That is enough to recompute accuracy but NOT enough
to run the two analyses the dumps exist for: a clip-level McNemar test needs a
stable clip identity, and a source-video clustered bootstrap needs the group id
(UCF101's g<NN> token). Both are recoverable exactly by replaying the same
dataset construction, so nothing here is re-inferred or re-estimated -- the
prediction arrays are copied through byte-for-byte.

Usage:
    PYTHONPATH=. python provenance/build_predictions.py \
        --perclip /path/to/perclip --ucf101 /path/to/UCF101_full \
        --out results/predictions
"""
import argparse, glob, hashlib, json, os
from pathlib import Path

import numpy as np

from src.training.exp_base import UCF11VideoDataset


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--perclip', required=True)
    ap.add_argument('--ucf101', required=True)
    ap.add_argument('--out', default='results/predictions')
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    ds = UCF11VideoDataset(a.ucf101, sequence_length=16, sample_fps=5.0,
                           augment=False)
    root = os.path.abspath(a.ucf101)

    def rel(i):
        p = ds.samples[i]
        p = p[0] if isinstance(p, (list, tuple)) else p
        return os.path.relpath(str(p), root).replace(os.sep, '/')

    idx_to_class = {v: k for k, v in ds.class_to_idx.items()}
    manifest = {}
    for src in sorted(glob.glob(os.path.join(a.perclip, '*.npz'))):
        d = np.load(src, allow_pickle=True)
        ti = d['test_idx']
        clip_id = np.array([rel(int(i)) for i in ti])
        group_id = np.array([str(ds.groups[int(i)]) for i in ti])
        payload = {k: d[k] for k in d.files}
        payload['clip_id'] = clip_id
        payload['group_id'] = group_id
        payload['label_name'] = np.array(
            [idx_to_class[int(y)] for y in d['labels']])
        name = os.path.basename(src)
        dst = out / name
        np.savez_compressed(dst, **payload)
        manifest[name] = {
            'file': name, 'source_dump': os.path.basename(src),
            'n_clips': int(ti.size),
            'n_groups': int(len(set(group_id.tolist()))),
            'arrays': sorted(payload),
            'acc_certain': float(d['acc_certain']),
            'acc_hybrid': float(d['acc_hybrid']),
            'sha256': sha256(dst),
        }
        print('%-64s clips=%d groups=%d' % (name, ti.size,
                                            manifest[name]['n_groups']))
    (out / 'PREDICTIONS_MANIFEST.json').write_text(
        json.dumps(manifest, indent=2) + '\n')
    print('wrote', out / 'PREDICTIONS_MANIFEST.json', '(%d files)' % len(manifest))


if __name__ == '__main__':
    main()
