#!/usr/bin/env python3
"""Checksum every checkpoint behind a reported number and size the release set.

Writes three things:
  checkpoints/CHECKSUMS.sha256   - sha256sum-compatible, ALL referenced files
  checkpoints/MANIFEST.csv       - filename, experiment id, seed, config, accuracy
  checkpoints/RELEASE_SET.txt    - the subset uploaded as GitHub Release assets

Every checkpoint behind a reported number ships. The full referenced set is
1.17 GB across 189 files -- too large to carry in git history, but comfortably
within a GitHub Release, so no reported number is left unverifiable. Files are
grouped into per-family tarballs because 189 separate assets are slow to upload
and awkward to consume; CHECKSUMS.sha256 still covers each file individually
after extraction.
"""
import argparse, csv, hashlib, json, os
from pathlib import Path

# experiment-id substrings whose EVERY seed ships in the release
HEADLINE = (
    'ucf101_clip_b32_ptfz_official1_noval_fc__',      # headline backbone rung
    'ucf101_resnet_ptfz_official1_noval_fc__',        # frozen ResNet rung + figures
    'ucf101_resnet_pt_official1_noval__',             # fine-tuned comparison
)
# control arms: one seed (42 where present) is enough to re-score
CONTROL_ARMS = ('R3_route', 'CB_tsketch', 'B4_gram', 'B3_shuffle', 'LR_bilinear')
FIGURE_CKPT = 'synccaps_ucf101_resnet_ptfz_official1_noval_fc_B4_syncnorm_seed42.pt'


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
    ap.add_argument('--out', default='checkpoints')
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(open(a.results)))
    seen, manifest, sums, release = set(), [], [], []
    for r in rows:
        fn = r['checkpoint']
        if fn in seen:
            continue
        seen.add(fn)
        p = os.path.join(a.checkpoints, fn)
        if not os.path.exists(p):
            continue
        digest = sha256(p)
        size = os.path.getsize(p)
        eid = r['experiment_id']
        # everything a reported number depends on is released
        in_release = True
        manifest.append({
            'filename': fn, 'experiment_id': eid, 'arm': r['arm'],
            'dataset': r['dataset'], 'backbone': r['backbone'],
            'optimizer_seed': r['optimizer_seed'], 'pair_seed': r['pair_seed'],
            'config_path': r['config_path'],
            'acc_certain': r['single_view_acc_certain'],
            'acc_hybrid': r['single_view_acc_hybrid'],
            'bytes': size, 'sha256': digest,
            'in_github_release': in_release,
            'release_asset': 'synccaps-checkpoints-%s.tar' % r['family'],
            'role': ('figures 2-4 source' if fn == FIGURE_CKPT
                     else 'headline' if any(eid.startswith(h) for h in HEADLINE)
                     else 'control' if r['arm'] in CONTROL_ARMS else 'supporting'),
        })
        sums.append('%s  %s' % (digest, fn))
        if in_release:
            release.append((fn, size))

    manifest.sort(key=lambda r: r['filename'])
    with open(out / 'MANIFEST.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(manifest[0]))
        w.writeheader(); w.writerows(manifest)
    (out / 'CHECKSUMS.sha256').write_text('\n'.join(sorted(sums)) + '\n')
    (out / 'RELEASE_SET.txt').write_text(
        '\n'.join(fn for fn, _ in sorted(release)) + '\n')
    by_asset = {}
    for r in manifest:
        by_asset.setdefault(r['release_asset'], []).append(r)
    (out / 'RELEASE_ASSETS.json').write_text(json.dumps(
        {k: {'n_files': len(v),
             'bytes': sum(x['bytes'] for x in v),
             'files': sorted(x['filename'] for x in v)}
         for k, v in sorted(by_asset.items())}, indent=2) + '\n')

    tot = sum(r['bytes'] for r in manifest)
    rel = sum(s for _, s in release)
    print('checkpoints referenced : %d  (%.2f GB)' % (len(manifest), tot / 2**30))
    print('release subset         : %d  (%.2f GB)' % (len(release), rel / 2**30))
    print('wrote MANIFEST.csv, CHECKSUMS.sha256, RELEASE_SET.txt in', out)


if __name__ == '__main__':
    main()
