#!/usr/bin/env python3
"""Compare the packaged src/ against the experiment source on `experiment-source`.

Round-2 audit finding B1 asks that a reviewer be able to recover the exact code
revision the results came from. The `experiment-source` branch carries that
source with per-file SHA-256; this script diffs it against the repackaged code
on `main` and classifies every difference, so the repackaging is auditable
rather than asserted.

  IDENTICAL      byte-for-byte equal
  IMPORTS_ONLY   differs only in import lines (flat -> packaged module paths)
  PATHS_ONLY     differs only in hardcoded input/output paths or comments
  EXTENDED       functional change, deliberately made for the release
  MISSING        present in one tree only

Exits non-zero if any MODEL or TRAINING file differs beyond imports, because
that would mean the released code no longer computes what produced the numbers.

  git worktree add ../expsrc experiment-source
  PYTHONPATH=. python provenance/verify_source_equivalence.py --snapshot ../expsrc
"""
import argparse, difflib, hashlib, json, os, re, sys

# Diffs are published, so they must not carry machine-specific absolute paths
# from either side of the comparison -- including ones the release DELETED.
REDACT = [
    (re.compile(r'/tmp/[^\s"\']*'), '<local-scratch-path>'),
    (re.compile(r'/home/[^\s"\']*'), '<local-home-path>'),
    (re.compile(r'/mnt/[a-z]/[^\s"\']*'), '<local-mount-path>'),
]


def redact(lines):
    out = []
    for l in lines:
        for rx, rep in REDACT:
            l = rx.sub(rep, l)
        out.append(l)
    return out

MAP = {
    'exp_base.py': 'src/training/exp_base.py',
    'synccaps_probe_experiment.py': 'src/training/synccaps_probe_experiment.py',
    'synccaps_followup_experiment.py': 'src/training/synccaps_followup_experiment.py',
    'synccaps_precompute_stem.py': 'src/training/synccaps_precompute_stem.py',
    'synccaps_multiclip_eval.py': 'src/evaluation/synccaps_multiclip_eval.py',
    'synccaps_perclip_dump.py': 'src/evaluation/synccaps_perclip_dump.py',
    'synccaps_route_probe.py': 'src/evaluation/synccaps_route_probe.py',
    'synccaps_matrix_table.py': 'src/evaluation/synccaps_matrix_table.py',
    'synccaps_repair_report.py': 'src/evaluation/synccaps_repair_report.py',
    'synccaps_mcnemar.py': 'src/evaluation/synccaps_mcnemar.py',
    'synccaps_readout_profile.py': 'src/profiling/synccaps_readout_profile.py',
    'src/models/sync_caps.py': 'src/models/sync_caps.py',
    'src/models/temporal_routing.py': 'src/models/temporal_routing.py',
    'src/models/capsule_layers.py': 'src/models/capsule_layers.py',
    'docs/paper/make_tick_strategy_figure.py': 'figures/scripts/make_tick_strategy_figure.py',
    'docs/paper/make_neuron_dynamics_figure.py': 'figures/scripts/make_neuron_dynamics_figure.py',
    'docs/paper/make_activity_web_grid.py': 'figures/scripts/make_activity_web_grid.py',
    'docs/paper/make_activity_web_grid_peaktick.py': 'figures/scripts/make_activity_web_grid_peaktick.py',
    'docs/paper/make_fixedmap_figure.py': 'figures/scripts/make_fixedmap_figure.py',
}
CRITICAL = ['src/models/sync_caps.py', 'src/models/temporal_routing.py',
            'src/models/capsule_layers.py', 'src/training/exp_base.py',
            'src/training/synccaps_probe_experiment.py',
            'src/training/synccaps_followup_experiment.py']
IMPORT_RE = re.compile(r'^\s*(from|import)\s')
PATHY_RE = re.compile(r'docs/paper|gating_results|figures/rendered|results/|perclip|/tmp/|scratchpad')


def classify(a_lines, b_lines):
    diff = [l for l in difflib.unified_diff(a_lines, b_lines, n=0)
            if l.startswith(('+', '-')) and not l.startswith(('+++', '---'))]
    if not diff:
        return 'IDENTICAL', 0, []
    body = [l[1:] for l in diff]
    if all(IMPORT_RE.match(l) for l in body):
        return 'IMPORTS_ONLY', len(diff), diff
    if all(IMPORT_RE.match(l) or PATHY_RE.search(l) or l.strip().startswith('#')
           or not l.strip() for l in body):
        return 'PATHS_ONLY', len(diff), diff
    return 'EXTENDED', len(diff), diff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--snapshot', required=True)
    ap.add_argument('--repo', default='.')
    ap.add_argument('--out', default='provenance/source_equivalence.json')
    a = ap.parse_args()
    report, counts = {}, {}
    for src, dst in sorted(MAP.items()):
        p, q = os.path.join(a.snapshot, src), os.path.join(a.repo, dst)
        if not os.path.exists(p) or not os.path.exists(q):
            report[dst] = {'origin': src, 'status': 'MISSING',
                           'snapshot_exists': os.path.exists(p),
                           'repo_exists': os.path.exists(q)}
            counts['MISSING'] = counts.get('MISSING', 0) + 1
            continue
        A, B = open(p, encoding='utf8').read(), open(q, encoding='utf8').read()
        st_, n, diff = classify(A.splitlines(), B.splitlines())
        counts[st_] = counts.get(st_, 0) + 1
        report[dst] = {'origin': src, 'status': st_, 'changed_lines': n,
                       'sha256_snapshot': hashlib.sha256(A.encode()).hexdigest(),
                       'sha256_repo': hashlib.sha256(B.encode()).hexdigest(),
                       'diff': redact(diff if st_ == 'EXTENDED' else diff[:20])}
        print('%-13s %-52s <- %s' % (st_, dst, src))
    json.dump({'summary': counts, 'critical_files': CRITICAL, 'files': report},
              open(a.out, 'w'), indent=2)
    print('\nsummary:', counts)
    print('wrote', a.out)
    bad = [c for c in CRITICAL
           if report.get(c, {}).get('status') not in ('IDENTICAL', 'IMPORTS_ONLY')]
    if bad:
        print('\nFAIL: model/training code differs beyond imports:', bad)
        return 1
    print('\nOK: all model and training code is identical to the experiment '
          'source modulo import paths.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
