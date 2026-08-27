"""synccaps_mcnemar.py — paired per-clip significance tests over the UCF101 test set.

Motivation (RUN-LEDGER Phase 11b): with 3 seeds the pipeline resolves ~+/-1 point,
so the +1.82 "frozen readout beats fine-tuning" result cannot be established by a
3-seed t-test and does not survive Bonferroni over the four planned comparisons.
Both arms are scored on the SAME 3783 clips, so a paired clip-level test is
available and is far more powerful.

TWO NULLS, REPORTED SEPARATELY, BECAUSE THEY ANSWER DIFFERENT QUESTIONS:

  seed-level t-test  H0: the METHODS do not differ in expectation over training
                     randomness. Honest about seed variance; n=3, underpowered.
  clip-level tests   H0: THESE TRAINED MODELS do not differ over the population
                     of clips. Powerful (n=3783), but conditions on the seeds
                     actually drawn -- it does NOT license a claim about the
                     method averaged over reinitialisation.

Neither subsumes the other. A claim is safe when both point the same way; when
they disagree, the seed-level result is the conservative one to quote. The fully
correct treatment crosses clips and seeds as random effects; with 3 seeds that
model is not identifiable, which is itself worth stating.

The bootstrap exploits the fact that the seed-averaged accuracy difference is the
mean of a per-clip quantity d_i = mean_s[correct_A(s,i)] - mean_s[correct_B(s,i)],
so resampling clips is exactly resampling d -- no need to rebuild the arms.
"""
import argparse
import glob
import itertools
import os

import numpy as np
from scipy import stats

# Defaults reproduce the pre-2026-08-21 analysis. --frozen-tag / --ft-tag
# override them, which is required after the frozen-BatchNorm fix: the corrected
# frozen dumps carry the `_fc` suffix, and loading the old tag here would
# silently pair corrected accuracies with train-mode-BN per-clip predictions.
TAG_FZ = 'ucf101_resnet_ptfz_official1_noval'
TAG_FT = 'ucf101_resnet_pt_official1_noval'


def load(tag, arm, seeds, policy, outdir='perclip'):
    """-> (correct [n_seeds, n_clips] bool, labels, test_idx). Asserts alignment."""
    cor, ref_lab, ref_idx = [], None, None
    for s in seeds:
        f = f'{outdir}/{tag}_{arm}_seed{s}.npz'
        if not os.path.exists(f):
            return None, None, None
        z = np.load(f)
        lab, idx = z['labels'], z['test_idx']
        if ref_lab is None:
            ref_lab, ref_idx = lab, idx
        else:
            # Different arms/seeds must be scored on the same clips in the same
            # order or the pairing is meaningless. Assert rather than trust.
            assert np.array_equal(lab, ref_lab), f'{f}: label vector differs'
            assert np.array_equal(idx, ref_idx), f'{f}: test index order differs'
        cor.append(z[f'preds_{policy}'] == lab)
    return np.array(cor), ref_lab, ref_idx


def mcnemar_exact(a, b):
    """Per-seed McNemar with an exact binomial on the discordant pairs."""
    out = []
    for s in range(a.shape[0]):
        n10 = int(np.sum(a[s] & ~b[s]))       # A right, B wrong
        n01 = int(np.sum(~a[s] & b[s]))       # A wrong, B right
        n = n10 + n01
        p = stats.binomtest(n10, n, 0.5).pvalue if n else 1.0
        out.append((n10, n01, n, p))
    return out


def clip_bootstrap(a, b, n_boot=100000, rng=None):
    """Bootstrap the SEED-AVERAGED accuracy difference over clips."""
    rng = rng or np.random.default_rng(0)
    d = a.mean(0).astype(np.float64) - b.mean(0).astype(np.float64)   # per clip
    n = d.size
    obs = d.mean() * 100
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = d[idx].mean(1) * 100
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p_two = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    se = d.std(ddof=1) / np.sqrt(n) * 100
    # When NO resample crosses zero the bootstrap cannot estimate p, it can only
    # bound it. Return the floor flagged as such so it is never written into a
    # table as if it were a measurement.
    floored = p_two == 0.0
    return obs, lo, hi, (1.0 / n_boot if floored else p_two), se, floored


def clip_tost(a, b, delta, n_boot=100000, rng=None):
    """Percentile-bootstrap TOST: is the difference EQUIVALENT to zero within +/-delta?

    A non-significant p is not evidence of no effect. For the non-additivity
    claim ("fine-tuning on top of the readout adds nothing") the paper needs the
    positive statement, which requires an equivalence test against a margin fixed
    in ADVANCE. We use delta = 1.0 point, the resolution this pipeline already
    states for itself in the ledger -- not a margin chosen after seeing the data.

    H0: |mu| >= delta   vs   H1: |mu| < delta, as two one-sided tests; equivalence
    is declared when the 90% CI lies entirely inside (-delta, +delta).
    """
    rng = rng or np.random.default_rng(1)
    d = a.mean(0).astype(np.float64) - b.mean(0).astype(np.float64)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    boots = d[idx].mean(1) * 100
    lo90, hi90 = np.percentile(boots, [5, 95])
    p1 = float((boots <= -delta).mean())     # H0: mu <= -delta
    p2 = float((boots >= delta).mean())      # H0: mu >= +delta
    p_tost = max(p1, p2)
    equivalent = (lo90 > -delta) and (hi90 < delta)
    return d.mean() * 100, lo90, hi90, p_tost, equivalent

def test_group_ids(dataset, test_idx, dumps=None):
    """Source-video (UCF101 `gNN`) identifier for each dumped test clip.

    UCF101 names clips `v_<Class>_g<group>_c<clip>.avi`, and the clips of one
    group are cuts of the SAME source video. Treating 3,783 such clips as
    independent understates every interval, because a model that gets one clip
    of a group right is far more likely to get its siblings right. The plan's
    inference section therefore asks for resampling at the group level.

    The released prediction dumps carry a `group_id` array (added by
    provenance/build_predictions.py), so prefer that: it lets the statistics be
    rerun WITHOUT a local copy of UCF101. Fall back to rebuilding the dataset
    index for dumps that predate the field.
    """
    if dumps:
        z = np.load(dumps[0], allow_pickle=True)
        if 'group_id' in z.files:
            assert np.array_equal(z['test_idx'], test_idx), (
                '%s: test index order differs from the loaded arms' % dumps[0])
            return z['group_id'].astype(str)
    from src.training.exp_base import UCF11VideoDataset
    from src.training.synccaps_followup_experiment import DATASETS
    path, _ = DATASETS[dataset]
    ds = UCF11VideoDataset(path, sequence_length=16, sample_fps=5.0,
                           augment=False, cache_dir='.cache')
    names = [os.path.basename(str(ds.samples[i][0])) for i in test_idx]
    # strip the `_cNN` clip suffix (and any extension) -> the source video id
    gids = [n.rsplit('_c', 1)[0] if '_c' in n else n for n in names]
    return np.array(gids)


def cluster_bootstrap(a, b, groups, n_boot=100000, delta=1.0, rng=None):
    """Bootstrap the seed-averaged difference by resampling GROUPS, not clips.

    Returns (obs, lo95, hi95, p_two_sided, lo90, hi90, p_tost, equivalent,
             n_groups). Unequal group sizes are handled by resampling each
    group's (sum, count) and dividing the totals, which is the cluster
    bootstrap of a mean rather than a mean of group means.
    """
    rng = rng or np.random.default_rng(1)
    d = a.mean(0).astype(np.float64) - b.mean(0).astype(np.float64)
    uniq, inv = np.unique(groups, return_inverse=True)
    G = uniq.size
    sums = np.bincount(inv, weights=d, minlength=G)
    cnts = np.bincount(inv, minlength=G).astype(np.float64)
    pick = rng.integers(0, G, size=(n_boot, G))
    boots = (sums[pick].sum(1) / cnts[pick].sum(1)) * 100
    lo95, hi95 = np.percentile(boots, [2.5, 97.5])
    lo90, hi90 = np.percentile(boots, [5, 95])
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    p_tost = max(float((boots <= -delta).mean()), float((boots >= delta).mean()))
    return (d.mean() * 100, lo95, hi95, p, lo90, hi90, p_tost,
            (lo90 > -delta) and (hi90 < delta), G)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--policy', default='certain', choices=['certain', 'final', 'hybrid'])
    ap.add_argument('--cluster-groups', action='store_true',
                    help='ALSO bootstrap by UCF101 source-video group. Clips of '
                         'one group are cuts of the same video, so the clip-level '
                         'interval is anti-conservative; this is the interval to '
                         'quote when the two disagree.')
    ap.add_argument('--perclip', default='results/predictions',
                    help='directory holding the per-clip prediction dumps')
    ap.add_argument('--dataset', default='ucf101',
                    help='only used to resolve group ids for --cluster-groups')
    ap.add_argument('--seeds', default='42,1337,7')
    ap.add_argument('--boot', type=int, default=100000)
    ap.add_argument('--equiv-margin', type=float, default=1.0,
                    help='TOST margin in accuracy points; pre-specified at the '
                         "pipeline's stated +/-1 point resolution")
    ap.add_argument('--frozen-tag', default=TAG_FZ,
                    help='perclip tag for the FROZEN arms; use '
                         '..._noval_fc after the frozen-BatchNorm fix')
    ap.add_argument('--ft-tag', default=TAG_FT,
                    help='perclip tag for the FINE-TUNED arms (unaffected by '
                         'the BatchNorm fix, since those stems legitimately '
                         'train their BatchNorm)')
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(',')]

    arms = {}
    ref_idx = None
    for tag, lbl in ((args.frozen_tag, 'frozen'), (args.ft_tag, 'finetuned')):
        for arm in ('B0_linear', 'B1_sync', 'B4_syncnorm', 'B4_gram',
                    'B3_shuffle'):
            c, _, idx = load(tag, arm, seeds, args.policy, outdir=args.perclip)
            if c is not None:
                arms[f'{lbl}/{arm}'] = c
                ref_idx = idx if ref_idx is None else ref_idx
    groups = None
    if args.cluster_groups:
        if ref_idx is None:
            raise SystemExit('--cluster-groups: no dumps loaded, nothing to group')
        groups = test_group_ids(
            args.dataset, ref_idx,
            dumps=sorted(glob.glob(os.path.join(args.perclip, '*.npz'))))
        print(f'[cluster] {ref_idx.size} test clips -> '
              f'{np.unique(groups).size} source-video groups\n')

    print(f'policy = {args.policy}_mv (3 clips x 1 crop), seeds {seeds}\n')
    print('available arms:')
    for k, v in arms.items():
        print(f'  {k:<24} acc {v.mean() * 100:6.2f}  (per seed '
              f'{", ".join(f"{x:.2f}" for x in v.mean(1) * 100)})')

    COMPARISONS = [
        ('frozen/B4_syncnorm', 'frozen/B0_linear',    'readout on frozen features (B4-B0)'),
        ('finetuned/B0_linear', 'frozen/B0_linear',   'fine-tuning helps the linear probe'),
        ('frozen/B4_syncnorm', 'finetuned/B0_linear', 'HEADLINE: frozen readout vs fine-tuned linear'),
        ('finetuned/B4_syncnorm', 'frozen/B4_syncnorm', 'fine-tuning ON TOP of the readout'),
        ('frozen/B1_sync', 'frozen/B0_linear',        'raw sync statistic (B1-B0)'),
        ('frozen/B4_syncnorm', 'frozen/B1_sync',      'normalisation (B4-B1)'),
        # Pre-registered EQUIVALENCE test (Phase 13), not a difference test: the
        # claim is that pinning rho=0 costs nothing, so read the TOST line, not
        # the p-value. A non-significant p alone would not support the claim.
        ('frozen/B4_gram', 'frozen/B4_syncnorm',       'rho=0 freeze: does the learned decay buy anything?'),
        # Order control. Seed-level n=3 cannot resolve the +/-1 margin, but the
        # test partition can: this is the contrast that lets the shuffle "tie"
        # be stated as an equivalence result rather than a failure to detect.
        ('frozen/B1_sync', 'frozen/B3_shuffle',          'frame shuffling: does temporal order matter?'),
    ]
    planned = 4          # the four pre-planned comparisons -> Bonferroni family
    alpha_bonf = 0.05 / planned

    rows = []
    for A, B, name in COMPARISONS:
        if A not in arms or B not in arms:
            print(f'\n--- {name}: SKIPPED (missing {A if A not in arms else B}) ---')
            continue
        a, b = arms[A], arms[B]
        print(f'\n--- {name} ---')
        print(f'    {A}  minus  {B}')

        # seed-level (what the ledger currently quotes)
        per_seed = (a.mean(1) - b.mean(1)) * 100
        t, p_t = stats.ttest_1samp(per_seed, 0)
        n_s = len(per_seed)
        print(f'  seed-level  n={n_s}   {per_seed.mean():+6.2f} +/- '
              f'{per_seed.std(ddof=1):4.2f}  '
              f'({int((per_seed > 0).sum())}/{n_s})  p={p_t:.3f}')

        # clip-level
        print('  clip-level  McNemar (exact binomial on discordant pairs), per seed:')
        for s, (n10, n01, n, p) in zip(seeds, mcnemar_exact(a, b)):
            star = '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else 'ns'
            print(f'      seed {s:<5} A-only {n10:4d} | B-only {n01:4d} | discordant {n:4d} | p={p:.2e} {star}')

        obs, lo, hi, p_b, se, floored = clip_bootstrap(a, b, args.boot)
        print(f'  clip-level  bootstrap ({args.boot:,} resamples of 3783 clips), seed-averaged:')
        pstr = f'<{p_b:.0e}' if floored else f'={p_b:.2e}'
        print(f'      diff {obs:+.2f}   95% CI [{lo:+.2f}, {hi:+.2f}]   p{pstr}   (analytic SE {se:.2f})')
        dm = args.equiv_margin
        _, l90, h90, p_tost, equiv = clip_tost(a, b, dm, args.boot)
        verdict = (f'EQUIVALENT within +/-{dm:.1f}' if equiv
                   else f'NOT shown equivalent within +/-{dm:.1f}')
        print(f'  clip-level  TOST vs +/-{dm:.1f} pt: 90% CI [{l90:+.2f}, {h90:+.2f}]  '
              f'p_TOST={p_tost:.2e}  -> {verdict}')
        if groups is not None:
            (g_obs, g_lo, g_hi, g_p, g_l90, g_h90,
             g_ptost, g_equiv, nG) = cluster_bootstrap(a, b, groups, args.boot, dm)
            print(f'  GROUP-level bootstrap ({args.boot:,} resamples of {nG} '
                  f'source videos), seed-averaged:')
            print(f'      diff {g_obs:+.2f}   95% CI [{g_lo:+.2f}, {g_hi:+.2f}]   '
                  f'p={g_p:.2e}')
            g_verdict = (f'EQUIVALENT within +/-{dm:.1f}' if g_equiv
                         else f'NOT shown equivalent within +/-{dm:.1f}')
            print(f'  GROUP-level TOST vs +/-{dm:.1f} pt: 90% CI '
                  f'[{g_l90:+.2f}, {g_h90:+.2f}]  p_TOST={g_ptost:.2e}  '
                  f'-> {g_verdict}')
        rows.append((name, per_seed.mean(), p_t, obs, lo, hi, p_b, floored))

    print('\n' + '=' * 100)
    print(f'SUMMARY   Bonferroni over the {planned} planned comparisons: alpha = {alpha_bonf:.4f}')
    print('=' * 100)
    print(f"{'comparison':<48}{'diff':>7}{'seed p':>9}{'clip 95% CI':>20}{'clip p':>11}  verdict")
    for name, d, p_t, obs, lo, hi, p_b, floored in rows:
        ok = 'SURVIVES Bonferroni' if p_b < alpha_bonf else 'below threshold'
        pstr = ('<' if floored else ' ') + f'{p_b:.0e}'
        print(f'{name[:47]:<48}{obs:+7.2f}{p_t:>9.3f}   [{lo:+5.2f},{hi:+5.2f}]{pstr:>11}  {ok}')
    print('\n"<" marks a bootstrap p at its resolution floor (1/n_boot): no resample')
    print('crossed zero, so the value is a BOUND, not an estimate. Quote it as "<".')
    print('\nRows beyond the first four were not in the planned family; the Bonferroni')
    print('label is applied to them anyway, which is conservative, not correct-by-design.')
    print('\nRead the two p-columns as different questions, not as a consistency check:')
    print('  seed p = do the METHODS differ over training randomness (n=3, weak).')
    print('  clip p = do THESE MODELS differ over clips (n=3783, strong, conditions on seeds).')
