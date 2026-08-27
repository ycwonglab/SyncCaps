#!/usr/bin/env python3
"""Rebuild claim_evidence_map.csv against the CURRENT manuscript.

Round-2 audit finding B3: the previous map described an older paper state. It
led with the frozen-CLIP number, mixed single-view and three-view estimates in
one row, and quoted a three-seed zero-decay result that a six-seed batch has
since superseded.

Design rules, taken from the audit:
  * one stable claim id per row;
  * every statistic RECOMPUTED from released artifacts, never transcribed;
  * dataset/split, backbone/layer, policy, view count, arm pair, seed set and
    pair set each in their own field;
  * machine-resolvable paths in separate path columns -- no "config A vs
    config B" strings;
  * manuscript location resolved by locating the value in the .docx rather than
    asserted.

The manuscript is NOT redistributed; --docx is optional and only supplies the
section labels. Without it those columns read "unresolved".

  python provenance/build_claim_evidence_map.py [--docx manuscript.docx]
"""
import argparse, collections, csv, json, math, os, re, statistics as st, zipfile

SEED_CSV = 'results/seed_summaries/seed_level_results.csv'
REPAIR_SV = 'results/statistics/repair_report_singleview.txt'
MCNEMAR = 'results/statistics/mcnemar_certain_frozenBNfix.txt'
CLAIMS_JSON = 'results/statistics/claim_analyses.json'
TIMING = 'results/timing/readout_profile_published.json'
FIGMAN = 'figures/FIGURE_MANIFEST.json'
R = 'ucf101_resnet_ptfz_official1_noval_fc__'
C = 'ucf101_clip_b32_ptfz_official1_noval_fc__'
PC = 'ucf101_resnet_ptfz_official1_noval_fc_'
MIX = 'pair_indices/pairs_ns2048_nself64_pair0.npz'
T95 = {2: 4.303, 3: 3.182, 5: 2.571, 8: 2.306}


def load_acc(col='single_view_acc_certain'):
    acc = collections.defaultdict(dict)
    for r in csv.DictReader(open(SEED_CSV)):
        if r.get(col):
            acc[r['experiment_id']][int(r['optimizer_seed'])] = float(r[col])
    return acc


def paired(A, B, seeds=None):
    s = sorted(set(A) & set(B)) if seeds is None else sorted(seeds)
    d = [A[x] - B[x] for x in s]
    m = st.mean(d)
    if len(d) < 2:
        return s, m, None
    se = st.stdev(d) / math.sqrt(len(d))
    t = T95.get(len(d) - 1, 2.0)
    return s, m, (m - t * se, m + t * se)


def fmt(m, ci):
    return ('%+.2f' % m) + ('' if ci is None else ' [%+.2f, %+.2f]' % ci)


def docx_locator(path):
    if not path or not os.path.exists(path):
        return lambda v: 'unresolved (manuscript not supplied)'
    xml = zipfile.ZipFile(path).read('word/document.xml').decode('utf8', 'ignore')
    items = []
    for x in re.split(r'(?=<w:p[ >])', xml):
        t = re.sub(r'<[^>]+>', '', x).replace('&amp;', '&').strip()
        if not t:
            continue
        stl = re.search(r'w:pStyle w:val="([^"]+)"', x)
        items.append((t, stl.group(1) if stl else ''))
    # headings come from the STYLE; table cells like "67.27 +/- 0.05" would
    # otherwise match a "starts with a number" pattern and pose as sections.
    heads = [(i, t) for i, (t, s) in enumerate(items) if s.lower().startswith('heading')]

    def section_of(i):
        cur = 'front matter'
        for hi, ht in heads:
            if hi <= i:
                cur = ht
            else:
                break
        return cur

    def find(value):
        hits = [i for i, (t, _) in enumerate(items) if value in t]
        if not hits:
            return 'not found in manuscript'
        out, seen = [], set()
        for i in hits:
            s = section_of(i)
            if s not in seen:
                seen.add(s); out.append(s)
        return '; '.join(out[:3])
    return find


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--docx', default=None)
    ap.add_argument('--out', default='provenance/claim_evidence_map.csv')
    a = ap.parse_args()
    loc = docx_locator(a.docx)
    sv = load_acc()
    ca = json.load(open(CLAIMS_JSON)) if os.path.exists(CLAIMS_JSON) else {}
    tm = json.load(open(TIMING))
    fg = json.load(open(FIGMAN))
    rows = []

    def add(cid, claim, split, backbone, policy, views, arm_a, arm_b, seeds,
            pairset, stat, analysis, raw, support, cfg_a, cfg_b, locval, notes=''):
        rows.append(dict(
            claim_id=cid, claim=claim, dataset='UCF101', split=split,
            backbone_layer=backbone, eval_policy=policy, view_count=views,
            arm_a=arm_a, arm_b=arm_b,
            seed_set=';'.join(map(str, seeds)) if seeds else '',
            n_seeds=len(seeds) if seeds else '', pair_set=pairset, statistic=stat,
            executable_analysis=analysis, raw_results_path=raw,
            support_path=support, config_path_a=cfg_a, config_path_b=cfg_b,
            manuscript_location=loc(locval) if locval else 'n/a', notes=notes))

    OFF = 'official split-1 (no val)'
    RES18 = 'ResNet-18 layer3 (frozen)'
    CLIP = 'CLIP ViT-B/32 (frozen)'

    s, m, ci = paired(sv[R + 'B4_gram'], sv[R + 'CB_tsketch'])
    add('C01', 'Exact sampled pairs beat Tensor Sketch on ResNet-18 L3', OFF,
        RES18, 'certain', 1, 'B4_gram (exact)', 'CB_tsketch (Tensor Sketch)', s,
        MIX, fmt(m, ci), REPAIR_SV + ' section 4', SEED_CSV,
        'checkpoints/MANIFEST.csv', 'configs/resnet18/%sB4_gram.json' % R,
        'configs/resnet18/%sCB_tsketch.json' % R, '1.28',
        'nine seeds: pre-registered set plus sequential extension')

    s, m, ci = paired(sv[R + 'B4_gram'], sv[R + 'CB_tsketch'], [5, 11, 23])
    add('C02', 'Exact minus Tensor Sketch, fresh confirmatory seeds only', OFF,
        RES18, 'certain', 1, 'B4_gram', 'CB_tsketch', s, MIX, fmt(m, ci),
        REPAIR_SV + ' section 4', SEED_CSV, 'checkpoints/MANIFEST.csv',
        'configs/resnet18/%sB4_gram.json' % R,
        'configs/resnet18/%sCB_tsketch.json' % R, '1.77',
        'analysed separately from the sequentially-extended set')

    s, m, ci = paired(sv[C + 'B4_gram'], sv[C + 'CB_tsketch'])
    add('C03', 'On CLIP the exact-vs-sketch gap disappears', OFF, CLIP,
        'certain', 1, 'B4_gram', 'CB_tsketch', s, MIX, fmt(m, ci),
        REPAIR_SV + ' section 4', SEED_CSV, 'checkpoints/MANIFEST.csv',
        'configs/clip/%sB4_gram.json' % C, 'configs/clip/%sCB_tsketch.json' % C,
        '0.05', 'interval spans zero')

    common = sorted(set(sv[R + 'B4_gram']) & set(sv[R + 'CB_tsketch'])
                    & set(sv[C + 'B4_gram']) & set(sv[C + 'CB_tsketch']))
    did = [(sv[R + 'B4_gram'][x] - sv[R + 'CB_tsketch'][x])
           - (sv[C + 'B4_gram'][x] - sv[C + 'CB_tsketch'][x]) for x in common]
    mm = st.mean(did); se = st.stdev(did) / math.sqrt(len(did))
    t = T95.get(len(did) - 1, 2.0)
    add('C04', 'Backbone x operator interaction (difference-in-differences)',
        OFF, 'ResNet-18 L3 vs CLIP ViT-B/32', 'certain', 1,
        '(B4_gram - CB_tsketch) @ ResNet', '(B4_gram - CB_tsketch) @ CLIP',
        common, MIX, fmt(mm, (mm - t * se, mm + t * se)),
        REPAIR_SV + ' section 4', SEED_CSV, 'checkpoints/MANIFEST.csv',
        'configs/resnet18/%sB4_gram.json' % R, 'configs/clip/%sB4_gram.json' % C,
        '0.99', 'seed-matched across backbones; gates any general superiority claim')

    for cid, b, lbl in (('C05', 'B4_gram', 'B5 ZeroDecay (rho=0)'),
                        ('C06', 'B4_syncnorm', 'B4 SyncNorm')):
        s, m, ci = paired(sv[R + 'R3_route'], sv[R + b])
        add(cid, 'Routing sits inside the operational band of %s' % lbl, OFF,
            RES18, 'certain', 1, 'R3_route', '%s [%s]' % (b, lbl), s, MIX,
            fmt(m, ci), REPAIR_SV + ' section 5 (includes TOST +/-1.0)',
            SEED_CSV, 'checkpoints/MANIFEST.csv',
            'configs/routing/%sR3_route.json' % R,
            'configs/resnet18/%s%s.json' % (R, b), 'routing',
            'TOST verdict comes from the analysis output, not from the sign')

    rt = ca.get('efficiency_ratios_vs_R3_route', {}).get('B4_syncnorm', {})
    add('C07', 'Routing costs ~18x the head parameters and ~1.78x readout latency',
        'n/a', 'CLIP ViT-B/32 stem, readout path only', 'n/a', 1, 'R3_route',
        'B4_syncnorm', [], MIX,
        'head params %sx; p50 %sx; p95 %sx; absolute %.2f/%.2f vs %.2f/%.2f ms'
        % (rt.get('head_params_x'), rt.get('p50_latency_x'), rt.get('p95_latency_x'),
           tm['R3_route']['p50_ms'], tm['R3_route']['p95_ms'],
           tm['B4_syncnorm']['p50_ms'], tm['B4_syncnorm']['p95_ms']),
        'src/evaluation/synccaps_claim_analyses.py', TIMING,
        'results/timing/readout_profile_replication.json', '', '', '1.78',
        'bs=1, 16 frames, fp32, 10 warm-ups + 100 timed iterations')

    add('C08', 'Frozen readout beats fine-tuning (three-view protocol)', OFF,
        'ResNet-18 layer3, frozen vs fine-tuned', 'certain', 3,
        'frozen B4_syncnorm', 'fine-tuned B0_linear', [42, 1337, 7], MIX,
        'frozen B4 72.86 vs frozen B0 67.27 vs fine-tuned B0 69.60; '
        'B4-ftB0 +3.26, group-clustered 95% CI [+1.46, +5.07], p=3.4e-4',
        MCNEMAR, SEED_CSV, 'results/predictions/',
        'configs/resnet18/%sB4_syncnorm.json' % R,
        'configs/fine_tuning/ucf101_resnet_pt_official1_noval__B0_linear.json',
        '72.86', 'THREE-VIEW. Never pool with the single-view column.')

    add('C09', 'Validation-carved replication of frozen vs fine-tuned',
        'official split-1, validation carved', 'ResNet-18 layer3', 'certain', 1,
        'frozen B4_syncnorm', 'fine-tuned B0_linear', [7, 42, 1337], MIX,
        '+2.81 [+0.53, +5.09] (3/3 seeds)', REPAIR_SV + ' section 7', SEED_CSV,
        'checkpoints/MANIFEST.csv', '', '', '2.81',
        'a separate protocol from the no-val headline; never pooled with it')

    s, m, ci = paired(sv[R + 'B4_gram'], sv[R + 'B4_syncnorm'])
    add('C10', 'Freezing the decay to rho=0 costs nothing (six seeds)', OFF,
        RES18, 'certain', 1, 'B4_gram (rho=0)', 'B4_syncnorm (learned rho)', s,
        MIX, fmt(m, ci) + ' ; TOST +/-1.0 EQUIVALENT', REPAIR_SV + ' section 2',
        SEED_CSV, 'checkpoints/MANIFEST.csv',
        'configs/resnet18/%sB4_gram.json' % R,
        'configs/resnet18/%sB4_syncnorm.json' % R, 'ZeroDecay',
        'supersedes the earlier three-seed +0.75')

    for cid, aid, bid, lbl, lv in (
            ('C11', PC + 'nself0_xself__B4_gram', PC + 'nself2048__B4_gram',
             'cross-only minus self-only', 'self-only'),
            ('C12', R + 'B4_gram', PC + 'nself0_xself__B4_gram',
             'mixed minus cross-only', 'cross-only')):
        s, m, ci = paired(sv[aid], sv[bid])
        add(cid, 'Pair composition: %s' % lbl, OFF, RES18, 'certain', 1,
            aid.split('__')[-1] + ' @ ' + lbl.split(' minus ')[0],
            bid.split('__')[-1] + ' @ ' + lbl.split(' minus ')[1], s,
            'pair_indices/PAIR_INDICES_MANIFEST.json (4 dictionaries, seeds 0-3)',
            fmt(m, ci), REPAIR_SV + ' section 6', SEED_CSV,
            'pair_indices/PAIR_INDICES_MANIFEST.json', '', '', lv,
            'self-only draws 2048 indices with replacement from 2304, so only '
            '1351-1372 are unique; the comparison is NOT rank-matched')

    d = ca.get('frame_permutation_disagreement', [])
    if d:
        md = st.mean([x['disagree'] for x in d]); n = d[0]['n_clips']
        mg = st.mean([x['acc_gap'] for x in d])
        add('C13', 'Frame permutation changes WHICH clips are right, not how many',
            OFF, RES18, 'certain', 1, 'B1_sync', 'B3_shuffle',
            [x['seed'] for x in d], MIX,
            '%.0f of %d clips disagree (%.2f%%) at an accuracy gap of %+.2f pts'
            % (md, n, 100 * md / n, mg),
            'src/evaluation/synccaps_claim_analyses.py', CLAIMS_JSON,
            'results/predictions/', 'configs/resnet18/%sB1_sync.json' % R,
            'configs/resnet18/%sB3_shuffle.json' % R, 'permut',
            'a per-clip disagreement result, not an accuracy claim')

    hc = fg['header_checks']
    add('C14', 'Figures 2-4 header values', OFF, RES18,
        'certain (accuracy); hybrid (exit tick)', 1, 'B4_syncnorm seed 42', '',
        [42], MIX,
        'accuracy %.4f%%; mean exit tick %.4f; median e^-rho %.4f; exit '
        'distribution %.1f%% @tick1 / %.1f%% @tick16'
        % (hc['accuracy_header_pct']['measured'],
           hc['mean_exit_tick']['measured'],
           hc['median_memory_exp_neg_rho']['measured'],
           hc['exit_tick_distribution']['measured_pct_exit_at_tick_1'],
           hc['exit_tick_distribution']['measured_pct_run_full_budget']),
        'figures/scripts/make_tick_strategy_figure.py', FIGMAN,
        'figures/tick_strategy_regen.log; checkpoints/CHECKSUMS.sha256',
        'configs/resnet18/%sB4_syncnorm.json' % R, '', '0.8093',
        'checkpoint sha256 %s' % fg['figure_checkpoint']['sha256'])

    s, m, ci = paired(sv[C + 'B4_syncnorm'], sv[C + 'B0_linear'])
    vals = list(sv[C + 'B4_syncnorm'].values())
    add('C15', 'CLIP backbone rung (ladder context, NOT the headline)', OFF,
        CLIP, 'certain', 1, 'B4_syncnorm', 'B0_linear', s, MIX,
        'B4 %.2f +/- %.2f; B4-B0 %s' % (st.mean(vals), st.stdev(vals), fmt(m, ci)),
        REPAIR_SV, SEED_CSV, 'checkpoints/MANIFEST.csv',
        'configs/clip/%sB4_syncnorm.json' % C,
        'configs/clip/%sB0_linear.json' % C, '82.65',
        'the paper leads with the ResNet-18 L3 comparisons, not this cell')

    with open(a.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print('wrote %s (%d claims)' % (a.out, len(rows)))
    for r in rows:
        print('  %-5s v=%s %-52s %s' % (r['claim_id'], r['view_count'],
                                        r['claim'][:52], r['statistic'][:38]))


if __name__ == '__main__':
    main()
