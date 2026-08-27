# Response to the Stage 4.5 round-2 artifact audit

> **Scope.** This is a *process record*: what an internal integrity audit asked
> for on 2026-08-27 and what changed in response. It is **not** documentation of
> the artifacts. Everything a reviewer needs in order to read this package
> correctly lives in [`README.md`](README.md) and
> [`environment/system-info.txt`](environment/system-info.txt); this file points
> at those rather than restating them, so the two cannot drift apart.
>
> If you are here to evaluate the research, read `README.md` instead.

Audited: `main` at `3eea479` and release `v1.0-artifacts` (both pre-rewrite SHAs;
see the note at the end).

---

## B1 — the recorded source revision was not public

**The finding was right, but its premise was wrong, so the fix is different from
the one requested.**

The audit asked that commit `5f174669d96ac2957ce4787653c1df4270f18b02` be made
fetchable, and warned against replacing the recorded hash unless byte-level
comparison established equivalence. Running that comparison showed **the recorded
commit is not the experiment source.** At that commit, `exp_base.py`:

- has no `_group_of` — the source-video group id used for leak-free splitting;
- has no `make_official_split1` — UCF101's published split-1, which every
  `official1` result depends on;
- has no `FrozenFeatureDataset` — the frozen-stem feature cache;
- defines `make_stratified_splits` as the **old clip-level split** — the leaky
  protocol that was retired — where the source that ran aliases that name to the
  group-aware `make_group_splits`.

Eight further files are untracked at that commit, including
`synccaps_readout_profile.py` (all timing evidence) and
`synccaps_precompute_stem.py` (the feature cache).

Publishing that commit would therefore have satisfied the request literally while
making provenance **worse**: a reviewer who checked it out and re-ran would
silently reproduce the retired leaky protocol.

**What was done instead.** The bytes that actually ran are published on the
[`experiment-source`](../../tree/experiment-source) branch, with per-file
SHA-256, the git blob SHA-1 for the 12 files byte-identical to the commit, and
complete unified diffs for the two that are not.
`provenance/verify_source_equivalence.py` diffs packaged `src/` against that
branch and **exits non-zero if any model or training file differs beyond import
paths**. Current result: `IDENTICAL 5 | IMPORTS_ONLY 6 | EXTENDED 8`, critical
set clean.

`git cat-file -t 5f17466` still will not resolve, and should not: that is the
finding above, not an unfinished item.

Full history is not published because at that commit it tracks 390 MB over 339
files, including a 131 MB checkpoint and seven `.docx` manuscript revisions under
review.

## B2 — the release tag predated the final corrections

`v1.0-artifacts` was not moved. `v1.0.1-artifacts` was cut at the corrected tip
and carries the same five checkpoint archives (189 files, 1.17 GB) so it is
self-contained.

## B3 — the claim map described an older paper state

Rebuilt by `provenance/build_claim_evidence_map.py`, which **recomputes** every
statistic from released artifacts rather than transcribing it. 15 claims
(C01–C15) in the requested schema, with separate fields for dataset, split,
backbone/layer, evaluation policy, view count, arm pair, seed set, pair set,
statistic, executable analysis, raw results path, support path, and two
machine-resolvable config paths. Manuscript locations are resolved by finding
each value in the `.docx` and mapping it to the enclosing heading.

The three specific defects named:

- CLIP 82.65 is now **C15, labelled a backbone-ladder rung, not the headline**
  (it resolves to "5.1. The backbone ladder").
- Single-view and three-view estimates are separated by a `view_count` column;
  C08 is marked THREE-VIEW with "never pool with the single-view column".
- The stale three-seed `+0.75` zero-decay figure is replaced by the six-seed
  result, **+0.30 [−0.17, +0.76], TOST ±1.0 EQUIVALENT** (C10); "shuffle ties" is
  replaced by the measured per-clip disagreement (C13).

Every value the audit listed reproduced exactly: nine-seed **+1.28 [0.71, 1.85]**;
fresh-seed **+1.77**; CLIP **+0.05 [−0.20, 0.29]**; difference-in-differences
**+0.99 [0.06, 1.91]**; val-carved **+2.81**; routing band **−0.21** and **+0.09**,
both TOST-equivalent; **17.81×** head parameters and **1.78×** p50 latency.

## Non-blocking items

| # | item | disposition |
|---|---|---|
| 1 | `reproduce_statistics.sh` overclaimed | Now also runs the seed-level repair report (both protocols) and the new `synccaps_claim_analyses.py`; its header states what is and is not covered. Unedited outputs committed under `results/statistics/`. |
| 2 | "60 configurations" was wrong | There are **55** configs and 60 manifest rows; `README.md` explains why. |
| 3 | ambiguous joins | `result_batch_id` added to the manifest and the seed-level CSV; each config carries a `seed_batches` array. |
| 4 | pair uniqueness is archived | Already in `pair_indices/PAIR_INDICES_MANIFEST.json` (self-only: 1351–1372 unique of 2048 draws); restated in the C11/C12 notes, which also record that the comparison is not rank-matched. |
| 5 | Data Availability / Zenodo DOI | **Not actionable here** — minting a DOI needs the authors' Zenodo account. |
| 6 | Figures 2–4 not reproducible from dumps alone | `reproduce_figures.sh` now says so; the dumps hold per-clip predictions, not per-tick logits. |
| 7 | `reproduce_tables.sh` called a CLIP cell the headline | Corrected; ResNet-18 L3 is now the primary entry point, including the two-batch nine-seed run. |
| 8 | not all archives independently downloaded | Noted; no action required. |

### A defect the audit's config count exposed

Item 2 was not only a documentation error. The config generator **overwrote** a
config whenever a second results file shared an experiment id, so five configs
silently listed only one seed batch — including **both arms of the nine-seed
exact-vs-sketch comparison**, which is assembled from seeds `7,42,1337` plus
`5,11,19,23,101,2026`. Configs now merge batches and list the union.

## Round 3 (2026-08-27) — crossed four-dictionary pair analysis

The round-3 re-audit cleared B1, B2 and most of B3, and found one genuine
claim-to-evidence mismatch, which was mine.

The manuscript reports the pair-composition contrasts **crossed over four
independent pair dictionaries**: cross-only − self-only `+3.17 [+2.71, +3.62]`
and mixed − cross-only `+0.10 [−0.22, +0.42]`. The released analysis computed
only the **pair_seed 0** result (`+3.44 [+2.13, +4.74]` and
`+0.20 [−1.01, +1.42]`), yet C11/C12 labelled `pair_set` as "4 dictionaries,
seeds 0–3". The map therefore contradicted the manuscript while appearing to
support its scope. The raw results for all four dictionaries were already
released; only the analysis and the label were wrong.

Fixed by adding **section 6b** to `synccaps_repair_report.py`, which forms each
contrast *inside* a dictionary (averaging over that dictionary's optimizer
seeds) and then summarises across the four dictionary-level contrasts with the
report's existing paired t-interval. It reproduces the manuscript values
exactly, per dictionary and in summary:

| contrast | pair0 | pair1 | pair2 | pair3 | summary | 95% CI |
|---|---:|---:|---:|---:|---:|---|
| cross-only − self-only | +3.44 | +2.80 | +3.35 | +3.08 | **+3.17** | [+2.71, +3.62] |
| mixed(64) − cross-only | +0.20 | +0.27 | +0.12 | −0.19 | **+0.10** | [−0.22, +0.42] |

C11/C12 now recompute the crossed values, and `n` is stated as 4 dictionaries
rather than 12 runs — pairing optimizer seeds *across* dictionaries would be
meaningless, since seed 42 under `pair_seed 1` shares nothing with seed 42 under
`pair_seed 2` but the optimizer stream. Section 6 is retained and relabelled as
the pair_seed 0 result; `README.md` states that the two units of replication are
not interchangeable.

The release description was also corrected: it had retained pre-rewrite
identifiers (commit `8e33872`, "368 files").

## Manuscript

Untouched, by instruction. The audit's "manuscript consequences" (Section 7.2
wording, Data Availability, the pair-uniqueness limitation, and a code/data
citation) remain for the authors.

## Note on commit references

History was later rewritten at the authors' request to purge two withdrawn
documents. The audited SHAs `3eea479` and `512e9c2` therefore no longer resolve;
their rewritten equivalents are `9488281` and `b3e7487`, with identical trees
apart from the withdrawn files.
