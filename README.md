# `experiment-source` — the code that produced the results

This orphan branch exists to answer one question a reviewer must be able to
answer: **exactly which source produced the numbers on `main`?**

## Read this first: the recorded commit is NOT the experiment source

The package originally recorded commit
`5f174669d96ac2957ce4787653c1df4270f18b02` as the source revision. Checking that
claim showed it is **wrong in a way that matters**: the reported runs were made
from a *working tree* that differed from that commit, and the differences are
not cosmetic.

At that commit, `exp_base.py`:

- has **no `_group_of`** — the source-video group id used for leak-free splitting;
- has **no `make_official_split1`** — UCF101's published split-1 protocol, which
  every `official1` result depends on;
- has **no `FrozenFeatureDataset`** — the frozen-stem feature cache;
- defines `make_stratified_splits` as the **old clip-level stratified split** —
  the *leaky* protocol that was retired, in which clips cut from one source video
  can land on both sides of the split.

In the working tree, `make_stratified_splits` is an alias for the group-aware
`make_group_splits`. So checking out that commit and re-running would silently
reproduce the **retired leaky protocol**, not the published results.

Five further files are untracked at that commit and exist only in the working
tree, including `synccaps_readout_profile.py` (all timing evidence) and
`synccaps_precompute_stem.py` (the feature cache).

Publishing that commit object would therefore have satisfied the letter of the
request while making the provenance *worse*. This branch publishes the actual
bytes instead.

## What is here

`SOURCE_MANIFEST.json` lists every file with its SHA-256 and its status against
the nearest ancestor commit:

| status | count | meaning |
|---|---:|---|
| `identical_to_commit` | 12 | byte-identical to `5f17466`; the blob SHA-1 is recorded |
| `MODIFIED_since_commit` | 2 | `exp_base.py`, `synccaps_probe_experiment.py` — 401 inserted / 34 deleted lines |
| `UNTRACKED_at_commit` | 8 | never committed; exist only in the working tree |

`diff_vs_commit/` carries the complete unified diff for both modified files, so
nothing is concealed by publishing the newer bytes.

## Verifying against `main`

`main` repackages this code into `src/{models,training,evaluation,profiling}`,
which required rewriting import lines. That repackaging is machine-checkable:

```bash
git clone https://github.com/ycwonglab/SyncCaps.git && cd SyncCaps
git worktree add ../expsrc experiment-source
PYTHONPATH=. python provenance/verify_source_equivalence.py --snapshot ../expsrc
```

It classifies every file as `IDENTICAL`, `IMPORTS_ONLY`, `PATHS_ONLY` or
`EXTENDED`, writes `provenance/source_equivalence.json`, and **exits non-zero if
any model or training file differs beyond imports**. Current result:

```
IDENTICAL 5 | IMPORTS_ONLY 6 | EXTENDED 8
OK: all model and training code is identical to the experiment revision
   modulo import paths.
```

The eight `EXTENDED` files are deliberate release changes, never model logic:
five figure scripts (output/import paths made repo-relative), `synccaps_mcnemar.py`
(`--perclip`, and reading group ids from the dumps so the statistics run without
UCF101), `synccaps_readout_profile.py` (persists the raw 100 timings and the GPU
state), and `synccaps_repair_report.py` (`--results-dir`).

## Why the full git history is not published

The repository history at that commit tracks 390 MB across 339 files, including
a 131 MB checkpoint and seven `.docx` manuscript revisions that are under review.
This branch carries the source only.
