# BPR Track

**Status: Work in Progress.** This track is currently maintained by the BPR owner. The method and results are still being adjusted and may change; it is not finalized or production-ready.

## Files and relationship

- `src/baseline_bpr.py` is the original implementation and remains the source of truth.
- `src/baseline_bpr_3neg.py` is retained as the separate 3-negative experiment entry, although its current code should be treated as WIP and verified before claiming a 3-negative implementation.
- `tracks/bpr/train_bpr.py` and `train_bpr_3neg.py` are wrappers so the track has a clear entry point without changing the original training logic.
- `tracks/bpr/model.py` only re-exports `FM`; model logic has not been moved or rewritten.

Shared imports are `src/data.py` for loading/encoding and `src/evaluate.py` for metrics. BPR does not depend on `src/baseline.py`.

## Protocol

Use the KuaiRand-Pure data under `KuaiRand-Pure/data/`. The fixed split is train `20220408–20220421`, valid `20220422–20220428`, and test `20220429–20220508`. Train on train only, select checkpoints by validation primary, and reserve test for final confirmation.

The first-stage experiments use `seed=0`, `src/evaluate.py`, and primary `mean(GAUC, nDCG@5)`. Compare gains against the shared baseline in `src/baseline.py`.

## Current execution

From the repository root:

```bash
PYTHONPATH=src python tracks/bpr/train_bpr.py --data_dir ./KuaiRand-Pure/data --model fm --seed 0
PYTHONPATH=src python tracks/bpr/train_bpr_3neg.py --data_dir ./KuaiRand-Pure/data --model fm --seed 0
```

These commands use the existing source files through wrappers and do not imply that the current 3-negative code is finalized.

## Recording experiments

Record track, experiment id, method, hyperparameters, seed, best epoch, validation GAUC, validation nDCG@5, validation primary, checkpoint path, git commit, training time, and notes in the shared experiment record. Detailed BPR artifacts belong in `results_by_track/bpr/`.

Other members may read and reference this track, but should not directly modify `tracks/bpr/`.
