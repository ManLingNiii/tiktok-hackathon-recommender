# Final History Track

This is the final end-to-end History model entry point. It trains six separate
candidate models, selects the top-1 feature by validation primary, and saves
only the selected candidate as the final model. The latest seed-0 run selected
`author_affinity_rate`.

- `same_author_as_candidate`
- `author_interaction_count`
- `author_interaction_ratio`
- `author_affinity_rate`
- `recent_engagement_rate`
- `dur_bucket_affinity`

The selected history field is a numeric FM field with a learned linear weight
and learned interaction embedding. There is no fixed `0.25` history weight.
History is constructed from chronological training interactions only;
validation selects the best epoch. Test is not used.

Run from the repository root:

```bash
PYTHONPATH=src python3 tracks/history/train.py \
  --data_dir "./KuaiRand-Pure/data copy"
```

This file contains the complete model implementation and is the source of
truth for the final track.

## Validation results

Shared FM baseline, seed `0`:

- GAUC: `0.66710`
- nDCG@5: `0.53580`
- Primary: `0.60150`

Each candidate below was independently trained end-to-end with the same
hyperparameters and evaluated on the official validation split:

| Candidate history feature | Valid GAUC | Valid nDCG@5 | Valid primary | Learned linear weight |
|---|---:|---:|---:|---:|
| **`author_affinity_rate`** | **0.66840** | **0.53670** | **0.60255** | **-0.1997** |
| `recent_engagement_rate` | 0.66746 | 0.53654 | 0.60200 | -0.2587 |
| `dur_bucket_affinity` | 0.66669 | 0.53603 | 0.60136 | -0.1955 |
| `author_interaction_ratio` | 0.66697 | 0.53571 | 0.60134 | -0.1367 |
| `author_interaction_count` | 0.66606 | 0.53533 | 0.60070 | -0.1660 |
| `same_author_as_candidate` | 0.66588 | 0.53535 | 0.60062 | -0.1671 |

Final selected model:

- Selected feature: `author_affinity_rate`
- Valid primary: `0.60255`
- Gain versus shared baseline: `+0.00105`
- Best epoch: `7`
- Seed: `0`

## Training configuration

- Base FM fields: `user_id`, `video_id`, `author_id`, `tab`, `dur_bucket`
- History source: chronological official training split only
- Embedding dimension: `k=16`
- Learning rate: `0.001`
- L2: `1e-6`
- Batch size: `8192`
- Maximum epochs: `15`
- Loss: pointwise `long_view` log-loss
- Sampling: all train rows, reshuffled each epoch; no negative sampling
- Selection metric: validation primary = mean(GAUC, nDCG@5)

## Artifacts

- Checkpoint: `results_by_track/history/checkpoints/history_top1_seed0.npz`
- Metrics: `results_by_track/history/metrics/history_top1_seed0.json`
- Manifest: `results_by_track/history/manifest.json`
- Feature-selection helper: `feature_selection.py`
- Safer protocol experiments: `protocols.py`
