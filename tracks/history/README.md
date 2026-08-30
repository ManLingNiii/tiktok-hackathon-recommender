# Final History Track

This is the final end-to-end History model entry point. The selected top-1
feature is `author_affinity_rate`, chosen by the completed seed-0 model-based
comparison. Final training uses only this train-only history field:

- `same_author_as_candidate`
- `author_interaction_count`
- `author_interaction_ratio`
- `author_affinity_rate`
- `recent_engagement_rate`
- `dur_bucket_affinity`

The selected history field is a numeric FM field with a learned linear weight
and learned interaction embedding. History is constructed from chronological
training interactions only; validation selects the best epoch. Test is not
used.

Run from the repository root:

```bash
PYTHONPATH=src python3 tracks/history/train.py \
  --data_dir "./KuaiRand-Pure/data copy"
```

This file contains the complete model implementation and is the source of
truth for the final track. The final checkpoint is written under
`results_by_track/history/checkpoints/history_author_affinity_seed0.npz`.
