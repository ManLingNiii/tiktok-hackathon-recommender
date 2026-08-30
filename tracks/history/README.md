# Final History Track

This is the History track pipeline. The six features below are the intended
output of the feature-proposal stage: they are candidate computations that an
LLM should propose after reasoning over the schema and history constraints.
They are not manually selected model inputs in the `history_with_LLM` branch.
That branch captures the LLM's structured proposal, validates it, assigns
stable canonical IDs, and passes the resulting candidates through training and
top-1 selection.

The older `history-final` workflow starts directly with these six expected
proposal outputs. It is the pre-LLM comparison branch and does not claim that
a live LLM generated the list during execution.

- `same_author_as_candidate`
- `author_interaction_count`
- `author_interaction_ratio`
- `author_affinity_rate`
- `recent_engagement_rate`
- `dur_bucket_affinity`

The selected history field is a numeric FM field with a learned linear weight
and learned interaction embedding. There is no fixed `0.25` history weight or
preassigned top-1 feature.
History is constructed from chronological training interactions only;
validation selects the best epoch. Test is not used.

Run the pre-LLM comparison workflow from the repository root:

```bash
PYTHONPATH=src python3 tracks/history/train.py \
  --data_dir "./KuaiRand-Pure/data copy"
```

`train.py` contains the reusable model and history-feature computation. The
complete proposal-to-selection entry point is `workflow.py`; it is the source
of truth when running the autonomous workflow below.

## Workflow

Use `feature_proposal.py` to generate the schema-constrained prompt. Save the
actual LLM JSON response, then run:

```bash
PYTHONPATH=src python3 tracks/history/workflow.py \
  --data_dir "./KuaiRand-Pure/data copy" \
  --proposals proposals.json
```

`workflow.py` never trusts an LLM-provided name. It validates each computation
spec, derives a canonical ID from the spec content, maps the validated spec to
the feature computation adapter, trains every candidate, and selects by
validation primary. The LLM proposal response should be committed as a run
input alongside the metrics for reproducibility.

The proposal file is intentionally an explicit input and represents a captured
one-time LLM run; this repository does not contain credentials or pretend to
make a live per-run LLM call. Generate it by sending
`llm_feature_proposal.proposal_prompt()` to the chosen LLM, save its JSON-only
response, and pass that file to `workflow.py`. Thus the six candidates are no
longer authored as the execution list; they are simply the historical set of
valid proposals that the adapter can reproduce.

Validation rejects proposals that violate the ranking constraint. In
particular, a statistic using only user-side fields is rejected because it is
constant across a user's candidates and cannot change their ranking. The
workflow also saves the exact proposal response and prompt to
`results_by_track/history/metrics/raw_llm_response.json` and
`results_by_track/history/metrics/llm_prompt.txt`.

The expected proposal output is the six candidate computations listed above.
The LLM must return structured specifications rather than bare names. The
pipeline rejects invalid or duplicate proposals before training.

## Feature proposal stage

`feature_proposal.py` defines the source schema and constraints. An agent may
replace or extend the proposal list, provided each candidate uses only fields
available at prediction time and train-only history. The existing candidate
training and validation selection then runs unchanged.

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
- Feature proposal: `feature_proposal.py`
- Safer protocol experiments: `protocols.py`
