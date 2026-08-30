# History Track: LLM-Driven Feature Proposal Workflow

This document describes the LLM-enabled version of the History track. The
purpose of the LLM is to propose *what history computation should be tested*.
It does not choose the final feature by itself. Every accepted proposal is
passed through the same model-training and validation process, and the final
top-1 feature is selected by validation primary.

## Workflow

```text
schema and constraints
        ↓
LLM proposes structured feature specifications
        ↓
validate, reject leakage/invalid proposals, canonicalize IDs
        ↓
compute each accepted history feature
        ↓
train one end-to-end model per feature, seed=0
        ↓
evaluate on the official validation split
        ↓
select top-1 by validation primary
```

The LLM proposes candidate features; gradient descent learns the feature's
model parameters and weighting; validation determines which candidate is
promoted.

## Step 1: LLM feature proposal

`llm_feature_proposal.py` defines the schema and produces the prompt through
`proposal_prompt()`. The LLM must return JSON with a `features` array. Each
feature is a computation specification containing:

- `source_columns`: raw columns used by the computation
- `history_scope`: which prior interactions are included
- `time_order`: how chronological history is defined
- `window`: all history, recent history, or another explicit window
- `aggregation`: count, rate, ratio, affinity, or another supported operation
- `formula`: an unambiguous computation description
- `uses_current_row`: must be `false`
- `rationale`: why the feature may affect candidate ranking

The LLM does not provide the feature ID. This prevents arbitrary names from
creating duplicate representations of the same computation.

## Constraints and safeguards

Each proposal must pass `validate_spec()` before it can reach model training.
The validator rejects a proposal when:

- a required specification field is missing;
- a source column is outside the declared schema;
- a forbidden field or external data source is used;
- `uses_current_row` is not `false`;
- the history scope does not explicitly use the official training history;
- the feature is a pure user-side statistic with no candidate-side reference;
- the proposal is a duplicate of another computation specification.

The candidate-side safeguard is important for within-user ranking. A value
that is identical for every candidate shown to one user cannot change the
ranking order, so the pipeline rejects such a feature rather than silently
spending a training run on a feature known to be ineffective.

History is computed chronologically from training interactions only. Validation
rows do not contribute labels or future interactions to feature construction,
and the test set is not used for proposal generation, tuning, or checkpoint
selection.

## Stable feature identifiers

After validation, the complete specification is serialized canonically and
hashed:

```text
feature_id = hist_<hash of canonical specification>
```

Therefore, the identifier depends on the source columns, window, aggregation,
formula, and other specification content—not on an LLM-generated display name.
Equivalent names for the same computation cannot create separate candidates.

## Step 2: Per-feature model training

`workflow.py` reads the captured proposal JSON and sends every accepted
proposal through the existing History-FM training engine in `train.py`.

For each feature, the model contains:

- FM categorical fields: `user_id`, `video_id`, `author_id`, `tab`,
  `dur_bucket`;
- one proposed numeric history feature;
- a learned linear coefficient for that history feature;
- a learned history embedding for FM interactions;
- pointwise `long_view` log-loss;
- seed `0`, identical data splits, and identical training settings.

The history coefficient is learned during training. No fixed `0.25` weight is
added manually.

Each candidate is independently trained and evaluated on validation. The
workflow records the feature ID, display name, full specification, learned
weight, checkpoint information, GAUC, nDCG@5, primary score, and best epoch.

## Step 3: Top-1 selection

Candidates are sorted by validation primary:

```text
primary = mean(GAUC, nDCG@5)
```

The highest-scoring valid candidate becomes `selected_top1`. Only this
candidate is promoted as the History model. The validation set is used for
checkpoint and top-1 selection; the test set remains untouched until the
final confirmation stage.

## Reproducible execution

The repository intentionally does not embed credentials or pretend to make a
live LLM API call. Generate a one-time response by sending
`proposal_prompt()` to the selected LLM and save its JSON-only response, for
example as `proposals.json`. Then run:

```bash
PYTHONPATH=src python3 tracks/history/workflow.py \
  --data_dir "./KuaiRand-Pure/data copy" \
  --proposals proposals.json
```

The workflow writes:

- `results_by_track/history/metrics/workflow_seed0.json` — all candidates and
  the selected top-1;
- `results_by_track/history/metrics/raw_llm_response.json` — exact proposal
  input;
- `results_by_track/history/metrics/llm_prompt.txt` — prompt contract;
- `results_by_track/history/manifest.json` — selected candidate manifest.

This separation makes the LLM's proposal auditable while keeping model
training, feature weighting, validation, and top-1 selection deterministic
under seed `0`.
