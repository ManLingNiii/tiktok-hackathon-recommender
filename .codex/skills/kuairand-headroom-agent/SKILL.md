---
name: kuairand-headroom-agent
description: Run autonomous KuaiRand ranking headroom experiments through reviewed roles, controlled cross-method composition, and validation-only development.
metadata:
  short-description: Govern five-role KuaiRand headroom experiments
---

# KuaiRand Headroom Agent

Use this skill for competition-style autonomous ML iterations on the KuaiRand project.

## Contract

- Target: `long_view`; metrics: GAUC and nDCG@5; primary: their mean.
- Use train and validation only. Never load, score, or inspect test/hidden-test data.
- Do not modify the official evaluator or bypass `agent/validation_experiment_runner.py`.
- New code belongs under `agent/modules/` and must be allowlisted.
- Model parameters, architecture variants, and cross-method composition graphs may only come from
  `agent/configs/search_space.json`; arbitrary code generation is forbidden.
- Each model family keeps one canonical best checkpoint under `outputs/`,
  overwritten only when that family's validation primary improves.  The
  family metadata and metrics are stored in `runs/best_models.json`.  A
  warm-start from that checkpoint is explicit opt-in.
- Record hypothesis, changed files, metrics, failures, recovery events, and the
  planner decision in `runs/`; each result must also include a validation-only
  diagnosis and the next research strategy.
- Autonomous execution may call only `agent/gemini_planner.py` and
  `agent/validation_experiment_runner.py`; it must never execute planner text
  as a shell command or modify the registry, evaluator, or data split.

## Reviewed roles

1. **Loss Researcher**: compare BPR and listwise losses using within-user exposure groups.
2. **History Researcher**: add leakage-safe prior user/tab/author history features.
3. **Multi-task Researcher**: keep `long_view` primary and use other feedback only as auxiliary targets.
4. **Watch-time Researcher**: test censored watch-time objectives with one-sided penalties.
5. **Composition Researcher**: combine outputs from registered families using an allowlisted weighted graph; never invent a new executable module.
6. **Skill Maintainer / Governor**: validate plans, reject test access and unreviewed paths, compare validation results, diagnose failures, and update strategy only after evidence.

## Autonomous loop

Before Task 1, run **Task 0: pure-feature governance**. Task 0 is a
pre-experiment audit and does not count toward the benchmark's 50 iterations.
It must create or validate a versioned safe feature catalog from the
KuaiRand-Pure training schema. Task 0 must:

1. Enumerate raw fields and train-derived candidate features from the approved
   training split only.
2. Record each candidate's source, dtype, missing-ratio policy, unique-count
   policy, and inference availability.
3. Define how train-derived statistics are fitted from training rows and how
   validation rows use only statistics available at the end of training.
4. Audit label leakage, future-history leakage, hidden-test access, and
   user-level constant or zero-variance behavior; reject unsafe candidates.
5. Record the approved feature-to-family mapping and metadata in a versioned
   catalog such as `runs/pure/feature_catalog.json`.
6. Keep `selected_features` empty at the start of Task 1. The catalog is the
   approved candidate universe, not a selection result.

The catalog may be expanded only through a reviewed Skill Maintainer/Governor
change followed by the same audit. Gemini may select among approved entries
but may not invent an unreviewed feature during an experiment.

The agent follows this fixed competition-safe pipeline:

```text
checkpoint registry / manifest
        ↓
Agent 分析 GAUC、nDCG、Primary 與失敗原因
        ↓
自動選擇跨 family 組合
        ↓
validation + confirmation 評估
        ↓
保存最佳 composition manifest
        ↓
submission adapter + local checker
```

The composition manifest is a separate prepared, not-submitted artifact; the
existing frozen submission manifest is never overwritten automatically.
Composition IDs are fixed and recorded in every recipe: `1=bpr_fm`,
`2=listwise_fm`, `3=history_fm`, `4=multitask_fm`, and `5=cwm_fm`. Every new
composition recipe must use all five IDs in order and code `11111`; this code
is an audit label, never a family selector. The five canonical checkpoints from
the offline team seed-0 runs are loaded from
`submission_ready/checkpoint_registry.json` and remain frozen during composition.
The planner generates exactly one diagnosis-aware composition candidate per
iteration from the current task template instead of materializing the full
search space. A promoted candidate is recorded but never stops the loop.

The task workflow has a hard maximum of 50 iterations split into reviewed
budgets: five-family weight learning (12), additive/interaction learning (16),
DNN composition (13), and multi-seed confirmation (9). Each iteration includes a
structured hypothesis, previous metrics/loss/correlation/variance/recovery
evidence, one allowlisted change plan, and a next action. Validation-primary
improvement is measured against the task best: `>=0.002` resets stagnation and
keeps the task; `<0.002` increments stagnation; three consecutive low improvements
mark the current strategy as converged. The controller first tries an untried
allowlisted strategy in the same task; only strategy exhaustion or the task
budget transitions to the next task. Task convergence never ends the global
workflow.

1. Run or load Task 0's versioned safe feature catalog before any model
   iteration; fail closed if the catalog is missing or unsafe.
2. Establish or load the validation-only checkpoint registry/manifest and leaderboard.
3. Analyze GAUC, nDCG@5, primary, confirmation drift, and failure/recovery causes.
4. Ask Gemini (or the deterministic local planner) to propose exactly one allowlisted module,
   hypothesis, or cross-family composition.
5. The Governor resolves that proposal to one entry in
   `agent/experiment_specs.json`; reject ambiguous, unknown, or unsafe plans.
6. Execute the fixed runner in a subprocess with a timeout and capture logs.
7. Parse GAUC, nDCG@5, and primary; evaluate both validation and confirmation splits;
   append the proposal/result/recovery event.
8. Feed the diagnosis back to the next planner decision. The first composition
   layer is a trainable nonnegative linear fusion of all five user-normalized
   predictions plus bias. Its fixed target is `long_view` and its fixed loss is
   `0.6 * within-user listwise CE + 0.4 * same-user BPR`. Later reviewed task
   templates may add only the named pure features/interactions or compare the
   small `gated_linear`/`small_mlp` variants. All pure features are computed from
   train-derived statistics and validation uses only statistics available at the
   end of train. Family checkpoints receive no composition gradient.
9. Save the best composition manifest only when its validation recipe improves the
   previous composition; then validate predictions through the submission adapter and
   local checker. This remains a prepared, not-submitted artifact.
10. Promote only when primary is at least baseline plus `epsilon=0.002` and
   the fixed validation confirmation guard passes the same gate.
11. Continue through the registered experiment/config search space until a
   confirmed promotion or safe exhaustion; never reset and repeat the same
   configs to chase validation noise. On planner/runner failure, stop and
   emit `recovery_required` for human review.

Run the loop with:

```powershell
python -u agent/autonomous_agent.py --max-iterations 5
```

## References

- Interfaces: `agent/modules/base.py`
- Implementations: `agent/modules/headroom_modules.py`
- Losses: `agent/modules/loss_adapter.py`
- Policy: `agent/headroom_registry.json`
- Execution: `agent/validation_experiment_runner.py`
