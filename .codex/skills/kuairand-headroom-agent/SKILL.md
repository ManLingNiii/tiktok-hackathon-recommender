---
name: kuairand-headroom-agent
description: Run autonomous KuaiRand ranking headroom experiments through five reviewed roles while enforcing validation-only development.
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
- Model parameters and architecture variants may only come from
  `agent/configs/search_space.json`; arbitrary code generation is forbidden.
- Each model family keeps one canonical best checkpoint under `outputs/`,
  overwritten only when that family's validation primary improves.  The
  family metadata and metrics are stored in `runs/best_models.json`.  A
  warm-start from that checkpoint is explicit opt-in.
- Record hypothesis, changed files, metrics, failures, recovery events, and the
  planner decision in `runs/`.
- Autonomous execution may call only `agent/gemini_planner.py` and
  `agent/validation_experiment_runner.py`; it must never execute planner text
  as a shell command or modify the registry, evaluator, or data split.

## Five roles

1. **Loss Researcher**: compare BPR and listwise losses using within-user exposure groups.
2. **History Researcher**: add leakage-safe prior user/tab/author history features.
3. **Multi-task Researcher**: keep `long_view` primary and use other feedback only as auxiliary targets.
4. **Watch-time Researcher**: test censored watch-time objectives with one-sided penalties.
5. **Skill Maintainer / Governor**: validate plans, reject test access and unreviewed paths, compare validation results, and update specs only after evidence.

## Autonomous loop

1. Establish or load the validation-only baseline and leaderboard.
2. Ask Gemini to propose exactly one allowlisted module and hypothesis.
3. The Governor resolves that proposal to one entry in
   `agent/experiment_specs.json`; reject ambiguous, unknown, or unsafe plans.
4. Execute the fixed runner in a subprocess with a timeout and capture logs.
5. Parse GAUC, nDCG@5, and primary; append the proposal/result/recovery event.
6. Promote only when primary is at least baseline plus `epsilon=0.002` and
   the fixed validation confirmation guard passes the same gate.
7. Continue through the registered experiment/config search space until a
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
