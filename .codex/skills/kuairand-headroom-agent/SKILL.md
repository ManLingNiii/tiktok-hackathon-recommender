---
name: kuairand-headroom-agent
description: Run controlled KuaiRand ranking headroom experiments through five reviewed roles while enforcing validation-only development.
metadata:
  short-description: Govern five-role KuaiRand headroom experiments
---

# KuaiRand Headroom Agent

Use this skill for controlled KuaiRand headroom experiments.

## Contract

- Target: `long_view`; metrics: GAUC and nDCG@5; primary: their mean.
- Use train and validation only. Never load, score, or inspect test/hidden-test data.
- Do not modify the official evaluator or bypass `agent/validation_experiment_runner.py`.
- New code belongs under `agent/modules/` and must be allowlisted.
- Record hypothesis, changed files, metrics, failures, and recovery events.

## Five roles

1. **Loss Researcher**: compare BPR and listwise losses using within-user exposure groups.
2. **History Researcher**: add leakage-safe prior user/tab/author history features.
3. **Multi-task Researcher**: keep `long_view` primary and use other feedback only as auxiliary targets.
4. **Watch-time Researcher**: test censored watch-time objectives with one-sided penalties.
5. **Skill Maintainer / Governor**: validate plans, reject test access and unreviewed paths, compare validation results, and update specs only after evidence.

## Loop

Each role proposes one small hypothesis. The Governor validates it, the fixed
runner executes it, and the team keeps a change only when validation primary
improves beyond tolerance or the result is a documented research finding.

## References

- Interfaces: `agent/modules/base.py`
- Implementations: `agent/modules/headroom_modules.py`
- Losses: `agent/modules/loss_adapter.py`
- Policy: `agent/headroom_registry.json`
- Execution: `agent/validation_experiment_runner.py`
