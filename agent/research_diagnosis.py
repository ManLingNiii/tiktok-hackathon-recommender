"""Validation-only diagnosis used to guide the next controlled experiment."""

PLATEAU_THRESHOLD = 0.002
PLATEAU_WINDOW = 3


def validation_plateau(rows, threshold=PLATEAU_THRESHOLD, window=PLATEAU_WINDOW):
    """Measure consecutive successful validation-primary improvements.

    A failed/recovery iteration resets the streak.  This makes the unlock
    policy conservative: three actual, consecutive validation results must
    each improve by no more than ``threshold`` before another task family is
    considered.  Negative improvements count as no improvement, as intended
    by the plateau rule.
    """
    improvements = []
    streak = 0
    previous = None
    for row in rows:
        if row.get("status") != "success":
            previous = None
            streak = 0
            improvements = []
            continue
        primary = (row.get("metrics") or {}).get("primary")
        if primary is None:
            previous = None
            streak = 0
            improvements = []
            continue
        primary = float(primary)
        if previous is None:
            previous = primary
            continue
        delta = primary - previous
        improvements.append(delta)
        if delta <= threshold:
            streak += 1
        else:
            streak = 0
            improvements = []
        previous = primary
    recent = improvements[-window:]
    return {
        "threshold": float(threshold),
        "window": int(window),
        "streak": int(streak),
        "recent_improvements": [float(x) for x in recent],
        "triggered": len(recent) == window and all(x <= threshold for x in recent),
    }


def diagnose(rows):
    successful = [x for x in rows if x.get("status") == "success" and x.get("metrics")]
    if not successful:
        return {"focus": "baseline", "conclusion": "no successful evidence yet",
                "next_strategy": "establish_baseline",
                "plateau": validation_plateau(rows)}
    latest = successful[-1]
    metrics = latest.get("metrics", {})
    confirmation = latest.get("confirmation_metrics", {}) or {}
    ga = float(metrics.get("GAUC", 0.0)); nd = float(metrics.get("nDCG@5", 0.0))
    primary = float(metrics.get("primary", 0.0))
    cprimary = confirmation.get("primary")
    gap = primary - float(cprimary) if cprimary is not None else None
    mean = (ga + nd) / 2.0
    plateau = validation_plateau(rows)
    if plateau["triggered"]:
        focus = "task_expansion"
        conclusion = ("validation Primary plateaued for three consecutive "
                      "improvements; unlock another reviewed task family")
        strategy = "unlock_auxiliary_tasks"
    elif gap is not None and gap > 0.008:
        focus, conclusion = "generalization", "validation gain is not confirmed; reduce coupling/capacity"
        strategy = "prefer_confirmation_safe_composition"
    elif nd < ga - 0.12:
        focus, conclusion = "local_ranking", "nDCG lags GAUC; prioritize listwise/ranking components"
        strategy = "compose_ranking_heads"
    elif ga < nd - 0.12:
        focus, conclusion = "user_discrimination", "GAUC lags nDCG; prioritize history or calibrated components"
        strategy = "compose_generalization_heads"
    else:
        focus, conclusion = "balanced", "metrics are balanced; explore an untried controlled combination"
        strategy = "explore_untried_composition"
    return {"focus": focus, "conclusion": conclusion, "next_strategy": strategy,
            "latest_experiment": latest.get("experiment"),
            "latest_primary": primary, "latest_confirmation_primary": cprimary,
            "confirmation_gap": gap, "metric_mean": mean, "plateau": plateau,
            "plateau_streak": plateau["streak"],
            "plateau_triggered": plateau["triggered"]}
