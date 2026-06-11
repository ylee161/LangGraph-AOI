"""Unit tests for shared/curve_abort.py — power-law learning-curve early-abort.

Covers the projection fit (project_series) and the abort decision
(evaluate_curve_abort), including the conservative "must be worse on both axes"
guard that protects a recall-for-overkill trade from being killed.
"""

from __future__ import annotations

from mle_star_agent.shared.curve_abort import evaluate_curve_abort, project_series


def _logs(pairs):
    """Build EPOCH_LOG rows from (val_ng_recall, val_overkill) pairs."""
    return [
        {"epoch": i, "val_ng_recall": r, "val_overkill": o}
        for i, (r, o) in enumerate(pairs)
    ]


# ── project_series ──────────────────────────────────────────────────────────

def test_project_rising_curve_asymptote_above_points():
    asymptote, r2, ok = project_series([0.5, 0.7, 0.8, 0.85])
    assert ok
    assert r2 > 0.9
    assert asymptote > 0.85  # ceiling is above the last observed point


def test_project_flat_curve_is_trustworthy():
    asymptote, r2, ok = project_series([0.6, 0.6, 0.6, 0.6])
    assert ok
    assert r2 == 1.0
    assert abs(asymptote - 0.6) < 1e-9


def test_project_too_few_points_returns_not_ok():
    asymptote, r2, ok = project_series([0.5, 0.7])  # < CURVE_ABORT_MIN_EPOCHS
    assert not ok
    assert asymptote is None


def test_project_asymptote_clamped_to_unit_interval():
    # Steeply rising curve whose naive intercept would exceed 1.0.
    asymptote, _, ok = project_series([0.90, 0.97, 0.99, 0.995])
    assert ok
    assert 0.0 <= asymptote <= 1.0


# ── evaluate_curve_abort ────────────────────────────────────────────────────

_BEST = {"ng_recall": 0.95, "overkill_rate": 0.05}


def test_abort_on_hopeless_curve():
    # Recall plateaus ~0.62, overkill stuck ~0.30 — worse on both axes.
    res = evaluate_curve_abort(
        _logs([(0.55, 0.35), (0.60, 0.32), (0.61, 0.31), (0.62, 0.30)]), _BEST
    )
    assert res["abort"] is True
    assert res["reasons"]


def test_no_abort_on_promising_curve():
    # Recall climbing toward ~0.97, overkill falling toward ~0.05.
    res = evaluate_curve_abort(
        _logs([(0.70, 0.40), (0.85, 0.20), (0.92, 0.10), (0.96, 0.06)]), _BEST
    )
    assert res["abort"] is False


def test_no_abort_when_too_few_epochs():
    res = evaluate_curve_abort(_logs([(0.55, 0.35), (0.60, 0.32)]), _BEST)
    assert res["abort"] is False


def test_no_abort_without_best_reference():
    res = evaluate_curve_abort(
        _logs([(0.55, 0.35), (0.60, 0.32), (0.61, 0.31)]), best_metrics=None
    )
    assert res["abort"] is False


def test_no_abort_on_recall_for_overkill_trade():
    # Recall projects a touch below best (0.92 vs 0.95) but overkill is much
    # better (→0.02 vs best 0.05). The "not better on the other axis" guard must
    # keep this alive — it is a legitimate acceptance-distance trade.
    best = {"ng_recall": 0.95, "overkill_rate": 0.05}
    res = evaluate_curve_abort(
        _logs([(0.85, 0.20), (0.89, 0.10), (0.91, 0.05), (0.915, 0.025)]), best
    )
    assert res["abort"] is False


def test_no_abort_on_noisy_low_fit_curve():
    # Non-monotone / noisy trajectory → low R² → projection not trusted.
    res = evaluate_curve_abort(
        _logs([(0.60, 0.30), (0.40, 0.45), (0.75, 0.20), (0.50, 0.40)]), _BEST
    )
    assert res["abort"] is False


def test_disabled_via_config(monkeypatch):
    from mle_star_agent import config
    monkeypatch.setattr(config, "CURVE_ABORT_ENABLED", False)
    res = evaluate_curve_abort(
        _logs([(0.55, 0.35), (0.60, 0.32), (0.61, 0.31), (0.62, 0.30)]), _BEST
    )
    assert res["abort"] is False
