"""shared/curve_abort.py — power-law learning-curve early-abort.

A cheap smoke micro-run (``config.CURVE_ABORT_DEBUG_EPOCHS`` epochs on a 5 %
data subset, produced by ``code_runner.apply_debug_patches``) emits a short
per-epoch ``EPOCH_LOG`` curve. This module fits a saturating curve to that
trajectory and projects each validation metric's asymptotic (best-case) value.
If even the optimistic projection is worse than the current best-so-far — and
the fit is trustworthy (R² ≥ ``CURVE_ABORT_MIN_FIT``) — the expensive full run
is skipped.

Design notes
------------
* The model is ``y(t) = c + a / rank`` where ``rank`` is the 1-based epoch
  position (1, 2, 3, …). This is the classic learning-curve approximation,
  *linear in 1/t*, so it is fit by ordinary least squares with **two**
  parameters. Two params over ``MIN_EPOCHS ≥ 3`` points keeps R² informative
  (a 3-parameter nonlinear fit over 3–4 points would be exactly determined and
  report a meaningless R²=1).
* The asymptote ``c`` (t → ∞) is the *optimistic* ceiling for a rising metric
  (ng_recall) and the *optimistic* floor for a falling metric (overkill). An
  abort therefore means "even the best case this curve can reach is worse than
  what we already have", which is deliberately conservative.
* Aborts require the run to be worse-or-not-better on **both** axes, so a run
  that trades a little recall for much better overkill (a legitimate move under
  the acceptance-distance objective) is never killed.

All thresholds come from ``config.CURVE_ABORT_*``. The module never raises on
malformed input — it returns ``abort=False`` (inconclusive) instead.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from mle_star_agent import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Curve fitting
# ---------------------------------------------------------------------------

def _extract_series(epoch_logs: list[dict], key: str) -> list[float]:
    """Return the per-epoch values for ``key`` (epoch order preserved), dropping
    entries where the value is missing or non-numeric."""
    series: list[float] = []
    for row in epoch_logs:
        if not isinstance(row, dict):
            continue
        val = row.get(key)
        if val is None:
            continue
        try:
            series.append(float(val))
        except (TypeError, ValueError):
            continue
    return series


def project_series(values: list[float]) -> tuple[Optional[float], float, bool]:
    """Fit ``y = c + a / rank`` and return ``(asymptote, r2, ok)``.

    ``asymptote`` is the t → ∞ limit (the intercept ``c``), clamped to [0, 1].
    ``ok`` is False — and ``asymptote`` None — when there are too few points to
    fit. A perfectly flat curve is treated as a trustworthy fit (r2 = 1.0) whose
    asymptote is the constant value, since a flat trajectory is a clean signal
    that the metric will not move.
    """
    n = len(values)
    if n < int(config.CURVE_ABORT_MIN_EPOCHS):
        return None, 0.0, False

    ys = np.asarray(values, dtype=float)
    # Use 1-based rank rather than the raw epoch number so the fit is robust to
    # 0-indexed logs (1/0 → inf) and to any gaps in epoch numbering.
    ranks = np.arange(1, n + 1, dtype=float)
    xs = 1.0 / ranks

    ss_tot = float(np.sum((ys - ys.mean()) ** 2))
    if ss_tot < 1e-12:
        # Flat curve: no variance to explain. Projection is the constant value.
        return float(np.clip(ys.mean(), 0.0, 1.0)), 1.0, True

    # Ordinary least squares for [intercept, slope] of y ~ c + a * x.
    design = np.vstack([np.ones_like(xs), xs]).T
    coef, *_ = np.linalg.lstsq(design, ys, rcond=None)
    intercept, slope = float(coef[0]), float(coef[1])

    y_pred = intercept + slope * xs
    ss_res = float(np.sum((ys - y_pred) ** 2))
    r2 = 1.0 - ss_res / ss_tot

    asymptote = float(np.clip(intercept, 0.0, 1.0))
    return asymptote, r2, True


# ---------------------------------------------------------------------------
# Abort decision
# ---------------------------------------------------------------------------

def evaluate_curve_abort(
    epoch_logs: list[dict],
    best_metrics: Optional[dict],
    *,
    min_fit: Optional[float] = None,
    recall_margin: Optional[float] = None,
    overkill_margin: Optional[float] = None,
) -> dict[str, Any]:
    """Decide whether a run should be aborted based on its short learning curve.

    Args:
        epoch_logs: parsed ``EPOCH_LOG`` rows (need ``val_ng_recall`` and/or
            ``val_overkill``).
        best_metrics: the current best-so-far metrics to beat. When None or
            empty, no abort is possible (returns ``abort=False``).
        min_fit / recall_margin / overkill_margin: overrides for the
            corresponding ``config.CURVE_ABORT_*`` thresholds (mainly for tests).

    Returns a diagnostic dict with ``abort`` (bool), ``reasons`` (list[str]),
    the projected metrics, and their R² values.
    """
    result: dict[str, Any] = {
        "abort": False,
        "reasons": [],
        "projected_ng_recall": None,
        "projected_overkill": None,
        "r2_recall": None,
        "r2_overkill": None,
        "best_ng_recall": None,
        "best_overkill": None,
    }

    if not getattr(config, "CURVE_ABORT_ENABLED", True):
        return result
    if not epoch_logs or not best_metrics:
        return result

    min_fit = float(config.CURVE_ABORT_MIN_FIT if min_fit is None else min_fit)
    recall_margin = float(config.CURVE_ABORT_MARGIN if recall_margin is None else recall_margin)
    overkill_margin = float(
        config.CURVE_ABORT_OVERKILL_MARGIN if overkill_margin is None else overkill_margin
    )

    best_recall = best_metrics.get("ng_recall")
    best_overkill = best_metrics.get("overkill_rate", best_metrics.get("overkill"))
    try:
        best_recall = float(best_recall) if best_recall is not None else None
        best_overkill = float(best_overkill) if best_overkill is not None else None
    except (TypeError, ValueError):
        best_recall = best_overkill = None
    result["best_ng_recall"] = best_recall
    result["best_overkill"] = best_overkill

    proj_recall, r2_recall, ok_recall = project_series(
        _extract_series(epoch_logs, "val_ng_recall")
    )
    proj_overkill, r2_overkill, ok_overkill = project_series(
        _extract_series(epoch_logs, "val_overkill")
    )
    result.update(
        projected_ng_recall=proj_recall,
        projected_overkill=proj_overkill,
        r2_recall=round(r2_recall, 4) if ok_recall else None,
        r2_overkill=round(r2_overkill, 4) if ok_overkill else None,
    )

    # Per-axis "clearly worse than best" verdicts (only when fit is trustworthy).
    recall_worse = (
        ok_recall and best_recall is not None
        and r2_recall >= min_fit
        and (best_recall - proj_recall) >= recall_margin
    )
    overkill_worse = (
        ok_overkill and best_overkill is not None
        and r2_overkill >= min_fit
        and (proj_overkill - best_overkill) >= overkill_margin
    )

    # "Not better" guards: only abort when the other axis is not compensating,
    # so a recall-for-overkill (or vice versa) trade is never killed.
    recall_not_better = (
        True if not ok_recall or best_recall is None else proj_recall <= best_recall
    )
    overkill_not_better = (
        True if not ok_overkill or best_overkill is None else proj_overkill >= best_overkill
    )

    reasons: list[str] = []
    if recall_worse and overkill_not_better:
        reasons.append(
            f"projected ng_recall ceiling {proj_recall:.3f} is ≥{recall_margin:.3f} "
            f"below best {best_recall:.3f} (R²={r2_recall:.2f}) and overkill is not improving"
        )
    if overkill_worse and recall_not_better:
        reasons.append(
            f"projected overkill floor {proj_overkill:.3f} is ≥{overkill_margin:.3f} "
            f"above best {best_overkill:.3f} (R²={r2_overkill:.2f}) and recall is not improving"
        )

    if reasons:
        result["abort"] = True
        result["reasons"] = reasons
        logger.info("Curve-abort fired: %s", "; ".join(reasons))

    return result
