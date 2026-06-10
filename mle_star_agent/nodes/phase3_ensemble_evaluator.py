"""nodes/phase3_ensemble_evaluator.py — Phase 3 Ensemble Evaluator Node.

Sub-steps:
  5.2.1 Check validation cache
  5.2.2 Run script via code_runner + metric_guard
  5.2.3 is_acceptance_improvement; update best if improved
  5.2.4 Append to tried_ensemble_approaches with fingerprint; persist to CKPT_TRIED_ENSEMBLE_APPROACHES
  5.2.5 Increment ensemble_iteration; set stop_ensemble_loop if cap hit or no-improvement
  5.2.6 Return {ensemble_models, latest_metrics, ensemble_iteration, tried_ensemble_approaches}
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from mle_star_agent import config
from mle_star_agent.guards.code_validator import check_validation_cache, store_validation_cache
from mle_star_agent.shared.acceptance_scoring import is_acceptance_improvement
from mle_star_agent.shared.checkpoint_io import checkpoint_exists, load_checkpoint, save_checkpoint
from mle_star_agent.shared.code_runner import run_script
from mle_star_agent.shared.metric_guard import guard_metrics
from mle_star_agent.shared.metrics_parser import metrics_to_dict, parse_metrics
from mle_star_agent.state import AgentState

logger = logging.getLogger(__name__)


# ─── Persistence helpers ───────────────────────────────────────────────────────

def _save_best_ensemble(
    n: int,
    script: str,
    score: float,
    overkill: float,
    metrics_dict: Optional[dict],
) -> None:
    """Write CKPT_ENSEMBLE with the best ensemble result so far."""
    save_checkpoint(config.CKPT_ENSEMBLE, {
        "ensemble_iteration":   n,
        "ensemble_best_score":  score,
        "ensemble_best_overkill": overkill,
        "ensemble_best_accuracy": (metrics_dict or {}).get("accuracy", 0.0),
        "ensemble_best_f1":     (metrics_dict or {}).get("f1", 0.0),
        "ensemble_script":      script,
        "metrics":              metrics_dict,
    })


def _record_tried_ensemble_approach(
    state: AgentState,
    n: int,
    metrics_dict: Optional[dict],
    improved: bool,
    run_ok: bool,
    failure_reason: Optional[str],
) -> dict:
    """Build, persist, and return a tried_ensemble_approaches entry."""
    strategy = state.get("ensemble_strategy") or {}
    entry = {
        "ensemble_iteration": n,
        "strategy_name":      strategy.get("strategy_name", "unknown"),
        "combination_method": strategy.get("combination_method", ""),
        "strategy_fingerprint": strategy.get("strategy_fingerprint"),
        "result": {
            "ng_recall":  round(float((metrics_dict or {}).get("ng_recall", 0.0)), 4),
            "miss_rate":  round(float((metrics_dict or {}).get("miss_rate", 1.0)), 4),
            "overkill":   round(float((metrics_dict or {}).get("overkill_rate", 1.0)), 4),
            "accuracy":   round(float((metrics_dict or {}).get("accuracy", 0.0)), 4),
            "improved":   improved,
        },
        "failure_reason": (
            "accepted"        if improved else
            "execution_failed" if not run_ok else
            (failure_reason or "degenerate_or_no_improvement")
        ),
    }

    # Disk is the deduplication authority — always merge with existing entries.
    existing: list = []
    if checkpoint_exists(config.CKPT_TRIED_ENSEMBLE_APPROACHES):
        existing = list(
            load_checkpoint(config.CKPT_TRIED_ENSEMBLE_APPROACHES)
            .get("tried_ensemble_approaches", []) or []
        )
    existing.append(entry)
    save_checkpoint(
        config.CKPT_TRIED_ENSEMBLE_APPROACHES,
        {"tried_ensemble_approaches": existing},
    )
    return entry


def _is_degenerate_ensemble(metrics_dict: dict) -> bool:
    """True for all-NG ensembles or ensembles with near-zero NG recall."""
    ng_recall    = float(metrics_dict.get("ng_recall", 0.0))
    overkill_rate = float(metrics_dict.get("overkill_rate", 0.0))
    if ng_recall >= 1.0 and overkill_rate >= 1.0:
        return True
    if ng_recall <= 0.50:
        return True
    return False


# ─── Node ──────────────────────────────────────────────────────────────────────

def phase3_ensemble_evaluator_node(state: AgentState) -> dict[str, Any]:
    """Phase 3 ensemble evaluator node.

    Validates and runs the ensemble script, compares against the Phase 3 running best,
    records the attempt, and sets stop_ensemble_loop when exit conditions are met.
    """
    ensemble_script = state.get("ensemble_script", "")
    n = int(state.get("ensemble_iteration", 0))

    if not ensemble_script:
        logger.warning("Ensemble evaluator: no ensemble_script in state — stopping loop.")
        return {
            "ensemble_iteration": n + 1,
            "stop_ensemble_loop": True,
        }

    # ─── 5.2.1 Check validation cache ────────────────────────────────────────
    cached_status = check_validation_cache(ensemble_script)
    if cached_status == "VALIDATION_FAILED":
        logger.warning("Ensemble script previously failed validation; skipping execution.")
        new_entry = _record_tried_ensemble_approach(
            state, n, None, False, False, "validation_failed"
        )
        n_next = n + 1
        return {
            "ensemble_iteration":    n_next,
            "ensemble_no_improve_count": int(state.get("ensemble_no_improve_count", 0) or 0) + 1,
            "tried_ensemble_approaches": [new_entry],
            "stop_ensemble_loop":    n_next >= config.ENSEMBLE_LOOP_MAX,
        }

    # ─── 5.2.2 Run script + metric_guard ─────────────────────────────────────
    run_kwargs: dict[str, Any] = {}
    if config.DEBUG_MODE:
        run_kwargs = {
            "debug_mode": True,
            "timeout": config.DEBUG_CHECK_TIMEOUT_SECONDS,
            "env": {"DRY_RUN": "1", "DRY_RUN_EPOCHS": "1", "DRY_RUN_SAMPLES": "10"},
        }

    logger.info("Ensemble evaluator: running script (iteration=%d)", n)
    run_result = run_script(ensemble_script, **run_kwargs)

    # Persist stdout/stderr for diagnostics (partial write; updated with outcome below)
    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    attempt_data: dict[str, Any] = {
        "ensemble_iteration": n,
        "returncode":   run_result.returncode,
        "timed_out":    run_result.timed_out,
        "duration_ms":  round(run_result.duration_ms, 1),
        "stdout_tail":  run_result.stdout[-3000:],
        "stderr_tail":  run_result.stderr[-1000:],
    }
    save_checkpoint(config.ckpt_ensemble_attempt(n), attempt_data)

    # Cache that we ran this script
    if cached_status is None:
        store_validation_cache(ensemble_script, "VALIDATED")

    metrics_raw = parse_metrics(run_result.stdout)
    metrics_raw = guard_metrics(
        metrics_raw, run_result.duration_ms, context=f"phase3 ensemble iter {n}"
    )
    metrics_dict: Optional[dict] = metrics_to_dict(metrics_raw) if metrics_raw is not None else None
    run_ok = run_result.returncode == 0 and metrics_dict is not None

    # ─── 5.2.3 is_acceptance_improvement; update best if improved ────────────
    ensemble_best_score    = float(state.get("ensemble_best_score", 0.0) or 0.0)
    ensemble_best_overkill = float(state.get("ensemble_best_overkill", 1.0) or 1.0)
    ensemble_best_accuracy = float(state.get("ensemble_best_accuracy", 0.0) or 0.0)
    ensemble_best_f1       = float(state.get("ensemble_best_f1", 0.0) or 0.0)

    current_ensemble_metrics = {
        "ng_recall":     ensemble_best_score,
        "miss_rate":     max(0.0, 1.0 - ensemble_best_score),
        "overkill_rate": ensemble_best_overkill,
        "accuracy":      ensemble_best_accuracy,
        "f1":            ensemble_best_f1,
    }

    improved = False
    failure_reason: Optional[str] = None

    if run_ok and metrics_dict:
        if _is_degenerate_ensemble(metrics_dict):
            failure_reason = "degenerate_ensemble"
            logger.warning("Ensemble iter %d: degenerate output (all-NG or low recall).", n)
        elif is_acceptance_improvement(metrics_dict, current_ensemble_metrics):
            improved = True
        else:
            failure_reason = "no_acceptance_improvement"
    elif not run_ok:
        failure_reason = "execution_failed"

    best_updates: dict[str, Any] = {}
    if improved and metrics_dict:
        new_score    = float(metrics_dict.get("ng_recall", 0.0))
        new_overkill = float(metrics_dict.get("overkill_rate", 1.0))
        best_updates = {
            "ensemble_best_score":    new_score,
            "ensemble_best_overkill": new_overkill,
            "ensemble_best_accuracy": float(metrics_dict.get("accuracy", 0.0)),
            "ensemble_best_f1":       float(metrics_dict.get("f1", 0.0)),
            # REPLACE semantics: defines the complete current ensemble member set.
            "ensemble_models": [{"script": ensemble_script, "metrics": metrics_dict}],
        }
        _save_best_ensemble(n, ensemble_script, new_score, new_overkill, metrics_dict)
        logger.info(
            "New ensemble best: ng_recall=%.4f overkill=%.4f (was %.4f / %.4f)",
            new_score, new_overkill, ensemble_best_score, ensemble_best_overkill,
        )
    else:
        logger.info(
            "No improvement: ng_recall=%.4f overkill=%.4f; "
            "ensemble_best=%.4f / %.4f; run_ok=%s; reason=%s",
            float((metrics_dict or {}).get("ng_recall", 0.0)),
            float((metrics_dict or {}).get("overkill_rate", 1.0)),
            ensemble_best_score, ensemble_best_overkill,
            run_ok, failure_reason,
        )

    # ─── 5.2.4 Record tried_ensemble_approaches ──────────────────────────────
    new_entry = _record_tried_ensemble_approach(
        state, n, metrics_dict, improved, run_ok, failure_reason
    )

    # Update per-attempt checkpoint with final outcome
    attempt_data.update({
        "improved":              improved,
        "failure_reason":        failure_reason,
        "metrics":               metrics_dict,
        "ensemble_best_score":   best_updates.get("ensemble_best_score", ensemble_best_score),
        "ensemble_best_overkill": best_updates.get("ensemble_best_overkill", ensemble_best_overkill),
    })
    save_checkpoint(config.ckpt_ensemble_attempt(n), attempt_data)

    # Fallback: guarantee CKPT_ENSEMBLE exists so phase4_submit never hits FileNotFoundError.
    if not config.CKPT_ENSEMBLE.exists():
        fallback_script = (state.get("best_pipeline") or {}).get("script", "")
        _save_best_ensemble(n, fallback_script, ensemble_best_score, ensemble_best_overkill, None)
        logger.info("Fallback ensemble.json written using best_pipeline.")

    # ─── 5.2.5 Increment ensemble_iteration; set exit signal ─────────────────
    n_next = n + 1
    no_improve = int(state.get("ensemble_no_improve_count", 0) or 0)
    if improved:
        no_improve = 0
    else:
        no_improve += 1

    stop_ensemble_loop = bool(state.get("stop_ensemble_loop", False))

    # Token budget hard stop
    if int(state.get("tokens_used", 0)) >= config.TOKEN_BUDGET:
        stop_ensemble_loop = True
        logger.warning("Ensemble: token budget exhausted — stop_ensemble_loop set.")

    # Iteration cap
    if n_next >= config.ENSEMBLE_LOOP_MAX:
        stop_ensemble_loop = True
        logger.info("Ensemble iteration cap: %d >= ENSEMBLE_LOOP_MAX=%d.", n_next, config.ENSEMBLE_LOOP_MAX)

    # No-improvement patience: allow at least 3 iterations (exploring diverse strategies)
    # before firing, then exit after 2 consecutive non-improving runs.
    if run_ok and not improved and n_next >= 3 and no_improve >= 2:
        stop_ensemble_loop = True
        logger.info("Ensemble no-improvement cap: %d consecutive non-improving runs.", no_improve)

    # ─── 5.2.6 Return ────────────────────────────────────────────────────────
    updates: dict[str, Any] = {
        "ensemble_iteration":       n_next,
        "ensemble_no_improve_count": no_improve,
        "stop_ensemble_loop":       stop_ensemble_loop,
        "tried_ensemble_approaches": [new_entry],  # appended via operator.add reducer
    }

    if metrics_dict:
        updates["latest_metrics"] = metrics_dict

    updates.update(best_updates)
    return updates
