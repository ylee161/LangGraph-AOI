"""nodes/phase4_submit.py — Phase 4 Submission Node.

Sub-steps:
  6.1.1 Lineage check — load CKPT_SUBMISSION if pipeline script SHA-256 matches
  6.1.2 Run final best pipeline on test split (not val)
  6.1.3 Parse metrics with metric_guard
  6.1.4 Check both acceptance tiers: relaxed §9.1 + final §9.2
  6.1.5 Save CKPT_SUBMISSION; set submission_passed, submission_report

  6.3   Retry reset — when submission fails and retries remain:
  6.3.1   Archive Phase 2/3/4 checkpoints to checkpoints/retry_archives/attempt_N/
  6.3.2   Reset loop counters (outer/inner/ensemble/no_improve/token budget)
  6.3.3   Preserve best_pipeline_script + all best_* snapshot fields
  6.3.4   Preserve tried_approaches across retries
  6.3.5   Do NOT re-run Phase 1 — skip gate is on disk
"""

from __future__ import annotations

import glob
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

from mle_star_agent import config
from mle_star_agent.shared.acceptance_scoring import (
    passes_final_acceptance,
    passes_relaxed_acceptance,
)
from mle_star_agent.shared.checkpoint_io import (
    checkpoint_exists,
    load_checkpoint,
    save_checkpoint,
)
from mle_star_agent.shared.code_runner import run_script
from mle_star_agent.shared.metric_guard import guard_metrics
from mle_star_agent.shared.metrics_parser import metrics_to_dict, parse_metrics
from mle_star_agent.state import AgentState

logger = logging.getLogger(__name__)


# ─── SHA-256 lineage helpers ──────────────────────────────────────────────────

def _text_sha256(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def _submission_lineage(script: str) -> dict:
    """Build a lineage dict keyed on the submission script hash."""
    return {"submission_script_sha256": _text_sha256(script)}


def _lineage_matches(saved: dict | None, current: dict) -> bool:
    if not isinstance(saved, dict):
        return False
    return all(saved.get(k) == current.get(k) for k in current)


# ─── Script selection ─────────────────────────────────────────────────────────

def _select_submission_script(state: AgentState) -> tuple[str, str]:
    """Return (script, source_label), preferring ensemble over best_pipeline."""
    # 1. Ensemble checkpoint
    if checkpoint_exists(config.CKPT_ENSEMBLE):
        data = load_checkpoint(config.CKPT_ENSEMBLE)
        script = data.get("ensemble_script", "")
        if script:
            return script, "ensemble_checkpoint"

    # 2. Ensemble state
    script = state.get("ensemble_script", "")
    if script:
        return script, "ensemble_state"

    # 3. Best pipeline checkpoint. phase2_evaluator writes this with key "script";
    #    phase4 retry reset writes it with key "best_pipeline_script". Read both.
    if checkpoint_exists(config.CKPT_BEST_PIPELINE):
        data = load_checkpoint(config.CKPT_BEST_PIPELINE)
        script = data.get("best_pipeline_script") or data.get("script", "")
        if script:
            return script, "best_pipeline_checkpoint"

    # 4. Best pipeline state
    script = (state.get("best_pipeline") or {}).get("script", "")
    if not script:
        script = state.get("best_pipeline_script", "")
    if script:
        return script, "best_pipeline_state"

    return "", ""


# ─── Acceptance check ─────────────────────────────────────────────────────────

def _build_acceptance(metrics_dict: Optional[dict], threshold: Any) -> dict:
    """Return acceptance check results for both §9.1 relaxed and §9.2 final tiers."""
    if not metrics_dict:
        return {
            "relaxed_minimum_pass": False,
            "final_target_pass": False,
            "reasons": ["metrics_missing"],
            "checks": {},
            "final_checks": {},
        }

    relaxed_pass = passes_relaxed_acceptance(metrics_dict)
    final_pass = passes_final_acceptance(metrics_dict)

    checks = {
        "accuracy":           metrics_dict.get("accuracy", 0.0)      >= config.ACCURACY_RELAXED_MIN,
        "ng_recall":          metrics_dict.get("ng_recall", 0.0)      >= config.NG_RECALL_RELAXED_MIN,
        "miss_rate":          metrics_dict.get("miss_rate", 1.0)      <= config.MISS_RATE_RELAXED_MAX,
        "overkill_rate":      metrics_dict.get("overkill_rate", 1.0)  <= config.OVERKILL_RELAXED_MAX,
        "threshold_recorded": threshold is not None,
    }
    final_checks = {
        "ng_recall":     metrics_dict.get("ng_recall", 0.0)     >= config.NG_RECALL_FINAL_MIN,
        "miss_rate":     metrics_dict.get("miss_rate", 1.0)      <= config.MISS_RATE_FINAL_MAX,
        "overkill_rate": metrics_dict.get("overkill_rate", 1.0)  <= config.OVERKILL_FINAL_MAX,
        "accuracy":      metrics_dict.get("accuracy", 0.0)       >= config.ACCURACY_FINAL_MIN,
    }
    reasons = [name for name, passed in checks.items() if not passed]

    return {
        "relaxed_minimum_pass": relaxed_pass,
        "final_target_pass":    final_pass,
        "checks":               checks,
        "final_checks":         final_checks,
        "reasons":              reasons,
    }


# ─── Attempt log ─────────────────────────────────────────────────────────────

def _append_attempt_log(attempt: int, pass_fail: dict, metrics: dict) -> None:
    try:
        config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        existing: list = []
        if checkpoint_exists(config.CKPT_SUBMISSION_ATTEMPTS):
            existing = list(load_checkpoint(config.CKPT_SUBMISSION_ATTEMPTS))
        existing.append({"attempt": attempt, "pass_fail": pass_fail, "metrics": metrics})
        save_checkpoint(config.CKPT_SUBMISSION_ATTEMPTS, existing)
    except Exception as exc:
        logger.warning("Could not update submission_attempts.json: %s", exc)


# ─── Retry reset (6.3) ────────────────────────────────────────────────────────

def _archive_glob(pattern: str, archive_dir: Path) -> None:
    """Move checkpoints matching glob pattern into archive_dir."""
    archive_dir.mkdir(parents=True, exist_ok=True)
    for p in glob.glob(pattern):
        try:
            src = Path(p)
            dest = archive_dir / src.name
            if dest.exists():
                dest = archive_dir / f"{src.stem}.{int(os.path.getmtime(src))}{src.suffix}"
            os.replace(src, dest)
            logger.debug("Archived checkpoint: %s -> %s", src, dest)
        except OSError as exc:
            logger.error("Could not archive checkpoint %s: %s", p, exc)


def _reset_for_retry(state: AgentState, attempt: int) -> dict[str, Any]:
    """Archive Phase 2/3/4 checkpoints and return state delta for retry.

    Preserves:
      - best_pipeline_script and all best_* snapshot fields (§6.3.3)
      - tried_approaches (§6.3.4)

    Does NOT re-run Phase 1 — the skip gate on disk ensures this (§6.3.5).
    """
    ckpt_dir = str(config.CHECKPOINT_DIR)
    archive_dir = config.CHECKPOINT_DIR / "retry_archives" / f"attempt_{attempt}"

    # 6.3.1 — Archive Phase 2/3/4 checkpoints
    for pattern in [
        f"{ckpt_dir}/submission.json",
        f"{ckpt_dir}/ensemble.json",
        f"{ckpt_dir}/ensemble_*.json",
        f"{ckpt_dir}/ablation_*.json",
        f"{ckpt_dir}/refinement_*.json",
        f"{ckpt_dir}/error_analysis_*.json",
        f"{ckpt_dir}/diagnosis_*.json",
    ]:
        _archive_glob(pattern, archive_dir)

    # Preserve best snapshot values from state
    best_script   = state.get("best_pipeline_script", "") or (state.get("best_pipeline") or {}).get("script", "")
    best_score    = float(state.get("current_best_score", 0.0) or 0.0)
    best_overkill = float(state.get("best_overkill_rate", 1.0) or 1.0)
    best_miss     = float(state.get("best_miss_rate", max(0.0, 1.0 - best_score)) or 0.0)
    best_accuracy = float(state.get("best_accuracy", 0.0) or 0.0)
    best_f1       = float(state.get("best_f1", 0.0) or 0.0)
    best_name     = state.get("best_candidate_name", "")

    # Rewrite best_pipeline.json with reset counters but preserved script
    save_checkpoint(config.CKPT_BEST_PIPELINE, {
        "outer_iteration":    0,
        "inner_iteration":    0,
        "no_improve_count":   0,
        "current_best_score": best_score,
        "best_overkill_rate": best_overkill,
        "best_miss_rate":     best_miss,
        "best_accuracy":      best_accuracy,
        "best_f1":            best_f1,
        "best_candidate_name": best_name,
        "best_pipeline_script": best_script,
        "ensemble_best_score":    best_score,
        "ensemble_best_overkill": best_overkill,
        "ensemble_best_accuracy": best_accuracy,
        "ensemble_best_f1":       best_f1,
        "stop_outer_loop":  False,
    })

    logger.info(
        "Retry %d: archived checkpoints and prepared reset. "
        "best_score=%.4f script=%s",
        attempt,
        best_score,
        "present" if best_script else "MISSING",
    )

    # 6.3.2 — Reset loop counters; 6.3.3 — preserve best_* snapshot
    reset: dict[str, Any] = {
        # Loop counters
        "outer_iteration":          0,
        "inner_iteration":          0,
        "ensemble_iteration":       0,
        "no_improve_count":         0,
        "ensemble_no_improve_count": 0,
        "stop_outer_loop":          False,
        "stop_ensemble_loop":       False,
        # Token budget — reset so flash-downgrade doesn't bleed across attempts
        "tokens_used":              0,
        # Phase 2/3 transient state
        "ablation_results":         [],
        "target_component":         "",
        "target_block_code":        "",
        "diagnosis":                "",
        "error_analysis":           None,
        "error_analysis_report":    None,
        "latest_error_analysis":    None,
        "refinement_plan":          "",
        "error_analysis_instrumentation_required": False,
        "error_analysis_repair_attempted":         False,
        "error_analysis_blocked":                  False,
        # Phase 3 — seed from Phase 2 best so Phase 3 only accepts genuine improvement
        "ensemble_script":          "",
        "ensemble_strategy":        None,
        "ensemble_models":          [],
        "ensemble_best_score":      best_score,
        "ensemble_best_overkill":   best_overkill,
        "ensemble_best_accuracy":   best_accuracy,
        "ensemble_best_f1":         best_f1,
        # Phase 4 result state
        "submission_passed":        False,
        # 6.3.3 — Preserve best snapshot so next attempt starts from best found, not L0
        "current_best_score":       best_score,
        "best_overkill_rate":       best_overkill,
        "best_miss_rate":           best_miss,
        "best_accuracy":            best_accuracy,
        "best_f1":                  best_f1,
        "best_candidate_name":      best_name,
        "best_pipeline_script":     best_script,
        # NOTE: tried_approaches is intentionally NOT reset (§6.3.4 — planner reads
        # it across retries to avoid repeating failed strategies).
    }
    return reset


# ─── Node ─────────────────────────────────────────────────────────────────────

def phase4_submit_node(state: AgentState) -> dict[str, Any]:
    """Phase 4 submission node.

    Runs the best available script on the test split, checks both acceptance
    tiers, saves CKPT_SUBMISSION, and — when a retry is needed — archives
    Phase 2/3/4 checkpoints and resets loop counters so the next attempt
    starts cleanly from the best pipeline found so far.
    """
    submission_retry = int(state.get("submission_retry", 0) or 0)

    # ─── 6.1.1 Select script ─────────────────────────────────────────────────
    script, source = _select_submission_script(state)
    if not script:
        logger.error("Phase 4: no submission script found in ensemble or best_pipeline.")
        return {
            "submission_passed":  False,
            "submission_report":  "ERROR: no submission script available.",
            "submission_retry":   submission_retry + 1,
        }

    # ─── 6.1.1 Lineage check — skip re-run if checkpoint is current ──────────
    current_lineage = _submission_lineage(script)
    if checkpoint_exists(config.CKPT_SUBMISSION):
        saved_data = load_checkpoint(config.CKPT_SUBMISSION)
        if _lineage_matches(saved_data.get("lineage"), current_lineage):
            pass_fail    = saved_data.get("pass_fail", {})
            metrics_dict = saved_data.get("metrics") or {}
            relaxed_pass = bool(pass_fail.get("relaxed_minimum_pass", False))
            final_pass   = bool(pass_fail.get("final_target_pass", False))
            logger.info(
                "Phase 4: CHECKPOINT_FOUND (lineage match). "
                "relaxed_pass=%s final_pass=%s",
                relaxed_pass, final_pass,
            )
            report = _build_report(metrics_dict, relaxed_pass, final_pass, source, cached=True)
            new_retry = submission_retry + (0 if relaxed_pass else 1)
            result: dict[str, Any] = {
                "submission_passed":  relaxed_pass,
                "submission_report":  report,
                "submission_retry":   new_retry,
                "latest_metrics":     metrics_dict,
            }
            if not relaxed_pass and new_retry <= config.SUBMISSION_RETRY_MAX:
                result.update(_reset_for_retry(state, new_retry))
                result["submission_retry"] = new_retry
            return result

    # ─── 6.1.2 Run script on test split ──────────────────────────────────────
    run_kwargs: dict[str, Any] = {
        "env": {"TEST_SPLIT_ONLY": "1"},
    }
    if config.DEBUG_MODE:
        run_kwargs["debug_mode"] = True
        run_kwargs["timeout"]    = config.DEBUG_CHECK_TIMEOUT_SECONDS
        run_kwargs["env"].update({"DRY_RUN": "1", "DRY_RUN_EPOCHS": "1", "DRY_RUN_SAMPLES": "10"})
    else:
        run_kwargs["timeout"] = config.TIMEOUT_SECONDS

    logger.info("Phase 4: running submission script from '%s'.", source)
    run_result = run_script(script, **run_kwargs)

    # ─── 6.1.3 Parse metrics + metric_guard ──────────────────────────────────
    metrics_raw  = parse_metrics(run_result.stdout)
    metrics_raw  = guard_metrics(metrics_raw, run_result.duration_ms, context="phase4 submission")
    metrics_dict = metrics_to_dict(metrics_raw) if metrics_raw is not None else None

    threshold  = (metrics_dict or {}).get("threshold")

    # ─── 6.1.4 Acceptance check ───────────────────────────────────────────────
    pass_fail    = _build_acceptance(metrics_dict, threshold)
    relaxed_pass = pass_fail["relaxed_minimum_pass"]
    final_pass   = pass_fail["final_target_pass"]

    # ─── 6.1.5 Save CKPT_SUBMISSION ──────────────────────────────────────────
    submission_record = {
        "script_source":  source,
        "lineage":        current_lineage,
        "returncode":     run_result.returncode,
        "timed_out":      run_result.timed_out,
        "duration_ms":    round(run_result.duration_ms, 1),
        "metrics":        metrics_dict,
        "pass_fail":      pass_fail,
        "stdout_tail":    run_result.stdout[-3000:],
        "stderr_tail":    run_result.stderr[-1000:],
    }
    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    save_checkpoint(config.CKPT_SUBMISSION, submission_record)

    new_retry = submission_retry + (0 if relaxed_pass else 1)
    _append_attempt_log(new_retry, pass_fail, metrics_dict or {})

    report = _build_report(metrics_dict, relaxed_pass, final_pass, source, cached=False)
    logger.info("Phase 4: %s", report)

    updates: dict[str, Any] = {
        "submission_passed":  relaxed_pass,
        "submission_report":  report,
        "submission_retry":   new_retry,
    }
    if metrics_dict:
        updates["latest_metrics"] = metrics_dict

    # ─── 6.3 Retry reset — only when failing and retries remain ──────────────
    if not relaxed_pass and new_retry <= config.SUBMISSION_RETRY_MAX:
        reset_delta = _reset_for_retry(state, new_retry)
        updates.update(reset_delta)
        # Restore submission_retry and submission_passed (reset_delta doesn't touch these)
        updates["submission_retry"]  = new_retry
        updates["submission_passed"] = False

    return updates


# ─── Report builder ───────────────────────────────────────────────────────────

def _build_report(
    metrics: Optional[dict],
    relaxed_pass: bool,
    final_pass: bool,
    source: str,
    cached: bool,
) -> str:
    prefix = "[CACHED] " if cached else ""
    if not metrics:
        return f"{prefix}Submission FAILED: script produced no METRICS (source={source})."
    status = "PASSED" if relaxed_pass else "FAILED"
    tier   = "§9.2 final" if final_pass else ("§9.1 relaxed" if relaxed_pass else "neither tier")
    return (
        f"{prefix}Submission {status} ({tier}): "
        f"ng_recall={metrics.get('ng_recall', 0):.4f}, "
        f"miss_rate={metrics.get('miss_rate', 1):.4f}, "
        f"overkill={metrics.get('overkill_rate', 1):.4f}, "
        f"accuracy={metrics.get('accuracy', 0):.4f}, "
        f"threshold={metrics.get('threshold')} "
        f"(source={source})."
    )
