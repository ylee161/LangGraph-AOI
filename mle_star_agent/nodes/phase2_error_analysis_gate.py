"""nodes/phase2_error_analysis_gate.py — Phase 2 Error Analysis Gate Node.

Guards against blind refinement after the first inner iteration.

Sub-steps:
  4.3.1  Iteration 0: skip gate (set inner_iteration = 0), pass through to planner
  4.3.2  Subsequent iterations: check if last script emitted PREDICTIONS per-sample output
  4.3.3  First missing evidence: set error_analysis_instrumentation_required = True;
         allow one repair iteration; coder must emit PREDICTIONS
  4.3.4  Second missing evidence (repair attempted): escalate — block inner loop
         ("blind refinement" prevention)
"""

from __future__ import annotations

import logging
from typing import Any

from mle_star_agent import config
from mle_star_agent.shared.checkpoint_io import checkpoint_exists, load_checkpoint
from mle_star_agent.state import AgentState

logger = logging.getLogger(__name__)


# Gate result codes (not written to state — used only for log readability)
_ALLOW = "ALLOW"
_ALLOW_NO_EVIDENCE = "ALLOW_NO_EVIDENCE"
_ALLOW_CONSISTENCY_WARNING = "ALLOW_CONSISTENCY_WARNING"
_BLOCK_NO_EVIDENCE = "BLOCK_NO_EVIDENCE"


def phase2_error_analysis_gate_node(state: AgentState) -> dict[str, Any]:
    """Phase 2 error analysis gate node.

    Checks whether the previous refinement iteration emitted valid per-sample
    PREDICTIONS evidence.  On first inner iteration, always passes through.
    On subsequent iterations, allows one instrumentation-repair pass before
    blocking blind refinement.

    Returns a partial state update dict.
    """
    inner_m = int(state.get("inner_iteration", 0))
    outer_n = int(state.get("outer_iteration", 0))

    # ------------------------------------------------------------------
    # 4.3.1 — Iteration 0: pass through; initialise inner counter
    # ------------------------------------------------------------------
    if inner_m <= 0:
        logger.info(
            "phase2_error_analysis_gate: inner_iteration=%d — first iteration, allowing.",
            inner_m,
        )
        return {
            "inner_iteration": 0,
            "error_analysis_instrumentation_required": False,
            "error_analysis_blocked": False,
        }

    # ------------------------------------------------------------------
    # 4.3.2 — Subsequent iterations: check error_analysis evidence
    # ------------------------------------------------------------------
    # Prefer state; fall back to checkpoint from the previous inner step
    report = state.get("error_analysis")

    # Try to load structured evidence from checkpoint if not in state
    if not isinstance(report, dict):
        ckpt_path = config.ckpt_error_analysis(outer_n, inner_m - 1)
        if checkpoint_exists(ckpt_path):
            try:
                data = load_checkpoint(ckpt_path)
                report = data if isinstance(data, dict) else None
            except Exception as exc:
                logger.warning(
                    "Failed to load error_analysis checkpoint (%s): %s", ckpt_path, exc
                )

    evidence_available = _evidence_is_available(report)
    repair_already_attempted = bool(state.get("error_analysis_repair_attempted", False))

    if evidence_available:
        # Check metrics consistency (soft warning — don't block on this alone)
        consistency = (report or {}).get("metrics_consistency") or {}
        if consistency.get("matches_metrics") is False:
            logger.warning(
                "phase2_error_analysis_gate: metrics_consistency failed for outer=%d inner=%d "
                "— allowing with warning (FP/FN counts may be approximate).",
                outer_n, inner_m,
            )
            return _allow_consistency_warning(inner_m)

        logger.info(
            "phase2_error_analysis_gate: valid evidence for outer=%d inner=%d — allowing.",
            outer_n, inner_m,
        )
        # Clear repair flags on success
        return {
            "error_analysis_instrumentation_required": False,
            "error_analysis_repair_attempted": False,
            "error_analysis_blocked": False,
        }

    # Evidence is missing
    if repair_already_attempted:
        # 4.3.4 — Second missing evidence: escalate / block
        logger.error(
            "phase2_error_analysis_gate: BLOCK — repair was already attempted at "
            "outer=%d inner=%d but PREDICTIONS evidence is still missing. "
            "Blocking blind refinement.",
            outer_n, inner_m,
        )
        return {
            "error_analysis_blocked": True,
            "error_analysis_instrumentation_required": False,
        }

    # 4.3.3 — First missing evidence: allow one repair iteration
    logger.warning(
        "phase2_error_analysis_gate: no evidence at outer=%d inner=%d — "
        "setting instrumentation_required=True for one repair pass.",
        outer_n, inner_m,
    )
    return {
        "error_analysis_instrumentation_required": True,
        "error_analysis_repair_attempted": True,
        "error_analysis_blocked": False,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _evidence_is_available(report: Any) -> bool:
    """Return True when the report dict has usable per-sample evidence."""
    if not isinstance(report, dict):
        return False
    # ADK-style structured report
    if "evidence_available" in report:
        return report.get("evidence_available") is True
    # Flat evidence dict from error_analysis_node (has fp_count / fn_count)
    if "fp_count" in report or "fn_count" in report:
        return True
    # Available field
    if report.get("available") is True:
        return True
    return False


def _allow_consistency_warning(inner_m: int) -> dict:
    return {
        "error_analysis_instrumentation_required": False,
        "error_analysis_blocked": False,
    }
