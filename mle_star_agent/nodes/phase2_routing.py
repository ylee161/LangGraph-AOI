"""nodes/phase2_routing.py — Phase 2 routing functions and outer gate node.

Task 4.9:
  4.9.1  route_inner_loop — conditional edge after phase2_error_analysis
         "continue" → phase2_error_analysis_gate
         "exit"     → phase2_outer_gate
         Exit when: error_analysis_blocked | inner_iteration >= INNER_LOOP_MAX
                  | stop_outer_loop (early-stop from evaluator)

  4.9.2  route_outer_loop — conditional edge after phase2_outer_gate
         "continue" → phase2_ablation
         "exit"     → phase3_ensemble_coder
         Exit when: stop_outer_loop | outer_iteration >= OUTER_LOOP_MAX
                  | tokens_used >= TOKEN_BUDGET
                  | passes_final_acceptance  + no_improve >= NO_IMPROVE_MAX
                  | passes_relaxed_acceptance + no_improve >= NO_IMPROVE_MAX_CONSTRAINED

Also defines phase2_outer_gate_node — the required named node between the inner
loop and the outer conditional edges.  It increments outer_iteration and resets
inner_iteration so the next ablation call uses the correct checkpoint index and
the inner loop starts fresh.
"""

from __future__ import annotations

import logging
from typing import Any

from mle_star_agent import config
from mle_star_agent.shared.acceptance_scoring import (
    passes_final_acceptance,
    passes_relaxed_acceptance,
)
from mle_star_agent.state import AgentState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 4.9.1 — Inner-loop router
# ---------------------------------------------------------------------------

def route_inner_loop(state: AgentState) -> str:
    """Conditional edge function called after phase2_error_analysis.

    Returns "continue" to loop back to phase2_error_analysis_gate, or
    "exit" to proceed to phase2_outer_gate.

    Exit conditions (any one is sufficient):
    - error_analysis_blocked: blind-refinement prevention triggered
    - inner_iteration >= INNER_LOOP_MAX: iteration cap reached
    - stop_outer_loop: early-stop signal set by evaluator (patience / token budget)
    """
    outer = int(state.get("outer_iteration", 0))
    inner = int(state.get("inner_iteration", 0))

    if state.get("error_analysis_blocked", False):
        logger.info(
            "route_inner_loop: EXIT — error_analysis_blocked "
            "(outer=%d inner=%d)",
            outer, inner,
        )
        return "exit"

    if inner >= config.INNER_LOOP_MAX:
        logger.info(
            "route_inner_loop: EXIT — inner_iteration=%d >= INNER_LOOP_MAX=%d "
            "(outer=%d)",
            inner, config.INNER_LOOP_MAX, outer,
        )
        return "exit"

    if state.get("stop_outer_loop", False):
        logger.info(
            "route_inner_loop: EXIT — stop_outer_loop signal "
            "(outer=%d inner=%d)",
            outer, inner,
        )
        return "exit"

    logger.info(
        "route_inner_loop: CONTINUE — outer=%d inner=%d / %d",
        outer, inner, config.INNER_LOOP_MAX,
    )
    return "continue"


# ---------------------------------------------------------------------------
# 4.9.2 — Outer-loop router
# ---------------------------------------------------------------------------

def route_outer_loop(state: AgentState) -> str:
    """Conditional edge function called after phase2_outer_gate.

    Returns "continue" to loop back to phase2_ablation, or
    "exit" to proceed to phase3_ensemble_coder.

    Exit conditions (any one is sufficient):
    - stop_outer_loop flag set by evaluator or ablation
    - outer_iteration >= OUTER_LOOP_MAX
    - tokens_used >= TOKEN_BUDGET
    - passes_final_acceptance AND no_improve_count >= NO_IMPROVE_MAX
    - passes_relaxed_acceptance AND no_improve_count >= NO_IMPROVE_MAX_CONSTRAINED
    """
    if state.get("stop_outer_loop", False):
        logger.info("route_outer_loop: EXIT — stop_outer_loop flag set")
        return "exit"

    outer = int(state.get("outer_iteration", 0))
    if outer >= config.OUTER_LOOP_MAX:
        logger.info(
            "route_outer_loop: EXIT — outer_iteration=%d >= OUTER_LOOP_MAX=%d",
            outer, config.OUTER_LOOP_MAX,
        )
        return "exit"

    tokens = int(state.get("tokens_used", 0))
    if tokens >= config.TOKEN_BUDGET:
        logger.info(
            "route_outer_loop: EXIT — token budget exhausted (%d >= %d)",
            tokens, config.TOKEN_BUDGET,
        )
        return "exit"

    no_improve = int(state.get("no_improve_count", 0))
    best_metrics = _build_best_metrics(state)

    if passes_final_acceptance(best_metrics) and no_improve >= config.NO_IMPROVE_MAX:
        logger.info(
            "route_outer_loop: EXIT — final acceptance met, patience exhausted "
            "(no_improve=%d >= NO_IMPROVE_MAX=%d)",
            no_improve, config.NO_IMPROVE_MAX,
        )
        return "exit"

    if passes_relaxed_acceptance(best_metrics) and no_improve >= config.NO_IMPROVE_MAX_CONSTRAINED:
        logger.info(
            "route_outer_loop: EXIT — relaxed acceptance met, constrained patience exhausted "
            "(no_improve=%d >= NO_IMPROVE_MAX_CONSTRAINED=%d)",
            no_improve, config.NO_IMPROVE_MAX_CONSTRAINED,
        )
        return "exit"

    logger.info(
        "route_outer_loop: CONTINUE — outer=%d no_improve=%d",
        outer, no_improve,
    )
    return "continue"


# ---------------------------------------------------------------------------
# Outer gate node
# ---------------------------------------------------------------------------

def phase2_outer_gate_node(state: AgentState) -> dict[str, Any]:
    """Named node required by LangGraph between the inner loop exit and the
    outer conditional edge (route_outer_loop).

    Increments outer_iteration so the next ablation call uses the correct
    checkpoint index, and resets inner_iteration to 0 so the inner loop
    restarts fresh on the next outer pass.

    LangGraph requires that conditional edges originate from a named node,
    not directly from a routing function, hence this explicit node.
    """
    outer = int(state.get("outer_iteration", 0))
    logger.info(
        "phase2_outer_gate: outer_iteration %d → %d, inner_iteration reset to 0",
        outer, outer + 1,
    )
    return {
        "outer_iteration": outer + 1,
        "inner_iteration": 0,
    }


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _build_best_metrics(state: AgentState) -> dict:
    """Build a best-metrics view from the snapshot fields in state."""
    def _f(key: str, default: float) -> float:
        v = state.get(key)
        return float(v) if v is not None else default

    return {
        "ng_recall":     _f("current_best_score", 0.0),
        "miss_rate":     _f("best_miss_rate", 1.0),
        "overkill_rate": _f("best_overkill_rate", 1.0),
        "accuracy":      _f("best_accuracy", 0.0),
        "f1":            _f("best_f1", 0.0),
    }
