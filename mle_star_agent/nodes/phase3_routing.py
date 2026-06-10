"""nodes/phase3_routing.py — Phase 3 routing function.

Task 5.3:
  route_ensemble_loop — conditional edge after phase3_ensemble_evaluator
    "continue" → phase3_ensemble_coder
    "exit"     → phase4_submit

  Exit when: stop_ensemble_loop | ensemble_iteration >= ENSEMBLE_LOOP_MAX
           | tokens_used >= TOKEN_BUDGET
"""

from __future__ import annotations

import logging

from mle_star_agent import config
from mle_star_agent.state import AgentState

logger = logging.getLogger(__name__)


def route_ensemble_loop(state: AgentState) -> str:
    """Conditional edge function called after phase3_ensemble_evaluator.

    Returns "continue" to loop back to phase3_ensemble_coder, or
    "exit" to proceed to phase4_submit.

    Exit conditions (any one is sufficient):
    - stop_ensemble_loop: flag set by evaluator (iteration cap, no-improvement patience,
      token budget, or validation failure)
    - ensemble_iteration >= ENSEMBLE_LOOP_MAX: hard iteration cap
    - tokens_used >= TOKEN_BUDGET: global token budget exhausted
    """
    if state.get("stop_ensemble_loop", False):
        logger.info("route_ensemble_loop: EXIT — stop_ensemble_loop flag set")
        return "exit"

    n = int(state.get("ensemble_iteration", 0))
    if n >= config.ENSEMBLE_LOOP_MAX:
        logger.info(
            "route_ensemble_loop: EXIT — ensemble_iteration=%d >= ENSEMBLE_LOOP_MAX=%d",
            n, config.ENSEMBLE_LOOP_MAX,
        )
        return "exit"

    tokens = int(state.get("tokens_used", 0))
    if tokens >= config.TOKEN_BUDGET:
        logger.info(
            "route_ensemble_loop: EXIT — token budget exhausted (%d >= %d)",
            tokens, config.TOKEN_BUDGET,
        )
        return "exit"

    logger.info(
        "route_ensemble_loop: CONTINUE — ensemble_iteration=%d / %d",
        n, config.ENSEMBLE_LOOP_MAX,
    )
    return "continue"
