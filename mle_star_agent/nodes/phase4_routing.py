"""nodes/phase4_routing.py — Phase 4 routing function.

Task 6.2:
  route_after_submit — conditional edge after phase4_submit
    "END"            → phase passed relaxed §9.1 acceptance
    "phase2_ablation" → failed, retry budget remains
    "END"            → failed, retry budget exhausted

  The retry reset (state delta) was already applied inside phase4_submit_node
  before returning — so by the time this routing function is called the state
  is already prepped for the next Phase 2 cycle.
"""

from __future__ import annotations

import logging

from langgraph.graph import END

from mle_star_agent import config
from mle_star_agent.state import AgentState

logger = logging.getLogger(__name__)


def route_after_submit(state: AgentState) -> str:
    """Conditional edge called after phase4_submit_node.

    Returns:
      END              — submission passed, or retry budget exhausted
      "phase2_ablation" — submission failed, retry budget still available
                          (reset already applied by the node itself)
    """
    if state.get("submission_passed", False):
        logger.info("route_after_submit: END — submission passed.")
        return END

    retry = int(state.get("submission_retry", 0) or 0)
    if retry > config.SUBMISSION_RETRY_MAX:
        logger.info(
            "route_after_submit: END — retry budget exhausted "
            "(submission_retry=%d > SUBMISSION_RETRY_MAX=%d).",
            retry, config.SUBMISSION_RETRY_MAX,
        )
        return END

    logger.info(
        "route_after_submit: RETRY — submission_retry=%d / %d, "
        "routing back to phase2_ablation.",
        retry, config.SUBMISSION_RETRY_MAX,
    )
    return "phase2_ablation"
