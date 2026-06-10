"""nodes/phase2_strategy_gate.py — Phase 2 Strategy Gate Node (MLE-STAR Section 3.2).

Sub-steps:
  4.6.1  Validate proposed strategy against KNOWN_FAILED_STRATEGY_FINGERPRINTS and small-data constraints
  4.6.2  Accept or request re-plan (internally loops/rewrites plan if rejected)
"""

from __future__ import annotations

import json
import logging
from typing import Any

from mle_star_agent import config
from mle_star_agent.shared.llm import build_messages, call_llm_json
from mle_star_agent.state import AgentState
from mle_star_agent.shared.small_data_strategy_validator import KNOWN_FAILED_STRATEGY_FINGERPRINTS

logger = logging.getLogger(__name__)

_SYSTEM = """You are the Strategy Gate Agent.
Your job is to review a proposed refinement plan and ensure it does NOT violate small-data constraints
or use any KNOWN_FAILED_STRATEGY_FINGERPRINTS.

If the plan is VALID, return it unchanged.
If the plan is INVALID (violates constraints), rewrite the plan to be valid and propose a new strategy name.

Return JSON strictly following this schema:
{
  "is_valid": true/false,
  "reason": "<explain why if invalid, or 'ok' if valid>",
  "strategy_name": "<new or original strategy name>",
  "refinement_plan": "<new or original refinement plan>"
}
"""


def phase2_strategy_gate_node(state: AgentState) -> dict[str, Any]:
    """Phase 2 strategy gate node.

    Validates proposed plan against blacklists and rewrites if invalid.
    """
    token_state: dict = {"token_count": int(state.get("tokens_used", 0) or 0)}

    proposed_plan = state.get("refinement_plan", "")
    tried_approaches = state.get("tried_approaches", [])

    # Get the latest proposed strategy name
    latest_strategy_name = "unknown"
    if tried_approaches:
        latest_strategy_name = tried_approaches[-1].get("strategy_name", "")

    blacklist_formatted = [str(x) for x in KNOWN_FAILED_STRATEGY_FINGERPRINTS]

    user_prompt = f"""
Proposed Strategy Name: {latest_strategy_name}
Proposed Plan:
{proposed_plan}

KNOWN_FAILED_STRATEGY_FINGERPRINTS (DO NOT USE):
{json.dumps(blacklist_formatted, indent=2)}

Small Data Constraints:
- Dataset is small (approx 287 samples).
- Do not add massive capacity without freeze/dropout/weight_decay.
- Avoid purely unsupervised anomaly detection models (can lead to AUC leak).
- Avoid local-patch MIL routes (causes overkill).

Evaluate the plan. If it violates any blacklist item or constraint, rewrite it to be a valid alternative targeting the same component.
"""

    messages = build_messages(_SYSTEM, user_prompt)

    try:
        response_data = call_llm_json(
            messages,
            model=config.MODEL_PRO,
            temperature=0.2,
            token_state=token_state,
        )
    except Exception as e:
        logger.error("Strategy Gate LLM call failed: %s", e)
        # Fallback to accepting the plan if LLM fails
        response_data = {
            "is_valid": True,
            "strategy_name": latest_strategy_name,
            "refinement_plan": proposed_plan,
        }

    if not isinstance(response_data, dict):
        logger.warning("Strategy gate LLM returned non-dict JSON (%s); accepting plan.", type(response_data).__name__)
        response_data = {}

    is_valid = response_data.get("is_valid", True)
    final_plan = response_data.get("refinement_plan", proposed_plan)
    final_strategy = response_data.get("strategy_name", latest_strategy_name)

    if not is_valid:
        logger.warning(
            "Strategy Gate REJECTED plan. Reason: %s. Rewrote as: %s",
            response_data.get("reason"), final_strategy
        )
        # Append the revised approach so history is accurate for the coder
        new_entry = {
            "strategy_name": f"{final_strategy}_(revised)",
            "refinement_plan": final_plan,
            "target_block_code": state.get("target_block_code", "unknown"),
        }
        return {
            "refinement_plan": final_plan,
            "tokens_used": token_state["token_count"],
            "tried_approaches": [new_entry],
        }

    logger.info("Strategy Gate ACCEPTED plan: %s", final_strategy)

    return {
        "refinement_plan": final_plan,
        "tokens_used": token_state["token_count"],
    }
