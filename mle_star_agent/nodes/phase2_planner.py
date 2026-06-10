"""nodes/phase2_planner.py — Phase 2 Planner Node (MLE-STAR Section 3.2).

Sub-steps:
  4.5.1  Load-on-demand: target block c_t, prior attempts, error analysis, strategy history
  4.5.2  Integrate kb_semantic (v2)
  4.5.3  Integrate ideator_agent (v2)
  4.5.4  Propose plan p_k; return {refinement_plan: p_k}
"""

from __future__ import annotations

import json
import logging
from typing import Any

from mle_star_agent import config
from mle_star_agent.shared.llm import build_messages, call_llm_json
from mle_star_agent.state import AgentState

logger = logging.getLogger(__name__)

_SYSTEM = """You are the Planner Agent for an AOI (Automated Optical Inspection) pipeline refinement loop.

You are given:
1. A target code block `c_t` that needs improvement.
2. The current error analysis (FP/FN breakdown).
3. A history of previous strategies tried during this phase.
4. A semantic knowledge base of what worked/failed in the past.
5. A blacklist of KNOWN_FAILED_STRATEGY_FINGERPRINTS.

Your job is to propose a new refinement plan p_k that targets the identified code block, improves on past attempts, and strictly avoids the blacklisted strategies.

Guidelines:
- The plan must be concrete and actionable for the coder. Focus on changes to `c_t`.
- Do NOT propose any strategy present in the blacklist.
- Avoid repeating strategies that were already tried in the provided history.
- Return JSON strictly following this schema:
{
  "strategy_name": "<short string label identifying the technique>",
  "refinement_plan": "<detailed string describing what code changes to implement next>"
}
"""


def _build_planner_prompt(
    target_block_code: str,
    error_analysis: str | dict,
    tried_approaches: list[dict],
    kb_semantic: dict,
    retrieved_technique_hints: str,
    blacklist: set,
) -> str:
    lines = [
        f"Target Code Block (c_t): {target_block_code}",
        "",
        "Error Analysis:",
        json.dumps(error_analysis, indent=2) if isinstance(error_analysis, dict) else str(error_analysis),
        "",
        "Previous attempts this loop (tried_approaches):",
        json.dumps(tried_approaches, indent=2) if tried_approaches else "None",
        "",
    ]

    if kb_semantic:
        lines += [
            "Semantic Knowledge Base (similar past failures):",
            json.dumps(kb_semantic, indent=2),
            "",
        ]

    if retrieved_technique_hints:
        lines += [
            "Retrieved Technique Hints (from literature):",
            retrieved_technique_hints,
            "",
        ]

    # Convert tuples to strings for JSON serialization if necessary
    blacklist_formatted = [str(x) for x in blacklist]
    lines += [
        "KNOWN_FAILED_STRATEGY_FINGERPRINTS (DO NOT USE):",
        json.dumps(blacklist_formatted, indent=2),
        "",
        "Propose the next actionable plan targeting the code block. Return JSON.",
    ]
    return "\n".join(lines)


def phase2_planner_node(state: AgentState) -> dict[str, Any]:
    """Phase 2 planner node.

    Proposes the next plan p_k based on error analysis and past attempts.
    """
    token_state: dict = {"token_count": int(state.get("tokens_used", 0) or 0)}

    # Load-on-demand context
    target_block_code = state.get("target_block_code", "unknown_block")
    error_analysis = state.get("error_analysis", "No detailed error analysis provided.")
    tried_approaches = state.get("tried_approaches", [])
    kb_semantic = state.get("knowledge_base", {})
    retrieved_technique_hints = state.get("retrieved_technique_hints", "")

    # Retrieve blacklist
    from mle_star_agent.shared.small_data_strategy_validator import KNOWN_FAILED_STRATEGY_FINGERPRINTS

    user_prompt = _build_planner_prompt(
        target_block_code=target_block_code,
        error_analysis=error_analysis,
        tried_approaches=tried_approaches,
        kb_semantic=kb_semantic,
        retrieved_technique_hints=retrieved_technique_hints,
        blacklist=KNOWN_FAILED_STRATEGY_FINGERPRINTS,
    )

    messages = build_messages(_SYSTEM, user_prompt)

    try:
        response_data = call_llm_json(
            messages,
            model=config.MODEL_PRO,
            temperature=0.7,
            token_state=token_state,
        )
    except Exception as e:
        logger.error("Failed to call LLM in planner: %s", e)
        response_data = {
            "refinement_plan": f"Fallback plan: Implement minor tweaks to {target_block_code} to address current errors.",
            "strategy_name": "fallback_tweak",
        }

    if not isinstance(response_data, dict):
        logger.warning("Planner LLM returned non-dict JSON (%s); using fallback.", type(response_data).__name__)
        response_data = {}

    plan = response_data.get("refinement_plan", "")
    strategy_name = response_data.get("strategy_name", "unknown_strategy")

    logger.info("Proposed plan for block '%s': %s", target_block_code, strategy_name)

    new_entry = {
        "strategy_name": strategy_name,
        "refinement_plan": plan,
        "target_block_code": target_block_code,
    }

    return {
        "refinement_plan": plan,
        "tokens_used": token_state["token_count"],
        "tried_approaches": [new_entry],
    }
