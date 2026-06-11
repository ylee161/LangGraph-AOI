"""nodes/phase2_coder.py — Phase 2 Coder Node.

Sub-steps:
  4.7.1 Load-on-demand: full best pipeline script (state or CKPT_BEST_PIPELINE fallback),
        diagnosis, error analysis, p_k, FP/FN per-sample evidence (capped), population summary.
  4.7.2 LLM implements p_k as refined block c_t^k; surgical replacement: s_t^k = s_t.replace(c_t, c_t^k).
  4.7.3 Pass through code_validator; return rejection reasons if fails.
  4.7.4 Return {candidate_scripts: [s_t^k]}.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from mle_star_agent import config
from mle_star_agent.guards.code_validator import validate_script
from mle_star_agent.shared.llm import build_messages, call_llm_json
from mle_star_agent.shared.checkpoint_io import checkpoint_exists, load_checkpoint
from mle_star_agent.state import AgentState

logger = logging.getLogger(__name__)

_SYSTEM = """You are the Coder Agent for an AOI pipeline refinement loop.

You will be given:
1. The full current pipeline script.
2. The target block of code (c_t) that needs to be replaced.
3. The refinement plan (p_k) detailing what changes to make.
4. Error analysis and context from previous attempts.

Your job is to rewrite ONLY the target code block (c_t) according to the refinement plan.
The new block (c_t^k) MUST be a direct, drop-in string replacement for `c_t`.
Ensure the indentation and variables match the surrounding script context.

IMPORTANT — the replacement is applied as a literal string substitution:
    new_script = pipeline_script.replace(target_block_verbatim, new_code_block)
So `target_block_verbatim` MUST be copied **character-for-character** (including
exact indentation and whitespace) from the Full Pipeline Script shown to you. If it
does not appear verbatim in the script, the refinement is a no-op and will be rejected.
Pick a self-contained span that is unique within the script.

Return JSON strictly following this schema:
{
  "target_block_verbatim": "<exact code span copied verbatim from the pipeline script that you will replace>",
  "new_code_block": "<the new code text to substitute in its place>"
}
"""

def _build_coder_prompt(
    pipeline_script: str,
    target_block_code: str,
    refinement_plan: str,
    diagnosis: str,
    error_analysis: str | dict,
    population_summary: list[dict],
) -> str:
    lines = [
        "Full Pipeline Script (for context):",
        "```python",
        pipeline_script,
        "```",
        "",
        "Target Code Block (c_t) to replace:",
        "```python",
        target_block_code,
        "```",
        "",
        f"Refinement Plan (p_k): {refinement_plan}",
        "",
        "Diagnosis:",
        diagnosis,
        "",
        "Error Analysis Context:",
        json.dumps(error_analysis, indent=2) if isinstance(error_analysis, dict) else str(error_analysis),
        "",
        "Population Summary (Previous attempts):",
        json.dumps(population_summary, indent=2) if population_summary else "None",
        "",
        "Please provide the new code block that exactly implements the Refinement Plan.",
        "It will be injected as: s_t.replace(c_t, c_t^k). Make sure the indentation matches.",
    ]
    return "\n".join(lines)


def phase2_coder_node(state: AgentState) -> dict[str, Any]:
    """Phase 2 coder node.

    Implements the chosen strategy as a surgical replacement in the pipeline.
    Passes the refined script through the code validator.
    """
    token_state: dict = {"token_count": int(state.get("tokens_used", 0) or 0)}

    # Load-on-demand context
    target_block_code = state.get("target_block_code", "")
    refinement_plan = state.get("refinement_plan", "")
    diagnosis = state.get("diagnosis", "")
    error_analysis = state.get("error_analysis") or ""
    tried_approaches = state.get("tried_approaches", [])

    # Load best pipeline script. CKPT_BEST_PIPELINE is written with key "script"
    # by phase2_evaluator but with key "best_pipeline_script" by phase4 retry reset,
    # so read both spellings to avoid silently falling back to the L0 baseline.
    pipeline_script = ""
    best_pipeline = state.get("best_pipeline")
    if best_pipeline and "script" in best_pipeline:
        pipeline_script = best_pipeline["script"]
    elif checkpoint_exists(config.CKPT_BEST_PIPELINE):
        _bp = load_checkpoint(config.CKPT_BEST_PIPELINE)
        pipeline_script = _bp.get("script") or _bp.get("best_pipeline_script", "")
    elif checkpoint_exists(config.CKPT_L0):
        pipeline_script = load_checkpoint(config.CKPT_L0).get("script", "")

    if not pipeline_script:
        logger.error("No base pipeline script found for coder.")
        return {"error": "No base pipeline script found."}
        
    if not target_block_code:
        logger.error("No target block code defined for replacement.")
        return {"error": "No target block code defined."}

    user_prompt = _build_coder_prompt(
        pipeline_script=pipeline_script,
        target_block_code=target_block_code,
        refinement_plan=refinement_plan,
        diagnosis=diagnosis,
        error_analysis=error_analysis,
        population_summary=tried_approaches,
    )

    # Retry loop: re-prompt on a no-op replacement (anchor not found verbatim) or a
    # validation failure, feeding the reason back to the LLM each attempt.
    new_script = ""
    validation_result = None
    last_failure = "Unknown error."
    for attempt in range(config.DEBUGGER_RETRY_CAP):
        messages = build_messages(_SYSTEM, user_prompt)
        try:
            response_data = call_llm_json(
                messages,
                model=config.MODEL_PRO,
                max_tokens=config.SCRIPT_MAX_TOKENS,
                temperature=0.2,
                token_state=token_state,
            )
        except Exception as e:
            logger.error("Failed to call LLM in coder: %s", e)
            return {"error": f"LLM error: {e}", "tokens_used": token_state["token_count"]}

        if not isinstance(response_data, dict):
            logger.warning("Coder LLM returned non-dict JSON (%s); retrying.", type(response_data).__name__)
            last_failure = "Response was not a JSON object."
            user_prompt += "\n\nYour previous response was not a JSON object. Return JSON matching the schema."
            continue

        new_code_block = response_data.get("new_code_block", "")
        if not new_code_block:
            logger.error("LLM did not return a new code block.")
            return {"error": "Empty code block returned by LLM.", "tokens_used": token_state["token_count"]}

        # Choose the anchor to replace: prefer the verbatim span the LLM copied from
        # the script; fall back to the diagnosis target_block_code hint. target_block_code
        # is often a label (e.g. "# threshold_selection") that does not appear in the
        # script, which would make .replace() a silent no-op — guard against that.
        verbatim = response_data.get("target_block_verbatim") or ""
        anchor = ""
        if verbatim and verbatim in pipeline_script:
            anchor = verbatim
        elif target_block_code and target_block_code in pipeline_script:
            anchor = target_block_code

        if not anchor:
            logger.warning(
                "Coder: replacement anchor not found in script (attempt %d). "
                "verbatim_present=%s target_block_code=%r",
                attempt + 1, bool(verbatim), target_block_code,
            )
            last_failure = "target_block_verbatim did not match any span in the pipeline script."
            user_prompt += (
                f"\n\nAttempt {attempt + 1} failed: the target_block_verbatim you provided was not "
                "found in the pipeline script. Copy an EXACT span (character-for-character, including "
                "indentation) from the Full Pipeline Script above and return valid JSON."
            )
            continue

        # Surgical replacement
        new_script = pipeline_script.replace(anchor, new_code_block)
        if new_script == pipeline_script:
            logger.warning("Coder: replacement produced an identical script (attempt %d).", attempt + 1)
            last_failure = "Replacement produced an identical script (no change applied)."
            user_prompt += (
                f"\n\nAttempt {attempt + 1} failed: your replacement did not change the script. "
                "Provide a new_code_block that materially differs from the target block."
            )
            continue

        # Validate
        validation_result = validate_script(new_script)
        if validation_result.valid:
            logger.info("Generated script passed validation.")
            return {
                "candidate_scripts": [new_script],
                "tokens_used": token_state["token_count"],
            }

        last_failure = "; ".join(validation_result.rejection_reasons)
        logger.warning("Validation failed (attempt %d): %s", attempt + 1, validation_result.rejection_reasons)
        user_prompt += (
            f"\n\nValidation Failed on attempt {attempt + 1}:\n"
            + "\n".join(validation_result.rejection_reasons)
            + "\nPlease fix the code block and return the valid JSON."
        )

    # Failed after all retries — do NOT emit the unchanged script as a candidate.
    return {
        "error": f"Coder failed after {config.DEBUGGER_RETRY_CAP} attempts: {last_failure}",
        "tokens_used": token_state["token_count"],
    }
