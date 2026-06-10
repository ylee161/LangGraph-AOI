"""nodes/phase2_ablation.py — Phase 2 Ablation Node (MLE-STAR Section 3.2).

Implements the phase2_ablation LangGraph node.  For each outer iteration it
runs a fixed set of AOI-specific ablation variants against the current best
pipeline script to identify which components contribute most to performance.

Sub-steps implemented here:
  4.1.1  Define the 6 fixed AOI ablation variants (ABLATION_VARIANTS)
  4.1.2  Run each variant with its own per-variant checkpoint so partial
         ablation runs survive crashes and can be resumed
  4.1.3  Run each variant script via code_runner + metric_guard; compute
         delta vs current best
  4.1.4  Pass previous ablation summaries as context so outer iterations
         target different components
  4.1.5  Save aggregate checkpoint to ckpt_ablation(outer_iteration)
  4.1.6  Set stop_outer_loop = True when outer exit conditions are met;
         ablation is the outer loop's exit signal source
  4.1.7  Return {ablation_results, target_component, stop_outer_loop}
         ranked by impact
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Optional

from mle_star_agent import config
from mle_star_agent.guards.code_validator import validate_script
from mle_star_agent.shared import code_runner, metric_guard
from mle_star_agent.shared.acceptance_scoring import acceptance_distance
from mle_star_agent.shared.checkpoint_io import (
    checkpoint_exists,
    load_checkpoint,
    save_checkpoint,
)
from mle_star_agent.shared.llm import (
    build_messages,
    call_llm_json,
)
from mle_star_agent.shared.metrics_parser import metrics_to_dict, parse_metrics
from mle_star_agent.state import AgentState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 4.1.1 — Ablation variant definitions
# ---------------------------------------------------------------------------

ABLATION_VARIANTS: list[dict[str, str]] = [
    {
        "name": "no_stereo_fusion",
        "description": (
            "Remove stereo fusion: use only the left (_L) image instead of combining "
            "L and R images.  Drop all code that loads or merges the _R image."
        ),
    },
    {
        "name": "no_weighted_loss",
        "description": (
            "Remove weighted loss: replace the class-weighted cross-entropy loss "
            "with a standard unweighted cross-entropy loss."
        ),
    },
    {
        "name": "no_threshold_sweep",
        "description": (
            "Remove threshold sweep: use a fixed classification threshold of 0.5 "
            "instead of sweeping thresholds on the validation set."
        ),
    },
    {
        "name": "no_augmentation",
        "description": (
            "Remove data augmentation: train without any augmentation transforms.  "
            "Apply only basic normalisation / resize."
        ),
    },
    {
        "name": "threshold_acceptance_distance",
        "description": (
            "Diagnostic threshold probe: keep the trained model architecture and data pipeline, "
            "but replace any recall-only or F1-only threshold selection with validation-set "
            "threshold selection that minimises acceptance distance.  The threshold objective must "
            "jointly consider miss_rate <= 0.03, ng_recall >= 0.97, overkill_rate <= 0.08, "
            "and accuracy >= 0.92, with miss_rate as the first priority and overkill_rate as "
            "the next practical gap to close."
        ),
    },
    {
        "name": "fp_penalty_loss",
        "description": (
            "Diagnostic G false-positive probe: keep stereo loading and threshold search, but "
            "modify the loss, sampler, or validation selection so false positives on G samples "
            "are explicitly penalised.  The goal is to reduce overkill_rate without allowing "
            "miss_rate to exceed 0.03."
        ),
    },
]

NUM_ABLATION_VARIANTS = len(ABLATION_VARIANTS)

# ---------------------------------------------------------------------------
# Lineage helpers — detect stale checkpoints when the best script changes
# ---------------------------------------------------------------------------

def _script_sha256(script: str) -> str:
    return hashlib.sha256(script.encode("utf-8")).hexdigest()[:16]


def _ablation_lineage(best_pipeline_script: str) -> dict:
    return {
        "best_pipeline_script_sha256": _script_sha256(best_pipeline_script),
        "ablation_variant_count": NUM_ABLATION_VARIANTS,
        "variant_names_sha256": _script_sha256(
            json.dumps([v["name"] for v in ABLATION_VARIANTS])
        ),
    }


def _lineage_matches(stored: Optional[dict], current: dict) -> bool:
    if not stored or not current:
        return False
    return (
        stored.get("best_pipeline_script_sha256") == current.get("best_pipeline_script_sha256")
        and stored.get("ablation_variant_count") == current.get("ablation_variant_count")
    )


def _is_complete(results: list[dict]) -> bool:
    """True when all variant indices are represented (skipped counts as present)."""
    seen = {
        int(r.get("variant_index", -1))
        for r in results
        if isinstance(r, dict) and r.get("variant_index") is not None
    }
    return seen == set(range(NUM_ABLATION_VARIANTS))


# ---------------------------------------------------------------------------
# 4.1.2/4.1.3 — Per-variant script generation + execution
# ---------------------------------------------------------------------------

_SCRIPT_GEN_SYSTEM = """You are an expert PyTorch engineer generating an AOI training script variant.
You will be given a complete baseline training script and a single diagnostic change to apply.
Produce a COMPLETE, self-contained Python training script with EXACTLY that one change applied.

Requirements (must all be preserved unless the variant explicitly removes them):
- DRY_RUN / DRY_RUN_EPOCHS / DRY_RUN_SAMPLES env-var support with the ternary: epochs = DRY_RUN_EPOCHS if DRY_RUN else 20
- Load data_split from DATA_SPLIT_PATH (JSON file)
- Print all required output markers: PROBE_METRICS, EPOCH_LOG, METRICS, CALIBRATION_STATS, THRESHOLD_CURVE, PREDICTIONS
- For no_stereo_fusion: load ONLY _L image; remove all _R loading and diff computation; use 3-channel input
- For all other variants: keep loading both _L and _R stereo images (9-channel fusion)
- PyTorch LR schedule with scheduler.step() is MANDATORY (unless changing it IS the variant)
- METRICS dict must contain: accuracy, ng_recall, miss_rate, overkill_rate, f1, avg_latency_ms, threshold, ng_count, g_count, tp, tn, fp, fn, roc_auc, prob_gap
- Define ABLATION_VARIANT_NAME = "<variant_name>" near the top of the script

Return a JSON object with key "script" containing the FULL Python source code.
"""


def _generate_variant_script(
    variant: dict,
    best_script: str,
    previous_ablation_summaries: list[dict],
    token_state: dict,
) -> Optional[str]:
    """Ask the LLM to produce one ablated version of best_script."""
    variant_name = variant["name"]
    variant_desc = variant["description"]

    context_lines = []
    if previous_ablation_summaries:
        context_lines.append("Previous ablation summaries (for context only — do not repeat):")
        for s in previous_ablation_summaries[-3:]:
            context_lines.append(f"  outer_iter={s.get('outer_iteration')} top_target={s.get('target_component')}")

    context_block = "\n".join(context_lines) if context_lines else ""

    user_prompt = (
        f"Variant to implement: **{variant_name}**\n"
        f"Change description: {variant_desc}\n\n"
        + (f"{context_block}\n\n" if context_block else "")
        + f"Baseline script:\n```python\n{best_script}\n```\n\n"
        "Apply EXACTLY the single change described above.  "
        "Return JSON: {{\"script\": \"<full python source>\"}}"
    )

    try:
        response = call_llm_json(
            build_messages(_SCRIPT_GEN_SYSTEM, user_prompt),
            model=config.MODEL_PRO,
            max_tokens=8192,
            temperature=0.2,
            token_state=token_state,
        )
        if isinstance(response, dict):
            script = response.get("script", "")
            if script and len(script) > 200:
                return script
    except Exception as exc:
        logger.warning("Variant script generation failed for %s: %s", variant_name, exc)

    return None


def _run_variant(
    variant: dict,
    variant_index: int,
    script: str,
    lineage: dict,
) -> dict:
    """Execute ablated script, parse metrics, return result dict."""
    variant_name = variant["name"]
    logger.info("Running ablation variant %d: %s", variant_index, variant_name)

    result = code_runner.run_script(script, timeout=config.TIMEOUT_SECONDS)
    parsed = parse_metrics(result.stdout)
    guarded = metric_guard.guard_metrics(
        parsed, result.duration_ms, context=f"phase2 ablation variant {variant_index}"
    ) if parsed else None

    metrics_dict = metrics_to_dict(guarded) if guarded else None
    status = "success" if (result.returncode == 0 and metrics_dict is not None) else "failed"

    return {
        "variant_index": variant_index,
        "name": variant_name,
        "returncode": result.returncode,
        "timed_out": result.timed_out,
        "duration_ms": round(result.duration_ms, 1),
        "stdout_tail": result.stdout[-3000:],
        "stderr_tail": result.stderr[-1000:],
        "metrics": metrics_dict,
        "status": status,
        "lineage": lineage,
    }


# ---------------------------------------------------------------------------
# 4.1.3 — Delta computation
# ---------------------------------------------------------------------------

def _compute_delta(variant_metrics: Optional[dict], current_best: dict) -> Optional[float]:
    """Return acceptance_distance delta (negative = variant is WORSE than best)."""
    if variant_metrics is None:
        return None
    best_dist = acceptance_distance(current_best)
    var_dist = acceptance_distance(variant_metrics)
    return round(best_dist - var_dist, 4)


# ---------------------------------------------------------------------------
# 4.1.7 — Target component identification via LLM
# ---------------------------------------------------------------------------

_AGGREGATOR_SYSTEM = """You are an AOI pipeline ablation analyst.

Given:
- The current best pipeline performance (ng_recall, miss_rate, overkill_rate, accuracy, f1)
- Ablation variant results showing what happens when each component is disabled or changed

Your job is to identify:
1. The component (target_component) most likely to improve the pipeline toward meeting ALL acceptance criteria:
   - ng_recall >= 1.00 (final) / >= 0.97 (relaxed)
   - miss_rate <= 0.00 (final) / <= 0.03 (relaxed)
   - overkill_rate <= 0.05 (final) / <= 0.08 (relaxed)
   - accuracy >= 0.97 (final) / >= 0.92 (relaxed)

2. Ranking of ablation results by impact (highest-impact component to target first)

Rules:
- For "no_*" removal variants: a large regression when a component is REMOVED means that component is valuable and may have headroom for improvement
- For diagnostic probe variants (threshold_acceptance_distance, fp_penalty_loss): if the probe IMPROVES metrics, that mechanism should be adopted/refined
- Prefer the target that closes the dominant gap (miss_rate first, then overkill_rate)
- Do NOT return a target that has already been extensively tried in prior iterations (check previous_targets_tried)

Return JSON:
{
  "target_component": "<component name or mechanism, e.g. 'threshold_selection', 'loss_function', 'stereo_fusion', 'augmentation'>",
  "reasoning": "<1-2 sentence explanation>",
  "ranked_variants": [
    {"name": "<variant_name>", "impact_score": <float>, "recommendation": "<adopt|investigate|ignore>"}
  ]
}
"""


def _identify_target_component(
    results: list[dict],
    current_best_metrics: dict,
    previous_ablation_summaries: list[dict],
    token_state: dict,
) -> str:
    """Use LLM (flash) to identify the best target component from ablation results."""
    # Build a compact summary of results for the LLM
    summary_rows = []
    for r in results:
        m = r.get("metrics") or {}
        summary_rows.append({
            "name": r.get("name"),
            "status": r.get("status"),
            "ng_recall": m.get("ng_recall"),
            "miss_rate": m.get("miss_rate"),
            "overkill_rate": m.get("overkill_rate"),
            "accuracy": m.get("accuracy"),
            "f1": m.get("f1"),
            "delta_acceptance_distance": _compute_delta(m or None, current_best_metrics),
        })

    prev_targets = [s.get("target_component") for s in previous_ablation_summaries if s.get("target_component")]

    user_prompt = (
        f"Current best metrics: {json.dumps(current_best_metrics)}\n\n"
        f"Ablation variant results:\n{json.dumps(summary_rows, indent=2)}\n\n"
        f"Previous targets already tried: {prev_targets}\n\n"
        "Identify the best target_component and rank variants by impact.  Return JSON."
    )

    try:
        response = call_llm_json(
            build_messages(_AGGREGATOR_SYSTEM, user_prompt),
            model=config.MODEL_FLASH,
            max_tokens=2048,
            temperature=0.1,
            token_state=token_state,
        )
        if isinstance(response, dict) and response.get("target_component"):
            return str(response["target_component"])
    except Exception as exc:
        logger.warning("Target component LLM call failed: %s", exc)

    # Heuristic fallback: pick the variant with the largest positive delta
    # whose removal caused the biggest regression (highest negative delta = most important)
    best_name = "threshold_selection"
    best_delta = None
    for r in results:
        if r.get("status") != "success":
            continue
        m = r.get("metrics") or {}
        delta = _compute_delta(m, current_best_metrics)
        if delta is not None:
            # Most negative delta = removing this component hurts the most = target it
            if best_delta is None or delta < best_delta:
                best_delta = delta
                # Map variant name to a component name
                name_map = {
                    "no_stereo_fusion": "stereo_fusion",
                    "no_weighted_loss": "loss_function",
                    "no_threshold_sweep": "threshold_selection",
                    "no_augmentation": "augmentation",
                    "threshold_acceptance_distance": "threshold_selection",
                    "fp_penalty_loss": "loss_function",
                }
                best_name = name_map.get(r["name"], r["name"])

    return best_name


# ---------------------------------------------------------------------------
# 4.1.6 — Outer loop exit condition check
# ---------------------------------------------------------------------------

def _should_stop_outer_loop(state: AgentState) -> bool:
    """True when any outer-loop exit condition is met."""
    if state.get("stop_outer_loop", False):
        return True
    outer = int(state.get("outer_iteration", 0))
    if outer >= config.OUTER_LOOP_MAX:
        logger.info("Outer loop cap reached: outer_iteration=%d >= %d", outer, config.OUTER_LOOP_MAX)
        return True
    tokens = int(state.get("tokens_used", 0))
    if tokens >= config.TOKEN_BUDGET:
        logger.info("Token budget exhausted: %d >= %d", tokens, config.TOKEN_BUDGET)
        return True
    no_improve = int(state.get("no_improve_count", 0))
    # Determine which patience tier applies
    from mle_star_agent.shared.acceptance_scoring import (
        passes_final_acceptance,
        passes_relaxed_acceptance,
    )
    best_metrics: dict = {}
    if state.get("current_best_score") is not None:
        best_metrics = {
            "ng_recall": state.get("current_best_score", 0.0),
            "miss_rate": state.get("best_miss_rate", 1.0),
            "overkill_rate": state.get("best_overkill_rate", 1.0),
            "accuracy": state.get("best_accuracy", 0.0),
            "f1": state.get("best_f1", 0.0),
        }
    if passes_final_acceptance(best_metrics) and no_improve >= config.NO_IMPROVE_MAX:
        logger.info(
            "Final acceptance met + patience exhausted: no_improve=%d >= %d",
            no_improve, config.NO_IMPROVE_MAX,
        )
        return True
    if passes_relaxed_acceptance(best_metrics) and no_improve >= config.NO_IMPROVE_MAX_CONSTRAINED:
        logger.info(
            "Relaxed acceptance met + constrained patience exhausted: no_improve=%d >= %d",
            no_improve, config.NO_IMPROVE_MAX_CONSTRAINED,
        )
        return True
    return False


# ---------------------------------------------------------------------------
# 4.1.4 — Previous ablation summaries (context for diagnosis rotation)
# ---------------------------------------------------------------------------

def _load_previous_summaries(outer_iteration: int) -> list[dict]:
    """Load aggregated ablation summaries from prior outer iterations."""
    summaries = []
    for n in range(outer_iteration):
        ckpt = config.ckpt_ablation(n)
        if checkpoint_exists(ckpt):
            try:
                data = load_checkpoint(ckpt)
                summaries.append({
                    "outer_iteration": n,
                    "target_component": data.get("target_component"),
                    "results_summary": [
                        {
                            "name": r.get("name"),
                            "status": r.get("status"),
                            "ng_recall": (r.get("metrics") or {}).get("ng_recall"),
                            "miss_rate": (r.get("metrics") or {}).get("miss_rate"),
                            "overkill_rate": (r.get("metrics") or {}).get("overkill_rate"),
                        }
                        for r in (data.get("ablation_results") or [])
                    ],
                })
            except Exception as exc:
                logger.warning("Failed to load ablation summary for iteration %d: %s", n, exc)
    return summaries


# ---------------------------------------------------------------------------
# Main node
# ---------------------------------------------------------------------------

def phase2_ablation_node(state: AgentState) -> dict[str, Any]:
    """Phase 2 ablation node: runs fixed AOI ablation variants and identifies
    the target component for the inner refinement loop.

    Returns a partial state update dict.
    """
    token_state: dict = {"token_count": int(state.get("tokens_used", 0) or 0)}
    outer_n = int(state.get("outer_iteration", 0))
    debug_mode: bool = bool(state.get("debug_mode", config.DEBUG_MODE))

    # ------------------------------------------------------------------
    # 4.1.6 — Early exit if stop_outer_loop already set
    # ------------------------------------------------------------------
    if _should_stop_outer_loop(state):
        logger.info("phase2_ablation: stop condition already met — returning stop signal.")
        return {
            "stop_outer_loop": True,
            "tokens_used": token_state["token_count"],
        }

    # ------------------------------------------------------------------
    # Retrieve context
    # ------------------------------------------------------------------
    best_pipeline: dict = state.get("best_pipeline") or {}
    best_pipeline_script: str = best_pipeline.get("script", "")

    if not best_pipeline_script:
        # Try disk fallback
        if checkpoint_exists(config.CKPT_L0):
            l0 = load_checkpoint(config.CKPT_L0)
            best_pipeline_script = l0.get("script", "")

    if not best_pipeline_script:
        logger.error("phase2_ablation: no best_pipeline_script found — aborting ablation.")
        return {
            "stop_outer_loop": True,
            "error": "phase2_ablation: no best_pipeline_script available",
            "tokens_used": token_state["token_count"],
        }

    def _f(key: str, default: float) -> float:
        v = state.get(key)
        return float(v) if v is not None else default

    current_best_metrics: dict = {
        "ng_recall":     _f("current_best_score", 0.0),
        "miss_rate":     _f("best_miss_rate", 1.0),
        "overkill_rate": _f("best_overkill_rate", 1.0),
        "accuracy":      _f("best_accuracy", 0.0),
        "f1":            _f("best_f1", 0.0),
    }

    data_split_meta: dict = (state.get("data_split") or {}).get("metadata", {})
    input_modality: str = data_split_meta.get("input_modality", "stereo")

    current_lineage = _ablation_lineage(best_pipeline_script)

    # 4.1.4 — Load previous ablation summaries
    previous_summaries = _load_previous_summaries(outer_n)

    # ------------------------------------------------------------------
    # 4.1.5 — Check aggregate checkpoint
    # ------------------------------------------------------------------
    agg_ckpt = config.ckpt_ablation(outer_n)
    if checkpoint_exists(agg_ckpt):
        agg_data = load_checkpoint(agg_ckpt)
        if _lineage_matches(agg_data.get("lineage"), current_lineage):
            results = agg_data.get("ablation_results", [])
            if _is_complete(results):
                logger.info(
                    "phase2_ablation: loaded complete checkpoint for outer_iteration=%d (%d results).",
                    outer_n, len(results),
                )
                stop = _should_stop_outer_loop(state)
                return {
                    "ablation_results": results,
                    "target_component": agg_data.get("target_component", ""),
                    "stop_outer_loop": stop,
                    "tokens_used": token_state["token_count"],
                }
            else:
                logger.info(
                    "Aggregate checkpoint for iteration %d is incomplete (%d/%d); re-running missing variants.",
                    outer_n, len(results), NUM_ABLATION_VARIANTS,
                )
        else:
            logger.info(
                "Aggregate checkpoint for iteration %d has lineage mismatch — discarding.", outer_n
            )

    # ------------------------------------------------------------------
    # 4.1.2 — Run each variant (with per-variant checkpoint recovery)
    # ------------------------------------------------------------------
    results: list[dict] = []

    for i, variant in enumerate(ABLATION_VARIANTS):
        variant_name = variant["name"]

        # 4.1.2 — Check per-variant checkpoint
        var_ckpt = config.ckpt_ablation_variant(outer_n, i)
        if checkpoint_exists(var_ckpt):
            var_data = load_checkpoint(var_ckpt)
            if _lineage_matches(var_data.get("lineage"), current_lineage):
                logger.info("Loaded per-variant checkpoint for variant %d (%s).", i, variant_name)
                results.append(var_data)
                continue
            else:
                logger.info(
                    "Per-variant checkpoint for variant %d (%s) has lineage mismatch — re-running.",
                    i, variant_name,
                )

        # Modality guard: no_stereo_fusion is meaningless for mono input
        if input_modality == "mono" and variant_name == "no_stereo_fusion":
            skip_result = {
                "variant_index": i,
                "name": variant_name,
                "status": "skipped",
                "reason": "stereo_ablation_not_relevant_for_mono",
                "metrics": None,
                "lineage": current_lineage,
            }
            save_checkpoint(var_ckpt, skip_result)
            results.append(skip_result)
            logger.info("Skipped %s (mono input).", variant_name)
            continue

        # In debug_mode: skip LLM + execution, emit a stub result
        if debug_mode:
            stub_result = {
                "variant_index": i,
                "name": variant_name,
                "status": "skipped",
                "reason": "debug_mode",
                "metrics": None,
                "lineage": current_lineage,
            }
            save_checkpoint(var_ckpt, stub_result)
            results.append(stub_result)
            logger.info("Debug mode: skipping variant execution for %s.", variant_name)
            continue

        # 4.1.1/4.1.2 — Generate ablated script via LLM
        logger.info("Generating ablated script for variant %d: %s", i, variant_name)
        ablated_script = _generate_variant_script(
            variant, best_pipeline_script, previous_summaries, token_state
        )

        if ablated_script is None:
            fail_result = {
                "variant_index": i,
                "name": variant_name,
                "status": "failed",
                "reason": "script_generation_failed",
                "metrics": None,
                "lineage": current_lineage,
            }
            save_checkpoint(var_ckpt, fail_result)
            results.append(fail_result)
            logger.warning("Script generation failed for variant %s.", variant_name)
            continue

        # Validate the ablated script (structural checks + dry-run)
        val_result = validate_script(ablated_script, input_modality=input_modality)
        if not val_result.valid:
            logger.warning(
                "Variant %s failed validation: %s — running anyway.",
                variant_name, val_result.rejection_reasons,
            )

        # 4.1.3 — Execute variant script
        run_result = _run_variant(variant, i, ablated_script, current_lineage)

        # 4.1.2 — Save per-variant checkpoint immediately
        config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        save_checkpoint(var_ckpt, run_result)
        logger.info("Saved per-variant checkpoint for variant %d (%s).", i, variant_name)

        # Log key metrics
        m = run_result.get("metrics") or {}
        if m:
            logger.info(
                "Variant %s: ng_recall=%.3f miss_rate=%.3f overkill=%.3f f1=%.3f",
                variant_name,
                m.get("ng_recall", 0.0),
                m.get("miss_rate", 1.0),
                m.get("overkill_rate", 1.0),
                m.get("f1", 0.0),
            )
        else:
            logger.info("Variant %s: FAILED (rc=%d).", variant_name, run_result.get("returncode", -1))

        results.append(run_result)

    # ------------------------------------------------------------------
    # 4.1.3 — Compute acceptance distance deltas
    # ------------------------------------------------------------------
    for r in results:
        if r.get("metrics"):
            r["delta_acceptance_distance"] = _compute_delta(r["metrics"], current_best_metrics)

    # Rank successful results by impact (most-impactful component first)
    # "Most impactful removal" = biggest acceptance distance regression when removed
    def _rank_key(r: dict) -> float:
        if r.get("status") != "success":
            return 0.0
        delta = r.get("delta_acceptance_distance")
        if delta is None:
            return 0.0
        # Negative delta = variant performed WORSE than baseline = component is valuable
        return -delta

    results.sort(key=_rank_key)

    # ------------------------------------------------------------------
    # 4.1.7 — Identify target component
    # ------------------------------------------------------------------
    target_component = _identify_target_component(
        results, current_best_metrics, previous_summaries, token_state
    )
    logger.info("phase2_ablation: identified target_component=%r for outer_iteration=%d", target_component, outer_n)

    # ------------------------------------------------------------------
    # 4.1.5 — Save aggregate checkpoint
    # ------------------------------------------------------------------
    if _is_complete(results):
        agg_payload = {
            "outer_iteration": outer_n,
            "lineage": current_lineage,
            "target_component": target_component,
            "ablation_results": results,
        }
        config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        save_checkpoint(agg_ckpt, agg_payload)
        logger.info("Saved aggregate ablation checkpoint: ablation_%d.json", outer_n)
    else:
        missing = sorted(
            set(range(NUM_ABLATION_VARIANTS))
            - {int(r.get("variant_index", -1)) for r in results if r.get("variant_index") is not None}
        )
        logger.warning(
            "phase2_ablation: %d/%d variants completed; missing indices %s — not saving aggregate checkpoint.",
            len(results), NUM_ABLATION_VARIANTS, missing,
        )

    # ------------------------------------------------------------------
    # 4.1.6 — Check outer loop exit conditions now that we've consumed tokens
    # ------------------------------------------------------------------
    # Merge token count back into a temporary state view for the check
    _state_view = dict(state)
    _state_view["tokens_used"] = token_state["token_count"]
    stop = _should_stop_outer_loop(_state_view)  # type: ignore[arg-type]

    return {
        "ablation_results": results,
        "target_component": target_component,
        "stop_outer_loop": stop,
        "tokens_used": token_state["token_count"],
    }
