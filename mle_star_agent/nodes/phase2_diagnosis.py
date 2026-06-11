"""nodes/phase2_diagnosis.py — Phase 2 Diagnosis Node (MLE-STAR Section 3.2).

Sub-steps:
  4.2.1  Checkpoint gate with lineage check — load diagnosis_N.json if SHA-256 matches
  4.2.2  LLM reads ablation summary → identifies target code block c_t + initial plan p_0
  4.2.3  Return {diagnosis_report, target_component, target_block_code, refinement_plan: p_0}
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Optional

from mle_star_agent import config
from mle_star_agent.shared.checkpoint_io import (
    checkpoint_exists,
    load_checkpoint,
    save_checkpoint,
)
from mle_star_agent.shared.diagnosis_scorer import generate_diagnosis_brief
from mle_star_agent.shared.llm import build_messages, call_llm_json
from mle_star_agent.state import AgentState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lineage helpers
# ---------------------------------------------------------------------------

def _stable_sha256(obj: Any) -> str:
    raw = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _diagnosis_lineage(ablation_results: list, best_pipeline_script: str) -> dict:
    return {
        "ablation_results_sha256": _stable_sha256(ablation_results),
        "ablation_result_count": len(ablation_results or []),
        "best_pipeline_script_sha256": _stable_sha256(best_pipeline_script),
    }


def _lineage_matches(stored: Optional[dict], current: dict) -> bool:
    if not stored or not current:
        return False
    return (
        stored.get("ablation_results_sha256") == current.get("ablation_results_sha256")
        and stored.get("ablation_result_count") == current.get("ablation_result_count")
    )


# ---------------------------------------------------------------------------
# LLM prompt
# ---------------------------------------------------------------------------

_SYSTEM = """You are the Diagnosis Agent for Phase 2 Refinement of an AOI (Automated Optical Inspection) pipeline.

You have received a structured diagnosis brief computed by a deterministic scorer.  The brief
contains a failure classification, ablation ranking, and recommended target.

Your job:
1. Review the brief and confirm or adjust the recommended target_component
2. Identify the specific code block c_t in the best pipeline script that should be refined
3. Write an initial refinement plan p_0 (3-5 concrete bullet points)

Metric priority (industrial spec §8, highest first):
  P0 miss_rate  >  P1 ng_recall  >  P2 overkill_rate  >  P3 latency  >  P4 accuracy  >  P5 model_size
NEVER propose a change that regresses a higher-priority metric to improve a lower-priority one
(e.g. do not trade away ng_recall / raise miss_rate just to cut overkill or lift accuracy).
A model with high accuracy but an unacceptable miss_rate is NOT qualified.

Guidelines:
- If miss_rate > 0.03: prioritise NG recall over overkill reduction
- If overkill_rate > 0.08 and miss_rate <= 0.03: prioritise false-positive control
- If acceptance_distance < 0.5: fine-tune threshold and calibration
- Prefer ablation_ranking evidence over your own assumptions
- target_block_code should be the EXACT name/label of the code section to improve
  (e.g. "loss_function", "threshold_sweep", "stereo_fusion", "augmentation", "backbone")

Return JSON:
{
  "target_component": "<component>",
  "target_block_code": "<code block label>",
  "impact_summary": "<one paragraph>",
  "recommended_changes": "<3-5 concrete bullet points as a string>",
  "refinement_plan": "<initial plan p_0: what to implement first>",
  "prediction_contract": {
    "expected_overkill_rate_max": <float>,
    "expected_ng_recall_min": <float>,
    "expected_miss_rate_max": <float>,
    "failure_if": "<condition string>"
  }
}
"""


def _build_diagnosis_prompt(
    brief: dict,
    outer_n: int,
    previous_targets: list[str],
    best_script_snippet: str,
) -> str:
    lines = [
        f"outer_iteration: {outer_n}",
        f"previous_targets_tried: {previous_targets}",
        "",
        "Deterministic diagnosis brief:",
        json.dumps(brief, indent=2, default=str),
    ]
    if best_script_snippet:
        lines += [
            "",
            "Best pipeline script (first 3000 chars for code block identification):",
            best_script_snippet[:3000],
        ]
    lines.append("\nIdentify the target component and write the initial refinement plan. Return JSON.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Previous target history helper
# ---------------------------------------------------------------------------

def _previous_target_components(outer_n: int) -> list[str]:
    targets = []
    for n in range(outer_n):
        ckpt = config.ckpt_diagnosis(n)
        if checkpoint_exists(ckpt):
            try:
                data = load_checkpoint(ckpt)
                t = (data.get("diagnosis_report") or {}).get("target_component")
                if t:
                    targets.append(t)
            except Exception:
                pass
    return targets


# ---------------------------------------------------------------------------
# Main node
# ---------------------------------------------------------------------------

def phase2_diagnosis_node(state: AgentState) -> dict[str, Any]:
    """Phase 2 diagnosis node: reads ablation results, classifies failure mode,
    identifies target code block c_t, and writes an initial plan p_0.

    Returns a partial state update dict.
    """
    token_state: dict = {"token_count": int(state.get("tokens_used", 0) or 0)}
    outer_n = int(state.get("outer_iteration", 0))
    debug_mode: bool = bool(state.get("debug_mode", config.DEBUG_MODE))

    # Gather context
    ablation_results: list = state.get("ablation_results") or []
    best_pipeline: dict = state.get("best_pipeline") or {}
    best_pipeline_script: str = best_pipeline.get("script", "")

    if not best_pipeline_script and checkpoint_exists(config.CKPT_L0):
        try:
            l0 = load_checkpoint(config.CKPT_L0)
            best_pipeline_script = l0.get("script", "")
        except Exception:
            pass

    current_lineage = _diagnosis_lineage(ablation_results, best_pipeline_script)

    # ------------------------------------------------------------------
    # 4.2.1 — Checkpoint gate with lineage check
    # ------------------------------------------------------------------
    ckpt_path = config.ckpt_diagnosis(outer_n)
    if checkpoint_exists(ckpt_path):
        try:
            data = load_checkpoint(ckpt_path)
            if _lineage_matches(data.get("lineage"), current_lineage):
                report = data.get("diagnosis_report", {})
                # Also reject if ablation_ranking is empty but we have ablation data
                ranking = report.get("ablation_ranking", [])
                if not ranking and ablation_results:
                    logger.info(
                        "diagnosis_%d.json has empty ablation_ranking but ablation_results exist — recomputing.",
                        outer_n,
                    )
                else:
                    logger.info(
                        "phase2_diagnosis: loaded checkpoint for outer_iteration=%d target='%s'.",
                        outer_n, report.get("target_component", ""),
                    )
                    return {
                        "diagnosis": json.dumps(report),
                        "target_component": report.get("target_component", ""),
                        "target_block_code": report.get("target_block_code", ""),
                        "refinement_plan": report.get("refinement_plan", ""),
                        "tokens_used": token_state["token_count"],
                    }
            else:
                logger.info("diagnosis_%d.json lineage mismatch — recomputing.", outer_n)
        except Exception as exc:
            logger.warning("Failed to load diagnosis checkpoint for iteration %d: %s", outer_n, exc)

    # ------------------------------------------------------------------
    # 4.2.2 — Compute deterministic diagnosis brief, then ask LLM
    # ------------------------------------------------------------------
    def _f(key: str, default: float) -> float:
        v = state.get(key)
        return float(v) if v is not None else default

    baseline_metrics: dict = {
        "ng_recall":     _f("current_best_score", 0.0),
        "miss_rate":     _f("best_miss_rate", 1.0),
        "overkill_rate": _f("best_overkill_rate", 1.0),
        "accuracy":      _f("best_accuracy", 0.0),
        "f1":            _f("best_f1", 0.0),
    }

    data_split_meta: dict = (state.get("data_split") or {}).get("metadata", {})
    input_modality: str = data_split_meta.get("input_modality", "stereo")

    calibration_stats: Optional[dict] = state.get("latest_metrics", {})
    error_analysis: Optional[dict] = None
    threshold_curve: Optional[list] = None

    brief = generate_diagnosis_brief(
        ablation_results,
        baseline_metrics,
        calibration_stats=calibration_stats,
        error_analysis=error_analysis,
        threshold_curve=threshold_curve,
        input_modality=input_modality,
    )

    previous_targets = _previous_target_components(outer_n)

    if debug_mode:
        # In debug mode, skip LLM and use the scorer's recommendation directly
        failure = brief.get("failure_classification", {})
        target_component = str(brief.get("recommended_target", "threshold_selection"))
        target_block_code = f"# {target_component}"
        refinement_plan = str(brief.get("recommended_action", "Apply recommended changes."))
        impact_summary = (
            f"[debug] failure_mode={failure.get('failure_mode')} confidence={failure.get('confidence')}"
        )
        logger.info("phase2_diagnosis: debug mode — using scorer recommendation, skipping LLM.")
    else:
        user_prompt = _build_diagnosis_prompt(
            brief, outer_n, previous_targets, best_pipeline_script
        )
        try:
            response = call_llm_json(
                build_messages(_SYSTEM, user_prompt),
                model=config.MODEL_PRO,
                max_tokens=4096,
                temperature=0.2,
                token_state=token_state,
            )
            if isinstance(response, dict):
                target_component = str(response.get("target_component") or brief.get("recommended_target", "threshold_selection"))
                # Use marker as default if target_block_code is missing or just the component name
                target_block_code = str(response.get("target_block_code") or f"# {target_component}")
                if target_block_code == target_component:
                    target_block_code = f"# {target_component}"
                refinement_plan = str(response.get("refinement_plan") or response.get("recommended_changes") or "")
                impact_summary = str(response.get("impact_summary") or "")
            else:
                raise ValueError(f"Unexpected LLM response type: {type(response)}")
        except Exception as exc:
            logger.warning("Diagnosis LLM call failed: %s — falling back to scorer.", exc)
            failure = brief.get("failure_classification", {})
            target_component = str(brief.get("recommended_target", "threshold_selection"))
            target_block_code = f"# {target_component}"
            refinement_plan = str(brief.get("recommended_action", "Apply recommended changes."))
            impact_summary = (
                f"[fallback] failure_mode={failure.get('failure_mode')} confidence={failure.get('confidence')}"
            )

    # ------------------------------------------------------------------
    # Build and persist diagnosis report
    # ------------------------------------------------------------------
    report: dict = {
        "outer_iteration": outer_n,
        "target_component": target_component,
        "target_block_code": target_block_code,
        "impact_summary": impact_summary,
        "refinement_plan": refinement_plan,
        "ablation_ranking": brief.get("ablation_ranking", []),
        "failure_classification": brief.get("failure_classification"),
        "baseline_metrics": brief.get("baseline_metrics"),
        "baseline_acceptance_distance": brief.get("baseline_acceptance_distance"),
    }

    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    save_checkpoint(ckpt_path, {"lineage": current_lineage, "diagnosis_report": report})
    logger.info(
        "phase2_diagnosis: outer_iteration=%d target_component='%s' target_block_code='%s'",
        outer_n, target_component, target_block_code,
    )

    # ------------------------------------------------------------------
    # 4.2.3 — Return state update
    # ------------------------------------------------------------------
    return {
        "diagnosis": json.dumps(report),
        "target_component": target_component,
        "target_block_code": target_block_code,
        "refinement_plan": refinement_plan,
        "tokens_used": token_state["token_count"],
    }
