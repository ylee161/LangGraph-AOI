"""nodes/phase3_ensemble_coder.py — Phase 3 Ensemble Coder Node.

Sub-steps:
  5.1.1 Load-on-demand: best pipeline script, Phase 1 candidate scores summary, ablation
        results, diagnosis, calibration stats, tried_ensemble_approaches history with fingerprints.
  5.1.2 Iteration 0: propose baseline e_0 (simple weighted-average ensemble).
  5.1.3 Subsequent iterations: propose e_r based on full {strategy, score} history;
        avoid previously-tried strategy fingerprints.
  5.1.4 Pass through code_validator.
  5.1.5 Return {ensemble_script, ensemble_strategy: {strategy_name, combination_method, strategy_fingerprint}}.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from mle_star_agent import config
from mle_star_agent.guards.code_validator import validate_script
from mle_star_agent.shared.checkpoint_io import checkpoint_exists, load_checkpoint
from mle_star_agent.shared.llm import build_messages, call_llm, call_llm_json
from mle_star_agent.state import AgentState

logger = logging.getLogger(__name__)


# ─── Strategy helpers ──────────────────────────────────────────────────────────

def _strategy_fingerprint(strategy_name: str, combination_method: str) -> str:
    return f"{strategy_name.strip().lower()}::{combination_method.strip().lower()}"


def _failed_ensemble_fingerprints(state: AgentState) -> set[str]:
    """Return fingerprints of previously-tried ensemble strategies that did not improve."""
    history = list(state.get("tried_ensemble_approaches", []) or [])
    if checkpoint_exists(config.CKPT_TRIED_ENSEMBLE_APPROACHES):
        ckpt = load_checkpoint(config.CKPT_TRIED_ENSEMBLE_APPROACHES)
        history.extend(ckpt.get("tried_ensemble_approaches", []) or [])
    failed: set[str] = set()
    for entry in history:
        result = entry.get("result") or {}
        if result.get("improved") is False:
            failed.add(_strategy_fingerprint(
                entry.get("strategy_name", ""),
                entry.get("combination_method", ""),
            ))
    return failed


def _tried_ensemble_history(state: AgentState) -> list[dict]:
    """Return tried_ensemble_approaches from state + disk, deduplicated by fingerprint."""
    seen: set[str] = set()
    merged: list[dict] = []
    sources = [state.get("tried_ensemble_approaches", [])]
    if checkpoint_exists(config.CKPT_TRIED_ENSEMBLE_APPROACHES):
        sources.append(
            load_checkpoint(config.CKPT_TRIED_ENSEMBLE_APPROACHES)
            .get("tried_ensemble_approaches", [])
        )
    for source in sources:
        for entry in (source or []):
            fp = entry.get("strategy_fingerprint", "")
            if fp not in seen:
                seen.add(fp)
                merged.append(entry)
    return merged


def _candidate_scores_summary(candidate_scores: list) -> list[dict]:
    return [
        {"name": e.get("name"), "metrics": e.get("metrics", {})}
        for e in (candidate_scores or [])
        if isinstance(e, dict)
    ]


# ─── Prompts ───────────────────────────────────────────────────────────────────

_STRATEGY_SYSTEM = """\
You are the Ensemble Strategy Planner for an AOI (Automated Optical Inspection) binary
inspection task.

Based on the provided context, choose a novel ensemble strategy and return JSON only.
Do NOT reuse any strategy whose fingerprint appears in FAILED_FINGERPRINTS.

Return JSON exactly matching this schema:
{
  "strategy_name": "<short kebab-case identifier, e.g. weighted-avg-dual-backbone>",
  "combination_method": "<how predictions are merged, e.g. weighted average of NG prob>",
  "component_descriptions": "<numbered list of model/pipeline components as a single string>"
}
"""

_SCRIPT_SYSTEM = (
    "You are the Ensemble Coder Agent for an AOI binary inspection pipeline.\n\n"
    "Implement the given ensemble strategy as a single self-contained Python script.\n\n"
    "CRITICAL output rules:\n"
    "1. Output ONLY raw Python code — no markdown fences, no commentary, no trailing text.\n"
    "2. The first character of your response must be the first character of Python code.\n"
    "3. The script must be complete and self-contained — no placeholders, no ellipsis.\n\n"
    "STARTING POINT — the user message includes BEST_PIPELINE_SCRIPT, which ALREADY passes the\n"
    "validator below. Reuse its data-loading, label-reading, LR-schedule, degenerate-prediction\n"
    "guard, and ALL of its print-marker scaffolding VERBATIM. Only add/modify what the ensemble\n"
    "strategy requires (extra models, the combination step). Do NOT drop any compliant scaffolding.\n\n"
    "VALIDATOR CONTRACT — the script is statically rejected unless EVERY item below is present.\n"
    "Treat each as a hard requirement, not a suggestion:\n\n"
    "[C1] Stereo input: load BOTH _L and _R images for every component that uses visual input\n"
    "     (reference the '_L' and '_R' path tokens explicitly).\n"
    "[C2] Labels: read the Excel ground-truth labels with pandas — the script text MUST contain a\n"
    "     literal `pd.read_excel(...)` call (the token 'read_excel' / '.xlsx' must appear). Loading\n"
    "     IDs from the split JSON is NOT a substitute for reading the Excel label column.\n"
    "[C3] Split: load the authoritative split and use its train/val/test IDs (no test-set leakage):\n"
    "       import json; data_split = json.load(open('DATA_SPLIT_PATH'))\n"
    "     The identifier `data_split` must appear in the script.\n"
    "[C4] Seed ALL RNGs from env BEFORE any model/dataloader is created:\n"
    "       import os, random\n"
    "       _seed = int(os.environ.get('AOI_RANDOM_SEED', os.environ.get('SEED', 42)))\n"
    "       random.seed(_seed); np.random.seed(_seed); torch.manual_seed(_seed); "
    "torch.cuda.manual_seed_all(_seed)\n"
    "[C4b] Device (target machine is Apple Silicon, no CUDA): select CUDA -> MPS -> CPU and move\n"
    "      the model AND all tensors onto it. Use float32 only (MPS has no float64):\n"
    "       device = torch.device('cuda' if torch.cuda.is_available() "
    "else ('mps' if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available() else 'cpu'))\n"
    "      Unsupported MPS ops auto-fall back to CPU (the runner sets PYTORCH_ENABLE_MPS_FALLBACK=1),\n"
    "      so prefer MPS — do not hard-code .cpu() for training.\n"
    "[C5] Epoch budget — include this line VERBATIM (the substring "
    "'epochs = DRY_RUN_EPOCHS if DRY_RUN else' is checked literally):\n"
    "       epochs = DRY_RUN_EPOCHS if DRY_RUN else 20\n"
    "     with DRY_RUN = int(os.environ.get('DRY_RUN', 0)) and "
    "DRY_RUN_EPOCHS = int(os.environ.get('DRY_RUN_EPOCHS', 1)).\n"
    "[C6] LR schedule: every trained component MUST construct a real scheduler "
    "(CosineAnnealingWarmRestarts or ReduceLROnPlateau) AND call scheduler.step().\n"
    "[C7] Degenerate-prediction guard: after scoring the validation set, compute the score spread\n"
    "     (e.g. `score_std = float(np.std(val_scores))` and `score_range = float(val_scores.max() - "
    "val_scores.min())`) and print `DEGENERATE_PREDICTION_WARNING` when the model collapses\n"
    "     (score_std < 1e-3 or all predictions on one side of the threshold). The literal token\n"
    "     `DEGENERATE_PREDICTION_WARNING` (or a `score_std` / `score_range` / `unique_scores` "
    "variable) MUST appear.\n"
    "[C8] Threshold: sweep the decision threshold on the VALIDATION set to maximise NG recall "
    "subject to overkill_rate <= 0.08.\n"
    "[C9] Print EXACTLY these six output markers (the literal 'LABEL:' prefixes are checked):\n"
    "       EPOCH_LOG: {\"epoch\": <int>, \"train_loss\": <float>, \"val_loss\": <float>, "
    "\"val_overkill\": <float>, \"val_ng_recall\": <float>}   # one line PER epoch\n"
    "       PROBE_METRICS: {\"g_prob_mean\": <float>, \"ng_prob_mean\": <float>, "
    "\"probability_gap\": <float>}\n"
    "       CALIBRATION_STATS: {\"g_prob_mean\": <float>, \"ng_prob_mean\": <float>}\n"
    "       THRESHOLD_CURVE: [{\"threshold\": <float>, \"ng_recall\": <float>, "
    "\"overkill\": <float>, \"miss_rate\": <float>}, ...]\n"
    "       PREDICTIONS: [{\"id\": <str>, \"label\": <str>, \"prob\": <float>, \"pred\": <str>}, ...]\n"
    '       METRICS: {"accuracy": ..., "ng_recall": ..., "miss_rate": ..., "overkill_rate": ..., '
    '"f1": ..., "roc_auc": ..., "prob_gap": ..., "avg_latency_ms": ..., "threshold": ...,\n'
    '                 "ng_count": ..., "g_count": ..., "tp": ..., "tn": ..., "fp": ..., "fn": ...}\n'
    "     (METRICS must include roc_auc and prob_gap, and a THRESHOLD_CURVE: line must be present —\n"
    "     these three together satisfy the metric-reporting check.)\n"
    "[C10] Keep total wall-clock time under 7200 s (2 hours).\n\n"
    "Before returning, self-check that C1–C9 each appear in your script text.\n"
).replace("DATA_SPLIT_PATH", str(config.CKPT_DATA_SPLIT))


def _build_context_block(
    n: int,
    pipeline_script: str,
    candidate_scores_summary: list,
    ablation_results: Any,
    diagnosis: str,
    calibration_stats: Any,
) -> str:
    return "\n\n".join([
        f"ENSEMBLE_ITERATION: {n}",
        f"CANDIDATE_SCORES (Phase 1 baselines): {json.dumps(candidate_scores_summary, default=str)}",
        f"ABLATION_RESULTS: {json.dumps(ablation_results, default=str)}",
        f"DIAGNOSIS: {diagnosis}",
        f"LATEST_CALIBRATION_STATS: {json.dumps(calibration_stats, default=str)}",
        f"BEST_PIPELINE_SCRIPT:\n{pipeline_script}",
    ])


def _build_strategy_context(
    n: int,
    tried_history: list[dict],
    failed_fingerprints: set[str],
) -> str:
    history_compact = [
        {
            "strategy_name": e.get("strategy_name"),
            "combination_method": e.get("combination_method"),
            "fingerprint": e.get("strategy_fingerprint"),
            "result": e.get("result", {}),
        }
        for e in tried_history
    ]
    history_str = json.dumps(history_compact, default=str)
    failed_str = ", ".join(sorted(failed_fingerprints)) if failed_fingerprints else "none"

    if n == 0:
        guidance = (
            "Iteration 0 — propose a simple weighted-average ensemble strategy:\n"
            "Train two complementary models (e.g. different stereo fusion approaches or "
            "backbones), compute a weighted average of their NG probability outputs, then "
            "sweep the combined threshold on the validation set."
        )
    elif n == 1:
        guidance = (
            "Iteration 1 — propose a second-stage verifier cascade:\n"
            "Train a high-recall first-stage detector (low threshold ~0.3), then train a "
            "second-stage verifier only on samples the first stage flags as NG. The verifier "
            "rescues obvious G false positives while preserving NG recall. Keep NG when stage1 "
            "is confident OR the verifier cannot confidently prove it is G."
        )
    elif n == 2:
        guidance = (
            "Iteration 2 — propose an augmentation-diversity ensemble:\n"
            "Train the best pipeline three times with different random augmentation seeds, "
            "average their predicted NG probabilities, then sweep the threshold on validation."
        )
    else:
        guidance = (
            f"Iteration {n} — prior strategies have been tried. Propose a new approach that "
            "combines elements from prior successful iterations or introduces stacking if "
            "diversity was observed. Prioritise simultaneously improving NG recall AND "
            "reducing overkill."
        )

    return (
        f"STRATEGY_GUIDANCE:\n{guidance}\n\n"
        f"TRIED_ENSEMBLE_APPROACHES:\n{history_str}\n\n"
        f"FAILED_FINGERPRINTS (do NOT reuse): {failed_str}"
    )


def _build_script_prompt(
    n: int,
    strategy: dict,
    pipeline_script: str,
    calibration_stats: Any,
    tried_history: list[dict],
) -> str:
    history_compact = json.dumps(
        [{"strategy_name": e.get("strategy_name"), "result": e.get("result", {})} for e in tried_history],
        default=str,
    )
    return "\n\n".join([
        f"ENSEMBLE_ITERATION: {n}",
        f"STRATEGY_NAME: {strategy['strategy_name']}",
        f"COMBINATION_METHOD: {strategy['combination_method']}",
        f"COMPONENT_DESCRIPTIONS: {strategy.get('component_descriptions', '')}",
        f"STRATEGY_FINGERPRINT: {strategy['strategy_fingerprint']}",
        f"LATEST_CALIBRATION_STATS: {json.dumps(calibration_stats, default=str)}",
        f"TRIED_ENSEMBLE_APPROACHES (reference):\n{history_compact}",
        "BEST_PIPELINE_SCRIPT (base each component on this — modify only what the strategy requires):",
        pipeline_script,
        (
            "Implement the ensemble strategy as a complete self-contained Python script.\n"
            "Follow ALL requirements from the system prompt.\n"
            "Output ONLY raw Python code — no markdown fences, no text before or after."
        ),
    ])


def _strip_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:python)?\s*\n?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


# ─── Node ──────────────────────────────────────────────────────────────────────

def phase3_ensemble_coder_node(state: AgentState) -> dict[str, Any]:
    """Phase 3 ensemble coder node.

    Designs and implements an ensemble strategy based on Phase 2 outputs.
    Validates the resulting script and returns it along with strategy metadata.
    """
    token_state: dict = {"token_count": int(state.get("tokens_used", 0) or 0)}
    n = int(state.get("ensemble_iteration", 0))

    # ─── 5.1.1 Load context ──────────────────────────────────────────────────
    pipeline_script = ""
    best_pipeline = state.get("best_pipeline")
    if best_pipeline and isinstance(best_pipeline, dict):
        pipeline_script = best_pipeline.get("script", "")
    if not pipeline_script and checkpoint_exists(config.CKPT_BEST_PIPELINE):
        _bp = load_checkpoint(config.CKPT_BEST_PIPELINE)
        pipeline_script = _bp.get("script") or _bp.get("best_pipeline_script", "")
    if not pipeline_script and checkpoint_exists(config.CKPT_L0):
        pipeline_script = load_checkpoint(config.CKPT_L0).get("script", "")

    if not pipeline_script:
        logger.error("Ensemble coder: no base pipeline script found.")
        return {"error": "No base pipeline script found for ensemble coder."}

    candidate_scores_summary = _candidate_scores_summary(state.get("candidate_scores", []))
    ablation_results = state.get("ablation_results", {})
    diagnosis = state.get("diagnosis", "")
    calibration_stats = state.get("latest_calibration_stats", {})
    tried_history = _tried_ensemble_history(state)
    failed_fingerprints = _failed_ensemble_fingerprints(state)

    # ─── 5.1.2 / 5.1.3 Plan strategy ────────────────────────────────────────
    context_block = _build_context_block(
        n=n,
        pipeline_script=pipeline_script,
        candidate_scores_summary=candidate_scores_summary,
        ablation_results=ablation_results,
        diagnosis=diagnosis,
        calibration_stats=calibration_stats,
    )
    strategy_block = _build_strategy_context(n, tried_history, failed_fingerprints)
    strategy_user_prompt = (
        f"{context_block}\n\n{strategy_block}\n\n"
        "Choose the best ensemble strategy based on this context. Return JSON only."
    )

    try:
        strategy_data = call_llm_json(
            build_messages(_STRATEGY_SYSTEM, strategy_user_prompt),
            model=config.MODEL_PRO,
            max_tokens=1024,
            temperature=0.2,
            token_state=token_state,
        )
    except Exception as e:
        logger.error("Ensemble coder: strategy planning LLM call failed: %s", e)
        return {"error": f"Strategy planning failed: {e}", "tokens_used": token_state["token_count"]}

    if not isinstance(strategy_data, dict):
        logger.error("Ensemble coder: strategy planning returned non-dict JSON (%s).", type(strategy_data).__name__)
        return {"error": "Strategy planning returned malformed JSON.", "tokens_used": token_state["token_count"]}

    strategy_name = strategy_data.get("strategy_name", f"ensemble-v{n}")
    combination_method = strategy_data.get("combination_method", "weighted average")
    component_descriptions = strategy_data.get("component_descriptions", "")
    fingerprint = _strategy_fingerprint(strategy_name, combination_method)

    # Retry once if the fingerprint was already tried and failed
    if fingerprint in failed_fingerprints:
        logger.warning(
            "LLM proposed already-failed fingerprint '%s' (iter=%d); retrying.", fingerprint, n
        )
        override_prompt = (
            f"{strategy_user_prompt}\n\n"
            f"REJECTED: fingerprint '{fingerprint}' already failed. "
            "Choose a materially different strategy. Return JSON only."
        )
        try:
            strategy_data = call_llm_json(
                build_messages(_STRATEGY_SYSTEM, override_prompt),
                model=config.MODEL_PRO,
                max_tokens=1024,
                temperature=0.4,
                token_state=token_state,
            )
            strategy_name = strategy_data.get("strategy_name", f"ensemble-v{n}-alt")
            combination_method = strategy_data.get("combination_method", "stacking")
            component_descriptions = strategy_data.get("component_descriptions", "")
            fingerprint = _strategy_fingerprint(strategy_name, combination_method)
        except Exception as e:
            logger.warning("Ensemble coder: strategy retry failed (%s); proceeding anyway.", e)

    ensemble_strategy = {
        "ensemble_iteration": n,
        "strategy_name": strategy_name,
        "combination_method": combination_method,
        "component_descriptions": component_descriptions,
        "strategy_fingerprint": fingerprint,
    }
    logger.info(
        "Ensemble coder (iter=%d): strategy='%s', method='%s', fingerprint='%s'",
        n, strategy_name, combination_method, fingerprint,
    )

    # ─── 5.1.4 Generate and validate script ──────────────────────────────────
    script_user_prompt = _build_script_prompt(
        n=n,
        strategy=ensemble_strategy,
        pipeline_script=pipeline_script,
        calibration_stats=calibration_stats,
        tried_history=tried_history,
    )

    script = ""
    valid_script = ""
    last_validation_result = None
    for attempt in range(config.DEBUGGER_RETRY_CAP):
        fix_ctx = ""
        if attempt > 0 and last_validation_result is not None:
            reasons = "\n".join(last_validation_result.rejection_reasons)
            fix_ctx = (
                f"\n\nPrevious script failed validation (attempt {attempt}):\n{reasons}\n"
                "Fix ALL issues listed above. Output ONLY the corrected Python script."
            )

        try:
            raw = call_llm(
                build_messages(_SCRIPT_SYSTEM, script_user_prompt + fix_ctx),
                model=config.MODEL_PRO,
                max_tokens=16384,
                temperature=0.2 if attempt == 0 else 0.3,
                token_state=token_state,
            )
        except Exception as e:
            logger.error("Ensemble coder: script generation failed (attempt %d): %s", attempt + 1, e)
            return {
                "error": f"Script generation failed: {e}",
                "tokens_used": token_state["token_count"],
            }

        script = _strip_fences(raw)
        if not script.strip():
            logger.warning("Ensemble coder: empty script (attempt %d).", attempt + 1)
            continue

        # ─── 5.1.4 code_validator ────────────────────────────────────────
        last_validation_result = validate_script(script)
        if last_validation_result.valid:
            logger.info("Ensemble coder: script passed validation (attempt %d).", attempt + 1)
            valid_script = script
            break

        logger.warning(
            "Ensemble coder: validation failed (attempt %d): %s",
            attempt + 1, last_validation_result.rejection_reasons,
        )

    # Only surface a VALIDATED script. Returning the last (invalid) attempt as
    # ensemble_script would poison state — phase4 would then run a known-bad script
    # instead of falling back to the validated best_pipeline.
    if not valid_script:
        reasons = (
            "; ".join(last_validation_result.rejection_reasons)
            if last_validation_result is not None else "no script produced"
        )
        logger.error("Ensemble coder: no valid script after %d attempts: %s",
                     config.DEBUGGER_RETRY_CAP, reasons)
        return {
            "error": f"Ensemble coder produced no valid script: {reasons}",
            "tokens_used": token_state["token_count"],
        }

    # ─── 5.1.5 Return ────────────────────────────────────────────────────────
    return {
        "ensemble_script": valid_script,
        "ensemble_strategy": ensemble_strategy,
        "tokens_used": token_state["token_count"],
    }
