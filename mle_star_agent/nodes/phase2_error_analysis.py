import hashlib
import logging
from typing import Any

from mle_star_agent import config
from mle_star_agent.shared.checkpoint_io import checkpoint_exists, load_checkpoint, save_checkpoint
from mle_star_agent.shared.metrics_parser import AOIMetrics, parse_error_analysis
from mle_star_agent.state import AgentState

logger = logging.getLogger(__name__)


def phase2_error_analysis(state: AgentState) -> dict[str, Any]:
    """
    Parse PREDICTIONS blocks from last stdout, extract detailed FP/FN breakdown,
    and lineage-check against the last script run to avoid re-parsing on resume.
    """
    outer_n = state.get("outer_iteration", 0)
    inner_m = state.get("inner_iteration", 0)

    # The last attempt index evaluated was inner_m - 1
    # If inner_iteration is 0, we shouldn't be here (gate skips), but handle safely
    last_idx = inner_m - 1
    if last_idx < 0:
        logger.warning("phase2_error_analysis called with inner_iteration 0, returning empty.")
        return {}

    candidate_scripts = state.get("candidate_scripts", [])
    if not candidate_scripts:
        logger.warning("No candidate scripts found in state for error analysis.")
        return {}

    last_script = candidate_scripts[-1]
    script_hash = hashlib.sha256(last_script.encode("utf-8")).hexdigest()

    ckpt_ea = config.ckpt_error_analysis(outer_n, last_idx)
    
    # 4.4.2 — Lineage-check against last script run; load if current
    if checkpoint_exists(ckpt_ea):
        try:
            data = load_checkpoint(ckpt_ea)
            if data.get("lineage") == script_hash:
                logger.info("Loaded error analysis from checkpoint %s.", ckpt_ea)
                report = data.get("report")
                return {
                    "error_analysis": report,
                    "error_analysis_report": report,
                    "latest_error_analysis": report,
                }
        except Exception as e:
            logger.warning("Failed to load error analysis checkpoint %s: %s", ckpt_ea, e)

    # 4.4.1 — Parse PREDICTIONS blocks from last stdout
    # Assume phase2_evaluator saved stdout to ckpt_refinement
    ckpt_ref = config.ckpt_refinement(outer_n, last_idx)
    stdout = ""
    if checkpoint_exists(ckpt_ref):
        try:
            ref_data = load_checkpoint(ckpt_ref)
            stdout = ref_data.get("stdout", "")
        except Exception as e:
            logger.warning("Failed to load refinement checkpoint %s: %s", ckpt_ref, e)
    
    if not stdout:
        logger.warning("No stdout found for inner iteration %d in checkpoint.", last_idx)

    # Get metrics object to pass to parse_error_analysis
    latest_metrics = state.get("latest_metrics", {})
    aoi_metrics = None
    if latest_metrics:
        aoi_metrics = AOIMetrics(
            accuracy=latest_metrics.get("accuracy", 0.0),
            ng_recall=latest_metrics.get("ng_recall", 0.0),
            miss_rate=latest_metrics.get("miss_rate", 0.0),
            overkill_rate=latest_metrics.get("overkill_rate", 0.0),
            f1=latest_metrics.get("f1", 0.0),
            avg_latency_ms=latest_metrics.get("avg_latency_ms", 0.0),
            threshold=latest_metrics.get("threshold", 0.5),
            ng_count=latest_metrics.get("ng_count", 0),
            g_count=latest_metrics.get("g_count", 0),
            tp=latest_metrics.get("tp", 0),
            tn=latest_metrics.get("tn", 0),
            fp=latest_metrics.get("fp", 0),
            fn=latest_metrics.get("fn", 0),
            roc_auc=latest_metrics.get("roc_auc", 0.0),
            prob_gap=latest_metrics.get("prob_gap", 0.0),
        )

    report = parse_error_analysis(stdout, aoi_metrics)

    # Apply cap to FP/FN samples (ERROR_ANALYSIS_SAMPLE_CAP)
    if "fp_samples" in report and report["fp_samples"]:
        report["fp_samples"] = report["fp_samples"][:config.ERROR_ANALYSIS_SAMPLE_CAP]
    if "fn_samples" in report and report["fn_samples"]:
        report["fn_samples"] = report["fn_samples"][:config.ERROR_ANALYSIS_SAMPLE_CAP]

    save_checkpoint(ckpt_ea, {
        "lineage": script_hash,
        "report": report
    })

    # 4.4.3 — Return {error_analysis_report, latest_error_analysis} 
    # and update `error_analysis` in AgentState.
    return {
        "error_analysis": report,
        "error_analysis_report": report,
        "latest_error_analysis": report,
    }
