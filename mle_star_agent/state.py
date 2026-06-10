"""
AgentState — the single shared state dict that flows through the LangGraph.

All nodes read from and write to this TypedDict.  Fields follow the same
naming convention as the ADK reference implementation (AOI agent/).
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class AgentState(TypedDict, total=False):
    # ── Run identity ────────────────────────────────────────────────────────
    run_id: str              # unique ID for this agent run
    goal: str                # natural-language performance target
    dataset_path: str        # absolute path to the dataset root

    # ── Phase tracking ──────────────────────────────────────────────────────
    # Counter names follow the ADK reference (outer_iteration / inner_iteration /
    # ensemble_iteration are the dominant keys there). Keep these names everywhere.
    current_phase: str       # "init" | "refine" | "ensemble" | "submit"
    outer_iteration: int     # outer refinement iteration counter
    inner_iteration: int     # inner (per-diagnosis) iteration counter
    ensemble_iteration: int  # Phase 3 iteration counter
    submission_retry: int    # Phase 4 retry counter
    no_improve_count: int    # consecutive non-improving iterations (patience)
    stop_outer_loop: bool    # set by route logic when the outer loop should end

    # ── Data split ──────────────────────────────────────────────────────────
    data_split: dict[str, Any]   # {train: [...], val: [...], test: [...]}

    # ── Candidate model tracking ─────────────────────────────────────────────
    # Accumulator fields: nodes APPEND via the operator.add reducer. A node that
    # must not append (e.g. a skip-check that loads from disk) returns nothing for
    # these keys rather than the full list, to avoid double-appending on resume.
    best_pipeline: dict[str, Any]                       # {script, metrics, threshold}
    candidate_scripts: Annotated[list[str], operator.add]   # generated training scripts
    candidate_scores: Annotated[list[dict], operator.add]   # evaluation metrics per candidate
    tried_approaches: Annotated[list[dict], operator.add]   # summaries of past attempts

    # ── Best-so-far snapshot (mirrors ADK session.state) ─────────────────────
    current_best_score: float
    best_miss_rate: float
    best_overkill_rate: float
    best_accuracy: float
    best_f1: float
    best_candidate_name: str

    # ── Latest metrics ───────────────────────────────────────────────────────
    # Canonical metric keys (match metrics_parser.py / acceptance_scoring.py):
    # ng_recall, miss_rate, overkill_rate, accuracy, f1.
    latest_metrics: dict[str, float]
    threshold: float                  # best decision threshold

    # ── Ablation / diagnosis / error analysis ────────────────────────────────
    ablation_results: list[dict[str, Any]]  # per-component ablation impact (one entry per variant)
    target_component: str             # weakest component identified by ablation
    target_block_code: str            # code block the inner loop will refine
    # NOTE: v1 uses free-text strings here. ADK instead passes structured
    # `diagnosis_report` / `error_analysis_report` dicts between agents — if a
    # ported node expects those dict fields, add them rather than re-typing these.
    diagnosis: str                    # LLM-written diagnosis of failures
    error_analysis: dict[str, Any]    # structured FP/FN breakdown (from error_analysis_node)
    error_analysis_report: dict[str, Any]  # alias returned by error_analysis node
    latest_error_analysis: dict[str, Any]  # most recent error analysis result
    refinement_plan: str              # chosen strategy for this inner iteration

    # ── Error analysis gate state (phase2_error_analysis_gate) ───────────────
    error_analysis_instrumentation_required: bool  # coder must emit PREDICTIONS
    error_analysis_repair_attempted: bool          # one repair pass already tried
    error_analysis_blocked: bool                   # blind refinement blocked

    # ── Ensemble ──────────────────────────────────────────────────────────────
    ensemble_script: str              # current ensemble training/inference script
    # REPLACE semantics (no reducer): each ensemble iteration defines its own full
    # member set, so last-write-wins is correct — do NOT add operator.add here, or
    # members from different attempts would be merged together.
    ensemble_models: list[dict]       # list of {path, weight, metrics} for the current ensemble
    ensemble_strategy: dict           # {strategy_name, combination_method, strategy_fingerprint, ...}

    # Phase 3 best-so-far snapshot (independent from Phase 2 current_best_score)
    ensemble_best_score: float        # best ng_recall achieved in Phase 3
    ensemble_best_overkill: float     # overkill for the Phase 3 best
    ensemble_best_accuracy: float     # accuracy for the Phase 3 best
    ensemble_best_f1: float           # f1 for the Phase 3 best
    ensemble_no_improve_count: int    # consecutive non-improving Phase 3 iterations
    stop_ensemble_loop: bool          # exit signal for route_ensemble_loop

    # Accumulator: nodes append one entry per iteration via operator.add.
    # Disk checkpoint (CKPT_TRIED_ENSEMBLE_APPROACHES) is the deduplication authority.
    tried_ensemble_approaches: Annotated[list[dict], operator.add]

    # ── Submission ────────────────────────────────────────────────────────────
    submission_passed: bool       # True when acceptance criteria are met
    submission_report: str        # human-readable final report

    # ── Token budget (for model switching) ───────────────────────────────────
    # Canonical name is `tokens_used`. The ADK reference calls this `token_count`;
    # ported nodes and the budget/route check must use `tokens_used`, NOT token_count,
    # or the budget gate silently reads 0 and never fires.
    tokens_used: int

    # ── Persistent knowledge base ────────────────────────────────────────────
    knowledge_base: dict[str, Any]   # semantic notes from past iterations

    # ── Misc ──────────────────────────────────────────────────────────────────
    debug_mode: bool        # mirror of config.DEBUG_MODE for this run
    messages: list[dict]    # LangChain message list (for LLM calls)
    error: str | None       # last unhandled error message
