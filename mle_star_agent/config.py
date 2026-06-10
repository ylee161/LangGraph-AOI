"""
config.py — central configuration for the LangGraph AOI MLE-STAR agent.

Mirrors the ADK reference implementation (AOI agent /mle_star_agent/config.py)
but swaps google.adk.models.lite_llm for raw LiteLLM / langchain-openai.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ─── LLM provider ─────────────────────────────────────────────────────────────
# NOTE: do NOT raise at import time — importing config (e.g. for the graph-compile
# smoke test, or any node that pulls in acceptance_scoring) must not require a key.
# Validate lazily, just before the first LLM call, via require_api_key().
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")


def require_api_key() -> str:
    """Return the DeepSeek API key, raising only when an LLM call actually needs it."""
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("DEEPSEEK_API_KEY is not set. Add it to .env before running.")
    return DEEPSEEK_API_KEY


# Model names (LiteLLM DeepSeek provider) — match ADK config.py exactly
MODEL_FLASH = "deepseek/deepseek-v4-flash"      # fast — evaluators, gates, bookkeeping; thinking disabled
MODEL_PRO   = "deepseek/deepseek-v4-pro"        # strong — coders, diagnosis, selection; extended thinking 16k budget

# ─── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# Dataset: symlinks inside the project point at the real lot folders.
# Override via env var DATASET_GLOB or edit the list below directly.
# NOTE: differs intentionally from the ADK glob ("aoi_agent_dataset/[SUP046]*")
# because this project symlinks the lot folders under different names.
DATASET_GLOB = os.environ.get("DATASET_GLOB", "dataset_SUP046_lot*")

# ─── Board grouping (consumed by shared/data_split.py) ───────────────────────
# data_split.py reads these via getattr() with a built-in fallback; declaring
# them here makes the grouped split deterministic instead of fallback-dependent.
BOARD_CODE_PATTERN            = r"VHB[A-Z0-9]+"
BOARD_CODE_STRIP_SUFFIX_DIGITS = 2

# ─── Label convention ─────────────────────────────────────────────────────────
FAIL_LABELS = {"fail", "ng", "1", "defect", "defective", "true", "positive"}
PASS_LABELS = {"pass", "ok", "g", "good", "0", "false", "negative"}

# ─── Acceptance criteria (MLE-STAR §9) ────────────────────────────────────────
# Relaxed minimums (§9.1) — loop continues until at least these are met
OVERKILL_RELAXED_MAX   = 0.08
NG_RECALL_RELAXED_MIN  = 0.97
MISS_RATE_RELAXED_MAX  = 0.03
ACCURACY_RELAXED_MIN   = 0.92
# Final targets (§9.2)
OVERKILL_FINAL_MAX     = 0.05
NG_RECALL_FINAL_MIN    = 1.00
MISS_RATE_FINAL_MAX    = 0.00
ACCURACY_FINAL_MIN     = 0.97

# ─── Loop caps ────────────────────────────────────────────────────────────────
OUTER_LOOP_MAX     = 10
INNER_LOOP_MAX     = 10
ENSEMBLE_LOOP_MAX  = 10
NO_IMPROVE_MAX             = 2   # patience once FINAL (§9.2) criteria are met
NO_IMPROVE_MAX_CONSTRAINED = 5   # patience once RELAXED (§9.1) but not final criteria are met
SUBMISSION_RETRY_MAX = 2

# ─── Training ─────────────────────────────────────────────────────────────────
MIN_EPOCHS              = 20
EARLY_STOPPING_PATIENCE = 3
TIMEOUT_SECONDS         = 7200   # 2 h per training script

# ─── Debug / dry-run ──────────────────────────────────────────────────────────
# DRY_RUN=1 (env) → debug_mode: generated scripts run with max_epochs=1 on ~5% of
# the data and a short timeout, so the full graph can be smoke-tested end-to-end.
# code_runner and the evaluator nodes must honor DEBUG_MODE.
DEBUG_MODE                  = os.environ.get("DRY_RUN", "0") not in ("0", "", "false", "False")
DEBUG_CHECK_TIMEOUT_SECONDS = 120   # cap for debug_mode smoke runs

# ─── Curve-abort / smoke debug ────────────────────────────────────────────────
CURVE_ABORT_DEBUG_EPOCHS       = 4     # epoch cap for debug micro-run so it emits a short curve
CURVE_ABORT_MIN_EPOCHS         = 3     # need >= this many per-epoch points to attempt a fit
CURVE_ABORT_MARGIN             = 0.05  # projected final must be worse than best by at least this
CURVE_ABORT_MIN_FIT            = 0.70  # min R² required to trust the projection
CURVE_ABORT_OVERKILL_MARGIN    = 0.10  # projected overkill must exceed best by at least this

# Egregious debug-run gates (loose — only abort on extreme failure)
DEBUG_PREDICT_OVERKILL_MAX     = 0.60   # abort only if micro-run already false-rejects most G
DEBUG_PREDICT_NG_RECALL_MIN    = 0.50   # abort only on severe NG-recall collapse

# Phase 1 smoke ranking
PHASE1_SMOKE_TOP_K             = 2      # full-run the top smoke-ranked initial candidates
PHASE1_SMOKE_UNCERTAINTY_BAND  = 0.05   # also full-run candidates within this score gap

ERROR_ANALYSIS_SAMPLE_CAP      = 10     # Cap for FP/FN samples reported


# Separability floor used by diagnosis / failure classifier
PROBE_PROBABILITY_GAP_MIN      = 0.01   # only block truly flat models; borderline cases fall through

# ─── Threshold sweep ──────────────────────────────────────────────────────────
THRESHOLD_MIN  = 0.10
THRESHOLD_MAX  = 0.90
THRESHOLD_STEP = 0.05

# ─── Token budget ────────────────────────────────────────────────────────────
TOKEN_BUDGET          = 10_000_000
TOKEN_LITE_THRESHOLD  =  7_000_000   # switch to flash once above this

# ─── Checkpoint filenames ─────────────────────────────────────────────────────
CKPT_DATA_SPLIT        = CHECKPOINT_DIR / "data_split_grouped.json"
CKPT_L0                = CHECKPOINT_DIR / "L0.json"   # Phase 1 baseline; skip-gate restores best_* from this
CKPT_CANDIDATE_SCRIPTS = CHECKPOINT_DIR / "candidate_scripts.json"
CKPT_CANDIDATE_SCORES  = CHECKPOINT_DIR / "candidate_scores.json"
CKPT_BEST_PIPELINE     = CHECKPOINT_DIR / "best_pipeline.json"
CKPT_ENSEMBLE          = CHECKPOINT_DIR / "ensemble.json"
CKPT_SUBMISSION        = CHECKPOINT_DIR / "submission.json"
CKPT_TRIED_APPROACHES        = CHECKPOINT_DIR / "tried_approaches.json"
CKPT_VALIDATION_CACHE        = CHECKPOINT_DIR / "validation_cache.json"
CKPT_FAILED_ARCHITECTURES    = CHECKPOINT_DIR / "failed_architectures.json"
CKPT_TRIED_ENSEMBLE_APPROACHES = CHECKPOINT_DIR / "tried_ensemble_approaches.json"
CKPT_SUBMISSION_ATTEMPTS      = CHECKPOINT_DIR / "submission_attempts.json"

# ─── Validation cache ─────────────────────────────────────────────────────────
# Maximum number of script-hash entries kept in the persistent validation cache
# (FIFO eviction). Prevents unbounded growth across long loop runs.
VALIDATION_CACHE_MAX = 200

# ─── Architectures permanently banned from candidate selection ────────────────
# Matched case-insensitively against script name AND architecture field.
# Add runtime failures to CKPT_FAILED_ARCHITECTURES (on disk) so they survive restarts.
HARD_EXCLUDED_ARCHITECTURES: list[str] = ["convnext", "deit"]

# ─── Code validator ───────────────────────────────────────────────────────────
DEBUGGER_RETRY_CAP = 3   # max dry-run attempts inside code_validator


def ckpt_ablation(n: int) -> Path:
    return CHECKPOINT_DIR / f"ablation_{n}.json"


def ckpt_ablation_variant(n: int, i: int) -> Path:
    return CHECKPOINT_DIR / f"ablation_variant_{n}_{i}.json"


def ckpt_diagnosis(n: int) -> Path:
    return CHECKPOINT_DIR / f"diagnosis_{n}.json"


def ckpt_refinement(n: int, m: int) -> Path:
    return CHECKPOINT_DIR / f"refinement_{n}_{m}.json"


def ckpt_error_analysis(n: int, m: int) -> Path:
    return CHECKPOINT_DIR / f"error_analysis_{n}_{m}.json"


def ckpt_ensemble_attempt(n: int) -> Path:
    return CHECKPOINT_DIR / f"ensemble_{n}.json"
