"""guards/code_validator.py — plain callable AOI script validator.

Ported from ADK guards/code_validator_agent.py (which was an LlmAgent).
This version is a deterministic, zero-LLM function that:
  1. Runs all static checks (marker contract, LR schedule, data usage,
     difference features, small-data strategy).
  2. Executes the script in dry-run mode (DRY_RUN=1, 1 epoch, 10 samples).
  3. Maintains a SHA-256 validation cache persisted to CKPT_VALIDATION_CACHE
     so identical scripts are never re-validated.

All static checks are run before execution. If any hard-gate check fails the
script is rejected without running — the caller (coder node) receives the
reasons and must fix them before resubmitting.

Public API
----------
validate_script(script, input_modality="stereo") -> ValidationResult
    Run the full validation pipeline. Returns a dataclass with .valid,
    .rejection_reasons, .static_checks, and .execution_result fields.

check_validation_cache(script) -> str | None
    Return "VALIDATED" or "VALIDATION_FAILED" for a previously seen script,
    or None for a cache miss.

store_validation_cache(script, status)
    Persist a validation outcome for future lookup (FIFO capped at
    config.VALIDATION_CACHE_MAX entries).
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from mle_star_agent import config
from mle_star_agent.shared import code_runner
from mle_star_agent.shared.checkpoint_io import (
    checkpoint_exists,
    load_checkpoint,
    save_checkpoint,
)
from mle_star_agent.shared.data_usage_validator import validate_data_usage_source
from mle_star_agent.shared.difference_feature_validator import validate_difference_feature_source
from mle_star_agent.shared.lr_schedule_validator import validate_lr_schedule_source
from mle_star_agent.shared.metrics_parser import REQUIRED_GENERATED_SCRIPT_MARKERS
from mle_star_agent.shared.small_data_strategy_validator import (
    validate_small_data_strategy_source,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DRY_RUN_ENV = {
    "DRY_RUN": "1",
    "DRY_RUN_EPOCHS": "1",
    "DRY_RUN_SAMPLES": "10",
    "AOI_RANDOM_SEED": "42",
    "PYTHONHASHSEED": "42",
    "SEED": "42",
}

_VALIDATOR_TIMEOUT = 120  # seconds

# Required structural ternary shape consumed by the debug-epoch rewriter.
_EPOCH_TERNARY_CONTRACT_RE = re.compile(
    r"\bepochs\s*=\s*DRY_RUN_EPOCHS\s+if\s+DRY_RUN\s+else\b"
)

# Markers that must appear as identifiers in the script text.
_FEATURE_DIFF_MARKERS = (
    "FEATURE_DIFF_CANDIDATE",
    "feature_diff",
    "feature-level difference",
    "siamese_difference",
    "siamese difference",
)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    valid: bool
    rejection_reasons: list[str] = field(default_factory=list)
    static_checks: dict[str, Any] = field(default_factory=dict)
    execution_result: Optional[dict] = None

    def as_dict(self) -> dict:
        return {
            "valid": self.valid,
            "rejection_reasons": self.rejection_reasons,
            "static_checks": self.static_checks,
            "execution_result": self.execution_result,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _script_hash(script: str) -> str:
    return hashlib.sha256(script.encode()).hexdigest()


def _missing_required_markers(script: str) -> list[str]:
    missing = []
    for marker in REQUIRED_GENERATED_SCRIPT_MARKERS:
        pattern = rf"(?<![A-Za-z0-9_]){re.escape(marker)}"
        if re.search(pattern, script) is None:
            missing.append(marker)
    return missing


def _declares_feature_diff_candidate(script: str) -> bool:
    lowered = script.lower()
    return any(marker.lower() in lowered for marker in _FEATURE_DIFF_MARKERS)


# ---------------------------------------------------------------------------
# Validation cache (§2.1.9)
# ---------------------------------------------------------------------------

def _load_cache() -> dict[str, str]:
    if checkpoint_exists(config.CKPT_VALIDATION_CACHE):
        try:
            return load_checkpoint(config.CKPT_VALIDATION_CACHE)
        except Exception:
            return {}
    return {}


def _save_cache(cache: dict[str, str]) -> None:
    try:
        save_checkpoint(config.CKPT_VALIDATION_CACHE, cache)
    except Exception as exc:
        logger.warning("Could not persist validation cache: %s", exc)


def check_validation_cache(script: str) -> Optional[str]:
    """Return "VALIDATED" | "VALIDATION_FAILED" for a known script, or None."""
    h = _script_hash(script)
    cache = _load_cache()
    return cache.get(h)


def store_validation_cache(script: str, status: str) -> None:
    """Persist a validation outcome. FIFO-evicts when cache exceeds the cap."""
    h = _script_hash(script)
    cache = _load_cache()
    cache.pop(h, None)
    cache[h] = status
    while len(cache) > config.VALIDATION_CACHE_MAX:
        oldest = next(iter(cache))
        cache.pop(oldest, None)
    _save_cache(cache)


# ---------------------------------------------------------------------------
# Static checks
# ---------------------------------------------------------------------------

def _static_marker_check(script: str) -> tuple[bool, list[str], dict]:
    """§2.1.3 + §2.1.8 — epoch ternary contract and required output markers."""
    missing_markers = _missing_required_markers(script)
    reasons: list[str] = [f"missing required marker {m}" for m in missing_markers]

    has_ternary = _EPOCH_TERNARY_CONTRACT_RE.search(script) is not None
    if not has_ternary:
        reasons.append("missing epoch ternary: epochs = DRY_RUN_EPOCHS if DRY_RUN else")

    return (not reasons), reasons, {
        "missing_markers": missing_markers,
        "has_epoch_ternary": has_ternary,
    }


def _static_data_usage_check(script: str, input_modality: str) -> tuple[bool, list[str], dict]:
    """§2.1.4 — both L+R stereo images must be loaded (data_usage_validator)."""
    report = validate_data_usage_source(script, input_modality=input_modality)
    reasons = [f"data_usage: {msg}" for msg in report.get("messages", [])]
    # Inconclusive results are informational only — do not hard-fail.
    if report.get("inconclusive"):
        return True, [], report
    return report["ok"], reasons, report


def _static_lr_schedule_check(script: str) -> tuple[bool, list[str], dict]:
    """§2.1.5 — scheduler.step() must be present (lr_schedule_validator)."""
    report = validate_lr_schedule_source(script)
    reasons = [f"lr_schedule: {msg}" for msg in report.get("messages", [])]
    return report["ok"], reasons, report


def _static_difference_feature_check(script: str) -> tuple[bool, list[str], dict]:
    """§2.1.6 — stereo difference features handled correctly (conditional check).

    Only enforced when the script opts into the feature-level Siamese difference
    family via FEATURE_DIFF_CANDIDATE or equivalent markers.
    """
    report = validate_difference_feature_source(script)
    applies = _declares_feature_diff_candidate(script)
    report["applies"] = applies

    if not applies:
        return True, [], report

    reasons: list[str] = []
    if not report.get("ok"):
        reasons = [f"difference_feature: {r}" for r in report.get("reasons", [])]
    return report.get("ok", False), reasons, report


def _static_small_data_check(script: str, input_modality: str) -> tuple[bool, list[str], dict]:
    """§2.1.7 — no KNOWN_FAILED_STRATEGY_FINGERPRINTS (small_data_strategy_validator)."""
    report = validate_small_data_strategy_source(script, input_modality=input_modality)
    reasons = [f"small_data_strategy: {r}" for r in report.get("reasons", [])]
    return report["ok"], reasons, report


# ---------------------------------------------------------------------------
# Dry-run execution (§2.1.2)
# ---------------------------------------------------------------------------

def _run_dry_run(script: str) -> dict:
    """Execute the script in dry-run mode and return a result dict."""
    result = code_runner.run_script(script, timeout=_VALIDATOR_TIMEOUT, env=_DRY_RUN_ENV)
    return {
        "returncode": result.returncode,
        "stdout": result.stdout[-5000:],
        "stderr": result.stderr[-5000:],
        "timed_out": result.timed_out,
        "duration_ms": round(result.duration_ms, 1),
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def validate_script(
    script: str,
    input_modality: str = "stereo",
) -> ValidationResult:
    """Run the full Code Validator pipeline on ``script``.

    Pipeline (all static checks run before execution):
      1. Check validation cache (§2.1.9) — return cached result immediately.
      2. §2.1.3 / §2.1.8  Epoch ternary + required output markers (hard gate).
      3. §2.1.4            Data usage (L+R stereo images loaded)   (hard gate).
      4. §2.1.5            LR schedule constructor + step()        (hard gate).
      5. §2.1.6            Feature-level Siamese difference         (conditional).
      6. §2.1.7            Small-data strategy fingerprints         (hard gate).
      7. §2.1.2            Dry-run execution (DRY_RUN=1)           (hard gate).

    Returns a ValidationResult. On cache hit the execution_result field is None
    and static_checks is empty — the caller should trust the cached outcome.
    """
    # ------------------------------------------------------------------
    # §2.1.9 — validation cache
    # ------------------------------------------------------------------
    cached = check_validation_cache(script)
    if cached is not None:
        logger.debug("Validation cache hit: %s (hash=%s)", cached, _script_hash(script)[:8])
        return ValidationResult(
            valid=(cached == "VALIDATED"),
            rejection_reasons=[] if cached == "VALIDATED" else ["cached: VALIDATION_FAILED"],
        )

    all_reasons: list[str] = []
    checks: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # §2.1.3 + §2.1.8 — marker + epoch-ternary contract
    # ------------------------------------------------------------------
    ok, reasons, detail = _static_marker_check(script)
    checks["marker_contract"] = detail
    if not ok:
        all_reasons.extend(reasons)

    # ------------------------------------------------------------------
    # §2.1.4 — data usage (L+R stereo)
    # ------------------------------------------------------------------
    ok, reasons, detail = _static_data_usage_check(script, input_modality)
    checks["data_usage"] = detail
    if not ok:
        all_reasons.extend(reasons)

    # ------------------------------------------------------------------
    # §2.1.5 — LR schedule
    # ------------------------------------------------------------------
    ok, reasons, detail = _static_lr_schedule_check(script)
    checks["lr_schedule"] = detail
    if not ok:
        all_reasons.extend(reasons)

    # ------------------------------------------------------------------
    # §2.1.6 — difference feature (conditional)
    # ------------------------------------------------------------------
    ok, reasons, detail = _static_difference_feature_check(script)
    checks["difference_feature"] = detail
    if not ok:
        all_reasons.extend(reasons)

    # ------------------------------------------------------------------
    # §2.1.7 — small-data strategy
    # ------------------------------------------------------------------
    ok, reasons, detail = _static_small_data_check(script, input_modality)
    checks["small_data_strategy"] = detail
    if not ok:
        all_reasons.extend(reasons)

    # Bail out before running if any hard-gate static check failed.
    if all_reasons:
        logger.info("Script rejected by static checks: %s", all_reasons)
        store_validation_cache(script, "VALIDATION_FAILED")
        return ValidationResult(
            valid=False,
            rejection_reasons=all_reasons,
            static_checks=checks,
        )

    # ------------------------------------------------------------------
    # §2.1.2 — dry-run execution
    # ------------------------------------------------------------------
    exec_result = _run_dry_run(script)
    exec_reasons: list[str] = []

    if exec_result["timed_out"]:
        exec_reasons.append(
            f"dry-run timed out after {_VALIDATOR_TIMEOUT}s"
        )
    elif exec_result["returncode"] != 0:
        snippet = (exec_result["stderr"] or exec_result["stdout"])[:500]
        exec_reasons.append(f"dry-run exited with code {exec_result['returncode']}: {snippet}")
    elif "Traceback" in exec_result.get("stderr", ""):
        snippet = exec_result["stderr"][:500]
        exec_reasons.append(f"dry-run raised an exception: {snippet}")
    elif "METRICS:" not in exec_result.get("stdout", ""):
        exec_reasons.append("dry-run produced no METRICS: output")

    all_reasons.extend(exec_reasons)
    valid = not all_reasons

    status = "VALIDATED" if valid else "VALIDATION_FAILED"
    store_validation_cache(script, status)

    if not valid:
        logger.info("Script rejected by dry-run: %s", exec_reasons)
    else:
        logger.debug("Script VALIDATED (hash=%s, %.0fms)", _script_hash(script)[:8], exec_result["duration_ms"])

    return ValidationResult(
        valid=valid,
        rejection_reasons=all_reasons,
        static_checks=checks,
        execution_result=exec_result,
    )
