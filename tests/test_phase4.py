"""Unit tests for Phase 4 nodes — Component 6 (tasks 6.2 + 6.4).

Coverage:

  6.2  route_after_submit
         – END when submission_passed is True
         – END when retry budget exhausted (retry > SUBMISSION_RETRY_MAX)
         – phase2_ablation when failed and retries remain
         – edge: retry exactly at SUBMISSION_RETRY_MAX (still retries)
         – edge: retry at SUBMISSION_RETRY_MAX + 1 (budget exhausted)

  6.4a phase4_submit — acceptance logic
         – relaxed §9.1 acceptance: all thresholds met
         – relaxed §9.1 acceptance: fails on miss_rate
         – relaxed §9.1 acceptance: fails on ng_recall
         – relaxed §9.1 acceptance: fails on overkill_rate
         – relaxed §9.1 acceptance: fails on accuracy
         – final §9.2 acceptance: stricter targets
         – final §9.2 fails while relaxed passes
         – metrics_missing → both tiers fail
         – lineage cache hit: skips re-run
         – lineage cache stale: re-runs script

  6.4b phase4_submit — retry reset
         – best_pipeline_script preserved across reset
         – best_* snapshot fields preserved across reset
         – tried_approaches NOT cleared on reset
         – loop counters reset to zero
         – ensemble state cleared on reset
         – tokens_used reset to zero on reset
         – no retry reset when submission_passed
         – no retry reset when retry budget exhausted
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from langgraph.graph import END


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _state(**kwargs: Any) -> dict:
    """Build a minimal AgentState-compatible dict."""
    defaults: dict = {
        "submission_retry":     0,
        "submission_passed":    False,
        "outer_iteration":      0,
        "inner_iteration":      0,
        "ensemble_iteration":   0,
        "no_improve_count":     0,
        "ensemble_no_improve_count": 0,
        "stop_outer_loop":      False,
        "stop_ensemble_loop":   False,
        "tokens_used":          5_000_000,
        "current_best_score":   0.85,
        "best_miss_rate":       0.15,
        "best_overkill_rate":   0.10,
        "best_accuracy":        0.88,
        "best_f1":              0.87,
        "best_candidate_name":  "efficientnet_b2",
        "best_pipeline_script": "print('best pipeline')",
        "ensemble_script":      "",
        "ensemble_strategy":    None,
        "ensemble_models":      [],
        "ensemble_best_score":  0.85,
        "ensemble_best_overkill": 0.10,
        "ensemble_best_accuracy": 0.88,
        "ensemble_best_f1":     0.87,
        "tried_approaches":     [{"plan": "prior_attempt_1"}],
        "debug_mode":           True,
        "latest_metrics":       {},
    }
    defaults.update(kwargs)
    return defaults


_RELAXED_PASS_METRICS = {
    "ng_recall":     0.97,
    "miss_rate":     0.03,
    "overkill_rate": 0.08,
    "accuracy":      0.92,
    "f1":            0.93,
    "threshold":     0.45,
}

_FINAL_PASS_METRICS = {
    "ng_recall":     1.00,
    "miss_rate":     0.00,
    "overkill_rate": 0.05,
    "accuracy":      0.97,
    "f1":            0.97,
    "threshold":     0.50,
}

_FAIL_MISS_RATE_METRICS = {
    "ng_recall":     0.97,
    "miss_rate":     0.10,   # >  MISS_RATE_RELAXED_MAX (0.03)
    "overkill_rate": 0.04,
    "accuracy":      0.93,
    "f1":            0.91,
    "threshold":     0.40,
}

_FAIL_NG_RECALL_METRICS = {
    "ng_recall":     0.90,   # < NG_RECALL_RELAXED_MIN (0.97)
    "miss_rate":     0.03,
    "overkill_rate": 0.04,
    "accuracy":      0.93,
    "f1":            0.91,
    "threshold":     0.50,
}

_FAIL_OVERKILL_METRICS = {
    "ng_recall":     0.98,
    "miss_rate":     0.02,
    "overkill_rate": 0.15,   # > OVERKILL_RELAXED_MAX (0.08)
    "accuracy":      0.92,
    "f1":            0.93,
    "threshold":     0.50,
}

_FAIL_ACCURACY_METRICS = {
    "ng_recall":     0.97,
    "miss_rate":     0.03,
    "overkill_rate": 0.06,
    "accuracy":      0.88,   # < ACCURACY_RELAXED_MIN (0.92)
    "f1":            0.91,
    "threshold":     0.45,
}

# Relaxed passes but final fails on ng_recall and miss_rate
_RELAXED_ONLY_METRICS = {
    "ng_recall":     0.98,
    "miss_rate":     0.02,
    "overkill_rate": 0.07,
    "accuracy":      0.93,
    "f1":            0.93,
    "threshold":     0.45,
}


def _make_run_result(stdout: str = "", returncode: int = 0) -> MagicMock:
    rr = MagicMock()
    rr.stdout = stdout
    rr.stderr = ""
    rr.returncode = returncode
    rr.timed_out = False
    rr.duration_ms = 200.0
    return rr


# ===========================================================================
# 6.2 — route_after_submit
# ===========================================================================

class TestRouteAfterSubmit:
    """Tests for route_after_submit (task 6.2)."""

    def test_end_when_passed(self):
        from mle_star_agent.nodes.phase4_routing import route_after_submit

        state = _state(submission_passed=True, submission_retry=0)
        assert route_after_submit(state) == END

    def test_end_when_max_retries_exhausted(self):
        from mle_star_agent.nodes.phase4_routing import route_after_submit
        from mle_star_agent import config

        state = _state(
            submission_passed=False,
            submission_retry=config.SUBMISSION_RETRY_MAX + 1,
        )
        assert route_after_submit(state) == END

    def test_retry_when_failed_and_budget_remains(self):
        from mle_star_agent.nodes.phase4_routing import route_after_submit

        state = _state(submission_passed=False, submission_retry=1)
        assert route_after_submit(state) == "phase2_ablation"

    def test_retry_at_exactly_max(self):
        """submission_retry == SUBMISSION_RETRY_MAX should still retry."""
        from mle_star_agent.nodes.phase4_routing import route_after_submit
        from mle_star_agent import config

        state = _state(
            submission_passed=False,
            submission_retry=config.SUBMISSION_RETRY_MAX,
        )
        assert route_after_submit(state) == "phase2_ablation"

    def test_retry_one_beyond_max_ends(self):
        """submission_retry == SUBMISSION_RETRY_MAX + 1 must end."""
        from mle_star_agent.nodes.phase4_routing import route_after_submit
        from mle_star_agent import config

        state = _state(
            submission_passed=False,
            submission_retry=config.SUBMISSION_RETRY_MAX + 1,
        )
        assert route_after_submit(state) == END

    def test_end_with_zero_retry_count_and_passed(self):
        from mle_star_agent.nodes.phase4_routing import route_after_submit

        state = _state(submission_passed=True, submission_retry=0)
        assert route_after_submit(state) == END


# ===========================================================================
# 6.4a — phase4_submit acceptance logic
# ===========================================================================

class TestPhase4AcceptanceLogic:
    """Tests for the acceptance check logic inside _build_acceptance and the node."""

    def test_relaxed_acceptance_all_pass(self):
        from mle_star_agent.nodes.phase4_submit import _build_acceptance

        result = _build_acceptance(_RELAXED_PASS_METRICS, threshold=0.45)
        assert result["relaxed_minimum_pass"] is True

    def test_relaxed_acceptance_fails_miss_rate(self):
        from mle_star_agent.nodes.phase4_submit import _build_acceptance

        result = _build_acceptance(_FAIL_MISS_RATE_METRICS, threshold=0.40)
        assert result["relaxed_minimum_pass"] is False
        assert "miss_rate" in result["reasons"]

    def test_relaxed_acceptance_fails_ng_recall(self):
        from mle_star_agent.nodes.phase4_submit import _build_acceptance

        result = _build_acceptance(_FAIL_NG_RECALL_METRICS, threshold=0.50)
        assert result["relaxed_minimum_pass"] is False
        assert "ng_recall" in result["reasons"]

    def test_relaxed_acceptance_fails_overkill(self):
        from mle_star_agent.nodes.phase4_submit import _build_acceptance

        result = _build_acceptance(_FAIL_OVERKILL_METRICS, threshold=0.50)
        assert result["relaxed_minimum_pass"] is False
        assert "overkill_rate" in result["reasons"]

    def test_relaxed_acceptance_fails_accuracy(self):
        from mle_star_agent.nodes.phase4_submit import _build_acceptance

        result = _build_acceptance(_FAIL_ACCURACY_METRICS, threshold=0.45)
        assert result["relaxed_minimum_pass"] is False
        assert "accuracy" in result["reasons"]

    def test_final_acceptance_stricter_targets(self):
        from mle_star_agent.nodes.phase4_submit import _build_acceptance

        result = _build_acceptance(_FINAL_PASS_METRICS, threshold=0.50)
        assert result["relaxed_minimum_pass"] is True
        assert result["final_target_pass"] is True

    def test_relaxed_passes_but_final_fails(self):
        from mle_star_agent.nodes.phase4_submit import _build_acceptance

        result = _build_acceptance(_RELAXED_ONLY_METRICS, threshold=0.45)
        assert result["relaxed_minimum_pass"] is True
        assert result["final_target_pass"] is False

    def test_no_metrics_both_tiers_fail(self):
        from mle_star_agent.nodes.phase4_submit import _build_acceptance

        result = _build_acceptance(None, threshold=None)
        assert result["relaxed_minimum_pass"] is False
        assert result["final_target_pass"] is False
        assert "metrics_missing" in result["reasons"]

    def test_threshold_recorded_flag_in_checks(self):
        from mle_star_agent.nodes.phase4_submit import _build_acceptance

        result_with = _build_acceptance(_RELAXED_PASS_METRICS, threshold=0.45)
        assert result_with["checks"]["threshold_recorded"] is True

        result_without = _build_acceptance(_RELAXED_PASS_METRICS, threshold=None)
        assert result_without["checks"]["threshold_recorded"] is False


class TestPhase4SubmitNode:
    """Integration tests for phase4_submit_node (task 6.4a)."""

    def test_submission_passed_when_metrics_meet_relaxed(self, tmp_path, monkeypatch):
        """When the script produces metrics that pass §9.1, submission_passed is True."""
        self._patch_config(monkeypatch, tmp_path)
        monkeypatch.setattr("mle_star_agent.config.DEBUG_MODE", False)

        from mle_star_agent.nodes.phase4_submit import phase4_submit_node

        with (
            patch("mle_star_agent.nodes.phase4_submit.run_script",
                  return_value=_make_run_result()),
            patch("mle_star_agent.nodes.phase4_submit.parse_metrics",
                  return_value=_RELAXED_PASS_METRICS),
            patch("mle_star_agent.nodes.phase4_submit.guard_metrics",
                  return_value=_RELAXED_PASS_METRICS),
            patch("mle_star_agent.nodes.phase4_submit.metrics_to_dict",
                  return_value=_RELAXED_PASS_METRICS),
        ):
            state = _state(
                best_pipeline_script="print('pipeline')",
                submission_retry=0,
            )
            result = phase4_submit_node(state)

        assert result["submission_passed"] is True
        assert (tmp_path / "submission.json").exists()

    def test_submission_failed_when_metrics_below_relaxed(self, tmp_path, monkeypatch):
        """When metrics fail §9.1, submission_passed must be False."""
        self._patch_config(monkeypatch, tmp_path)
        monkeypatch.setattr("mle_star_agent.config.DEBUG_MODE", False)

        from mle_star_agent.nodes.phase4_submit import phase4_submit_node

        with (
            patch("mle_star_agent.nodes.phase4_submit.run_script",
                  return_value=_make_run_result()),
            patch("mle_star_agent.nodes.phase4_submit.parse_metrics",
                  return_value=_FAIL_MISS_RATE_METRICS),
            patch("mle_star_agent.nodes.phase4_submit.guard_metrics",
                  return_value=_FAIL_MISS_RATE_METRICS),
            patch("mle_star_agent.nodes.phase4_submit.metrics_to_dict",
                  return_value=_FAIL_MISS_RATE_METRICS),
        ):
            state = _state(
                best_pipeline_script="print('pipeline')",
                submission_retry=0,
            )
            result = phase4_submit_node(state)

        assert result["submission_passed"] is False

    def test_no_metrics_means_failed(self, tmp_path, monkeypatch):
        """Script crash (no metrics) must set submission_passed=False."""
        self._patch_config(monkeypatch, tmp_path)
        monkeypatch.setattr("mle_star_agent.config.DEBUG_MODE", False)

        from mle_star_agent.nodes.phase4_submit import phase4_submit_node

        with (
            patch("mle_star_agent.nodes.phase4_submit.run_script",
                  return_value=_make_run_result("", returncode=1)),
            patch("mle_star_agent.nodes.phase4_submit.parse_metrics", return_value=None),
            patch("mle_star_agent.nodes.phase4_submit.guard_metrics", return_value=None),
            patch("mle_star_agent.nodes.phase4_submit.metrics_to_dict", return_value=None),
        ):
            state = _state(best_pipeline_script="print('fail')", submission_retry=0)
            result = phase4_submit_node(state)

        assert result["submission_passed"] is False

    def test_lineage_cache_hit_skips_execution(self, tmp_path, monkeypatch):
        """If submission.json lineage matches, the script must not be re-executed."""
        self._patch_config(monkeypatch, tmp_path)
        monkeypatch.setattr("mle_star_agent.config.DEBUG_MODE", False)

        script = "print('pipeline_cached')"
        sha = _sha256(script)
        lineage = {"submission_script_sha256": sha}
        saved = {
            "lineage":   lineage,
            "metrics":   _RELAXED_PASS_METRICS,
            "pass_fail": {
                "relaxed_minimum_pass": True,
                "final_target_pass":    False,
                "reasons":              [],
                "checks":               {},
                "final_checks":         {},
            },
        }
        (tmp_path / "submission.json").write_text(json.dumps(saved), encoding="utf-8")

        from mle_star_agent.nodes.phase4_submit import phase4_submit_node

        with patch("mle_star_agent.nodes.phase4_submit.run_script") as mock_run:
            state = _state(best_pipeline_script=script, submission_retry=0)
            result = phase4_submit_node(state)

        mock_run.assert_not_called()
        assert result["submission_passed"] is True

    def test_lineage_cache_stale_reruns_script(self, tmp_path, monkeypatch):
        """If lineage does NOT match, script must be re-executed."""
        self._patch_config(monkeypatch, tmp_path)
        monkeypatch.setattr("mle_star_agent.config.DEBUG_MODE", False)

        stale_script = "print('old_script')"
        stale_sha    = _sha256(stale_script)
        saved = {
            "lineage":   {"submission_script_sha256": stale_sha},
            "metrics":   _RELAXED_PASS_METRICS,
            "pass_fail": {"relaxed_minimum_pass": True, "final_target_pass": False},
        }
        (tmp_path / "submission.json").write_text(json.dumps(saved), encoding="utf-8")

        new_script = "print('new_and_different_script')"

        from mle_star_agent.nodes.phase4_submit import phase4_submit_node

        with (
            patch("mle_star_agent.nodes.phase4_submit.run_script",
                  return_value=_make_run_result()) as mock_run,
            patch("mle_star_agent.nodes.phase4_submit.parse_metrics",
                  return_value=_RELAXED_PASS_METRICS),
            patch("mle_star_agent.nodes.phase4_submit.guard_metrics",
                  return_value=_RELAXED_PASS_METRICS),
            patch("mle_star_agent.nodes.phase4_submit.metrics_to_dict",
                  return_value=_RELAXED_PASS_METRICS),
        ):
            state = _state(best_pipeline_script=new_script, submission_retry=0)
            phase4_submit_node(state)

        mock_run.assert_called_once()

    def test_submission_json_saved(self, tmp_path, monkeypatch):
        """CKPT_SUBMISSION must be written after a successful run."""
        self._patch_config(monkeypatch, tmp_path)
        monkeypatch.setattr("mle_star_agent.config.DEBUG_MODE", False)

        from mle_star_agent.nodes.phase4_submit import phase4_submit_node

        with (
            patch("mle_star_agent.nodes.phase4_submit.run_script",
                  return_value=_make_run_result()),
            patch("mle_star_agent.nodes.phase4_submit.parse_metrics",
                  return_value=_FINAL_PASS_METRICS),
            patch("mle_star_agent.nodes.phase4_submit.guard_metrics",
                  return_value=_FINAL_PASS_METRICS),
            patch("mle_star_agent.nodes.phase4_submit.metrics_to_dict",
                  return_value=_FINAL_PASS_METRICS),
        ):
            state = _state(best_pipeline_script="print('test')", submission_retry=0)
            phase4_submit_node(state)

        submission_ckpt = tmp_path / "submission.json"
        assert submission_ckpt.exists()
        data = json.loads(submission_ckpt.read_text())
        assert data["pass_fail"]["relaxed_minimum_pass"] is True
        assert data["pass_fail"]["final_target_pass"] is True

    def test_selects_ensemble_script_over_best_pipeline(self, tmp_path, monkeypatch):
        """Ensemble checkpoint must take priority over best_pipeline."""
        self._patch_config(monkeypatch, tmp_path)
        monkeypatch.setattr("mle_star_agent.config.DEBUG_MODE", False)

        ensemble_script = "print('ensemble_script')"
        ensemble_ckpt = {"ensemble_script": ensemble_script}
        (tmp_path / "ensemble.json").write_text(json.dumps(ensemble_ckpt), encoding="utf-8")

        from mle_star_agent.nodes.phase4_submit import phase4_submit_node

        captured_args = {}
        def capture_run(script, **kwargs):
            captured_args["script"] = script
            return _make_run_result()

        with (
            patch("mle_star_agent.nodes.phase4_submit.run_script", side_effect=capture_run),
            patch("mle_star_agent.nodes.phase4_submit.parse_metrics", return_value=_RELAXED_PASS_METRICS),
            patch("mle_star_agent.nodes.phase4_submit.guard_metrics", return_value=_RELAXED_PASS_METRICS),
            patch("mle_star_agent.nodes.phase4_submit.metrics_to_dict", return_value=_RELAXED_PASS_METRICS),
        ):
            state = _state(best_pipeline_script="print('should_not_use_this')", submission_retry=0)
            phase4_submit_node(state)

        assert captured_args.get("script") == ensemble_script

    def _patch_config(self, monkeypatch, tmp_path: Path) -> None:
        monkeypatch.setattr("mle_star_agent.config.CHECKPOINT_DIR", tmp_path)
        monkeypatch.setattr("mle_star_agent.config.CKPT_SUBMISSION", tmp_path / "submission.json")
        monkeypatch.setattr("mle_star_agent.config.CKPT_ENSEMBLE", tmp_path / "ensemble.json")
        monkeypatch.setattr("mle_star_agent.config.CKPT_BEST_PIPELINE", tmp_path / "best_pipeline.json")
        monkeypatch.setattr("mle_star_agent.config.CKPT_SUBMISSION_ATTEMPTS", tmp_path / "submission_attempts.json")
        monkeypatch.setattr("mle_star_agent.config.TIMEOUT_SECONDS", 10)


# ===========================================================================
# 6.4b — phase4_submit retry reset
# ===========================================================================

class TestPhase4RetryReset:
    """Tests for the retry reset behaviour inside phase4_submit_node (task 6.4b)."""

    def _patch_config(self, monkeypatch, tmp_path: Path) -> None:
        monkeypatch.setattr("mle_star_agent.config.CHECKPOINT_DIR", tmp_path)
        monkeypatch.setattr("mle_star_agent.config.CKPT_SUBMISSION", tmp_path / "submission.json")
        monkeypatch.setattr("mle_star_agent.config.CKPT_ENSEMBLE", tmp_path / "ensemble.json")
        monkeypatch.setattr("mle_star_agent.config.CKPT_BEST_PIPELINE", tmp_path / "best_pipeline.json")
        monkeypatch.setattr("mle_star_agent.config.CKPT_SUBMISSION_ATTEMPTS", tmp_path / "submission_attempts.json")
        monkeypatch.setattr("mle_star_agent.config.TIMEOUT_SECONDS", 10)
        monkeypatch.setattr("mle_star_agent.config.DEBUG_MODE", False)

    def _run_failed_submission(self, state: dict, tmp_path: Path, monkeypatch):
        """Helper: run a failed submission and return the result dict."""
        from mle_star_agent.nodes.phase4_submit import phase4_submit_node

        with (
            patch("mle_star_agent.nodes.phase4_submit.run_script",
                  return_value=_make_run_result()),
            patch("mle_star_agent.nodes.phase4_submit.parse_metrics",
                  return_value=_FAIL_MISS_RATE_METRICS),
            patch("mle_star_agent.nodes.phase4_submit.guard_metrics",
                  return_value=_FAIL_MISS_RATE_METRICS),
            patch("mle_star_agent.nodes.phase4_submit.metrics_to_dict",
                  return_value=_FAIL_MISS_RATE_METRICS),
        ):
            return phase4_submit_node(state)

    def test_best_pipeline_script_preserved(self, tmp_path, monkeypatch):
        """After reset, best_pipeline_script must still be the best script found."""
        self._patch_config(monkeypatch, tmp_path)
        state = _state(
            best_pipeline_script="print('my_best_script')",
            submission_retry=0,
        )
        result = self._run_failed_submission(state, tmp_path, monkeypatch)
        assert result.get("best_pipeline_script") == "print('my_best_script')"

    def test_best_snapshot_fields_preserved(self, tmp_path, monkeypatch):
        """All best_* metric snapshot fields must survive the retry reset."""
        self._patch_config(monkeypatch, tmp_path)
        state = _state(
            best_pipeline_script="print('x')",
            current_best_score=0.92,
            best_miss_rate=0.08,
            best_overkill_rate=0.06,
            best_accuracy=0.91,
            best_f1=0.90,
            best_candidate_name="vit_small",
            submission_retry=0,
        )
        result = self._run_failed_submission(state, tmp_path, monkeypatch)

        assert result.get("current_best_score") == pytest.approx(0.92)
        assert result.get("best_miss_rate") == pytest.approx(0.08)
        assert result.get("best_overkill_rate") == pytest.approx(0.06)
        assert result.get("best_accuracy") == pytest.approx(0.91)
        assert result.get("best_f1") == pytest.approx(0.90)
        assert result.get("best_candidate_name") == "vit_small"

    def test_tried_approaches_not_reset(self, tmp_path, monkeypatch):
        """tried_approaches must NOT appear in reset delta (planner reads it across retries)."""
        self._patch_config(monkeypatch, tmp_path)

        from mle_star_agent.nodes.phase4_submit import _reset_for_retry

        state = _state(
            tried_approaches=[{"plan": "strategy_A"}, {"plan": "strategy_B"}],
        )
        reset = _reset_for_retry(state, attempt=1)
        # tried_approaches must not be zeroed in the reset delta
        assert "tried_approaches" not in reset

    def test_loop_counters_reset_to_zero(self, tmp_path, monkeypatch):
        """All Phase 2/3 loop counters must be 0 after reset."""
        self._patch_config(monkeypatch, tmp_path)
        state = _state(
            outer_iteration=7,
            inner_iteration=5,
            ensemble_iteration=4,
            no_improve_count=3,
            ensemble_no_improve_count=2,
            submission_retry=0,
            best_pipeline_script="print('x')",
        )
        result = self._run_failed_submission(state, tmp_path, monkeypatch)

        assert result.get("outer_iteration")          == 0
        assert result.get("inner_iteration")          == 0
        assert result.get("ensemble_iteration")       == 0
        assert result.get("no_improve_count")         == 0
        assert result.get("ensemble_no_improve_count") == 0

    def test_stop_flags_reset(self, tmp_path, monkeypatch):
        """stop_outer_loop and stop_ensemble_loop must be False after reset."""
        self._patch_config(monkeypatch, tmp_path)
        state = _state(
            stop_outer_loop=True,
            stop_ensemble_loop=True,
            submission_retry=0,
            best_pipeline_script="print('x')",
        )
        result = self._run_failed_submission(state, tmp_path, monkeypatch)
        assert result.get("stop_outer_loop") is False
        assert result.get("stop_ensemble_loop") is False

    def test_tokens_used_reset_to_zero(self, tmp_path, monkeypatch):
        """tokens_used must be reset to 0 so flash-downgrade doesn't bleed across attempts."""
        self._patch_config(monkeypatch, tmp_path)
        state = _state(
            tokens_used=8_000_000,
            submission_retry=0,
            best_pipeline_script="print('x')",
        )
        result = self._run_failed_submission(state, tmp_path, monkeypatch)
        assert result.get("tokens_used") == 0

    def test_ensemble_state_cleared_on_reset(self, tmp_path, monkeypatch):
        """ensemble_script and ensemble_strategy must be cleared on reset."""
        self._patch_config(monkeypatch, tmp_path)
        state = _state(
            ensemble_script="print('old_ensemble')",
            ensemble_strategy={"strategy_name": "stacked"},
            submission_retry=0,
            best_pipeline_script="print('x')",
        )
        result = self._run_failed_submission(state, tmp_path, monkeypatch)
        assert result.get("ensemble_script") == ""
        assert result.get("ensemble_strategy") is None

    def test_no_reset_when_submission_passed(self, tmp_path, monkeypatch):
        """When submission passes, loop counters must NOT be reset."""
        self._patch_config(monkeypatch, tmp_path)

        from mle_star_agent.nodes.phase4_submit import phase4_submit_node

        with (
            patch("mle_star_agent.nodes.phase4_submit.run_script",
                  return_value=_make_run_result()),
            patch("mle_star_agent.nodes.phase4_submit.parse_metrics",
                  return_value=_RELAXED_PASS_METRICS),
            patch("mle_star_agent.nodes.phase4_submit.guard_metrics",
                  return_value=_RELAXED_PASS_METRICS),
            patch("mle_star_agent.nodes.phase4_submit.metrics_to_dict",
                  return_value=_RELAXED_PASS_METRICS),
        ):
            state = _state(
                outer_iteration=5,
                inner_iteration=3,
                tokens_used=6_000_000,
                submission_retry=0,
                best_pipeline_script="print('best')",
            )
            result = phase4_submit_node(state)

        assert result["submission_passed"] is True
        # Reset fields must NOT appear (or must be untouched)
        assert result.get("outer_iteration", 5) == 5 or "outer_iteration" not in result
        assert result.get("tokens_used", 6_000_000) == 6_000_000 or "tokens_used" not in result

    def test_no_reset_when_budget_exhausted(self, tmp_path, monkeypatch):
        """When retry budget is exhausted, reset must not be applied."""
        self._patch_config(monkeypatch, tmp_path)

        from mle_star_agent import config
        from mle_star_agent.nodes.phase4_submit import phase4_submit_node

        with (
            patch("mle_star_agent.nodes.phase4_submit.run_script",
                  return_value=_make_run_result()),
            patch("mle_star_agent.nodes.phase4_submit.parse_metrics",
                  return_value=_FAIL_MISS_RATE_METRICS),
            patch("mle_star_agent.nodes.phase4_submit.guard_metrics",
                  return_value=_FAIL_MISS_RATE_METRICS),
            patch("mle_star_agent.nodes.phase4_submit.metrics_to_dict",
                  return_value=_FAIL_MISS_RATE_METRICS),
        ):
            # Set retry to SUBMISSION_RETRY_MAX so after +1 it exceeds the budget
            state = _state(
                outer_iteration=7,
                inner_iteration=4,
                tokens_used=9_000_000,
                submission_retry=config.SUBMISSION_RETRY_MAX,
                best_pipeline_script="print('x')",
            )
            result = phase4_submit_node(state)

        assert result["submission_passed"] is False
        # outer_iteration must NOT be reset (budget exhausted, no reset)
        assert result.get("outer_iteration", 7) != 0 or "outer_iteration" not in result

    def test_archive_created_on_retry(self, tmp_path, monkeypatch):
        """Retry archive directory must be created when a reset occurs."""
        self._patch_config(monkeypatch, tmp_path)

        from mle_star_agent.nodes.phase4_submit import _reset_for_retry

        state = _state(best_pipeline_script="print('x')")
        _reset_for_retry(state, attempt=1)

        archive_dir = tmp_path / "retry_archives" / "attempt_1"
        assert archive_dir.exists()

    def test_best_pipeline_json_rewritten_on_reset(self, tmp_path, monkeypatch):
        """CKPT_BEST_PIPELINE must be rewritten with reset counters but preserved script."""
        self._patch_config(monkeypatch, tmp_path)

        from mle_star_agent.nodes.phase4_submit import _reset_for_retry
        import json

        script = "print('preserved_script')"
        state = _state(
            best_pipeline_script=script,
            current_best_score=0.91,
            outer_iteration=8,
        )
        _reset_for_retry(state, attempt=1)

        best_ckpt = tmp_path / "best_pipeline.json"
        assert best_ckpt.exists()
        data = json.loads(best_ckpt.read_text())
        assert data["best_pipeline_script"] == script
        assert data["outer_iteration"] == 0
        assert data["current_best_score"] == pytest.approx(0.91)
