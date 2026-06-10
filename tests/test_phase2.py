"""Unit tests for Phase 2 nodes — Component 4 (tasks 4.10.1 – 4.10.3).

Coverage:
  4.10.1  Ablation variant runner (mock code_runner)
            – debug_mode skips LLM + execution and emits stub results
            – lineage-matched aggregate checkpoint is loaded and returned early
            – all 6 variants produce results when run fresh

  4.10.2  Error analysis gate state machine (all 3/4 branches)
            – iteration 0 → always pass through (inner_iteration = 0)
            – iteration > 0, valid evidence present → allow, clear repair flags
            – iteration > 0, no evidence, first missing → require instrumentation
            – iteration > 0, no evidence, repair already attempted → block

  4.10.3  route_inner_loop + route_outer_loop routing logic
            – route_inner_loop: continue / exit on cap / exit on blocked / exit on stop signal
            – route_outer_loop: continue / exit on stop_outer_loop / exit on cap /
                                exit on token budget / exit on final patience /
                                exit on relaxed patience
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _state(**kwargs: Any) -> dict:
    """Build a minimal AgentState-compatible dict with sensible defaults."""
    defaults: dict = {
        "outer_iteration": 0,
        "inner_iteration": 0,
        "no_improve_count": 0,
        "tokens_used": 0,
        "stop_outer_loop": False,
        "error_analysis_blocked": False,
        "error_analysis_instrumentation_required": False,
        "error_analysis_repair_attempted": False,
        "current_best_score": 0.0,
        "best_miss_rate": 1.0,
        "best_overkill_rate": 1.0,
        "best_accuracy": 0.0,
        "best_f1": 0.0,
        "debug_mode": True,
        "best_pipeline": {"script": "print('stub')", "metrics": {}},
        "candidate_scripts": [],
        "tried_approaches": [],
    }
    defaults.update(kwargs)
    return defaults


_STUB_METRICS = {
    "ng_recall": 0.85,
    "miss_rate": 0.15,
    "overkill_rate": 0.04,
    "accuracy": 0.90,
    "f1": 0.87,
    "threshold": 0.5,
    "avg_latency_ms": 10.0,
    "ng_count": 20,
    "g_count": 80,
    "tp": 17,
    "tn": 77,
    "fp": 3,
    "fn": 3,
    "roc_auc": 0.91,
    "prob_gap": 0.3,
}

# Metrics that pass final acceptance (ng_recall=1.0, miss=0, overkill<=0.05, acc>=0.97)
_FINAL_PASSING_METRICS = {
    "ng_recall": 1.0,
    "miss_rate": 0.0,
    "overkill_rate": 0.03,
    "accuracy": 0.98,
    "f1": 0.99,
    "current_best_score": 1.0,
}

# Metrics that pass relaxed acceptance (ng_recall>=0.97, miss<=0.03, overkill<=0.08, acc>=0.92)
_RELAXED_PASSING_METRICS = {
    "ng_recall": 0.98,
    "miss_rate": 0.02,
    "overkill_rate": 0.06,
    "accuracy": 0.93,
    "f1": 0.95,
    "current_best_score": 0.98,
}


# ===========================================================================
# 4.10.1 — Ablation variant runner
# ===========================================================================

class TestAblationVariantRunner:
    """Tests for phase2_ablation_node variant generation and execution."""

    def test_debug_mode_skips_execution(self, tmp_path, monkeypatch):
        """In debug_mode all variants should be marked as 'skipped' (no LLM call)."""
        monkeypatch.setattr("mle_star_agent.config.CHECKPOINT_DIR", tmp_path)
        # Also patch ckpt helpers that use CHECKPOINT_DIR at module import time
        monkeypatch.setattr(
            "mle_star_agent.config.ckpt_ablation", lambda n: tmp_path / f"ablation_{n}.json"
        )
        monkeypatch.setattr(
            "mle_star_agent.config.ckpt_ablation_variant",
            lambda n, i: tmp_path / f"ablation_variant_{n}_{i}.json",
        )

        from mle_star_agent.nodes.phase2_ablation import NUM_ABLATION_VARIANTS, phase2_ablation_node

        state = _state(debug_mode=True)

        # Should not call code_runner at all
        with patch("mle_star_agent.nodes.phase2_ablation.code_runner") as mock_runner:
            result = phase2_ablation_node(state)

        mock_runner.run_script.assert_not_called()
        assert "ablation_results" in result
        results = result["ablation_results"]
        assert len(results) == NUM_ABLATION_VARIANTS
        for r in results:
            assert r["status"] in ("skipped", "failed"), (
                f"Expected skipped/failed in debug mode, got {r['status']!r} for {r.get('name')}"
            )

    def test_aggregate_checkpoint_loaded_when_lineage_matches(self, tmp_path, monkeypatch):
        """If a complete, lineage-matching checkpoint exists it must be returned early."""
        monkeypatch.setattr("mle_star_agent.config.CHECKPOINT_DIR", tmp_path)
        monkeypatch.setattr(
            "mle_star_agent.config.ckpt_ablation", lambda n: tmp_path / f"ablation_{n}.json"
        )
        monkeypatch.setattr(
            "mle_star_agent.config.ckpt_ablation_variant",
            lambda n, i: tmp_path / f"ablation_variant_{n}_{i}.json",
        )

        from mle_star_agent.nodes.phase2_ablation import (
            ABLATION_VARIANTS,
            NUM_ABLATION_VARIANTS,
            _ablation_lineage,
            phase2_ablation_node,
        )
        from mle_star_agent.shared.checkpoint_io import save_checkpoint

        script = "print('stub')"
        lineage = _ablation_lineage(script)
        # Build a fake complete aggregate checkpoint
        fake_results = [
            {
                "variant_index": i,
                "name": v["name"],
                "status": "skipped",
                "reason": "test",
                "metrics": None,
                "lineage": lineage,
            }
            for i, v in enumerate(ABLATION_VARIANTS)
        ]
        agg_payload = {
            "outer_iteration": 0,
            "lineage": lineage,
            "target_component": "threshold_selection",
            "ablation_results": fake_results,
        }
        save_checkpoint(tmp_path / "ablation_0.json", agg_payload)

        state = _state(debug_mode=False, best_pipeline={"script": script, "metrics": {}})

        with patch("mle_star_agent.nodes.phase2_ablation.call_llm_json") as mock_llm:
            result = phase2_ablation_node(state)

        mock_llm.assert_not_called()
        assert result["ablation_results"] == fake_results
        assert result["target_component"] == "threshold_selection"

    def test_all_variants_represented_in_debug_run(self, tmp_path, monkeypatch):
        """debug_mode run must produce exactly NUM_ABLATION_VARIANTS result entries."""
        monkeypatch.setattr("mle_star_agent.config.CHECKPOINT_DIR", tmp_path)
        monkeypatch.setattr(
            "mle_star_agent.config.ckpt_ablation", lambda n: tmp_path / f"ablation_{n}.json"
        )
        monkeypatch.setattr(
            "mle_star_agent.config.ckpt_ablation_variant",
            lambda n, i: tmp_path / f"ablation_variant_{n}_{i}.json",
        )

        from mle_star_agent.nodes.phase2_ablation import (
            ABLATION_VARIANTS,
            NUM_ABLATION_VARIANTS,
            phase2_ablation_node,
        )

        state = _state(debug_mode=True)
        result = phase2_ablation_node(state)

        variant_names = {v["name"] for v in ABLATION_VARIANTS}
        result_names = {r["name"] for r in result["ablation_results"]}
        assert result_names == variant_names
        assert len(result["ablation_results"]) == NUM_ABLATION_VARIANTS

    def test_stop_outer_loop_set_when_cap_reached(self, tmp_path, monkeypatch):
        """When outer_iteration >= OUTER_LOOP_MAX the node must return stop_outer_loop=True."""
        monkeypatch.setattr("mle_star_agent.config.CHECKPOINT_DIR", tmp_path)
        monkeypatch.setattr(
            "mle_star_agent.config.ckpt_ablation", lambda n: tmp_path / f"ablation_{n}.json"
        )
        monkeypatch.setattr(
            "mle_star_agent.config.ckpt_ablation_variant",
            lambda n, i: tmp_path / f"ablation_variant_{n}_{i}.json",
        )
        from mle_star_agent import config
        from mle_star_agent.nodes.phase2_ablation import phase2_ablation_node

        state = _state(debug_mode=True, outer_iteration=config.OUTER_LOOP_MAX)
        result = phase2_ablation_node(state)
        assert result.get("stop_outer_loop") is True


# ===========================================================================
# 4.10.2 — Error analysis gate state machine
# ===========================================================================

class TestErrorAnalysisGate:
    """Tests for phase2_error_analysis_gate_node — all 4 branches."""

    def test_branch1_iteration_zero_always_passes(self):
        """iteration 0 must always pass through regardless of evidence."""
        from mle_star_agent.nodes.phase2_error_analysis_gate import (
            phase2_error_analysis_gate_node,
        )

        state = _state(inner_iteration=0, error_analysis=None)
        result = phase2_error_analysis_gate_node(state)

        assert result["inner_iteration"] == 0
        assert result["error_analysis_instrumentation_required"] is False
        assert result["error_analysis_blocked"] is False

    def test_branch2_valid_evidence_allows(self):
        """iteration > 0 with valid evidence must clear flags and allow."""
        from mle_star_agent.nodes.phase2_error_analysis_gate import (
            phase2_error_analysis_gate_node,
        )

        # evidence_available when report has fp_count / fn_count
        evidence = {"fp_count": 3, "fn_count": 1, "available": True}
        state = _state(
            inner_iteration=2,
            error_analysis=evidence,
            error_analysis_repair_attempted=False,
        )
        result = phase2_error_analysis_gate_node(state)

        assert result.get("error_analysis_blocked") is False
        assert result.get("error_analysis_instrumentation_required") is False

    def test_branch3_first_missing_evidence_requires_instrumentation(self, tmp_path, monkeypatch):
        """First missing evidence: must set instrumentation_required=True and allow."""
        monkeypatch.setattr("mle_star_agent.config.CHECKPOINT_DIR", tmp_path)
        monkeypatch.setattr(
            "mle_star_agent.config.ckpt_error_analysis",
            lambda n, m: tmp_path / f"ea_{n}_{m}.json",
        )
        from mle_star_agent.nodes.phase2_error_analysis_gate import (
            phase2_error_analysis_gate_node,
        )

        state = _state(
            inner_iteration=1,
            error_analysis=None,  # no evidence in state
            error_analysis_repair_attempted=False,  # repair not yet attempted
        )
        result = phase2_error_analysis_gate_node(state)

        assert result["error_analysis_instrumentation_required"] is True
        assert result["error_analysis_repair_attempted"] is True
        assert result["error_analysis_blocked"] is False

    def test_branch4_second_missing_evidence_blocks(self, tmp_path, monkeypatch):
        """Second missing evidence (repair attempted): must block the inner loop."""
        monkeypatch.setattr("mle_star_agent.config.CHECKPOINT_DIR", tmp_path)
        monkeypatch.setattr(
            "mle_star_agent.config.ckpt_error_analysis",
            lambda n, m: tmp_path / f"ea_{n}_{m}.json",
        )
        from mle_star_agent.nodes.phase2_error_analysis_gate import (
            phase2_error_analysis_gate_node,
        )

        state = _state(
            inner_iteration=2,
            error_analysis=None,  # evidence still missing after repair
            error_analysis_repair_attempted=True,  # repair was already tried
        )
        result = phase2_error_analysis_gate_node(state)

        assert result["error_analysis_blocked"] is True
        assert result.get("error_analysis_instrumentation_required") is False

    def test_branch2_evidence_via_evidence_available_flag(self):
        """evidence_available=True in report dict must count as valid evidence."""
        from mle_star_agent.nodes.phase2_error_analysis_gate import (
            phase2_error_analysis_gate_node,
        )

        evidence = {"evidence_available": True, "fp_samples": [], "fn_samples": []}
        state = _state(inner_iteration=3, error_analysis=evidence)
        result = phase2_error_analysis_gate_node(state)

        assert result.get("error_analysis_blocked") is False

    def test_iteration_zero_clears_repair_state(self):
        """Even with repair flags set, iteration 0 must reset them."""
        from mle_star_agent.nodes.phase2_error_analysis_gate import (
            phase2_error_analysis_gate_node,
        )

        state = _state(
            inner_iteration=0,
            error_analysis_repair_attempted=True,  # leftover from last outer iter
        )
        result = phase2_error_analysis_gate_node(state)

        assert result["inner_iteration"] == 0
        assert result["error_analysis_blocked"] is False
        assert result["error_analysis_instrumentation_required"] is False


# ===========================================================================
# 4.10.3 — route_inner_loop + route_outer_loop
# ===========================================================================

class TestRouteInnerLoop:
    """Tests for route_inner_loop."""

    def test_continue_when_no_exit_conditions(self):
        from mle_star_agent.nodes.phase2_routing import route_inner_loop
        from mle_star_agent import config

        state = _state(
            inner_iteration=config.INNER_LOOP_MAX - 1,  # one below cap
            stop_outer_loop=False,
            error_analysis_blocked=False,
        )
        assert route_inner_loop(state) == "continue"

    def test_exit_when_inner_cap_reached(self):
        from mle_star_agent.nodes.phase2_routing import route_inner_loop
        from mle_star_agent import config

        state = _state(inner_iteration=config.INNER_LOOP_MAX)
        assert route_inner_loop(state) == "exit"

    def test_exit_when_inner_exceeds_cap(self):
        from mle_star_agent.nodes.phase2_routing import route_inner_loop
        from mle_star_agent import config

        state = _state(inner_iteration=config.INNER_LOOP_MAX + 5)
        assert route_inner_loop(state) == "exit"

    def test_exit_when_error_analysis_blocked(self):
        from mle_star_agent.nodes.phase2_routing import route_inner_loop
        from mle_star_agent import config

        state = _state(
            inner_iteration=1,  # well below cap
            error_analysis_blocked=True,
        )
        assert route_inner_loop(state) == "exit"

    def test_exit_when_stop_outer_loop_set(self):
        from mle_star_agent.nodes.phase2_routing import route_inner_loop
        from mle_star_agent import config

        state = _state(
            inner_iteration=1,
            stop_outer_loop=True,
            error_analysis_blocked=False,
        )
        assert route_inner_loop(state) == "exit"

    def test_blocked_takes_precedence_over_cap(self):
        """error_analysis_blocked should cause exit even when inner < cap."""
        from mle_star_agent.nodes.phase2_routing import route_inner_loop
        from mle_star_agent import config

        state = _state(
            inner_iteration=0,
            error_analysis_blocked=True,
            stop_outer_loop=False,
        )
        assert route_inner_loop(state) == "exit"

    def test_continue_at_zero_iterations(self):
        from mle_star_agent.nodes.phase2_routing import route_inner_loop

        state = _state(inner_iteration=0, stop_outer_loop=False, error_analysis_blocked=False)
        assert route_inner_loop(state) == "continue"


class TestRouteOuterLoop:
    """Tests for route_outer_loop."""

    def test_continue_with_no_exit_conditions(self):
        from mle_star_agent.nodes.phase2_routing import route_outer_loop
        from mle_star_agent import config

        state = _state(outer_iteration=1, no_improve_count=0, tokens_used=0)
        assert route_outer_loop(state) == "continue"

    def test_exit_when_stop_outer_loop_flag(self):
        from mle_star_agent.nodes.phase2_routing import route_outer_loop

        state = _state(stop_outer_loop=True, outer_iteration=0)
        assert route_outer_loop(state) == "exit"

    def test_exit_when_outer_cap_reached(self):
        from mle_star_agent.nodes.phase2_routing import route_outer_loop
        from mle_star_agent import config

        state = _state(outer_iteration=config.OUTER_LOOP_MAX, stop_outer_loop=False)
        assert route_outer_loop(state) == "exit"

    def test_exit_when_outer_cap_exceeded(self):
        from mle_star_agent.nodes.phase2_routing import route_outer_loop
        from mle_star_agent import config

        state = _state(outer_iteration=config.OUTER_LOOP_MAX + 3)
        assert route_outer_loop(state) == "exit"

    def test_exit_when_token_budget_exhausted(self):
        from mle_star_agent.nodes.phase2_routing import route_outer_loop
        from mle_star_agent import config

        state = _state(tokens_used=config.TOKEN_BUDGET, outer_iteration=0)
        assert route_outer_loop(state) == "exit"

    def test_exit_when_token_budget_exceeded(self):
        from mle_star_agent.nodes.phase2_routing import route_outer_loop
        from mle_star_agent import config

        state = _state(tokens_used=config.TOKEN_BUDGET + 1, outer_iteration=0)
        assert route_outer_loop(state) == "exit"

    def test_exit_when_final_acceptance_and_patience_exhausted(self):
        from mle_star_agent.nodes.phase2_routing import route_outer_loop
        from mle_star_agent import config

        state = _state(
            outer_iteration=1,
            no_improve_count=config.NO_IMPROVE_MAX,
            # Set best metrics to pass final acceptance
            current_best_score=_FINAL_PASSING_METRICS["ng_recall"],
            best_miss_rate=_FINAL_PASSING_METRICS["miss_rate"],
            best_overkill_rate=_FINAL_PASSING_METRICS["overkill_rate"],
            best_accuracy=_FINAL_PASSING_METRICS["accuracy"],
            best_f1=_FINAL_PASSING_METRICS["f1"],
        )
        assert route_outer_loop(state) == "exit"

    def test_continue_when_final_acceptance_but_patience_not_exhausted(self):
        from mle_star_agent.nodes.phase2_routing import route_outer_loop
        from mle_star_agent import config

        state = _state(
            outer_iteration=1,
            no_improve_count=config.NO_IMPROVE_MAX - 1,
            current_best_score=_FINAL_PASSING_METRICS["ng_recall"],
            best_miss_rate=_FINAL_PASSING_METRICS["miss_rate"],
            best_overkill_rate=_FINAL_PASSING_METRICS["overkill_rate"],
            best_accuracy=_FINAL_PASSING_METRICS["accuracy"],
            best_f1=_FINAL_PASSING_METRICS["f1"],
        )
        assert route_outer_loop(state) == "continue"

    def test_exit_when_relaxed_acceptance_and_constrained_patience_exhausted(self):
        from mle_star_agent.nodes.phase2_routing import route_outer_loop
        from mle_star_agent import config

        state = _state(
            outer_iteration=1,
            no_improve_count=config.NO_IMPROVE_MAX_CONSTRAINED,
            current_best_score=_RELAXED_PASSING_METRICS["ng_recall"],
            best_miss_rate=_RELAXED_PASSING_METRICS["miss_rate"],
            best_overkill_rate=_RELAXED_PASSING_METRICS["overkill_rate"],
            best_accuracy=_RELAXED_PASSING_METRICS["accuracy"],
            best_f1=_RELAXED_PASSING_METRICS["f1"],
        )
        assert route_outer_loop(state) == "exit"

    def test_continue_when_relaxed_acceptance_but_constrained_patience_not_exhausted(self):
        from mle_star_agent.nodes.phase2_routing import route_outer_loop
        from mle_star_agent import config

        state = _state(
            outer_iteration=1,
            no_improve_count=config.NO_IMPROVE_MAX_CONSTRAINED - 1,
            current_best_score=_RELAXED_PASSING_METRICS["ng_recall"],
            best_miss_rate=_RELAXED_PASSING_METRICS["miss_rate"],
            best_overkill_rate=_RELAXED_PASSING_METRICS["overkill_rate"],
            best_accuracy=_RELAXED_PASSING_METRICS["accuracy"],
            best_f1=_RELAXED_PASSING_METRICS["f1"],
        )
        assert route_outer_loop(state) == "continue"

    def test_continue_when_below_relaxed_acceptance_any_patience(self):
        """No acceptance tier met → patience caps should not trigger exit."""
        from mle_star_agent.nodes.phase2_routing import route_outer_loop
        from mle_star_agent import config

        # Very high no_improve but metrics not meeting any acceptance tier
        state = _state(
            outer_iteration=1,
            no_improve_count=100,
            current_best_score=0.5,
            best_miss_rate=0.5,
            best_overkill_rate=0.5,
            best_accuracy=0.5,
            best_f1=0.5,
        )
        # Should continue (no acceptance tier met, cap not hit, no stop flag, no budget)
        assert route_outer_loop(state) == "continue"

    def test_stop_flag_overrides_everything(self):
        """stop_outer_loop=True must exit even if all other counters are zero."""
        from mle_star_agent.nodes.phase2_routing import route_outer_loop

        state = _state(stop_outer_loop=True, outer_iteration=0, no_improve_count=0, tokens_used=0)
        assert route_outer_loop(state) == "exit"


# ===========================================================================
# 4.10  Outer gate node
# ===========================================================================

class TestOuterGateNode:
    """Tests for phase2_outer_gate_node."""

    def test_increments_outer_iteration(self):
        from mle_star_agent.nodes.phase2_routing import phase2_outer_gate_node

        state = _state(outer_iteration=3, inner_iteration=7)
        result = phase2_outer_gate_node(state)

        assert result["outer_iteration"] == 4
        assert result["inner_iteration"] == 0

    def test_resets_inner_iteration_to_zero(self):
        from mle_star_agent.nodes.phase2_routing import phase2_outer_gate_node

        state = _state(outer_iteration=0, inner_iteration=10)
        result = phase2_outer_gate_node(state)

        assert result["inner_iteration"] == 0

    def test_returns_only_necessary_fields(self):
        """Gate should be minimal — only outer_iteration and inner_iteration."""
        from mle_star_agent.nodes.phase2_routing import phase2_outer_gate_node

        state = _state(outer_iteration=1, inner_iteration=5, tokens_used=500)
        result = phase2_outer_gate_node(state)

        # Must contain these two; must not contain unrelated state mutations
        assert set(result.keys()) == {"outer_iteration", "inner_iteration"}
