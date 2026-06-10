"""Unit tests for Phase 3 nodes — Component 5 (tasks 5.3–5.4).

Coverage:
  5.3  route_ensemble_loop
         – continue when no exit conditions are met
         – exit when stop_ensemble_loop flag is set
         – exit when ensemble_iteration >= ENSEMBLE_LOOP_MAX
         – exit when ensemble_iteration exceeds ENSEMBLE_LOOP_MAX
         – exit when token budget is exhausted
         – stop_ensemble_loop overrides everything else

  5.4  Phase 3 node behaviour (mock script execution)
         – phase3_ensemble_coder: strategy fingerprint deduplication
           (previously-failed fingerprints are passed to LLM and excluded)
         – phase3_ensemble_evaluator: records tried_ensemble_approaches entry on
           success and on failure; stop_ensemble_loop set when cap reached;
           stop_ensemble_loop set when no-improvement patience exhausted;
           best ensemble updated only on genuine improvement;
           degenerate output (all-NG) is rejected even when run succeeds
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _state(**kwargs: Any) -> dict:
    """Build a minimal AgentState-compatible dict with sensible defaults."""
    defaults: dict = {
        "ensemble_iteration":        0,
        "ensemble_no_improve_count": 0,
        "ensemble_best_score":       0.0,
        "ensemble_best_overkill":    1.0,
        "ensemble_best_accuracy":    0.0,
        "ensemble_best_f1":          0.0,
        "stop_ensemble_loop":        False,
        "tokens_used":               0,
        "debug_mode":                True,
        "best_pipeline":             {"script": "print('stub')", "metrics": {}},
        "candidate_scripts":         [],
        "candidate_scores":          [],
        "tried_ensemble_approaches": [],
        "ensemble_script":           "",
        "ensemble_strategy":         None,
        "ablation_results":          [],
        "diagnosis_report":          "",
    }
    defaults.update(kwargs)
    return defaults


_GOOD_METRICS = {
    "ng_recall":     0.90,
    "miss_rate":     0.10,
    "overkill_rate": 0.04,
    "accuracy":      0.92,
    "f1":            0.91,
}

_BETTER_METRICS = {
    "ng_recall":     0.95,
    "miss_rate":     0.05,
    "overkill_rate": 0.03,
    "accuracy":      0.95,
    "f1":            0.94,
}

_DEGENERATE_METRICS = {
    "ng_recall":     1.0,
    "miss_rate":     0.0,
    "overkill_rate": 1.0,   # all-NG: overkill is also 1.0
    "accuracy":      0.5,
    "f1":            0.67,
}


# ===========================================================================
# 5.3 — route_ensemble_loop
# ===========================================================================

class TestRouteEnsembleLoop:
    """Tests for route_ensemble_loop (task 5.3)."""

    def test_continue_when_no_exit_conditions(self):
        from mle_star_agent.nodes.phase3_routing import route_ensemble_loop
        from mle_star_agent import config

        state = _state(
            ensemble_iteration=config.ENSEMBLE_LOOP_MAX - 1,
            stop_ensemble_loop=False,
            tokens_used=0,
        )
        assert route_ensemble_loop(state) == "continue"

    def test_exit_when_stop_ensemble_loop_set(self):
        from mle_star_agent.nodes.phase3_routing import route_ensemble_loop

        state = _state(stop_ensemble_loop=True, ensemble_iteration=0)
        assert route_ensemble_loop(state) == "exit"

    def test_exit_when_iteration_cap_reached(self):
        from mle_star_agent.nodes.phase3_routing import route_ensemble_loop
        from mle_star_agent import config

        state = _state(ensemble_iteration=config.ENSEMBLE_LOOP_MAX, stop_ensemble_loop=False)
        assert route_ensemble_loop(state) == "exit"

    def test_exit_when_iteration_cap_exceeded(self):
        from mle_star_agent.nodes.phase3_routing import route_ensemble_loop
        from mle_star_agent import config

        state = _state(ensemble_iteration=config.ENSEMBLE_LOOP_MAX + 5)
        assert route_ensemble_loop(state) == "exit"

    def test_exit_when_token_budget_exhausted(self):
        from mle_star_agent.nodes.phase3_routing import route_ensemble_loop
        from mle_star_agent import config

        state = _state(tokens_used=config.TOKEN_BUDGET, ensemble_iteration=0)
        assert route_ensemble_loop(state) == "exit"

    def test_exit_when_token_budget_exceeded(self):
        from mle_star_agent.nodes.phase3_routing import route_ensemble_loop
        from mle_star_agent import config

        state = _state(tokens_used=config.TOKEN_BUDGET + 1, ensemble_iteration=0)
        assert route_ensemble_loop(state) == "exit"

    def test_stop_flag_overrides_everything(self):
        """stop_ensemble_loop=True exits even when iteration=0 and budget=0."""
        from mle_star_agent.nodes.phase3_routing import route_ensemble_loop

        state = _state(stop_ensemble_loop=True, ensemble_iteration=0, tokens_used=0)
        assert route_ensemble_loop(state) == "exit"

    def test_continue_at_iteration_zero(self):
        from mle_star_agent.nodes.phase3_routing import route_ensemble_loop

        state = _state(ensemble_iteration=0, stop_ensemble_loop=False, tokens_used=0)
        assert route_ensemble_loop(state) == "continue"

    def test_continue_one_below_cap(self):
        from mle_star_agent.nodes.phase3_routing import route_ensemble_loop
        from mle_star_agent import config

        state = _state(
            ensemble_iteration=max(0, config.ENSEMBLE_LOOP_MAX - 1),
            stop_ensemble_loop=False,
            tokens_used=0,
        )
        assert route_ensemble_loop(state) == "continue"


# ===========================================================================
# 5.4 — Phase 3 node behaviour (mock script execution)
# ===========================================================================

class TestEnsembleCoderFingerprints:
    """Tests for strategy fingerprint deduplication in phase3_ensemble_coder (task 5.4)."""

    def test_failed_fingerprints_excluded_from_prompt(self, tmp_path, monkeypatch):
        """Fingerprints of previously-failed strategies must appear in the LLM prompt
        so the coder knows what to avoid."""
        monkeypatch.setattr("mle_star_agent.config.CHECKPOINT_DIR", tmp_path)
        monkeypatch.setattr(
            "mle_star_agent.config.CKPT_TRIED_ENSEMBLE_APPROACHES",
            tmp_path / "tried_ensemble_approaches.json",
        )

        from mle_star_agent.nodes.phase3_ensemble_coder import (
            _failed_ensemble_fingerprints,
            _strategy_fingerprint,
        )

        # Build a state with one failing and one improving approach
        tried = [
            {
                "strategy_name": "simple_average",
                "combination_method": "weighted_average",
                "strategy_fingerprint": _strategy_fingerprint("simple_average", "weighted_average"),
                "result": {"improved": False},
            },
            {
                "strategy_name": "max_vote",
                "combination_method": "majority_vote",
                "strategy_fingerprint": _strategy_fingerprint("max_vote", "majority_vote"),
                "result": {"improved": True},
            },
        ]
        state = _state(tried_ensemble_approaches=tried)

        failed = _failed_ensemble_fingerprints(state)

        assert _strategy_fingerprint("simple_average", "weighted_average") in failed
        assert _strategy_fingerprint("max_vote", "majority_vote") not in failed

    def test_fingerprint_deduplication_across_state_and_disk(self, tmp_path, monkeypatch):
        """_tried_ensemble_history must deduplicate entries that appear in both
        state and the on-disk checkpoint."""
        monkeypatch.setattr("mle_star_agent.config.CHECKPOINT_DIR", tmp_path)
        ckpt_path = tmp_path / "tried_ensemble_approaches.json"
        monkeypatch.setattr(
            "mle_star_agent.config.CKPT_TRIED_ENSEMBLE_APPROACHES",
            ckpt_path,
        )

        from mle_star_agent.nodes.phase3_ensemble_coder import (
            _strategy_fingerprint,
            _tried_ensemble_history,
        )
        from mle_star_agent.shared.checkpoint_io import save_checkpoint

        fp = _strategy_fingerprint("simple_average", "mean_pool")
        entry = {
            "strategy_name": "simple_average",
            "combination_method": "mean_pool",
            "strategy_fingerprint": fp,
            "result": {"improved": False},
        }

        # Same entry in both state and disk
        save_checkpoint(ckpt_path, {"tried_ensemble_approaches": [entry]})
        state = _state(tried_ensemble_approaches=[entry])

        history = _tried_ensemble_history(state)
        # Must appear exactly once despite being in both sources
        fps = [e["strategy_fingerprint"] for e in history]
        assert fps.count(fp) == 1

    def test_strategy_fingerprint_is_case_insensitive_and_normalised(self):
        """Fingerprints must be normalised so different casing compares equal."""
        from mle_star_agent.nodes.phase3_ensemble_coder import _strategy_fingerprint

        fp1 = _strategy_fingerprint("Simple_Average", "Weighted_Average")
        fp2 = _strategy_fingerprint("simple_average", "weighted_average")
        assert fp1 == fp2


class TestEnsembleEvaluatorBehaviour:
    """Tests for phase3_ensemble_evaluator_node (task 5.4)."""

    def _make_run_result(self, stdout: str = "", returncode: int = 0,
                         timed_out: bool = False) -> MagicMock:
        rr = MagicMock()
        rr.stdout = stdout
        rr.stderr = ""
        rr.returncode = returncode
        rr.timed_out = timed_out
        rr.duration_ms = 100.0
        return rr

    def _good_stdout(self) -> str:
        return (
            "NG_RECALL: 0.90\n"
            "MISS_RATE: 0.10\n"
            "OVERKILL_RATE: 0.04\n"
            "ACCURACY: 0.92\n"
            "F1: 0.91\n"
        )

    def _better_stdout(self) -> str:
        return (
            "NG_RECALL: 0.95\n"
            "MISS_RATE: 0.05\n"
            "OVERKILL_RATE: 0.03\n"
            "ACCURACY: 0.95\n"
            "F1: 0.94\n"
        )

    def test_tried_ensemble_approaches_recorded_on_success(self, tmp_path, monkeypatch):
        """A successful run must append an entry to tried_ensemble_approaches."""
        monkeypatch.setattr("mle_star_agent.config.CHECKPOINT_DIR", tmp_path)
        monkeypatch.setattr("mle_star_agent.config.CKPT_ENSEMBLE", tmp_path / "ensemble.json")
        monkeypatch.setattr(
            "mle_star_agent.config.CKPT_TRIED_ENSEMBLE_APPROACHES",
            tmp_path / "tried_ensemble_approaches.json",
        )
        monkeypatch.setattr(
            "mle_star_agent.config.ckpt_ensemble_attempt", lambda n: tmp_path / f"ea_{n}.json"
        )
        monkeypatch.setattr("mle_star_agent.config.DEBUG_MODE", False)

        from mle_star_agent.nodes.phase3_ensemble_evaluator import phase3_ensemble_evaluator_node

        strategy = {
            "strategy_name": "simple_average",
            "combination_method": "weighted_average",
            "strategy_fingerprint": "simple_average::weighted_average",
        }

        with (
            patch("mle_star_agent.nodes.phase3_ensemble_evaluator.check_validation_cache",
                  return_value=None),
            patch("mle_star_agent.nodes.phase3_ensemble_evaluator.store_validation_cache"),
            patch("mle_star_agent.nodes.phase3_ensemble_evaluator.run_script",
                  return_value=self._make_run_result(self._better_stdout())),
            patch("mle_star_agent.nodes.phase3_ensemble_evaluator.parse_metrics",
                  return_value=_BETTER_METRICS),
            patch("mle_star_agent.nodes.phase3_ensemble_evaluator.guard_metrics",
                  return_value=_BETTER_METRICS),
            patch("mle_star_agent.nodes.phase3_ensemble_evaluator.metrics_to_dict",
                  return_value=_BETTER_METRICS),
        ):
            state = _state(
                ensemble_script="print('ensemble')",
                ensemble_strategy=strategy,
                ensemble_iteration=0,
            )
            result = phase3_ensemble_evaluator_node(state)

        assert "tried_ensemble_approaches" in result
        entries = result["tried_ensemble_approaches"]
        assert len(entries) == 1
        assert entries[0]["strategy_name"] == "simple_average"

    def test_tried_ensemble_approaches_recorded_on_failure(self, tmp_path, monkeypatch):
        """Even a failed execution must record an entry in tried_ensemble_approaches."""
        monkeypatch.setattr("mle_star_agent.config.CHECKPOINT_DIR", tmp_path)
        monkeypatch.setattr("mle_star_agent.config.CKPT_ENSEMBLE", tmp_path / "ensemble.json")
        monkeypatch.setattr(
            "mle_star_agent.config.CKPT_TRIED_ENSEMBLE_APPROACHES",
            tmp_path / "tried_ensemble_approaches.json",
        )
        monkeypatch.setattr(
            "mle_star_agent.config.ckpt_ensemble_attempt", lambda n: tmp_path / f"ea_{n}.json"
        )
        monkeypatch.setattr("mle_star_agent.config.DEBUG_MODE", False)

        from mle_star_agent.nodes.phase3_ensemble_evaluator import phase3_ensemble_evaluator_node

        with (
            patch("mle_star_agent.nodes.phase3_ensemble_evaluator.check_validation_cache",
                  return_value=None),
            patch("mle_star_agent.nodes.phase3_ensemble_evaluator.store_validation_cache"),
            patch("mle_star_agent.nodes.phase3_ensemble_evaluator.run_script",
                  return_value=self._make_run_result("", returncode=1)),
            patch("mle_star_agent.nodes.phase3_ensemble_evaluator.parse_metrics",
                  return_value=None),
            patch("mle_star_agent.nodes.phase3_ensemble_evaluator.guard_metrics",
                  return_value=None),
            patch("mle_star_agent.nodes.phase3_ensemble_evaluator.metrics_to_dict",
                  return_value=None),
        ):
            state = _state(
                ensemble_script="print('fail')",
                ensemble_strategy={"strategy_name": "bad", "combination_method": "x",
                                   "strategy_fingerprint": "bad::x"},
                ensemble_iteration=0,
                ensemble_best_score=0.0,
            )
            result = phase3_ensemble_evaluator_node(state)

        assert "tried_ensemble_approaches" in result
        assert len(result["tried_ensemble_approaches"]) == 1
        entry = result["tried_ensemble_approaches"][0]
        assert entry["failure_reason"] == "execution_failed"

    def test_stop_ensemble_loop_set_when_cap_reached(self, tmp_path, monkeypatch):
        """stop_ensemble_loop must be True when iteration reaches ENSEMBLE_LOOP_MAX."""
        monkeypatch.setattr("mle_star_agent.config.CHECKPOINT_DIR", tmp_path)
        monkeypatch.setattr("mle_star_agent.config.CKPT_ENSEMBLE", tmp_path / "ensemble.json")
        monkeypatch.setattr(
            "mle_star_agent.config.CKPT_TRIED_ENSEMBLE_APPROACHES",
            tmp_path / "tried_ensemble_approaches.json",
        )
        monkeypatch.setattr(
            "mle_star_agent.config.ckpt_ensemble_attempt", lambda n: tmp_path / f"ea_{n}.json"
        )
        monkeypatch.setattr("mle_star_agent.config.DEBUG_MODE", False)

        from mle_star_agent import config
        from mle_star_agent.nodes.phase3_ensemble_evaluator import phase3_ensemble_evaluator_node

        # Start at cap - 1 so one more increment reaches cap
        n_start = config.ENSEMBLE_LOOP_MAX - 1

        with (
            patch("mle_star_agent.nodes.phase3_ensemble_evaluator.check_validation_cache",
                  return_value=None),
            patch("mle_star_agent.nodes.phase3_ensemble_evaluator.store_validation_cache"),
            patch("mle_star_agent.nodes.phase3_ensemble_evaluator.run_script",
                  return_value=self._make_run_result(self._good_stdout())),
            patch("mle_star_agent.nodes.phase3_ensemble_evaluator.parse_metrics",
                  return_value=_GOOD_METRICS),
            patch("mle_star_agent.nodes.phase3_ensemble_evaluator.guard_metrics",
                  return_value=_GOOD_METRICS),
            patch("mle_star_agent.nodes.phase3_ensemble_evaluator.metrics_to_dict",
                  return_value=_GOOD_METRICS),
        ):
            state = _state(
                ensemble_script="print('x')",
                ensemble_strategy={"strategy_name": "a", "combination_method": "b",
                                   "strategy_fingerprint": "a::b"},
                ensemble_iteration=n_start,
            )
            result = phase3_ensemble_evaluator_node(state)

        assert result["ensemble_iteration"] == config.ENSEMBLE_LOOP_MAX
        assert result["stop_ensemble_loop"] is True

    def test_best_ensemble_updated_on_improvement(self, tmp_path, monkeypatch):
        """When new metrics beat the current best, ensemble_best_score must be updated."""
        monkeypatch.setattr("mle_star_agent.config.CHECKPOINT_DIR", tmp_path)
        monkeypatch.setattr("mle_star_agent.config.CKPT_ENSEMBLE", tmp_path / "ensemble.json")
        monkeypatch.setattr(
            "mle_star_agent.config.CKPT_TRIED_ENSEMBLE_APPROACHES",
            tmp_path / "tried_ensemble_approaches.json",
        )
        monkeypatch.setattr(
            "mle_star_agent.config.ckpt_ensemble_attempt", lambda n: tmp_path / f"ea_{n}.json"
        )
        monkeypatch.setattr("mle_star_agent.config.DEBUG_MODE", False)

        from mle_star_agent.nodes.phase3_ensemble_evaluator import phase3_ensemble_evaluator_node

        with (
            patch("mle_star_agent.nodes.phase3_ensemble_evaluator.check_validation_cache",
                  return_value=None),
            patch("mle_star_agent.nodes.phase3_ensemble_evaluator.store_validation_cache"),
            patch("mle_star_agent.nodes.phase3_ensemble_evaluator.run_script",
                  return_value=self._make_run_result(self._better_stdout())),
            patch("mle_star_agent.nodes.phase3_ensemble_evaluator.parse_metrics",
                  return_value=_BETTER_METRICS),
            patch("mle_star_agent.nodes.phase3_ensemble_evaluator.guard_metrics",
                  return_value=_BETTER_METRICS),
            patch("mle_star_agent.nodes.phase3_ensemble_evaluator.metrics_to_dict",
                  return_value=_BETTER_METRICS),
        ):
            state = _state(
                ensemble_script="print('better')",
                ensemble_strategy={"strategy_name": "stacked", "combination_method": "stack",
                                   "strategy_fingerprint": "stacked::stack"},
                ensemble_iteration=0,
                ensemble_best_score=0.80,      # worse than _BETTER_METRICS
                ensemble_best_overkill=0.10,
            )
            result = phase3_ensemble_evaluator_node(state)

        assert "ensemble_best_score" in result
        assert result["ensemble_best_score"] == pytest.approx(0.95, abs=1e-4)

    def test_best_ensemble_not_updated_when_no_improvement(self, tmp_path, monkeypatch):
        """When new metrics do not beat the best, ensemble_best_score must be unchanged."""
        monkeypatch.setattr("mle_star_agent.config.CHECKPOINT_DIR", tmp_path)
        monkeypatch.setattr("mle_star_agent.config.CKPT_ENSEMBLE", tmp_path / "ensemble.json")
        monkeypatch.setattr(
            "mle_star_agent.config.CKPT_TRIED_ENSEMBLE_APPROACHES",
            tmp_path / "tried_ensemble_approaches.json",
        )
        monkeypatch.setattr(
            "mle_star_agent.config.ckpt_ensemble_attempt", lambda n: tmp_path / f"ea_{n}.json"
        )
        monkeypatch.setattr("mle_star_agent.config.DEBUG_MODE", False)

        from mle_star_agent.nodes.phase3_ensemble_evaluator import phase3_ensemble_evaluator_node

        with (
            patch("mle_star_agent.nodes.phase3_ensemble_evaluator.check_validation_cache",
                  return_value=None),
            patch("mle_star_agent.nodes.phase3_ensemble_evaluator.store_validation_cache"),
            patch("mle_star_agent.nodes.phase3_ensemble_evaluator.run_script",
                  return_value=self._make_run_result(self._good_stdout())),
            patch("mle_star_agent.nodes.phase3_ensemble_evaluator.parse_metrics",
                  return_value=_GOOD_METRICS),
            patch("mle_star_agent.nodes.phase3_ensemble_evaluator.guard_metrics",
                  return_value=_GOOD_METRICS),
            patch("mle_star_agent.nodes.phase3_ensemble_evaluator.metrics_to_dict",
                  return_value=_GOOD_METRICS),
        ):
            state = _state(
                ensemble_script="print('worse')",
                ensemble_strategy={"strategy_name": "simple", "combination_method": "avg",
                                   "strategy_fingerprint": "simple::avg"},
                ensemble_iteration=0,
                # Current best is already better than _GOOD_METRICS
                ensemble_best_score=0.98,
                ensemble_best_overkill=0.02,
            )
            result = phase3_ensemble_evaluator_node(state)

        # Best must not have been updated
        assert "ensemble_best_score" not in result or result["ensemble_best_score"] == pytest.approx(0.90, abs=1e-4)

    def test_degenerate_ensemble_rejected(self, tmp_path, monkeypatch):
        """Ensemble with ng_recall=1.0 AND overkill=1.0 (all-NG) must not update best."""
        monkeypatch.setattr("mle_star_agent.config.CHECKPOINT_DIR", tmp_path)
        monkeypatch.setattr("mle_star_agent.config.CKPT_ENSEMBLE", tmp_path / "ensemble.json")
        monkeypatch.setattr(
            "mle_star_agent.config.CKPT_TRIED_ENSEMBLE_APPROACHES",
            tmp_path / "tried_ensemble_approaches.json",
        )
        monkeypatch.setattr(
            "mle_star_agent.config.ckpt_ensemble_attempt", lambda n: tmp_path / f"ea_{n}.json"
        )
        monkeypatch.setattr("mle_star_agent.config.DEBUG_MODE", False)

        from mle_star_agent.nodes.phase3_ensemble_evaluator import phase3_ensemble_evaluator_node

        with (
            patch("mle_star_agent.nodes.phase3_ensemble_evaluator.check_validation_cache",
                  return_value=None),
            patch("mle_star_agent.nodes.phase3_ensemble_evaluator.store_validation_cache"),
            patch("mle_star_agent.nodes.phase3_ensemble_evaluator.run_script",
                  return_value=self._make_run_result("NG_RECALL: 1.0\nOVERKILL_RATE: 1.0\n")),
            patch("mle_star_agent.nodes.phase3_ensemble_evaluator.parse_metrics",
                  return_value=_DEGENERATE_METRICS),
            patch("mle_star_agent.nodes.phase3_ensemble_evaluator.guard_metrics",
                  return_value=_DEGENERATE_METRICS),
            patch("mle_star_agent.nodes.phase3_ensemble_evaluator.metrics_to_dict",
                  return_value=_DEGENERATE_METRICS),
        ):
            state = _state(
                ensemble_script="print('degen')",
                ensemble_strategy={"strategy_name": "degen", "combination_method": "all_ng",
                                   "strategy_fingerprint": "degen::all_ng"},
                ensemble_iteration=0,
                ensemble_best_score=0.0,
            )
            result = phase3_ensemble_evaluator_node(state)

        # ensemble_best_score must not be updated to 1.0 (degenerate)
        assert result.get("ensemble_best_score", 0.0) != pytest.approx(1.0, abs=1e-4)
        entry = result["tried_ensemble_approaches"][0]
        assert entry["result"]["improved"] is False

    def test_no_script_stops_loop(self, tmp_path, monkeypatch):
        """Missing ensemble_script must set stop_ensemble_loop=True immediately."""
        monkeypatch.setattr("mle_star_agent.config.CHECKPOINT_DIR", tmp_path)
        monkeypatch.setattr("mle_star_agent.config.CKPT_ENSEMBLE", tmp_path / "ensemble.json")
        monkeypatch.setattr(
            "mle_star_agent.config.CKPT_TRIED_ENSEMBLE_APPROACHES",
            tmp_path / "tried_ensemble_approaches.json",
        )
        monkeypatch.setattr(
            "mle_star_agent.config.ckpt_ensemble_attempt", lambda n: tmp_path / f"ea_{n}.json"
        )
        monkeypatch.setattr("mle_star_agent.config.DEBUG_MODE", False)

        from mle_star_agent.nodes.phase3_ensemble_evaluator import phase3_ensemble_evaluator_node

        state = _state(ensemble_script="", ensemble_iteration=0)
        result = phase3_ensemble_evaluator_node(state)

        assert result["stop_ensemble_loop"] is True

    def test_validation_failed_cache_skips_execution(self, tmp_path, monkeypatch):
        """A script with VALIDATION_FAILED in cache must not be re-executed."""
        monkeypatch.setattr("mle_star_agent.config.CHECKPOINT_DIR", tmp_path)
        monkeypatch.setattr("mle_star_agent.config.CKPT_ENSEMBLE", tmp_path / "ensemble.json")
        monkeypatch.setattr(
            "mle_star_agent.config.CKPT_TRIED_ENSEMBLE_APPROACHES",
            tmp_path / "tried_ensemble_approaches.json",
        )
        monkeypatch.setattr(
            "mle_star_agent.config.ckpt_ensemble_attempt", lambda n: tmp_path / f"ea_{n}.json"
        )
        monkeypatch.setattr("mle_star_agent.config.DEBUG_MODE", False)

        from mle_star_agent.nodes.phase3_ensemble_evaluator import phase3_ensemble_evaluator_node

        with (
            patch("mle_star_agent.nodes.phase3_ensemble_evaluator.check_validation_cache",
                  return_value="VALIDATION_FAILED"),
            patch("mle_star_agent.nodes.phase3_ensemble_evaluator.run_script") as mock_run,
        ):
            state = _state(
                ensemble_script="print('cached_fail')",
                ensemble_strategy={"strategy_name": "x", "combination_method": "y",
                                   "strategy_fingerprint": "x::y"},
                ensemble_iteration=0,
            )
            result = phase3_ensemble_evaluator_node(state)

        mock_run.assert_not_called()
        assert result["ensemble_no_improve_count"] == 1

    def test_no_improvement_patience_sets_stop_flag(self, tmp_path, monkeypatch):
        """After 2 consecutive non-improving runs (with n >= 3) stop flag must be set."""
        monkeypatch.setattr("mle_star_agent.config.CHECKPOINT_DIR", tmp_path)
        monkeypatch.setattr("mle_star_agent.config.CKPT_ENSEMBLE", tmp_path / "ensemble.json")
        monkeypatch.setattr(
            "mle_star_agent.config.CKPT_TRIED_ENSEMBLE_APPROACHES",
            tmp_path / "tried_ensemble_approaches.json",
        )
        monkeypatch.setattr(
            "mle_star_agent.config.ckpt_ensemble_attempt", lambda n: tmp_path / f"ea_{n}.json"
        )
        monkeypatch.setattr("mle_star_agent.config.DEBUG_MODE", False)

        from mle_star_agent.nodes.phase3_ensemble_evaluator import phase3_ensemble_evaluator_node

        with (
            patch("mle_star_agent.nodes.phase3_ensemble_evaluator.check_validation_cache",
                  return_value=None),
            patch("mle_star_agent.nodes.phase3_ensemble_evaluator.store_validation_cache"),
            patch("mle_star_agent.nodes.phase3_ensemble_evaluator.run_script",
                  return_value=self._make_run_result(self._good_stdout())),
            patch("mle_star_agent.nodes.phase3_ensemble_evaluator.parse_metrics",
                  return_value=_GOOD_METRICS),
            patch("mle_star_agent.nodes.phase3_ensemble_evaluator.guard_metrics",
                  return_value=_GOOD_METRICS),
            patch("mle_star_agent.nodes.phase3_ensemble_evaluator.metrics_to_dict",
                  return_value=_GOOD_METRICS),
        ):
            state = _state(
                ensemble_script="print('stagnant')",
                ensemble_strategy={"strategy_name": "s", "combination_method": "c",
                                   "strategy_fingerprint": "s::c"},
                # Simulate: iteration 3, already 1 prior non-improving run (so this makes 2)
                ensemble_iteration=3,
                ensemble_no_improve_count=1,
                # Current best is better → this run will not improve
                ensemble_best_score=0.98,
                ensemble_best_overkill=0.02,
            )
            result = phase3_ensemble_evaluator_node(state)

        assert result["stop_ensemble_loop"] is True
        assert result["ensemble_no_improve_count"] == 2
