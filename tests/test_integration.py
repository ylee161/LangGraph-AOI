"""tests/test_integration.py — Component 8 integration tests.

Coverage:

  8.1  Graph compilation test
         – build_graph() returns a compiled graph without an API key
         – compiled graph has all expected node names
         – phase1_init is reachable from START; phase4_submit has an outgoing edge

  8.2  Dry-run smoke test (DRY_RUN=1, all heavy nodes stubbed)
         – graph.invoke() completes all 4 phases in a single call
         – final state has submission_passed=True
         – Phase 1 best-snapshot fields are present in final state
         – phase2_ablation is called at least once (outer loop entered)
         – phase3_ensemble_coder is called at least once (ensemble phase entered)

  8.3  Checkpoint resume test (interrupt_after)
         – checkpoint contains best_pipeline_script after Phase 1 interrupt
         – current_best_score is preserved in checkpoint
         – resuming from Phase 1 interrupt completes the graph successfully
         – outer_iteration is 1 (incremented by outer_gate) after Phase 2 interrupt
         – inner_iteration is 0 (reset by outer_gate) after Phase 2 interrupt
         – best_pipeline_script is non-None in checkpoint after Phase 2

  8.4  Full run on SUP046 dataset (skipped unless DATASET_ROOT + DEEPSEEK_API_KEY set)
         – relaxed §9.1 acceptance met (ng_recall, miss_rate, overkill_rate, accuracy)
"""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import patch

import pytest

from langgraph.checkpoint.memory import MemorySaver


# ===========================================================================
# Stub node functions
#
# Each stub returns the minimal state delta needed to drive routing correctly
# through all 4 phases in a single graph.invoke() call.
#
# Routing exit signals used:
#   – phase2_ablation sets stop_outer_loop=True  → outer loop exits after 1 pass
#   – phase2_evaluator sets inner_iteration=INNER_LOOP_MAX → inner loop exits
#   – phase3_ensemble_evaluator sets stop_ensemble_loop=True → ensemble exits
#   – phase4_submit sets submission_passed=True → END via route_after_submit
# ===========================================================================

_STUB_METRICS: dict[str, Any] = {
    "ng_recall":     0.98,
    "miss_rate":     0.02,
    "overkill_rate": 0.05,
    "accuracy":      0.95,
    "f1":            0.96,
    "threshold":     0.45,
}


def _stub_phase1_init(state: dict) -> dict:
    return {
        "data_split":         {"train": [], "val": [], "test": []},
        "best_pipeline":      {
            "script":    "print('L0 baseline')",
            "metrics":   _STUB_METRICS,
            "threshold": 0.45,
        },
        "current_best_score": _STUB_METRICS["ng_recall"],
        "best_miss_rate":     _STUB_METRICS["miss_rate"],
        "best_overkill_rate": _STUB_METRICS["overkill_rate"],
        "best_accuracy":      _STUB_METRICS["accuracy"],
        "best_f1":            _STUB_METRICS["f1"],
        "best_candidate_name": "efficientnet_b2",
        "candidate_scripts":   ["print('stub_candidate')"],
        "candidate_scores":    [_STUB_METRICS],
        "latest_metrics":      _STUB_METRICS,
    }


def _stub_phase2_ablation(state: dict) -> dict:
    return {
        "ablation_results": {"no_stereo_fusion": {"delta_ng_recall": -0.05}},
        "target_component": "stereo_fusion",
        "stop_outer_loop":  True,   # → route_outer_loop exits after 1 pass
    }


def _stub_phase2_diagnosis(state: dict) -> dict:
    return {
        "target_block_code": "# stub target block",
        "refinement_plan":   "improve stereo fusion",
        "diagnosis":         "stereo fusion is the bottleneck",
    }


def _stub_phase2_error_analysis_gate(state: dict) -> dict:
    return {}


def _stub_phase2_planner(state: dict) -> dict:
    return {
        "refinement_plan":   "add channel attention to stereo fusion",
        "tried_approaches":  [{"plan": "attention_fusion", "score": 0.0}],
    }


def _stub_phase2_strategy_gate(state: dict) -> dict:
    return {}


def _stub_phase2_coder(state: dict) -> dict:
    return {"candidate_scripts": ["print('refined_script')"]}


def _stub_phase2_evaluator(state: dict) -> dict:
    from mle_star_agent import config
    return {
        "latest_metrics":   _STUB_METRICS,
        "inner_iteration":  config.INNER_LOOP_MAX,   # → route_inner_loop exits
        "no_improve_count": 0,
        "best_pipeline":    {
            "script":    "print('refined best pipeline')",
            "metrics":   _STUB_METRICS,
            "threshold": 0.45,
        },
        "current_best_score": _STUB_METRICS["ng_recall"],
        "best_miss_rate":     _STUB_METRICS["miss_rate"],
        "best_overkill_rate": _STUB_METRICS["overkill_rate"],
        "best_accuracy":      _STUB_METRICS["accuracy"],
        "best_f1":            _STUB_METRICS["f1"],
        "candidate_scores":   [_STUB_METRICS],
    }


def _stub_phase2_error_analysis(state: dict) -> dict:
    return {"error_analysis_report": {}, "latest_error_analysis": {}}


def _stub_phase3_ensemble_coder(state: dict) -> dict:
    return {"ensemble_script": "print('ensemble')"}


def _stub_phase3_ensemble_evaluator(state: dict) -> dict:
    return {
        "ensemble_iteration":        1,
        "stop_ensemble_loop":        True,   # → route_ensemble_loop exits
        "tried_ensemble_approaches": [
            {"strategy_name": "averaging", "strategy_fingerprint": "avg_v1"}
        ],
    }


def _stub_phase4_submit(state: dict) -> dict:
    return {
        "submission_passed": True,
        "submission_report": "Stub: all acceptance criteria met.",
    }


# Mapping of patch targets (names as imported into mle_star_agent.graph)
_STUB_NODES: dict[str, Any] = {
    "mle_star_agent.graph.phase1_init_node":                _stub_phase1_init,
    "mle_star_agent.graph.phase2_ablation_node":            _stub_phase2_ablation,
    "mle_star_agent.graph.phase2_diagnosis_node":           _stub_phase2_diagnosis,
    "mle_star_agent.graph.phase2_error_analysis_gate_node": _stub_phase2_error_analysis_gate,
    "mle_star_agent.graph.phase2_planner_node":             _stub_phase2_planner,
    "mle_star_agent.graph.phase2_strategy_gate_node":       _stub_phase2_strategy_gate,
    "mle_star_agent.graph.phase2_coder_node":               _stub_phase2_coder,
    "mle_star_agent.graph.phase2_evaluator_node":           _stub_phase2_evaluator,
    "mle_star_agent.graph.phase2_error_analysis":           _stub_phase2_error_analysis,
    "mle_star_agent.graph.phase3_ensemble_coder_node":      _stub_phase3_ensemble_coder,
    "mle_star_agent.graph.phase3_ensemble_evaluator_node":  _stub_phase3_ensemble_evaluator,
    "mle_star_agent.graph.phase4_submit_node":              _stub_phase4_submit,
}


_INITIAL_STATE: dict[str, Any] = {
    "dataset_path":     "/fake/dataset",
    "goal":             "NG recall >= 1.00",
    "outer_iteration":  0,
    "inner_iteration":  0,
    "ensemble_iteration":            0,
    "submission_retry":              0,
    "no_improve_count":              0,
    "ensemble_no_improve_count":     0,
    "tokens_used":                   0,
    "stop_outer_loop":               False,
    "stop_ensemble_loop":            False,
    "error_analysis_blocked":        False,
    "error_analysis_instrumentation_required": False,
    "error_analysis_repair_attempted":         False,
    "submission_passed":             False,
    "debug_mode":                    True,
    "messages":                      [],
    "tried_approaches":              [],
    "tried_ensemble_approaches":     [],
    "candidate_scripts":             [],
    "candidate_scores":              [],
}


def _start_patches(overrides: dict | None = None) -> list:
    """Start all node patches, optionally overriding specific stubs."""
    stubs = dict(_STUB_NODES)
    if overrides:
        stubs.update(overrides)
    patches = [patch(target, stub) for target, stub in stubs.items()]
    for p in patches:
        p.start()
    return patches


def _stop_patches(patches: list) -> None:
    for p in patches:
        p.stop()


# ===========================================================================
# 8.1 — Graph compilation
# ===========================================================================

class TestGraphCompilation:
    """8.1 — build_graph() compiles without errors and without an API key."""

    def test_compiles_without_api_key(self, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        from mle_star_agent.graph import build_graph

        graph = build_graph(checkpointer=MemorySaver())
        assert hasattr(graph, "invoke"), "Compiled graph must expose .invoke()"
        assert hasattr(graph, "stream"), "Compiled graph must expose .stream()"

    def test_graph_has_expected_nodes(self):
        from mle_star_agent.graph import build_graph

        graph = build_graph(checkpointer=MemorySaver())
        node_keys = set(graph.get_graph().nodes.keys())
        expected = {
            "phase1_init",
            "phase2_ablation",
            "phase2_diagnosis",
            "phase2_error_analysis_gate",
            "phase2_planner",
            "phase2_strategy_gate",
            "phase2_coder",
            "phase2_evaluator",
            "phase2_error_analysis",
            "phase2_outer_gate",
            "phase3_ensemble_coder",
            "phase3_ensemble_evaluator",
            "phase4_submit",
        }
        missing = expected - node_keys
        assert not missing, f"Nodes missing from compiled graph: {missing}"

    def test_phase1_is_entry_point(self):
        from mle_star_agent.graph import build_graph

        graph = build_graph(checkpointer=MemorySaver())
        edges = [(e.source, e.target) for e in graph.get_graph().edges]
        assert any(t == "phase1_init" for _, t in edges), \
            "phase1_init must be reachable from __start__"

    def test_phase4_has_outgoing_edge(self):
        from mle_star_agent.graph import build_graph

        graph = build_graph(checkpointer=MemorySaver())
        edges = [(e.source, e.target) for e in graph.get_graph().edges]
        assert any(s == "phase4_submit" for s, _ in edges), \
            "phase4_submit must have at least one outgoing edge"


# ===========================================================================
# 8.2 — Dry-run smoke test
# ===========================================================================

class TestDryRunSmoke:
    """8.2 — With all heavy nodes stubbed, a single graph.invoke() completes
    all 4 phases and terminates with submission_passed=True."""

    def test_graph_completes_with_submission_passed(self, monkeypatch):
        monkeypatch.setenv("DRY_RUN", "1")

        patches = _start_patches()
        try:
            from mle_star_agent.graph import build_graph

            graph = build_graph(checkpointer=MemorySaver())
            cfg = {"configurable": {"thread_id": "smoke_01"}}
            final = graph.invoke(_INITIAL_STATE.copy(), config=cfg)
        finally:
            _stop_patches(patches)

        assert final.get("submission_passed") is True, \
            f"Expected submission_passed=True, got {final.get('submission_passed')!r}"

    def test_phase1_best_snapshot_in_final_state(self, monkeypatch):
        monkeypatch.setenv("DRY_RUN", "1")

        patches = _start_patches()
        try:
            from mle_star_agent.graph import build_graph

            graph = build_graph(checkpointer=MemorySaver())
            cfg = {"configurable": {"thread_id": "smoke_02"}}
            final = graph.invoke(_INITIAL_STATE.copy(), config=cfg)
        finally:
            _stop_patches(patches)

        # best_pipeline_script is not a declared AgentState channel; nodes store
        # the script inside the best_pipeline dict instead.
        for field in ["best_pipeline", "current_best_score",
                      "best_miss_rate", "best_overkill_rate"]:
            assert field in final, f"Field '{field}' missing from final state"
        assert final["best_pipeline"].get("script"), \
            "best_pipeline dict must contain a non-empty 'script' key"

    def test_outer_loop_entered_at_least_once(self, monkeypatch):
        monkeypatch.setenv("DRY_RUN", "1")

        call_counts: dict[str, int] = {"ablation": 0}

        def counting_ablation(state):
            call_counts["ablation"] += 1
            return _stub_phase2_ablation(state)

        patches = _start_patches(
            {"mle_star_agent.graph.phase2_ablation_node": counting_ablation}
        )
        try:
            from mle_star_agent.graph import build_graph

            graph = build_graph(checkpointer=MemorySaver())
            cfg = {"configurable": {"thread_id": "smoke_03"}}
            graph.invoke(_INITIAL_STATE.copy(), config=cfg)
        finally:
            _stop_patches(patches)

        assert call_counts["ablation"] >= 1, \
            "phase2_ablation must be called at least once"

    def test_ensemble_phase_entered_at_least_once(self, monkeypatch):
        monkeypatch.setenv("DRY_RUN", "1")

        call_counts: dict[str, int] = {"ensemble_coder": 0}

        def counting_ensemble(state):
            call_counts["ensemble_coder"] += 1
            return _stub_phase3_ensemble_coder(state)

        patches = _start_patches(
            {"mle_star_agent.graph.phase3_ensemble_coder_node": counting_ensemble}
        )
        try:
            from mle_star_agent.graph import build_graph

            graph = build_graph(checkpointer=MemorySaver())
            cfg = {"configurable": {"thread_id": "smoke_04"}}
            graph.invoke(_INITIAL_STATE.copy(), config=cfg)
        finally:
            _stop_patches(patches)

        assert call_counts["ensemble_coder"] >= 1, \
            "phase3_ensemble_coder must be called at least once"

    def test_inner_loop_entered_at_least_once(self, monkeypatch):
        monkeypatch.setenv("DRY_RUN", "1")

        call_counts: dict[str, int] = {"evaluator": 0}

        def counting_evaluator(state):
            call_counts["evaluator"] += 1
            return _stub_phase2_evaluator(state)

        patches = _start_patches(
            {"mle_star_agent.graph.phase2_evaluator_node": counting_evaluator}
        )
        try:
            from mle_star_agent.graph import build_graph

            graph = build_graph(checkpointer=MemorySaver())
            cfg = {"configurable": {"thread_id": "smoke_05"}}
            graph.invoke(_INITIAL_STATE.copy(), config=cfg)
        finally:
            _stop_patches(patches)

        assert call_counts["evaluator"] >= 1, \
            "phase2_evaluator must be called at least once (inner loop entered)"


# ===========================================================================
# 8.3 — Checkpoint resume test
# ===========================================================================

def _build_interruptable_graph(checkpointer, interrupt_after=None):
    """Assemble the full graph topology with stub nodes and compile it with
    an optional interrupt_after list.  Routing functions are real code;
    all heavy node implementations are replaced with stubs.
    """
    from langgraph.graph import StateGraph, END

    from mle_star_agent.state import AgentState
    from mle_star_agent.nodes.phase2_routing import (
        route_inner_loop,
        route_outer_loop,
        phase2_outer_gate_node,
    )
    from mle_star_agent.nodes.phase3_routing import route_ensemble_loop
    from mle_star_agent.nodes.phase4_routing import route_after_submit

    g = StateGraph(AgentState)

    # Stub nodes
    g.add_node("phase1_init",                  _stub_phase1_init)
    g.add_node("phase2_ablation",              _stub_phase2_ablation)
    g.add_node("phase2_diagnosis",             _stub_phase2_diagnosis)
    g.add_node("phase2_error_analysis_gate",   _stub_phase2_error_analysis_gate)
    g.add_node("phase2_planner",               _stub_phase2_planner)
    g.add_node("phase2_strategy_gate",         _stub_phase2_strategy_gate)
    g.add_node("phase2_coder",                 _stub_phase2_coder)
    g.add_node("phase2_evaluator",             _stub_phase2_evaluator)
    g.add_node("phase2_error_analysis",        _stub_phase2_error_analysis)
    g.add_node("phase2_outer_gate",            phase2_outer_gate_node)   # real — counter math
    g.add_node("phase3_ensemble_coder",        _stub_phase3_ensemble_coder)
    g.add_node("phase3_ensemble_evaluator",    _stub_phase3_ensemble_evaluator)
    g.add_node("phase4_submit",                _stub_phase4_submit)

    # Topology (mirrors graph.py exactly)
    g.set_entry_point("phase1_init")
    g.add_edge("phase1_init",              "phase2_ablation")
    g.add_edge("phase2_ablation",          "phase2_diagnosis")
    g.add_edge("phase2_diagnosis",         "phase2_error_analysis_gate")
    g.add_edge("phase2_error_analysis_gate", "phase2_planner")
    g.add_edge("phase2_planner",           "phase2_strategy_gate")
    g.add_edge("phase2_strategy_gate",     "phase2_coder")
    g.add_edge("phase2_coder",             "phase2_evaluator")
    g.add_edge("phase2_evaluator",         "phase2_error_analysis")
    g.add_conditional_edges(
        "phase2_error_analysis",
        route_inner_loop,
        {"continue": "phase2_error_analysis_gate", "exit": "phase2_outer_gate"},
    )
    g.add_conditional_edges(
        "phase2_outer_gate",
        route_outer_loop,
        {"continue": "phase2_ablation", "exit": "phase3_ensemble_coder"},
    )
    g.add_edge("phase3_ensemble_coder", "phase3_ensemble_evaluator")
    g.add_conditional_edges(
        "phase3_ensemble_evaluator",
        route_ensemble_loop,
        {"continue": "phase3_ensemble_coder", "exit": "phase4_submit"},
    )
    g.add_conditional_edges(
        "phase4_submit",
        route_after_submit,
        {END: END, "phase2_ablation": "phase2_ablation"},
    )

    compile_kwargs: dict[str, Any] = {}
    if interrupt_after:
        compile_kwargs["interrupt_after"] = interrupt_after

    return g.compile(checkpointer=checkpointer, **compile_kwargs)


class TestCheckpointResume:
    """8.3 — Interrupt the graph at a node boundary; verify that state is
    persisted to the MemorySaver and correctly restored on resume.
    """

    def test_phase1_fields_in_checkpoint(self):
        """After a Phase 1 interrupt, the best_pipeline dict must be in the
        saved checkpoint with a non-empty script."""
        checkpointer = MemorySaver()
        graph = _build_interruptable_graph(
            checkpointer, interrupt_after=["phase1_init"]
        )
        cfg = {"configurable": {"thread_id": "resume_01"}}
        # First invoke: runs Phase 1, saves checkpoint, pauses
        graph.invoke(_INITIAL_STATE.copy(), config=cfg)

        ckpt = checkpointer.get(cfg)
        assert ckpt is not None, "No checkpoint found after Phase 1 interrupt"
        channel_values = ckpt["channel_values"]
        best_pipeline = channel_values.get("best_pipeline", {})
        assert best_pipeline.get("script") == "print('L0 baseline')", \
            f"best_pipeline.script not in checkpoint after Phase 1: {best_pipeline!r}"

    def test_current_best_score_in_checkpoint(self):
        """After a Phase 1 interrupt, current_best_score must be preserved."""
        checkpointer = MemorySaver()
        graph = _build_interruptable_graph(
            checkpointer, interrupt_after=["phase1_init"]
        )
        cfg = {"configurable": {"thread_id": "resume_02"}}
        graph.invoke(_INITIAL_STATE.copy(), config=cfg)

        ckpt = checkpointer.get(cfg)
        assert ckpt is not None
        saved_score = ckpt["channel_values"].get("current_best_score")
        assert saved_score == pytest.approx(0.98), \
            f"current_best_score not preserved: got {saved_score}"

    def test_graph_completes_after_resume_from_phase1(self):
        """Resuming from a Phase 1 interrupt (None input, same thread_id) must
        complete the full graph with submission_passed=True."""
        checkpointer = MemorySaver()
        graph = _build_interruptable_graph(
            checkpointer, interrupt_after=["phase1_init"]
        )
        cfg = {"configurable": {"thread_id": "resume_03"}}
        # Step 1: Phase 1 runs, pauses
        graph.invoke(_INITIAL_STATE.copy(), config=cfg)
        # Step 2: resume (None → continue from checkpoint)
        final = graph.invoke(None, config=cfg)

        assert final.get("submission_passed") is True, \
            "Graph must reach submission_passed=True after resuming from Phase 1"
        assert "best_pipeline" in final, \
            "best_pipeline must survive the Phase 1 → resume transition"

    def test_outer_iteration_incremented_after_outer_gate(self):
        """After Phase 2 outer gate runs, outer_iteration must be 1 in the
        checkpoint (incremented from 0 by phase2_outer_gate_node)."""
        checkpointer = MemorySaver()
        graph = _build_interruptable_graph(
            checkpointer, interrupt_after=["phase2_outer_gate"]
        )
        cfg = {"configurable": {"thread_id": "resume_04"}}
        # Runs Phase 1 + full Phase 2 inner loop + outer gate, then pauses
        graph.invoke(_INITIAL_STATE.copy(), config=cfg)

        ckpt = checkpointer.get(cfg)
        assert ckpt is not None, "No checkpoint found after phase2_outer_gate"
        outer = ckpt["channel_values"].get("outer_iteration", -1)
        assert outer == 1, \
            f"Expected outer_iteration=1 after outer_gate, got {outer}"

    def test_inner_iteration_reset_after_outer_gate(self):
        """phase2_outer_gate_node resets inner_iteration to 0; checkpoint must
        reflect this reset so the next ablation cycle starts fresh."""
        checkpointer = MemorySaver()
        graph = _build_interruptable_graph(
            checkpointer, interrupt_after=["phase2_outer_gate"]
        )
        cfg = {"configurable": {"thread_id": "resume_05"}}
        graph.invoke(_INITIAL_STATE.copy(), config=cfg)

        ckpt = checkpointer.get(cfg)
        assert ckpt is not None
        inner = ckpt["channel_values"].get("inner_iteration", -1)
        assert inner == 0, \
            f"Expected inner_iteration=0 after outer_gate reset, got {inner}"

    def test_best_pipeline_in_checkpoint_after_phase2(self):
        """After Phase 2 outer gate, the best_pipeline dict written by the
        evaluator stub must be in the checkpoint with a non-empty script."""
        checkpointer = MemorySaver()
        graph = _build_interruptable_graph(
            checkpointer, interrupt_after=["phase2_outer_gate"]
        )
        cfg = {"configurable": {"thread_id": "resume_06"}}
        graph.invoke(_INITIAL_STATE.copy(), config=cfg)

        ckpt = checkpointer.get(cfg)
        assert ckpt is not None
        best_pipeline = ckpt["channel_values"].get("best_pipeline", {})
        assert best_pipeline.get("script"), \
            "best_pipeline.script must be non-empty in checkpoint after Phase 2"


# ===========================================================================
# 8.4 — Full run on SUP046 dataset (requires env vars to run)
# ===========================================================================

_DATASET_ROOT = os.environ.get("DATASET_ROOT", "")
_HAS_DATASET  = bool(_DATASET_ROOT) and os.path.isdir(_DATASET_ROOT)
_HAS_API_KEY  = bool(os.environ.get("DEEPSEEK_API_KEY"))


@pytest.mark.slow
@pytest.mark.skipif(
    not _HAS_DATASET,
    reason=(
        "DATASET_ROOT env var not set or path does not exist — "
        "set DATASET_ROOT=/path/to/dataset_SUP046_lot1 to enable"
    ),
)
@pytest.mark.skipif(
    not _HAS_API_KEY,
    reason="DEEPSEEK_API_KEY not set — required for full LLM run",
)
class TestFullRunSup046:
    """8.4 — End-to-end run on the real SUP046 dataset.

    Prerequisites (both required):
        export DATASET_ROOT=/path/to/dataset_SUP046_lot1
        export DEEPSEEK_API_KEY=<your-key>

    Validate that the final submission meets relaxed §9.1 acceptance criteria.
    """

    def test_relaxed_acceptance_met(self):
        os.environ.setdefault("DRY_RUN", "0")

        from langgraph.checkpoint.sqlite import SqliteSaver
        from mle_star_agent.graph import build_graph
        from mle_star_agent import config

        os.makedirs("checkpoints", exist_ok=True)
        checkpointer = SqliteSaver.from_conn_string(
            "checkpoints/test_sup046_langgraph.db"
        )
        graph = build_graph(checkpointer=checkpointer)

        initial_state: dict[str, Any] = {
            "dataset_path":     _DATASET_ROOT,
            "goal":             "NG recall >= 1.00, overkill <= 0.05",
            "outer_iteration":  0,
            "inner_iteration":  0,
            "ensemble_iteration":            0,
            "submission_retry":              0,
            "no_improve_count":              0,
            "ensemble_no_improve_count":     0,
            "tokens_used":                   0,
            "stop_outer_loop":               False,
            "stop_ensemble_loop":            False,
            "error_analysis_blocked":        False,
            "error_analysis_instrumentation_required": False,
            "error_analysis_repair_attempted":         False,
            "submission_passed":             False,
            "debug_mode":                    config.DEBUG_MODE,
            "messages":                      [],
            "tried_approaches":              [],
            "tried_ensemble_approaches":     [],
            "candidate_scripts":             [],
            "candidate_scores":              [],
        }

        cfg = {"configurable": {"thread_id": "test_full_run_sup046"}}
        final = graph.invoke(initial_state, config=cfg)

        metrics = final.get("latest_metrics", {})
        assert metrics.get("ng_recall", 0.0) >= config.NG_RECALL_RELAXED_MIN, (
            f"ng_recall {metrics.get('ng_recall')!r} "
            f"< relaxed minimum {config.NG_RECALL_RELAXED_MIN}"
        )
        assert metrics.get("miss_rate", 1.0) <= config.MISS_RATE_RELAXED_MAX, (
            f"miss_rate {metrics.get('miss_rate')!r} "
            f"> relaxed maximum {config.MISS_RATE_RELAXED_MAX}"
        )
        assert metrics.get("overkill_rate", 1.0) <= config.OVERKILL_RELAXED_MAX, (
            f"overkill_rate {metrics.get('overkill_rate')!r} "
            f"> relaxed maximum {config.OVERKILL_RELAXED_MAX}"
        )
        assert metrics.get("accuracy", 0.0) >= config.ACCURACY_RELAXED_MIN, (
            f"accuracy {metrics.get('accuracy')!r} "
            f"< relaxed minimum {config.ACCURACY_RELAXED_MIN}"
        )
