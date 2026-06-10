"""
graph.py — top-level LangGraph StateGraph for the MLE-STAR AOI agent.

Full topology (mirrors ADK SequentialAgent + nested RetryLoopAgent):

  START → phase1_init → phase2_ablation

  ┌──────────────── Outer loop (route_outer_loop) ────────────────┐
  │  phase2_ablation → phase2_diagnosis → phase2_error_analysis_gate
  │                                                               │
  │  ┌─────────────── Inner loop (route_inner_loop) ──────────────┤
  │  │  → phase2_planner → phase2_strategy_gate → phase2_coder   │
  │  │    → phase2_evaluator → phase2_error_analysis             │
  │  │              ↑____________(continue)_______________________|
  │  │                                                exit ↓
  │  └────────────────────────────────── phase2_outer_gate ───────┘
  │                                              │ exit
  └──────────────────────────────────────────────┘

  phase3_ensemble_coder ↔ phase3_ensemble_evaluator (route_ensemble_loop)
              ↓ exit
         phase4_submit → route_after_submit → END | retry → phase2_ablation
"""
from __future__ import annotations

import os

from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from mle_star_agent.state import AgentState

# ── Node imports ──────────────────────────────────────────────────────────────
from mle_star_agent.nodes.phase1_init import phase1_init_node

from mle_star_agent.nodes.phase2_ablation import phase2_ablation_node
from mle_star_agent.nodes.phase2_diagnosis import phase2_diagnosis_node
from mle_star_agent.nodes.phase2_error_analysis_gate import phase2_error_analysis_gate_node
from mle_star_agent.nodes.phase2_planner import phase2_planner_node
from mle_star_agent.nodes.phase2_strategy_gate import phase2_strategy_gate_node
from mle_star_agent.nodes.phase2_coder import phase2_coder_node
from mle_star_agent.nodes.phase2_evaluator import phase2_evaluator_node
from mle_star_agent.nodes.phase2_error_analysis import phase2_error_analysis
from mle_star_agent.nodes.phase2_routing import (
    route_inner_loop,
    route_outer_loop,
    phase2_outer_gate_node,
)

from mle_star_agent.nodes.phase3_ensemble_coder import phase3_ensemble_coder_node
from mle_star_agent.nodes.phase3_ensemble_evaluator import phase3_ensemble_evaluator_node
from mle_star_agent.nodes.phase3_routing import route_ensemble_loop

from mle_star_agent.nodes.phase4_submit import phase4_submit_node
from mle_star_agent.nodes.phase4_routing import route_after_submit


# ── Graph assembly ────────────────────────────────────────────────────────────

def build_graph(checkpointer=None):
    """Build and compile the MLE-STAR StateGraph.

    Args:
        checkpointer: LangGraph checkpointer instance, or None.
            None  → MemorySaver() is used (Studio / tests / dry-run).
            Caller can pass SqliteSaver for persistent real runs.

    Returns:
        Compiled LangGraph CompiledGraph.
    """
    g = StateGraph(AgentState)

    # ── Register nodes ────────────────────────────────────────────────────────
    g.add_node("phase1_init",                  phase1_init_node)

    g.add_node("phase2_ablation",              phase2_ablation_node)
    g.add_node("phase2_diagnosis",             phase2_diagnosis_node)
    g.add_node("phase2_error_analysis_gate",   phase2_error_analysis_gate_node)
    g.add_node("phase2_planner",               phase2_planner_node)
    g.add_node("phase2_strategy_gate",         phase2_strategy_gate_node)
    g.add_node("phase2_coder",                 phase2_coder_node)
    g.add_node("phase2_evaluator",             phase2_evaluator_node)
    g.add_node("phase2_error_analysis",        phase2_error_analysis)
    g.add_node("phase2_outer_gate",            phase2_outer_gate_node)

    g.add_node("phase3_ensemble_coder",        phase3_ensemble_coder_node)
    g.add_node("phase3_ensemble_evaluator",    phase3_ensemble_evaluator_node)

    g.add_node("phase4_submit",                phase4_submit_node)

    # ── Entry point ───────────────────────────────────────────────────────────
    g.set_entry_point("phase1_init")

    # ── Phase 1 → Phase 2 ─────────────────────────────────────────────────────
    g.add_edge("phase1_init", "phase2_ablation")

    # ── Phase 2 outer loop ────────────────────────────────────────────────────
    g.add_edge("phase2_ablation",            "phase2_diagnosis")
    g.add_edge("phase2_diagnosis",           "phase2_error_analysis_gate")

    # ── Phase 2 inner loop ────────────────────────────────────────────────────
    g.add_edge("phase2_error_analysis_gate", "phase2_planner")
    g.add_edge("phase2_planner",             "phase2_strategy_gate")
    g.add_edge("phase2_strategy_gate",       "phase2_coder")
    g.add_edge("phase2_coder",               "phase2_evaluator")
    g.add_edge("phase2_evaluator",           "phase2_error_analysis")

    # Inner loop conditional: continue → error_analysis_gate | exit → outer_gate
    g.add_conditional_edges(
        "phase2_error_analysis",
        route_inner_loop,
        {"continue": "phase2_error_analysis_gate", "exit": "phase2_outer_gate"},
    )

    # Outer loop conditional: continue → ablation | exit → ensemble
    g.add_conditional_edges(
        "phase2_outer_gate",
        route_outer_loop,
        {"continue": "phase2_ablation", "exit": "phase3_ensemble_coder"},
    )

    # ── Phase 3 ensemble loop ─────────────────────────────────────────────────
    g.add_edge("phase3_ensemble_coder", "phase3_ensemble_evaluator")

    g.add_conditional_edges(
        "phase3_ensemble_evaluator",
        route_ensemble_loop,
        {"continue": "phase3_ensemble_coder", "exit": "phase4_submit"},
    )

    # ── Phase 4 submit + retry ────────────────────────────────────────────────
    g.add_conditional_edges(
        "phase4_submit",
        route_after_submit,
        {END: END, "phase2_ablation": "phase2_ablation"},
    )

    # ── Compile with checkpointer ─────────────────────────────────────────────
    ckpt = checkpointer if checkpointer is not None else MemorySaver()
    return g.compile(checkpointer=ckpt)


def make_graph():
    """Zero-arg factory for `langgraph dev` / LangGraph Platform (Studio).

    The platform manages persistence itself, so the graph must be compiled
    WITHOUT an explicit checkpointer. This must stay zero-arg: the platform
    treats any single unannotated factory parameter as its config dict and
    would pass that dict in (raising "Invalid checkpointer ... Received dict").
    """
    return build_graph(checkpointer=False)


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="LangGraph AOI MLE-STAR agent")
    parser.add_argument("--dataset", required=True,
                        help="Path to dataset root directory")
    parser.add_argument("--goal",    default="NG recall >= 1.00, overkill <= 0.05",
                        help="Natural-language goal description")
    parser.add_argument("--dry-run", action="store_true",
                        help="Enable dry-run mode (1 epoch / 10 samples / MemorySaver)")
    args = parser.parse_args()

    if args.dry_run:
        os.environ["DRY_RUN"] = "1"
        checkpointer = MemorySaver()
    else:
        from langgraph.checkpoint.sqlite import SqliteSaver
        os.makedirs("checkpoints", exist_ok=True)
        checkpointer = SqliteSaver.from_conn_string("checkpoints/langgraph.db")

    from mle_star_agent import config

    graph = build_graph(checkpointer=checkpointer)

    initial_state: AgentState = {
        "dataset_path":                 args.dataset,
        "goal":                         args.goal,
        "outer_iteration":              0,
        "inner_iteration":              0,
        "ensemble_iteration":           0,
        "submission_retry":             0,
        "no_improve_count":             0,
        "ensemble_no_improve_count":    0,
        "tokens_used":                  0,
        "stop_outer_loop":              False,
        "stop_ensemble_loop":           False,
        "error_analysis_blocked":       False,
        "error_analysis_instrumentation_required": False,
        "error_analysis_repair_attempted": False,
        "submission_passed":            False,
        "debug_mode":                   config.DEBUG_MODE,
        "messages":                     [],
        "tried_approaches":             [],
        "tried_ensemble_approaches":    [],
        "candidate_scripts":            [],
        "candidate_scores":             [],
    }

    run_tag = "dry-run" if args.dry_run else "full"
    print(f"\nStarting MLE-STAR loop [{run_tag}]")
    print(f"  goal:    {args.goal}")
    print(f"  dataset: {args.dataset}\n")

    thread_cfg = {"configurable": {"thread_id": "mle_star_run_1"}}
    final = graph.invoke(initial_state, config=thread_cfg)

    print("\nRun complete")
    skip_keys = {"messages", "knowledge_base", "candidate_scripts"}
    print(json.dumps(
        {k: v for k, v in final.items() if k not in skip_keys},
        indent=2, default=str,
    ))
