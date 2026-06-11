"""Unit tests for shared/knowledge_base.py — persistent refinement memory."""

from __future__ import annotations

import pytest

from mle_star_agent import config
from mle_star_agent.shared import knowledge_base as kb


@pytest.fixture
def kb_disk(tmp_path, monkeypatch):
    """Redirect KB persistence to a temp file."""
    path = tmp_path / "tried_approaches.json"
    monkeypatch.setattr(config, "CKPT_TRIED_APPROACHES", path)
    return path


def _entry(strategy, mechanism, outcome="rejected"):
    return kb.build_kb_entry(
        strategy_name=strategy,
        mechanism=mechanism,
        target_component="model_architecture",
        outcome=outcome,
        deltas={"delta_ng_recall": 0.01},
        outer_iteration=1,
        inner_iteration=2,
    )


# ── merge / dedup ───────────────────────────────────────────────────────────

def test_merge_concatenates_distinct_entries():
    a = {"g_ng_overlap": [_entry("ssim", "add ssim map")]}
    b = {"g_ng_overlap": [_entry("attention", "diff attention")]}
    merged = kb.merge_knowledge_base(a, b)
    assert len(merged["g_ng_overlap"]) == 2


def test_merge_dedups_same_strategy_and_mechanism():
    a = {"g_ng_overlap": [_entry("ssim", "add ssim map")]}
    b = {"g_ng_overlap": [_entry("ssim", "add ssim map")]}  # identical signature
    merged = kb.merge_knowledge_base(a, b)
    assert len(merged["g_ng_overlap"]) == 1


def test_merge_keeps_distinct_mechanism_same_strategy():
    a = {"k": [_entry("ssim", "variant A")]}
    b = {"k": [_entry("ssim", "variant B")]}
    merged = kb.merge_knowledge_base(a, b)
    assert len(merged["k"]) == 2


def test_merge_caps_history_depth():
    base: dict = {}
    for i in range(kb.KB_MAX_ENTRIES_PER_KEY + 10):
        base = kb.merge_knowledge_base(base, {"k": [_entry("s", f"mech-{i}")]})
    assert len(base["k"]) == kb.KB_MAX_ENTRIES_PER_KEY
    # Most-recent-wins: the last mechanism survives, the first is evicted.
    mechs = [e["mechanism"] for e in base["k"]]
    assert "mech-0" not in mechs
    assert f"mech-{kb.KB_MAX_ENTRIES_PER_KEY + 9}" in mechs


def test_merge_handles_none_and_non_list():
    assert kb.merge_knowledge_base(None, None) == {}
    merged = kb.merge_knowledge_base({"x": "scalar"}, {"y": [_entry("s", "m")]})
    assert merged["x"] == "scalar"
    assert len(merged["y"]) == 1


# ── build_kb_entry ──────────────────────────────────────────────────────────

def test_build_entry_truncates_mechanism_and_rounds_deltas():
    e = kb.build_kb_entry(
        strategy_name="s",
        mechanism="x" * 500,
        target_component="c",
        outcome="accepted",
        deltas={"d": 0.123456, "junk": "nope"},
        outer_iteration=0,
        inner_iteration=0,
    )
    assert len(e["mechanism"]) == 280
    assert e["deltas"] == {"d": 0.1235}  # non-numeric dropped, rounded


def test_kb_key_prefers_failure_mode():
    assert kb.kb_key("g_ng_overlap", "model") == "g_ng_overlap"
    assert kb.kb_key("", "model") == "model"
    assert kb.kb_key(None, None) == "general"


# ── persistence ─────────────────────────────────────────────────────────────

def test_persist_and_load_round_trip(kb_disk):
    data = {"k": [_entry("s", "m", outcome="accepted")]}
    kb.persist_kb(data)
    assert kb.load_kb_from_disk() == data


def test_load_missing_returns_empty(kb_disk):
    assert kb.load_kb_from_disk() == {}


def test_record_outcome_merges_state_and_disk(kb_disk):
    # Seed disk with a prior-run lesson.
    kb.persist_kb({"g_ng_overlap": [_entry("ssim", "add ssim map")]})
    # In-state KB holds a different lesson under the same key.
    state_kb = {"g_ng_overlap": [_entry("attention", "diff attention")]}
    merged = kb.record_outcome(
        state_kb, "g_ng_overlap", _entry("clahe", "per-lot normalize", outcome="accepted")
    )
    strategies = {e["strategy_name"] for e in merged["g_ng_overlap"]}
    assert strategies == {"ssim", "attention", "clahe"}
    # And it was persisted back to disk.
    assert kb.load_kb_from_disk() == merged
