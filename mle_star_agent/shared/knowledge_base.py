"""shared/knowledge_base.py — persistent semantic memory of refinement outcomes.

Records one structured note per refinement evaluation, keyed by the failure
mode / target component being addressed, so the Planner can see which
strategies have already been tried for a given failure and whether they helped.

Two properties the previous (dead) ``knowledge_base`` state field lacked:

* **Dedup** — entries are deduplicated by ``(strategy_name, mechanism)`` per key,
  so the same fix for the same failure is never recorded twice. Each key's
  history is capped at ``KB_MAX_ENTRIES_PER_KEY`` (most-recent-wins).
* **Cross-run persistence** — the merged KB is written to
  ``config.CKPT_TRIED_APPROACHES`` so the memory survives a fresh run, not just a
  resume of one LangGraph thread. ``load_kb_from_disk`` seeds state at startup.

The module never raises on malformed input or I/O errors — it logs and degrades
to an empty/unchanged KB.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from mle_star_agent import config
from mle_star_agent.shared.checkpoint_io import (
    checkpoint_exists,
    load_checkpoint,
    save_checkpoint,
)

logger = logging.getLogger(__name__)

# History depth per failure-mode key. Old, superseded notes are dropped so the
# Planner prompt stays compact and the most recent lessons dominate.
KB_MAX_ENTRIES_PER_KEY = 25


# ---------------------------------------------------------------------------
# Merge / dedup
# ---------------------------------------------------------------------------

def _entry_signature(entry: dict) -> tuple:
    """Identity used for dedup: same technique applied the same way."""
    return (entry.get("strategy_name"), entry.get("mechanism"))


def merge_knowledge_base(old: Optional[dict], new: Optional[dict]) -> dict:
    """Deep-merge two KB dicts.

    Per key, entry lists are concatenated, deduplicated by
    ``(strategy_name, mechanism)``, and capped to the most recent
    ``KB_MAX_ENTRIES_PER_KEY``. Non-list values are last-write-wins.
    """
    merged: dict[str, Any] = {
        k: (list(v) if isinstance(v, list) else v) for k, v in (old or {}).items()
    }
    for key, entries in (new or {}).items():
        if not isinstance(entries, list):
            merged[key] = entries
            continue
        bucket = merged.setdefault(key, [])
        seen = {_entry_signature(e) for e in bucket if isinstance(e, dict)}
        for e in entries:
            if not isinstance(e, dict):
                continue
            sig = _entry_signature(e)
            if sig in seen:
                continue
            bucket.append(e)
            seen.add(sig)
        if len(bucket) > KB_MAX_ENTRIES_PER_KEY:
            merged[key] = bucket[-KB_MAX_ENTRIES_PER_KEY:]
    return merged


# ---------------------------------------------------------------------------
# Entry construction
# ---------------------------------------------------------------------------

def kb_key(failure_mode: Optional[str], target_component: Optional[str]) -> str:
    """Key a note by failure mode (preferred) or the component being refined."""
    return str(failure_mode or target_component or "general")


def build_kb_entry(
    *,
    strategy_name: Optional[str],
    mechanism: Optional[str],
    target_component: Optional[str],
    outcome: str,
    deltas: Optional[dict],
    outer_iteration: int,
    inner_iteration: int,
) -> dict:
    """Build one structured KB note.

    ``outcome`` is one of ``accepted`` | ``rejected`` | ``curve_aborted`` |
    ``invalid``. ``mechanism`` is a short description of *how* the strategy
    changed the pipeline (truncated for prompt compactness).
    """
    return {
        "strategy_name": strategy_name or "unknown_strategy",
        "mechanism": (mechanism or "").strip()[:280],
        "target_component": target_component or "unknown",
        "outcome": outcome,
        "deltas": {k: round(float(v), 4) for k, v in (deltas or {}).items()
                   if isinstance(v, (int, float))},
        "outer_iteration": int(outer_iteration),
        "inner_iteration": int(inner_iteration),
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def load_kb_from_disk() -> dict:
    """Return the persisted KB, or ``{}`` when none exists / on error."""
    if checkpoint_exists(config.CKPT_TRIED_APPROACHES):
        try:
            data = load_checkpoint(config.CKPT_TRIED_APPROACHES)
            kb = data.get("knowledge_base")
            if isinstance(kb, dict):
                return kb
        except Exception as exc:
            logger.warning("Could not load knowledge base from disk: %s", exc)
    return {}


def persist_kb(kb: dict) -> None:
    """Write the KB to ``CKPT_TRIED_APPROACHES`` (best-effort)."""
    try:
        save_checkpoint(config.CKPT_TRIED_APPROACHES, {"knowledge_base": kb})
    except Exception as exc:
        logger.warning("Could not persist knowledge base: %s", exc)


def record_outcome(state_kb: Optional[dict], key: str, entry: dict) -> dict:
    """Merge ``entry`` (under ``key``) into the disk KB and the in-state KB,
    persist the result, and return the merged KB for the state update.

    Disk is merged in first so a fresh run inherits prior-run lessons; dedup
    keeps re-merged entries from accumulating.
    """
    base = merge_knowledge_base(load_kb_from_disk(), state_kb or {})
    merged = merge_knowledge_base(base, {key: [entry]})
    persist_kb(merged)
    return merged
