"""Unit tests for Phase 1 initialization node (Component 3.2).

Validates three checkpoints from MLE-STAR §3.1 running in DRY_RUN=1 mode
(no LLM calls, no real dataset I/O — everything is patched or stubbed):

  test_skip_check_restores_state
      When CKPT_L0 and CKPT_CANDIDATE_SCORES both exist on disk, the node
      must return early and restore all 6 best-snapshot fields from L0.json.

  test_data_split_created_when_missing
      When no CKPT_DATA_SPLIT exists, the node must call build_data_split
      and persist the result so subsequent runs can skip the split step.

  test_l0_saved_with_all_required_fields
      After a full DRY_RUN=1 pass the node must write CKPT_L0 containing
      best_candidate_name, script, current_best_score, best_miss_rate,
      best_overkill_rate, best_accuracy, and best_f1.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FAKE_L0 = {
    "best_candidate_name": "EfficientNet-B0",
    "script": "print('stub')",
    "current_best_score": 0.95,
    "best_miss_rate": 0.05,
    "best_overkill_rate": 0.03,
    "best_accuracy": 0.94,
    "best_f1": 0.93,
}

_FAKE_SCORES = {
    "scores": [
        {
            "index": 0,
            "name": "EfficientNet-B0",
            "status": "full_pass",
            "score": 0.95,
            "miss_rate": 0.05,
            "overkill_rate": 0.03,
            "accuracy": 0.94,
            "f1": 0.93,
        }
    ]
}

_FAKE_SPLIT = {
    "train": [{"sample_id": "s0", "img_l": "/x/a.png", "img_r": "/x/b.png", "label": "G"}],
    "val":   [{"sample_id": "s1", "img_l": "/x/c.png", "img_r": "/x/d.png", "label": "NG"}],
    "test":  [{"sample_id": "s2", "img_l": "/x/e.png", "img_r": "/x/f.png", "label": "G"}],
    "metadata": {"input_modality": "stereo"},
}

_FAKE_SCORED_CANDIDATES = [
    {
        "index": 0,
        "name": "EfficientNet-B0",
        "status": "full_pass",
        "script": "print('stub')",
        "architecture": "EfficientNet-B0",
        "metrics": {
            "ng_recall": 0.95,
            "miss_rate": 0.05,
            "overkill_rate": 0.03,
            "accuracy": 0.94,
            "f1": 0.93,
        },
    }
]


def _make_state(**overrides: Any) -> dict:
    """Return a minimal AgentState-compatible dict."""
    base = {
        "tokens_used": 0,
        "debug_mode": True,
        "dataset_path": "",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Fixture: redirect all checkpoint paths to a temp directory
# ---------------------------------------------------------------------------

@pytest.fixture()
def isolated_checkpoints(tmp_path, monkeypatch):
    """Patch every config.CKPT_* to live under tmp_path/checkpoints."""
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()

    import mle_star_agent.config as cfg
    import mle_star_agent.nodes.phase1_init as node_mod

    monkeypatch.setattr(cfg, "CHECKPOINT_DIR",           ckpt_dir)
    monkeypatch.setattr(cfg, "CKPT_L0",                  ckpt_dir / "L0.json")
    monkeypatch.setattr(cfg, "CKPT_CANDIDATE_SCORES",    ckpt_dir / "candidate_scores.json")
    monkeypatch.setattr(cfg, "CKPT_CANDIDATE_SCRIPTS",   ckpt_dir / "candidate_scripts.json")
    monkeypatch.setattr(cfg, "CKPT_DATA_SPLIT",          ckpt_dir / "data_split_grouped.json")
    monkeypatch.setattr(cfg, "CKPT_FAILED_ARCHITECTURES", ckpt_dir / "failed_architectures.json")
    monkeypatch.setattr(cfg, "CKPT_VALIDATION_CACHE",    ckpt_dir / "validation_cache.json")

    # _DATA_SPLIT_PATH is a module-level str set at import — redirect it too
    monkeypatch.setattr(node_mod, "_DATA_SPLIT_PATH", str(ckpt_dir / "data_split_grouped.json"))

    return ckpt_dir


# ---------------------------------------------------------------------------
# 3.2-A  Skip check
# ---------------------------------------------------------------------------

class TestSkipCheck:
    def test_skip_check_restores_state(self, isolated_checkpoints):
        """When CKPT_L0 and CKPT_CANDIDATE_SCORES exist, node must return early."""
        import mle_star_agent.config as cfg

        # Write the pre-existing checkpoints
        cfg.CKPT_L0.write_text(json.dumps(_FAKE_L0))
        cfg.CKPT_CANDIDATE_SCORES.write_text(json.dumps(_FAKE_SCORES))

        from mle_star_agent.nodes.phase1_init import phase1_init_node
        result = phase1_init_node(_make_state())

        assert result["current_phase"] == "refine"
        assert result["best_candidate_name"] == _FAKE_L0["best_candidate_name"]
        assert result["current_best_score"]  == pytest.approx(_FAKE_L0["current_best_score"])
        assert result["best_miss_rate"]      == pytest.approx(_FAKE_L0["best_miss_rate"])
        assert result["best_overkill_rate"]  == pytest.approx(_FAKE_L0["best_overkill_rate"])
        assert result["best_accuracy"]       == pytest.approx(_FAKE_L0["best_accuracy"])
        assert result["best_f1"]             == pytest.approx(_FAKE_L0["best_f1"])
        assert result["best_pipeline"]["script"] == _FAKE_L0["script"]

    def test_skip_check_absent_when_scores_missing(self, isolated_checkpoints):
        """If only CKPT_L0 exists but CKPT_CANDIDATE_SCORES is missing, no skip."""
        import mle_star_agent.config as cfg

        cfg.CKPT_L0.write_text(json.dumps(_FAKE_L0))
        # CKPT_CANDIDATE_SCORES intentionally NOT written

        from mle_star_agent.nodes.phase1_init import phase1_init_node

        with patch("mle_star_agent.nodes.phase1_init.build_data_split",
                   return_value=_FAKE_SPLIT) as mock_split, \
             patch("mle_star_agent.nodes.phase1_init._evaluate_candidates",
                   return_value=_FAKE_SCORED_CANDIDATES), \
             patch("mle_star_agent.nodes.phase1_init.validate_script") as mock_val:

            mock_val.return_value = MagicMock(valid=True, rejection_reasons=[])
            phase1_init_node(_make_state())

        # The node must have fallen through to the data-split step, not returned early.
        # build_data_split being called is proof the skip-check did NOT fire.
        mock_split.assert_called_once()


# ---------------------------------------------------------------------------
# 3.2-B  Data split persistence
# ---------------------------------------------------------------------------

class TestDataSplit:
    def test_data_split_created_when_missing(self, isolated_checkpoints):
        """Node must call build_data_split and persist CKPT_DATA_SPLIT."""
        import mle_star_agent.config as cfg

        assert not cfg.CKPT_DATA_SPLIT.exists(), "Pre-condition: no split on disk"

        from mle_star_agent.nodes.phase1_init import phase1_init_node

        with patch("mle_star_agent.nodes.phase1_init.build_data_split",
                   return_value=_FAKE_SPLIT) as mock_split, \
             patch("mle_star_agent.nodes.phase1_init._evaluate_candidates",
                   return_value=_FAKE_SCORED_CANDIDATES), \
             patch("mle_star_agent.nodes.phase1_init.validate_script") as mock_val:

            mock_val.return_value = MagicMock(valid=True, rejection_reasons=[])
            phase1_init_node(_make_state())

        mock_split.assert_called_once()
        assert cfg.CKPT_DATA_SPLIT.exists(), "CKPT_DATA_SPLIT must be written after split"
        saved = json.loads(cfg.CKPT_DATA_SPLIT.read_text())
        assert "train" in saved and "val" in saved and "test" in saved

    def test_data_split_loaded_from_checkpoint(self, isolated_checkpoints):
        """When CKPT_DATA_SPLIT already exists, build_data_split must NOT be called."""
        import mle_star_agent.config as cfg

        cfg.CKPT_DATA_SPLIT.write_text(json.dumps(_FAKE_SPLIT))

        from mle_star_agent.nodes.phase1_init import phase1_init_node

        with patch("mle_star_agent.nodes.phase1_init.build_data_split") as mock_split, \
             patch("mle_star_agent.nodes.phase1_init._evaluate_candidates",
                   return_value=_FAKE_SCORED_CANDIDATES), \
             patch("mle_star_agent.nodes.phase1_init.validate_script") as mock_val:

            mock_val.return_value = MagicMock(valid=True, rejection_reasons=[])
            phase1_init_node(_make_state())

        mock_split.assert_not_called()


# ---------------------------------------------------------------------------
# 3.2-C  L0 save
# ---------------------------------------------------------------------------

class TestL0Save:
    def test_l0_saved_with_all_required_fields(self, isolated_checkpoints):
        """After a full DRY_RUN=1 pass, CKPT_L0 must contain all 6 snapshot fields."""
        import mle_star_agent.config as cfg

        from mle_star_agent.nodes.phase1_init import phase1_init_node

        with patch("mle_star_agent.nodes.phase1_init.build_data_split",
                   return_value=_FAKE_SPLIT), \
             patch("mle_star_agent.nodes.phase1_init._evaluate_candidates",
                   return_value=_FAKE_SCORED_CANDIDATES), \
             patch("mle_star_agent.nodes.phase1_init.validate_script") as mock_val:

            mock_val.return_value = MagicMock(valid=True, rejection_reasons=[])
            result = phase1_init_node(_make_state())

        assert cfg.CKPT_L0.exists(), "CKPT_L0 must be written"
        l0 = json.loads(cfg.CKPT_L0.read_text())

        required_fields = [
            "best_candidate_name",
            "script",
            "current_best_score",
            "best_miss_rate",
            "best_overkill_rate",
            "best_accuracy",
            "best_f1",
        ]
        for field in required_fields:
            assert field in l0, f"CKPT_L0 missing field: {field}"

    def test_l0_values_match_return_state(self, isolated_checkpoints):
        """Values in CKPT_L0 must match what the node returns in state."""
        import mle_star_agent.config as cfg

        from mle_star_agent.nodes.phase1_init import phase1_init_node

        with patch("mle_star_agent.nodes.phase1_init.build_data_split",
                   return_value=_FAKE_SPLIT), \
             patch("mle_star_agent.nodes.phase1_init._evaluate_candidates",
                   return_value=_FAKE_SCORED_CANDIDATES), \
             patch("mle_star_agent.nodes.phase1_init.validate_script") as mock_val:

            mock_val.return_value = MagicMock(valid=True, rejection_reasons=[])
            result = phase1_init_node(_make_state())

        l0 = json.loads(cfg.CKPT_L0.read_text())
        assert result["current_best_score"] == pytest.approx(l0["current_best_score"])
        assert result["best_miss_rate"]     == pytest.approx(l0["best_miss_rate"])
        assert result["best_overkill_rate"] == pytest.approx(l0["best_overkill_rate"])
        assert result["best_accuracy"]      == pytest.approx(l0["best_accuracy"])
        assert result["best_f1"]            == pytest.approx(l0["best_f1"])
        assert result["best_candidate_name"] == l0["best_candidate_name"]

    def test_candidate_scores_written(self, isolated_checkpoints):
        """CKPT_CANDIDATE_SCORES must be written alongside CKPT_L0."""
        import mle_star_agent.config as cfg

        from mle_star_agent.nodes.phase1_init import phase1_init_node

        with patch("mle_star_agent.nodes.phase1_init.build_data_split",
                   return_value=_FAKE_SPLIT), \
             patch("mle_star_agent.nodes.phase1_init._evaluate_candidates",
                   return_value=_FAKE_SCORED_CANDIDATES), \
             patch("mle_star_agent.nodes.phase1_init.validate_script") as mock_val:

            mock_val.return_value = MagicMock(valid=True, rejection_reasons=[])
            phase1_init_node(_make_state())

        assert cfg.CKPT_CANDIDATE_SCORES.exists(), "CKPT_CANDIDATE_SCORES must be written"
        scores = json.loads(cfg.CKPT_CANDIDATE_SCORES.read_text())
        assert "scores" in scores
        assert len(scores["scores"]) >= 1
