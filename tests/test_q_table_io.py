"""
test_q_table_io.py
==================
Tests for Q-table save/load functions in btc_qlearning.
"""
import sys, os, tempfile, warnings
sys.path.insert(0, '/home/claude/quant_portfolio/src')
sys.path.insert(0, '/home/claude/quant_portfolio/src/strategies')

import numpy as np

from btc_qlearning import (
    BTCQLearningConfig, save_q_table, load_q_table, state_size, N_ACTIONS,
)


def make_dummy_q_table(seed=0):
    """A reproducible 'trained' Q-table for tests."""
    cfg = BTCQLearningConfig()
    np.random.seed(seed)
    return np.random.randn(state_size(cfg), N_ACTIONS), cfg


def test_save_and_load_round_trip():
    """Save then load should produce an equal Q-table."""
    Q, cfg = make_dummy_q_table()
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        path = f.name
    try:
        save_q_table(Q, cfg, path)
        Q_loaded, cfg_loaded, meta = load_q_table(path)
        assert np.array_equal(Q, Q_loaded)
        assert cfg_loaded.n_episodes == cfg.n_episodes
        assert cfg_loaded.learning_rate == cfg.learning_rate
    finally:
        os.unlink(path)


def test_metadata_preserved():
    """User-supplied metadata should survive save/load."""
    Q, cfg = make_dummy_q_table()
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        path = f.name
    try:
        meta = {"data_range": "2020-01-01 to 2023-12-31",
                "training_seed": 42}
        save_q_table(Q, cfg, path, metadata=meta)
        _, _, loaded_meta = load_q_table(path)
        assert loaded_meta["data_range"] == "2020-01-01 to 2023-12-31"
        assert loaded_meta["training_seed"] == 42
        # Auto-generated fields should also be present
        assert "saved_at" in loaded_meta
        assert "shape" in loaded_meta
    finally:
        os.unlink(path)


def test_strict_compatibility_check_passes_for_same_config():
    """Same config → load with strict=True should succeed."""
    Q, cfg = make_dummy_q_table()
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        path = f.name
    try:
        save_q_table(Q, cfg, path)
        # Should not raise
        Q_loaded, _, _ = load_q_table(path, expected_config=cfg, strict=True)
        assert np.array_equal(Q, Q_loaded)
    finally:
        os.unlink(path)


def test_strict_check_raises_on_state_field_mismatch():
    """Mismatched n_pct_bins should raise in strict mode."""
    from dataclasses import replace
    Q, cfg = make_dummy_q_table()
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        path = f.name
    try:
        save_q_table(Q, cfg, path)
        # Build an "expected" config with a different state space
        diff_cfg = replace(cfg, n_pct_bins=cfg.n_pct_bins + 1)
        try:
            load_q_table(path, expected_config=diff_cfg, strict=True)
            assert False, "Should have raised on mismatched n_pct_bins"
        except RuntimeError as e:
            assert "n_pct_bins" in str(e)
    finally:
        os.unlink(path)


def test_non_strict_only_warns_on_mismatch():
    """Non-strict mode should warn but not raise."""
    from dataclasses import replace
    Q, cfg = make_dummy_q_table()
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        path = f.name
    try:
        save_q_table(Q, cfg, path)
        diff_cfg = replace(cfg, n_pct_bins=cfg.n_pct_bins + 1)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            Q_loaded, _, _ = load_q_table(path, expected_config=diff_cfg, strict=False)
            # Should produce a UserWarning, not raise
            assert any("n_pct_bins" in str(w.message) for w in caught)
            # But still returns the Q-table
            assert np.array_equal(Q, Q_loaded)
    finally:
        os.unlink(path)


def test_load_nonexistent_file_raises():
    """Loading a missing file should raise FileNotFoundError."""
    try:
        load_q_table("/tmp/definitely_does_not_exist_xyz.pkl")
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError:
        pass


def test_load_invalid_file_raises():
    """Loading an arbitrary pickle that isn't a Q-table artifact should raise."""
    import pickle
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        pickle.dump({"unrelated": "data"}, f)
        path = f.name
    try:
        try:
            load_q_table(path)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Q-table" in str(e)
    finally:
        os.unlink(path)


def test_only_state_fields_block_compatibility():
    """Different non-state fields (learning_rate, n_episodes) should NOT block load."""
    from dataclasses import replace
    Q, cfg = make_dummy_q_table()
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        path = f.name
    try:
        save_q_table(Q, cfg, path)
        # Different non-state hyperparams — should load fine
        diff_cfg = replace(cfg, learning_rate=0.99, n_episodes=99999, discount=0.1)
        Q_loaded, _, _ = load_q_table(path, expected_config=diff_cfg, strict=True)
        assert np.array_equal(Q, Q_loaded)
    finally:
        os.unlink(path)


if __name__ == "__main__":
    import traceback
    tests = [
        test_save_and_load_round_trip,
        test_metadata_preserved,
        test_strict_compatibility_check_passes_for_same_config,
        test_strict_check_raises_on_state_field_mismatch,
        test_non_strict_only_warns_on_mismatch,
        test_load_nonexistent_file_raises,
        test_load_invalid_file_raises,
        test_only_state_fields_block_compatibility,
    ]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"✓ {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"✗ {t.__name__}: {e}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
