"""Smoke tests — verify the package imports and the data layer works end-to-end
on a synthetic dataset (no need for the real data CSV to be present)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lake.config import FEATURE_COLUMNS, TARGET_COLUMN, default_config
from lake.data.dataset import (
    LakeTorchDataset,
    normalize,
    split_timeseries_by_lake,
)


def _make_fake_dataframe(n_lakes: int = 3, n_years: int = 19) -> pd.DataFrame:
    """Build a synthetic frame with the expected columns."""
    rng = np.random.default_rng(0)
    rows = []
    for lake_id in range(n_lakes):
        for year in range(n_years):
            row = {"ID": lake_id, "Date": 2000 + year}
            for col in FEATURE_COLUMNS:
                row[col] = float(rng.standard_normal())
            row[TARGET_COLUMN] = float(rng.uniform(10.0, 1000.0))
            rows.append(row)
    return pd.DataFrame(rows)


def test_default_config_paths_resolvable():
    cfg = default_config()
    cfg.paths.ensure()
    # All output dirs should exist after `ensure()`.
    assert cfg.paths.figures_dir.exists()
    assert cfg.paths.outputs_dir.exists()


def test_split_shapes_match_year_counts():
    df = _make_fake_dataframe()
    splits = split_timeseries_by_lake(df)
    # 3 lakes * (10 train, 4 valid, 5 test) years.
    assert splits.train.X.shape == (3 * 10, len(FEATURE_COLUMNS))
    assert splits.valid.X.shape == (3 * 4, len(FEATURE_COLUMNS))
    assert splits.test.X.shape == (3 * 5, len(FEATURE_COLUMNS))


def test_normalize_train_mean_is_zero():
    df = _make_fake_dataframe()
    splits = split_timeseries_by_lake(df)
    splits_s, scaler = normalize(splits)
    # The scaler is fit on train, so its mean should be ~0 after transform.
    assert np.allclose(splits_s.train.X.mean(axis=0), 0.0, atol=1e-5)


def test_torch_dataset_returns_expected_keys():
    df = _make_fake_dataframe()
    splits_s, _ = normalize(split_timeseries_by_lake(df))
    ds = LakeTorchDataset(splits_s.train)
    sample = ds[0]
    assert set(sample.keys()) == {"X", "y", "ID", "year"}
    # Each sample should be (1, n_features) so the Transformer's seq-first
    # permute does the right thing.
    assert sample["X"].shape == (1, len(FEATURE_COLUMNS))


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
