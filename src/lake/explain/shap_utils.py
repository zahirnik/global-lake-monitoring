"""
SHAP helpers: averaged TreeExplainer values and CSV serialization.

The original notebook averaged SHAP values over multiple TreeExplainer calls
to smooth out the small amount of stochasticity in the algorithm. We keep the
same pattern but expose it as a documented function.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import shap


def shap_explain_tree(
    model,
    X: np.ndarray,
    num_iter: int = 10,
) -> np.ndarray:
    """Return SHAP values averaged over `num_iter` TreeExplainer evaluations.

    Parameters
    ----------
    model
        A fitted tree-based regressor (RF, GB, XGBoost).
    X
        Feature matrix the SHAP values should be computed for.
    num_iter
        Number of repeated SHAP computations to average over. The original
        code used 10.

    Notes
    -----
    For most tree models, ``TreeExplainer.shap_values`` is deterministic, so
    averaging is largely belt-and-braces. We keep it for parity with the
    legacy outputs.
    """
    if num_iter < 1:
        raise ValueError("num_iter must be >= 1")

    explainer = shap.TreeExplainer(model)
    accumulator = np.zeros_like(X, dtype=np.float64)
    for _ in range(num_iter):
        accumulator += explainer.shap_values(X)

    return accumulator / num_iter


def save_shap_table(
    shap_values: np.ndarray,
    feature_names: Sequence[str],
    output_csv: str | Path,
    extra_columns: dict | None = None,
) -> Path:
    """Write SHAP values to a tidy CSV next to any extra metadata columns.

    Parameters
    ----------
    shap_values
        Array of shape (n_samples, n_features).
    feature_names
        Column names for the SHAP columns; must have length n_features.
    output_csv
        Where to write the file.
    extra_columns
        Optional dict of name -> 1D array to prepend (e.g., ID, year, ytrue).
    """
    if shap_values.shape[1] != len(feature_names):
        raise ValueError(
            f"shap_values has {shap_values.shape[1]} columns but "
            f"{len(feature_names)} feature names were provided"
        )

    df = pd.DataFrame()
    if extra_columns:
        for name, col in extra_columns.items():
            df[name] = col
    for i, name in enumerate(feature_names):
        df[name] = shap_values[:, i]

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return output_csv
