"""
Per-lake explainability analysis.

For *each* lake in the dataset we:
1. Fit a Ridge regression on its history -> save coefficients.
2. Fit a Random Forest -> save feature importances + RF SHAP values.
3. Run permutation importance on the held-out test years.
4. Save per-step R^2 to a single results CSV.

This is a structured rewrite of `lake-main/src/lakewise.py`. Apart from
clearer organisation the code is logically equivalent — outputs land in
`outputs/explain/lakewise/` instead of the original hard-coded path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

# Feature columns used by the per-lake analysis. These are the original 9
# (no AreaBasin / Lat — those are lake-constant within a single time series).
_LAKEWISE_FEATURES: List[str] = [
    "Evapotranspiration",
    "Forest",
    "GRASS",
    "Crop",
    "Population",
    "Precipitation",
    "Temp_Mean",
    "Temp_Min",
    "Urban",
]

# Year-positional split shared with the main pipeline: 14 train + remainder test.
_LAKE_TRAIN_END = 14


def _split_one_lake(lake_df: pd.DataFrame):
    """Return Xtrain, Xtest, ytrain, ytest for a single lake's time series."""
    X = lake_df[_LAKEWISE_FEATURES].fillna(0)
    y = lake_df["Lake_Area"].to_numpy()

    Xtrain = X.iloc[:_LAKE_TRAIN_END]
    Xtest = X.iloc[_LAKE_TRAIN_END:]
    ytrain = y[:_LAKE_TRAIN_END]
    ytest = y[_LAKE_TRAIN_END:]
    return Xtrain, Xtest, ytrain, ytest


def _per_feature_standardize(Xtrain: pd.DataFrame, Xtest: pd.DataFrame):
    """Standardize each column using train stats.

    We use `np.errstate` and a manual `(x - mean)/std` so the same code path
    handles the constant-column case gracefully (std == 0 -> NaN -> 0).
    """
    mean = Xtrain.mean(axis=0)
    std = Xtrain.std(axis=0).replace(0, 1)  # avoid divide-by-zero
    Xtrain_s = ((Xtrain - mean) / std).fillna(0)
    Xtest_s = ((Xtest - mean) / std).fillna(0)
    # An ::inf can still slip through if the source columns contain inf.
    Xtrain_s = Xtrain_s.replace([np.inf, -np.inf], 0)
    Xtest_s = Xtest_s.replace([np.inf, -np.inf], 0)
    return Xtrain_s, Xtest_s


def lake_wise_analysis(
    dataframe: pd.DataFrame,
    output_dir: str | Path,
) -> Dict[str, Path]:
    """Run per-lake Ridge / RF / SHAP / permutation analysis.

    Parameters
    ----------
    dataframe
        The full Lake_Density dataset (must contain `ID`, `Lake_Area`, and the
        9 features in `_LAKEWISE_FEATURES`).
    output_dir
        Directory to write the per-analysis CSVs into. Will be created.

    Returns
    -------
    Dict mapping analysis name ('ridge', 'randomforest', 'permut', 'shap',
    'shap_mean', 'results') to the path of the CSV written.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pre-allocate accumulators. We use a dict-of-lists because that is
    # easier to extend than the original module-level naming explosion.
    ids: List = []
    ridge_coefs: List[np.ndarray] = []
    rf_importances: List[np.ndarray] = []
    permut_means: List[np.ndarray] = []
    train_r2_reg, test_r2_reg = [], []
    train_r2_rf, test_r2_rf = [], []

    shap_rows: List[np.ndarray] = []   # per-year SHAP values per lake
    shap_id_rows: List = []            # the corresponding lake ids
    shap_means: List[np.ndarray] = []  # one mean-row per lake

    # Re-usable model objects — we re-fit them on each lake. Hyperparameters
    # match the original `lakewise.py`.
    ridge_model = Ridge()
    rf_model = RandomForestRegressor(
        n_estimators=50, min_samples_split=2, min_samples_leaf=1,
        max_features="sqrt", max_depth=5, bootstrap=True,
    )

    for lake_id, lake_df in dataframe.groupby("ID", sort=False):
        # Skip lakes that don't have enough data for the split.
        if len(lake_df) <= _LAKE_TRAIN_END:
            continue

        Xtrain, Xtest, ytrain, ytest = _split_one_lake(lake_df)
        Xtrain_s, Xtest_s = _per_feature_standardize(Xtrain, Xtest)

        ids.append(lake_id)

        # ---- Ridge ----
        ridge_model.fit(Xtrain_s, ytrain)
        ridge_coefs.append(np.asarray(ridge_model.coef_))
        train_r2_reg.append(r2_score(ytrain, ridge_model.predict(Xtrain_s)))
        test_r2_reg.append(r2_score(ytest, ridge_model.predict(Xtest_s)))

        # ---- Random Forest ----
        rf_model.fit(Xtrain_s, ytrain)
        rf_importances.append(np.asarray(rf_model.feature_importances_))
        train_r2_rf.append(r2_score(ytrain, rf_model.predict(Xtrain_s)))
        test_r2_rf.append(r2_score(ytest, rf_model.predict(Xtest_s)))

        # ---- Permutation importance on the test years ----
        perm = permutation_importance(
            rf_model, Xtest_s, ytest, n_repeats=10, random_state=42, n_jobs=2,
        )
        permut_means.append(perm["importances_mean"])

        # ---- SHAP (per-year) ----
        explainer = shap.TreeExplainer(rf_model)
        sv_train = explainer.shap_values(Xtrain_s)
        sv_test = explainer.shap_values(Xtest_s)
        sv_all = np.concatenate([sv_train, sv_test], axis=0)
        shap_rows.append(sv_all)
        shap_id_rows.extend([lake_id] * sv_all.shape[0])
        shap_means.append(sv_all.mean(axis=0))

    # ------------------------------------------------------------------
    # Materialise results to CSVs
    # ------------------------------------------------------------------

    def _df(name_arrays):
        # Helper that builds a DataFrame keyed by ID with one column per feature.
        out = pd.DataFrame({"ID": ids})
        for col_name, col_vals in zip(_LAKEWISE_FEATURES, np.stack(name_arrays, axis=0).T):
            out[col_name] = col_vals
        return out

    paths: Dict[str, Path] = {}

    # Per-lake coefficient / importance tables.
    paths["ridge"] = output_dir / "ridge.csv"
    _df(ridge_coefs).to_csv(paths["ridge"], index=False)

    paths["randomforest"] = output_dir / "randomforest.csv"
    _df(rf_importances).to_csv(paths["randomforest"], index=False)

    paths["permut"] = output_dir / "permutation.csv"
    _df(permut_means).to_csv(paths["permut"], index=False)

    # Year-resolved SHAP table — one row per (lake, year).
    paths["shap"] = output_dir / "shap.csv"
    shap_array = np.concatenate(shap_rows, axis=0)
    shap_df = pd.DataFrame(shap_array, columns=_LAKEWISE_FEATURES)
    shap_df.insert(0, "ID", shap_id_rows)
    shap_df.to_csv(paths["shap"], index=False)

    # Mean SHAP per lake (one row per lake).
    paths["shap_mean"] = output_dir / "shap_mean.csv"
    _df(shap_means).to_csv(paths["shap_mean"], index=False)

    # Train/test R^2 per lake for both models.
    paths["results"] = output_dir / "results.csv"
    pd.DataFrame({
        "ID": ids,
        "train_r2_ridge": train_r2_reg,
        "test_r2_ridge": test_r2_reg,
        "train_r2_rf": train_r2_rf,
        "test_r2_rf": test_r2_rf,
    }).to_csv(paths["results"], index=False)

    return paths
