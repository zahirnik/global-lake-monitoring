"""
Tree-based regressors and a shared training helper.

We expose three flavours (RandomForest, GradientBoosting, XGBoost) using a
single `build_tree_regressor("RF" | "GB" | "XGB")` factory so calling code
doesn't need to know each library's import path.

The training helper :func:`train_tree_regressor` runs a quick repeated K-fold
score, fits on the training split, and returns predictions + the fitted model.
This mirrors the original `MLReg()` function but with cleaner inputs/outputs.
"""

from __future__ import annotations

from typing import Literal, Tuple

import numpy as np
from numpy import absolute
from sklearn import metrics
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import RepeatedKFold, cross_val_score
from xgboost import XGBRegressor


RegressorKind = Literal["RF", "GB", "XGB"]


def build_tree_regressor(kind: RegressorKind):
    """Return an unfit sklearn-compatible regressor of the requested kind.

    Hyperparameters mirror the values used in the original notebook so any
    saved SHAP outputs remain comparable.
    """
    if kind == "RF":
        # Random Forest — robust default, gives feature importances.
        return RandomForestRegressor(
            n_estimators=200,
            min_samples_split=12,
            min_samples_leaf=6,
            max_features="sqrt",
            max_depth=10,
            bootstrap=True,
        )
    if kind == "GB":
        # Gradient Boosting baseline.
        return GradientBoostingRegressor(
            n_estimators=200,
            min_samples_split=12,
            min_samples_leaf=6,
            max_features="sqrt",
            max_depth=10,
        )
    if kind == "XGB":
        # XGBoost — what the original notebook used for the final SHAP run.
        return XGBRegressor(n_estimators=200, max_depth=4)
    raise ValueError(f"Unknown regressor kind: {kind!r}")


def train_tree_regressor(
    Xtrain: np.ndarray,
    ytrain: np.ndarray,
    Xtest: np.ndarray,
    ytest: np.ndarray,
    kind: RegressorKind,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, object]:
    """Fit a tree regressor and return (ytrain_pred, ytest_pred, fitted_model).

    A 10-fold x 3-repeat cross-validated MAE is printed before the final fit
    so the caller can sanity-check stability; this matches what the original
    `MLReg()` did.
    """
    model = build_tree_regressor(kind)

    # Cross-validated MAE — useful diagnostic, not used for model selection.
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    cv_scores = cross_val_score(
        model, Xtrain, ytrain,
        scoring="neg_mean_absolute_error", cv=cv, n_jobs=-1,
    )
    cv_scores = absolute(cv_scores)
    if verbose:
        print(f"[{kind}] CV MAE: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

    # Final fit on the *full* training split.
    model.fit(Xtrain, ytrain)

    ytrain_pred = model.predict(Xtrain)
    ytest_pred = model.predict(Xtest)

    if verbose:
        train_r2 = metrics.r2_score(ytrain, ytrain_pred)
        test_r2 = metrics.r2_score(ytest, ytest_pred)
        print(f"[{kind}] R^2 train: {train_r2:.3f} | test: {test_r2:.3f}")

    return ytrain_pred, ytest_pred, model
