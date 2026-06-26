"""
Predicted-vs-true scatter plots with R^2/MAE/RMSE annotations.

Like the rest of the viz module, every function here saves into the
centralized figures directory and never calls `plt.show()` — that keeps the
package usable from non-interactive contexts (CI, batch jobs).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from lake.training.evaluate import regression_metrics


def _annotate(ax, ytrue: np.ndarray, ypred: np.ndarray) -> None:
    """Add an R^2/MAE/RMSE legend in the top-left of the axes."""
    r2, mae, rmse = regression_metrics(ytrue, ypred)
    label = f"R^2 = {r2:.2f}\nMAE = {mae:.2f}\nRMSE = {rmse:.2f}"
    ax.legend(title=label, loc="upper left", frameon=True)


def save_pred_vs_true_scatter(
    splits: Iterable[Tuple[str, np.ndarray, np.ndarray]],
    fig_path: str | Path,
) -> Path:
    """Save a multi-panel scatter plot (one panel per split).

    Parameters
    ----------
    splits
        Iterable of `(split_name, ytrue, ypred)` tuples — typically
        ``[("train", ...), ("valid", ...), ("test", ...)]``.
    fig_path
        Where to write the PNG.
    """
    splits = list(splits)
    if not splits:
        raise ValueError("Need at least one split to plot.")

    fig_path = Path(fig_path)
    fig_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, len(splits), figsize=(6 * len(splits), 6))
    if len(splits) == 1:
        axes = [axes]  # make iterable for the loop below

    for ax, (name, ytrue, ypred) in zip(axes, splits):
        sns.scatterplot(x=ytrue, y=ypred, ax=ax, alpha=0.4, s=12)

        # The perfect-prediction diagonal — uses the larger of the two ranges.
        hi = float(max(ytrue.max(), ypred.max()))
        ax.plot([0, hi], [0, hi], "--r", linewidth=1.5)

        ax.set_xlabel(f"Measured ({name})", fontsize=12)
        ax.set_ylabel("Predicted" if ax is axes[0] else "")
        _annotate(ax, ytrue, ypred)

    fig.tight_layout()
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)
    return fig_path
