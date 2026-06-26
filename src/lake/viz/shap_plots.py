"""
SHAP summary plots saved as PNGs into outputs/figures/.

We wrap `shap.summary_plot` with a `plt.savefig`/`plt.close` pair so the
caller never has to remember to manage the figure manually.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import shap


def save_shap_summary_plot(
    shap_values: np.ndarray,
    X: np.ndarray,
    feature_names: Sequence[str],
    fig_path: str | Path,
    plot_type: str | None = None,
) -> Path:
    """Save a SHAP summary plot (beeswarm by default; bar if `plot_type='bar'`).

    Parameters
    ----------
    shap_values, X
        Standard `shap.summary_plot` inputs.
    feature_names
        Column labels for `X`.
    fig_path
        Output PNG path.
    plot_type
        ``None`` (default beeswarm) or ``"bar"``.
    """
    fig_path = Path(fig_path)
    fig_path.parent.mkdir(parents=True, exist_ok=True)

    # `shap.summary_plot` draws on the current figure. We set `show=False`,
    # save explicitly, and close to avoid leaking figures in long sessions.
    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        shap_values,
        X,
        feature_names=list(feature_names),
        plot_type=plot_type,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close("all")
    return fig_path
