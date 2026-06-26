"""
Loss-curve plotting & CSV serialization.

Single concern: turn a `{"train": [...], "val": [...]}` dict into one CSV plus
one PNG. The PNG lives in `outputs/figures/` so every figure produced by the
package ends up in the same place.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def save_loss_curve(
    loss_stats: Dict[str, List[float]],
    csv_path: str | Path,
    fig_path: str | Path,
) -> None:
    """Persist train/val loss history to CSV and PNG.

    Parameters
    ----------
    loss_stats
        Dict with at least the keys "train" and "val", each mapping to a list
        of per-epoch mean losses.
    csv_path
        Output CSV path (in long format: columns epochs/variable/value).
    fig_path
        Output PNG path. Will be created with `dpi=200`.
    """
    csv_path = Path(csv_path)
    fig_path = Path(fig_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fig_path.parent.mkdir(parents=True, exist_ok=True)

    # Long-format DataFrame is the easiest input for seaborn.lineplot.
    df = (
        pd.DataFrame.from_dict(loss_stats)
        .reset_index()
        .melt(id_vars=["index"])
        .rename(columns={"index": "epoch"})
    )
    df.to_csv(csv_path, index=False)

    # Render the curve. We close the figure explicitly so the script doesn't
    # leak figures when called many times in a notebook.
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=df, x="epoch", y="value", hue="variable", ax=ax)
    ax.set_title("Train / Validation loss per epoch")
    ax.set_ylim(0, df["value"].max() * 1.05 if len(df) else 1.0)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)
