"""
Inference, metrics, and scatter-plot helpers.

Replaces the original `src/inference.py`. Two notable changes:
* No hard-coded `./EXPs/` paths — the caller supplies a `predictions_dir`.
* Predictions are un-scaled (multiplied back by `TARGET_RESCALE`) here
  rather than inside the model loop, keeping training code single-purpose.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader as TorchDataLoader

from lake.config import TARGET_RESCALE


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def regression_metrics(ytrue: np.ndarray, ypred: np.ndarray) -> Tuple[float, float, float]:
    """Return (R^2, MAE, RMSE) for a regression prediction."""
    r2 = r2_score(ytrue, ypred)
    mae = mean_absolute_error(ytrue, ypred)
    rmse = float(np.sqrt(mean_squared_error(ytrue, ypred)))
    return r2, mae, rmse


# ---------------------------------------------------------------------------
# Per-split prediction
# ---------------------------------------------------------------------------

def _predict_split(model: torch.nn.Module, loader: TorchDataLoader, device: str) -> pd.DataFrame:
    """Run a single split through the model and return a long DataFrame."""
    rows: List[Dict[str, np.ndarray]] = []
    model.eval()
    with torch.no_grad():
        for sample in loader:
            X = sample["X"].to(device)
            y_true = sample["y"].numpy()
            y_pred = model(X).squeeze(-1).detach().cpu().numpy()

            rows.append({
                "ID": np.asarray(sample["ID"]).flatten(),
                "year": np.asarray(sample["year"]).flatten(),
                # Un-scale to original units (km^2).
                "ytrue": (y_true * TARGET_RESCALE).flatten(),
                "ypred": (y_pred * TARGET_RESCALE).flatten(),
            })

    # Concatenate the per-batch arrays into one DataFrame.
    return pd.DataFrame({k: np.concatenate([r[k] for r in rows]) for k in rows[0]})


def predict_with_model(
    model: torch.nn.Module,
    ckpt_path: str | Path,
    loaders: Dict[str, TorchDataLoader],
    predictions_dir: str | Path,
    exp_name: str,
) -> Dict[str, Path]:
    """Load `ckpt_path` into `model` and write per-split prediction CSVs.

    Parameters
    ----------
    model
        An *instantiated* model with the same architecture as the checkpoint.
    ckpt_path
        Path to the `.pth` file saved during training.
    loaders
        Dict of `{split_name: DataLoader}` — usually train/valid/test.
    predictions_dir
        Where to write the `<split>_<exp_name>.csv` files.
    exp_name
        Used to namespace output files (so multiple experiments coexist).
    """
    predictions_dir = Path(predictions_dir)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    # Load the best weights into the existing model object.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)

    output_paths: Dict[str, Path] = {}
    for split_name, loader in loaders.items():
        df = _predict_split(model, loader, device)
        out_path = predictions_dir / f"{split_name}_{exp_name}.csv"
        df.to_csv(out_path, index=False)
        output_paths[split_name] = out_path

        # Print quick metrics for at-a-glance sanity checks.
        r2, mae, rmse = regression_metrics(df["ytrue"].values, df["ypred"].values)
        print(f"[{split_name}] R^2={r2:.3f} MAE={mae:.2f} RMSE={rmse:.2f} -> {out_path}")

    return output_paths
