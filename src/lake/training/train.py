"""
Training loop for the Transformer regressor.

This module is the package equivalent of the original `train.py` script, but:
* All paths/hyperparameters come from :class:`LakeConfig` (no hardcoding).
* The 500-row debug cap is gone (opt-in via `cfg.training.debug_row_limit`).
* Loss curves land in the central `outputs/figures/` folder.
* `train_transformer()` is importable from notebooks and tests, and is also
  invoked by the CLI (`lake train`).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader as TorchDataLoader

from lake.config import LakeConfig
from lake.data.dataset import LakeTorchDataset, load_dataset, normalize, split_timeseries_by_lake
from lake.models.transformer import Transformer1d
from lake.training.evaluate import predict_with_model
from lake.viz.loss_curves import save_loss_curve


def _device() -> str:
    """Return 'cuda' if available, else 'cpu'. Keeps the call sites readable."""
    return "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Early stopping callback
# ---------------------------------------------------------------------------

class EarlyStopping:
    """Stops training when validation loss stalls.

    Parameters
    ----------
    patience
        Number of consecutive epochs without improvement before stopping.
    min_delta
        Improvements smaller than this don't reset the counter.
    """

    def __init__(self, patience: int = 30, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def step(self, improved: bool) -> None:
        # `improved` is True when the latest val loss beat the best-so-far.
        if improved:
            self.counter = 0
        else:
            self.counter += 1
        if self.counter >= self.patience:
            self.early_stop = True


# ---------------------------------------------------------------------------
# The training entrypoint
# ---------------------------------------------------------------------------

def train_transformer(cfg: LakeConfig) -> Path:
    """Train the Transformer regressor end-to-end.

    Returns
    -------
    Path
        The path to the best checkpoint that was saved during training.
    """
    # Make sure all output dirs exist before any I/O happens.
    cfg.paths.ensure()

    # ------------------------------------------------------------------
    # 1. Data
    # ------------------------------------------------------------------
    df = load_dataset(cfg.paths.data_csv, debug_row_limit=cfg.training.debug_row_limit)
    splits = split_timeseries_by_lake(df)
    splits_scaled, _scaler = normalize(splits)
    print(
        f"[data] train={splits_scaled.train.X.shape} "
        f"valid={splits_scaled.valid.X.shape} "
        f"test={splits_scaled.test.X.shape}"
    )

    train_ds = LakeTorchDataset(splits_scaled.train)
    valid_ds = LakeTorchDataset(splits_scaled.valid)
    test_ds = LakeTorchDataset(splits_scaled.test)

    train_loader = TorchDataLoader(
        train_ds, batch_size=cfg.training.batch_size,
        shuffle=True, num_workers=cfg.training.num_workers,
    )
    valid_loader = TorchDataLoader(
        valid_ds, batch_size=cfg.training.batch_size,
        shuffle=False, num_workers=cfg.training.num_workers,
    )
    test_loader = TorchDataLoader(
        test_ds, batch_size=cfg.training.batch_size,
        shuffle=False, num_workers=cfg.training.num_workers,
    )

    # ------------------------------------------------------------------
    # 2. Model, loss, optimizer
    # ------------------------------------------------------------------
    device = _device()
    model = Transformer1d(cfg.transformer).to(device)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )

    early = EarlyStopping(
        patience=cfg.training.early_stop_patience,
        min_delta=cfg.training.early_stop_min_delta,
    )

    # Where the best checkpoint lives. Naming includes `exp_name` so multiple
    # experiments don't clobber each other.
    ckpt_path: Path = cfg.paths.checkpoints_dir / f"best_{cfg.exp_name}.pth"

    # ------------------------------------------------------------------
    # 3. Training loop
    # ------------------------------------------------------------------
    loss_stats: Dict[str, List[float]] = {"train": [], "val": []}
    # Float('inf') is a safer sentinel than the original `100000`.
    best_val_loss = float("inf")

    for epoch in range(1, cfg.training.epochs + 1):
        epoch_start = time.time()

        # ---- TRAIN ----
        model.train()
        train_epoch_loss = 0.0
        for sample in train_loader:
            X = sample["X"].to(device)
            y = sample["y"].to(device)

            optimizer.zero_grad()
            y_pred = model(X).squeeze(-1)
            loss = loss_fn(y, y_pred)
            loss.backward()
            optimizer.step()
            train_epoch_loss += loss.item()

        # ---- VALIDATE ----
        model.eval()
        val_epoch_loss = 0.0
        with torch.no_grad():
            for sample in valid_loader:
                X = sample["X"].to(device)
                y = sample["y"].to(device)
                y_pred = model(X).squeeze(-1)
                val_epoch_loss += loss_fn(y, y_pred).item()

        # Per-epoch mean losses for the loss-curve plot.
        train_epoch_loss /= max(len(train_loader), 1)
        val_epoch_loss /= max(len(valid_loader), 1)
        loss_stats["train"].append(train_epoch_loss)
        loss_stats["val"].append(val_epoch_loss)

        # ---- BOOK-KEEPING & CHECKPOINTING ----
        improved = val_epoch_loss < best_val_loss
        if improved:
            best_val_loss = val_epoch_loss
            torch.save(model.state_dict(), ckpt_path)

        early.step(improved)

        elapsed = time.time() - epoch_start
        print(
            f"epoch {epoch:03d} | {elapsed:5.1f}s | "
            f"train {train_epoch_loss:8.4f} | val {val_epoch_loss:8.4f} | "
            f"{'*saved*' if improved else 'no improvement'}"
        )

        if early.early_stop:
            print(f"[early stop] no improvement for {early.patience} epochs")
            break

    # ------------------------------------------------------------------
    # 4. Post-training artefacts
    # ------------------------------------------------------------------
    # Loss curve PNG goes into the central figures folder.
    fig_path = cfg.paths.figures_dir / f"loss_{cfg.exp_name}.png"
    csv_path = cfg.paths.outputs_dir / f"loss_{cfg.exp_name}.csv"
    save_loss_curve(loss_stats, csv_path=csv_path, fig_path=fig_path)

    # Reload the best checkpoint and write predictions for all three splits.
    predict_with_model(
        model=model,
        ckpt_path=ckpt_path,
        loaders={"train": train_loader, "valid": valid_loader, "test": test_loader},
        predictions_dir=cfg.paths.predictions_dir,
        exp_name=cfg.exp_name,
    )

    return ckpt_path
