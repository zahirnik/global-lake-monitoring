"""
Command-line interface for the lake package.

Once installed (`pip install -e .`), this is the entry point exposed by the
`lake` console script. The subcommands wire together the rest of the package
so the user can train, predict, and explain without writing any Python.

Examples
--------
::

    # Train a Transformer with the default config.
    lake train

    # Train with a custom YAML config.
    lake train --config configs/transformer.yaml

    # Run per-lake SHAP analysis on the Lake_Density dataset.
    lake explain lakewise

    # Run per-climate SHAP analysis.
    lake explain climatewise
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd

from lake.config import LakeConfig, default_config
from lake.explain.climatewise import climate_wise_analysis
from lake.explain.lakewise import lake_wise_analysis
from lake.explain.shap_utils import save_shap_table, shap_explain_tree
from lake.models.tree import train_tree_regressor
from lake.training.train import train_transformer


# ---------------------------------------------------------------------------
# Config loading shared across subcommands
# ---------------------------------------------------------------------------

def _load_config(yaml_path: Optional[str], exp_name: Optional[str]) -> LakeConfig:
    """Construct a LakeConfig from either a YAML path or the defaults."""
    if yaml_path:
        cfg = LakeConfig.from_yaml(yaml_path)
    else:
        cfg = default_config()
    # Let the CLI override the experiment name without needing a separate YAML.
    if exp_name:
        cfg.exp_name = exp_name
    return cfg


# ---------------------------------------------------------------------------
# `lake train`
# ---------------------------------------------------------------------------

def _cmd_train(args: argparse.Namespace) -> int:
    """Train the Transformer regressor end-to-end."""
    cfg = _load_config(args.config, args.exp_name)

    # CLI overrides for the most common hyperparameters. These are optional;
    # if absent we keep whatever came from the YAML/defaults.
    if args.epochs is not None:
        cfg.training.epochs = args.epochs
    if args.batch_size is not None:
        cfg.training.batch_size = args.batch_size
    if args.lr is not None:
        cfg.training.learning_rate = args.lr
    if args.debug_rows is not None:
        cfg.training.debug_row_limit = args.debug_rows

    print(f"[train] exp_name={cfg.exp_name}")
    print(f"[train] data_csv={cfg.paths.data_csv}")
    print(f"[train] figures will land in {cfg.paths.figures_dir}")

    ckpt = train_transformer(cfg)
    print(f"[train] best checkpoint -> {ckpt}")
    return 0


# ---------------------------------------------------------------------------
# `lake explain ...`
# ---------------------------------------------------------------------------

def _cmd_explain_lakewise(args: argparse.Namespace) -> int:
    """Run the per-lake explainability analysis."""
    cfg = _load_config(args.config, args.exp_name)
    cfg.paths.ensure()

    # The lake-density CSV is the input for this analysis.
    df = pd.read_csv(cfg.paths.lake_density_csv)
    out_dir = cfg.paths.shap_dir / "lakewise"
    paths = lake_wise_analysis(df, output_dir=out_dir)

    for name, p in paths.items():
        print(f"[explain:lakewise] {name:13s} -> {p}")
    return 0


def _cmd_explain_climatewise(args: argparse.Namespace) -> int:
    """Run the per-climate explainability analysis."""
    cfg = _load_config(args.config, args.exp_name)
    cfg.paths.ensure()

    df = pd.read_csv(cfg.paths.lake_density_csv)
    out_dir = cfg.paths.shap_dir / "climatewise"
    results = climate_wise_analysis(df, output_dir=out_dir)

    for climate, paths in results.items():
        print(f"[explain:climatewise] {climate}: {len(paths)} CSVs in {out_dir / climate}")
    return 0


def _cmd_explain_tree(args: argparse.Namespace) -> int:
    """Fit a tree regressor on the global dataset and dump SHAP values."""
    cfg = _load_config(args.config, args.exp_name)
    cfg.paths.ensure()

    # Lazy imports to avoid loading torch/etc. when only this subcommand is used.
    from lake.data.dataset import load_dataset, normalize, split_timeseries_by_lake
    from lake.config import FEATURE_COLUMNS

    df = load_dataset(cfg.paths.data_csv, debug_row_limit=cfg.training.debug_row_limit)
    splits = split_timeseries_by_lake(df)
    splits_s, _ = normalize(splits)

    # Combine train+valid for training (matches the original notebook).
    Xtr = splits_s.train.X
    ytr = splits_s.train.y
    Xte = splits_s.test.X
    yte = splits_s.test.y

    _, _, fitted = train_tree_regressor(Xtr, ytr, Xte, yte, kind=args.kind)

    shap_values = shap_explain_tree(fitted, Xte, num_iter=args.num_iter)
    out_csv = cfg.paths.shap_dir / f"global_{args.kind.lower()}_{cfg.exp_name}.csv"
    save_shap_table(
        shap_values,
        feature_names=FEATURE_COLUMNS,
        output_csv=out_csv,
        extra_columns={
            "ID": splits_s.test.ids[:, 0],
            "year": splits_s.test.ids[:, 1],
            "ytrue": yte,
        },
    )
    print(f"[explain:tree] SHAP table -> {out_csv}")
    return 0


# ---------------------------------------------------------------------------
# Argument parser construction
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lake",
        description="Lake-area regression and explainability.",
    )

    # Global flags. We attach them to every subcommand below for convenience.
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to a YAML config file (optional; defaults are used otherwise).",
    )
    parser.add_argument(
        "--exp-name", type=str, default=None,
        help="Override the experiment name (used in output filenames).",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---- train ----
    p_train = subparsers.add_parser("train", help="Train the Transformer regressor.")
    p_train.add_argument("--epochs", type=int, default=None)
    p_train.add_argument("--batch-size", type=int, default=None)
    p_train.add_argument("--lr", type=float, default=None)
    p_train.add_argument(
        "--debug-rows", type=int, default=None,
        help="Cap the dataset to N rows for fast iteration (omit for the full dataset).",
    )
    p_train.set_defaults(func=_cmd_train)

    # ---- explain ----
    p_explain = subparsers.add_parser("explain", help="SHAP / importance analyses.")
    explain_sub = p_explain.add_subparsers(dest="explain_command", required=True)

    p_lake = explain_sub.add_parser("lakewise", help="Per-lake Ridge/RF/SHAP analysis.")
    p_lake.set_defaults(func=_cmd_explain_lakewise)

    p_climate = explain_sub.add_parser("climatewise", help="Per-climate-class analysis.")
    p_climate.set_defaults(func=_cmd_explain_climatewise)

    p_tree = explain_sub.add_parser("tree", help="Global tree-based SHAP analysis.")
    p_tree.add_argument("--kind", choices=["RF", "GB", "XGB"], default="XGB")
    p_tree.add_argument("--num-iter", type=int, default=10)
    p_tree.set_defaults(func=_cmd_explain_tree)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """Entry point referenced by `pyproject.toml` (project.scripts.lake)."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
