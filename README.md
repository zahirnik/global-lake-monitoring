# Global Lake Monitoring

A clean Python package for predicting **lake surface area** from environmental
features (land cover, climate, hydrology) and explaining the predictions with
**SHAP** values.

> Importable as `import lake` — the Python package keeps the short name; the
> repository name reflects the project's intent.

This is a refactor of the original [`lake-main`](../lake-main) notebook-style
project into an installable package with a CLI, type-checked configs, and a
single output directory for all generated artefacts.

---

## Install

```bash
cd lake
pip install -e .          # editable install
pip install -e ".[dev]"   # includes pytest + jupyter
```

After install you get a `lake` console script.

---

## Project layout

```
lake/
├── pyproject.toml          # build config + dependencies + CLI entry point
├── configs/
│   └── transformer.yaml    # example training config
├── src/lake/
│   ├── config.py           # LakeConfig dataclass, paths, hyperparameters
│   ├── data/
│   │   └── dataset.py      # per-lake temporal split, scaler, PyTorch dataset
│   ├── models/
│   │   ├── transformer.py  # Transformer1d
│   │   └── tree.py         # RF / GB / XGB factory + training helper
│   ├── training/
│   │   ├── train.py        # train_transformer(), EarlyStopping
│   │   └── evaluate.py     # predict_with_model(), regression_metrics()
│   ├── explain/
│   │   ├── shap_utils.py   # averaged TreeExplainer SHAP
│   │   ├── lakewise.py     # per-lake Ridge/RF/SHAP/permutation
│   │   └── climatewise.py  # per-climate analysis (fixes bug in original)
│   ├── viz/
│   │   ├── loss_curves.py  # train/val loss PNG
│   │   ├── scatter.py      # measured-vs-predicted scatter plots
│   │   └── shap_plots.py   # SHAP summary / bar plots
│   └── cli.py              # `lake train`, `lake explain ...`
├── notebooks/
│   └── demo.ipynb          # thin demo that imports the package
└── outputs/                # everything generated lands here
    ├── figures/            # *** all PNGs *** (loss curves, scatter, SHAP)
    ├── checkpoints/        # best_*.pth files
    ├── predictions/        # train/valid/test prediction CSVs
    └── shap/               # SHAP values + per-lake / per-climate tables
```

---

## Usage

### CLI

```bash
# Train with defaults.
lake train

# Train with a YAML config and override the epoch count.
lake train --config configs/transformer.yaml --epochs 50

# Run per-lake Ridge/RF/SHAP analysis.
lake explain lakewise

# Run per-climate-class analysis.
lake explain climatewise

# Global tree-based SHAP analysis (XGBoost by default).
lake explain tree --kind XGB
```

### Library

```python
from lake import default_config, train_transformer

cfg = default_config(exp_name="my_run")
cfg.training.epochs = 50
cfg.training.debug_row_limit = 1000   # fast iteration
ckpt_path = train_transformer(cfg)
```

### Notebook

See [`notebooks/demo.ipynb`](notebooks/demo.ipynb) — it imports the package
and reproduces the original SHAP plots in <10 lines of code.

---

## Where do figures go?

**All** generated PNGs land under `outputs/figures/`:

| File                              | Produced by                       |
| --------------------------------- | --------------------------------- |
| `loss_<exp_name>.png`             | `lake.viz.save_loss_curve`        |
| `scatter_<exp_name>.png`          | `lake.viz.save_pred_vs_true_scatter` |
| `shap_summary_<exp_name>.png`     | `lake.viz.save_shap_summary_plot` |

No function in the package calls `plt.show()` — they all write to disk
explicitly, so the package works fine in headless / CI environments.

---

## Notable improvements vs. the original

* **Importable package** (`from lake.training import train_transformer`)
  instead of `from src.models import ...` (which required cwd to be the
  project root).
* **Single `LakeConfig`** for all paths and hyperparameters; no more magic
  constants spread across files. Loadable from YAML.
* **Central `outputs/figures/`** — every figure ends up in one folder.
* **CLI** with subcommands (`lake train`, `lake explain lakewise`, ...).
* **Bug fix:** `climate_wise_analysis` actually runs the per-climate analysis
  now (the legacy `__main__` accidentally called the lake-wise function).
* **Bug fix:** the debug 500-row cap in `train.py` is opt-in (`debug_row_limit`).
* **Better separation of concerns** — viz utilities can't crash the training
  loop, training code doesn't know about plotting libraries, etc.

---

## License

Not yet specified. Add a `LICENSE` file (e.g. MIT) before publishing.
