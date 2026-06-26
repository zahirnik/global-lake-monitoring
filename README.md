# Global Lake Monitoring

A Python package for predicting **lake surface area** from environmental
features (land cover, climate, hydrology) and explaining the predictions with
**SHAP** values. Built around a 1D Transformer regressor with XGBoost and
Random Forest baselines, applied to a global dataset of ~2,700 lakes.

> Importable as `import lake`. CLI exposed as `lake train …` and
> `lake explain …`.

---

## Pipeline walk-through

The following figures step through the project: data overview → EDA →
baseline models → global interpretability → stratified interpretability.

### 1 — Lake coverage across Köppen-Geiger climate classes

Of the ~2,700 lakes in the global panel, the majority sit in cold,
continental climates (`Dfb`, `Dfc`), with sparser representation in tropical
and polar regimes.

![lakes by climate](docs/figures/01_lakes_by_climate.png)

### 2 — Lake coverage across latitude bands

Mid-to-high northern latitudes dominate, reflecting where most monitored
freshwater lakes physically are.

![lakes by latitude band](docs/figures/02_lakes_by_latitude.png)

### 3 — Feature pair-plot (EDA)

Joint and marginal distributions of the environmental features used to
predict lake area — useful for spotting correlations and outlier behaviour
before modelling.

![EDA pair plot](docs/figures/03_eda_pairplot.png)

### 4 — Baseline: Random Forest

Measured vs. predicted lake area for the Random Forest baseline (train left,
test right). Strong fit on training; non-trivial residual structure on test.

![Random Forest scatter](docs/figures/04_scatter_random_forest.png)

### 5 — Baseline: XGBoost

The XGBoost baseline tightens training fit further (R² ≈ 1.0) and keeps a
held-out R² of ~0.85 on the temporal test split.

![XGBoost scatter](docs/figures/05_scatter_xgboost.png)

### 6 — Permutation feature importance

Permutation importance computed across 83 representative lakes for the RF
regressor — gives a model-agnostic ranking of which inputs matter most.

![permutation importance](docs/figures/06_permutation_importance.png)

### 7 — Global SHAP feature importance (mean |SHAP|)

Mean absolute SHAP value per feature. Forest cover, grassland, and human
population dominate the model's predictions of lake surface area.

![SHAP mean importance bar](docs/figures/07_shap_mean_importance.png)

### 8 — SHAP beeswarm: per-sample contributions

Each dot is one (lake, year). Horizontal position is the SHAP value (impact
on the prediction); colour is the feature value. Shows direction-of-effect
as well as magnitude.

![SHAP beeswarm](docs/figures/08_shap_beeswarm.png)

### 9 — SHAP waterfall: a single prediction

Decomposition of one prediction into per-feature contributions. Reads from
the model's baseline expectation (bottom) up to the final prediction (top).

![SHAP waterfall](docs/figures/09_shap_waterfall_single.png)

### 10 — SHAP dependence plot

Interaction view: SHAP value of `GRASS` vs. the `GRASS` feature value,
coloured by `Population`. Reveals non-linear regimes and where features
co-act.

![SHAP dependence plot](docs/figures/10_shap_dependence.png)

### 11 — SHAP distributions stratified by climate

Kernel density of SHAP values per feature, coloured by Köppen-Geiger climate
group. Shows the model relies on different drivers in different climate
regimes.

![SHAP by climate](docs/figures/11_shap_by_climate.png)

### 12 — SHAP distributions stratified by latitude band

Same idea as (11) but stratified geographically rather than climatically —
useful for sanity-checking that the model isn't latching onto purely
spatial confounds.

![SHAP by latitude](docs/figures/12_shap_by_latitude.png)

> All 92 raw figures from the original analysis notebooks are archived
> under [`outputs/figures/`](outputs/figures), grouped by notebook.

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

## Design notes

* **Importable package** — `from lake.training import train_transformer`.
* **Single `LakeConfig`** dataclass for all paths and hyperparameters,
  loadable from YAML.
* **Central `outputs/figures/`** — every figure produced by the package ends
  up in one folder, never via `plt.show()`, so it runs headlessly too.
* **CLI** with subcommands: `lake train`, `lake explain lakewise`,
  `lake explain climatewise`, `lake explain tree`.
* **Smoke tests** under `tests/` run on synthetic data and don't need the real
  CSV to pass.

---

## License

Not yet specified. Add a `LICENSE` file (e.g. MIT) before publishing.
