"""Plotting helpers — every function saves its figure under outputs/figures/."""

from lake.viz.loss_curves import save_loss_curve
from lake.viz.scatter import save_pred_vs_true_scatter
from lake.viz.shap_plots import save_shap_summary_plot

__all__ = [
    "save_loss_curve",
    "save_pred_vs_true_scatter",
    "save_shap_summary_plot",
]
