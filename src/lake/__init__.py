"""
lake
====

A clean Python package for predicting lake surface area from environmental
features (land cover, climate, hydrology) and explaining the predictions with
SHAP values.

Public surface
--------------
The most useful entry points are re-exported here so that downstream callers
can do `from lake import train_transformer, predict_with_model` instead of
walking the internal module tree.
"""

from importlib.metadata import PackageNotFoundError, version as _pkg_version

# Surface the installed version (falls back to "0.0.0" when running from a
# source checkout that has not been installed yet).
try:
    __version__ = _pkg_version("lake")
except PackageNotFoundError:  # pragma: no cover - only hit in dev checkouts
    __version__ = "0.0.0"

# Re-export the high-level helpers. We keep this list small on purpose:
# anything not exposed here is considered internal.
from lake.config import LakeConfig, default_config  # noqa: E402
from lake.training.train import train_transformer  # noqa: E402
from lake.training.evaluate import predict_with_model  # noqa: E402
from lake.explain.shap_utils import shap_explain_tree  # noqa: E402

__all__ = [
    "__version__",
    "LakeConfig",
    "default_config",
    "train_transformer",
    "predict_with_model",
    "shap_explain_tree",
]
