"""
1D Transformer regressor for lake-area prediction.

Architecture
------------
We treat the 11 environmental features as a *length-1 sequence of 11-dim
vectors* and feed it through a small TransformerEncoder stack. The encoder
output is mean-pooled and projected to a single scalar (the predicted lake
area, in the rescaled units defined by `TARGET_RESCALE`).

This is a faithful refactor of `lake-main/src/models.py:Transformer1d`, with
two improvements:
* Number of encoder layers comes from the config (previously hard-coded to 6).
* The forward pass docstring explicitly states the expected tensor shapes,
  which the original lacked.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from lake.config import TransformerConfig


class Transformer1d(nn.Module):
    """1D Transformer regressor.

    Input
    -----
    x : Tensor of shape (batch, 1, dim_model)
        A length-1 token sequence per sample. The leading "1" is the sequence
        dimension; if you reshape feature vectors yourself, keep it.

    Output
    ------
    y : Tensor of shape (batch, num_class)
        For regression, `num_class == 1`.
    """

    def __init__(self, cfg: TransformerConfig):
        super().__init__()

        # Stash the parts of cfg we actually use — makes `repr(model)` and
        # state-dict reloading more debuggable.
        self.d_model = cfg.dim_model
        self.nhead = cfg.num_heads
        self.dim_feedforward = cfg.mlp_dim
        self.dropout = cfg.dropout_rate
        self.activation = cfg.activation
        self.n_classes = cfg.num_class

        # Standard PyTorch encoder layer; we feed it a *sequence-first*
        # tensor below (shape (seq, batch, feat)) — matches the default
        # `batch_first=False` convention.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation=self.activation,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=cfg.num_layers
        )

        # Final projection to a scalar (for regression `n_classes == 1`).
        self.dense = nn.Linear(self.d_model, self.n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Incoming x is (batch, seq, feat). PyTorch's default Transformer
        # expects (seq, batch, feat) — permute to match.
        x = x.permute(1, 0, 2)

        # Encode then mean-pool across the sequence dimension. With seq=1
        # the mean is a no-op, but the same code works if downstream callers
        # ever pass multi-token sequences.
        out = self.transformer_encoder(x)
        out = out.mean(dim=0)

        # Final linear projection to the regression target.
        return self.dense(out)
