"""Small RealNVP normalizing flow for muon states (CPU, torch).

The flow maps a standardised feature vector ``x`` to ``z ~ N(0, I)`` through affine coupling layers with alternating
masks (Dinh et al., "Density estimation using Real NVP", 2017). It is deliberately small: the demo loop trains it in
seconds on a few hundred rows.

Notes
-----
Features are standardised with the training mean and standard deviation inside the flow. ``log_prob`` includes the
Jacobian of that standardisation, so it is a density in the original feature space and two flows fitted on
different data can be compared point by point (needed for NLL across splits and for importance weights q0/qg).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class _Coupling(nn.Module):
    def __init__(self, dim: int, mask: torch.Tensor, hidden: int):
        super().__init__()
        self.register_buffer("mask", mask)
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 2 * dim))

    def _st(self, kept: torch.Tensor):
        s, t = self.net(kept).chunk(2, dim=-1)
        free = 1 - self.mask
        return torch.tanh(s) * free, t * free

    def forward(self, x):  # x -> z, log|det dz/dx|
        s, t = self._st(x * self.mask)
        return x * self.mask + (1 - self.mask) * (x * torch.exp(s) + t), s.sum(-1)

    def inverse(self, z):
        s, t = self._st(z * self.mask)
        return z * self.mask + (1 - self.mask) * ((z - t) * torch.exp(-s))


class Flow(nn.Module):
    """RealNVP on ``dim`` features.

    Parameters
    ----------
    dim : int
        Number of features.
    layers : int
        Number of coupling layers; masks alternate between even and odd features.
    hidden : int
        Width of each coupling network.
    """

    def __init__(self, dim: int, layers: int = 6, hidden: int = 64):
        super().__init__()
        masks = [torch.tensor([(i + k) % 2 for i in range(dim)], dtype=torch.float32) for k in range(layers)]
        self.layers = nn.ModuleList(_Coupling(dim, m, hidden) for m in masks)
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("std", torch.ones(dim))

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        z, logdet = (x - self.mean) / self.std, torch.zeros(len(x))
        for layer in self.layers:
            z, ld = layer(z)
            logdet = logdet + ld
        return -0.5 * (z ** 2).sum(-1) - 0.5 * z.shape[1] * np.log(2 * np.pi) + logdet - torch.log(self.std).sum()

    @torch.no_grad()
    def sample(self, n: int, seed: int) -> np.ndarray:
        z = torch.randn(n, len(self.mean), generator=torch.Generator().manual_seed(seed))
        for layer in reversed(self.layers):
            z = layer.inverse(z)
        return (z * self.std + self.mean).numpy().astype(np.float64)


def fit(x: np.ndarray, seed: int, epochs: int = 400, lr: float = 3e-3) -> tuple[Flow, list[float]]:
    """Fit a :class:`Flow` to ``x`` (rows = samples) by maximum likelihood, full batch.

    Returns
    -------
    flow : Flow
    losses : list of float
        Mean negative log-likelihood per epoch, for the run report.
    """
    torch.manual_seed(seed)
    xt = torch.as_tensor(x, dtype=torch.float32)
    flow = Flow(x.shape[1])
    flow.mean.copy_(xt.mean(0))
    flow.std.copy_(xt.std(0).clamp_min(1e-6))
    opt = torch.optim.Adam(flow.parameters(), lr=lr)
    losses = []
    for _ in range(epochs):
        opt.zero_grad()
        loss = -flow.log_prob(xt).mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return flow, losses
