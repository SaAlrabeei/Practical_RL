"""Shared FCNet used by all solvers."""

import torch
import torch.nn as nn


class FCNet(nn.Module):
    """
    Fully-connected network.
    layer_sizes: e.g. [2, 64, 64, 64, 64, 1]
    activation:  'tanh' | 'relu' | 'swish' | 'gelu'
    """

    _ACT = {
        'tanh':    nn.Tanh,
        'relu':    nn.ReLU,
        'swish':   nn.SiLU,
        'gelu':    nn.GELU,
    }

    def __init__(self, layer_sizes: list, activation: str = 'tanh'):
        super().__init__()
        act_cls = self._ACT[activation]
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                layers.append(act_cls())
        self.net = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FourierFCNet(nn.Module):
    """
    Fourier-feature network for multi-scale PDEs.

    Input x is first encoded as:
        γ(x) = [sin(B·x), cos(B·x)]   where B ~ N(0, σ²) is FIXED (not trained).

    Then a standard FCNet maps γ(x) → output.

    The random Fourier projection lifts the input into a high-frequency basis,
    breaking the low-frequency spectral bias that standard MLPs exhibit.

    Reference: Tancik et al. NeurIPS 2020; Wang et al. CMAME 2021.
    """

    def __init__(self, in_dim: int, out_dim: int,
                 fourier_dim: int = 64,
                 sigma: float = 1.0,
                 hidden: list = None,
                 activation: str = 'tanh'):
        super().__init__()
        if hidden is None:
            hidden = [128, 128, 128, 128]
        B = torch.randn(in_dim, fourier_dim) * sigma
        self.register_buffer('B', B)
        self.net = FCNet([2 * fourier_dim] + hidden + [out_dim], activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = x @ self.B
        feat = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
        return self.net(feat)
