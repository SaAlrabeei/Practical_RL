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
