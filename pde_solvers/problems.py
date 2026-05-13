"""
PDE problem definitions  —  1-D and 2-D Poisson, 1-D Convection-Diffusion.

Every problem exposes:
  .dim          : int (1 or 2)
  .lb / .ub     : domain bounds (scalar)
  .eps          : diffusion coefficient   (default 1.0)
  .beta         : convection coefficient  (default 0.0 = pure diffusion)
  .name         : str
  .exact_u(x)   : exact solution        (N, 1)
  .source_f(x)  : RHS                   (N, 1)
  .mollifier(x) : boundary-vanishing fn  (N, 1)  used by solvers
  .interior_points(n)
  .boundary_points()
  .test_grid(n)
  .l2_rel(u_pred, u_true)

1-D Poisson  —  -u'' = f  on [0,1],  u(0)=u(1)=0
────────────────────────────────────────────────────────
  Level 1  SmoothPoisson1D          u = sin(πx)
  Level 2  LayerPoisson1D (k=15)    u = sin(πx)·tanh(k(x-½))
  Level 3  OscPoisson1D   (n=8)     u = sin(nπx)

1-D Convection-Diffusion  —  -ε·u'' + β·u' = f  on [0,1],  u(0)=u(1)=0
────────────────────────────────────────────────────────────────────────
  u = sin(πx)·(1 − e^((x−1)/ε))   — boundary layer of width ~ε at x=1
  Péclet number  Pe = β/ε  controls sharpness of the layer.
"""

import numpy as np
import torch


# ════════════════════════════════════════════════════════════════
# Base classes
# ════════════════════════════════════════════════════════════════

class PDE1D:
    """Abstract 1-D PDE on [0, 1] with homogeneous Dirichlet BC."""

    dim:  int   = 1
    lb:   float = 0.0
    ub:   float = 1.0
    eps:  float = 1.0   # diffusion coefficient
    beta: float = 0.0   # convection coefficient (0 = pure diffusion / Poisson)

    @property
    def name(self) -> str:
        raise NotImplementedError

    def exact_u(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N,1)  →  u: (N,1)"""
        raise NotImplementedError

    def source_f(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N,1)  →  f=-u'': (N,1)"""
        raise NotImplementedError

    def mollifier(self, x: torch.Tensor) -> torch.Tensor:
        """x(1-x)  — vanishes at 0 and 1."""
        return x * (1.0 - x)

    def interior_points(self, n: int) -> torch.Tensor:
        return torch.rand(n, 1)

    def boundary_points(self) -> tuple:
        x_bd = torch.tensor([[self.lb], [self.ub]])
        return x_bd, self.exact_u(x_bd)

    def test_grid(self, n: int = 300) -> tuple:
        x = torch.linspace(self.lb, self.ub, n).unsqueeze(1)
        return x, self.exact_u(x)

    def l2_rel(self, u_pred: torch.Tensor, u_true: torch.Tensor) -> float:
        return (torch.norm(u_pred - u_true) / torch.norm(u_true)).item()

    def default_layers(self) -> list:
        return [1, 64, 64, 64, 64, 1]


class PDE2D:
    """Abstract 2-D PDE on [-1, 1]² with homogeneous Dirichlet BC."""

    dim: int   = 2
    lb:  float = -1.0
    ub:  float =  1.0

    @property
    def name(self) -> str:
        raise NotImplementedError

    def exact_u(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def source_f(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def mollifier(self, x: torch.Tensor) -> torch.Tensor:
        """(1-x₁²)(1-x₂²) — vanishes on all four edges of [-1,1]²."""
        return (1.0 - x[:, 0:1]**2) * (1.0 - x[:, 1:2]**2)

    def interior_points(self, n: int) -> torch.Tensor:
        return torch.rand(n, 2) * (self.ub - self.lb) + self.lb

    def boundary_points(self, n_per_side: int = 100) -> tuple:
        t  = torch.linspace(self.lb, self.ub, n_per_side)
        lb = torch.full((n_per_side,), self.lb)
        ub = torch.full((n_per_side,), self.ub)
        x_bd = torch.cat([
            torch.stack([t,  lb], dim=1),
            torch.stack([t,  ub], dim=1),
            torch.stack([lb, t],  dim=1),
            torch.stack([ub, t],  dim=1),
        ])
        return x_bd, self.exact_u(x_bd)

    def test_grid(self, n: int = 60) -> tuple:
        t = torch.linspace(self.lb, self.ub, n)
        xx, yy = torch.meshgrid(t, t, indexing='ij')
        x = torch.stack([xx.flatten(), yy.flatten()], dim=1)
        return x, self.exact_u(x)

    def l2_rel(self, u_pred: torch.Tensor, u_true: torch.Tensor) -> float:
        return (torch.norm(u_pred - u_true) / torch.norm(u_true)).item()

    def default_layers(self) -> list:
        return [2, 64, 64, 64, 64, 1]


# ════════════════════════════════════════════════════════════════
# 1-D Poisson problems   -u'' = f  on [0,1],  u(0)=u(1)=0
# ════════════════════════════════════════════════════════════════

class SmoothPoisson1D(PDE1D):
    """
    Level 1 — Smooth.
    u = sin(πx),   f = π²·sin(πx)
    Infinitely differentiable — both methods should converge easily.
    """

    name = "Level 1 — Smooth  u=sin(πx)"

    def exact_u(self, x):
        return torch.sin(np.pi * x)

    def source_f(self, x):
        return (np.pi**2) * torch.sin(np.pi * x)


class LayerPoisson1D(PDE1D):
    """
    Level 2 — Interior layer.
    u = sin(πx)·tanh(k·(x-½))

    The tanh creates a steep sign-change transition at x=0.5.
    BC is satisfied exactly: sin(0)=sin(π)=0.
    Sharpness controlled by k; default k=15 gives layer width ≈ 1/k.
    f = -u'' computed analytically.
    """

    def __init__(self, k: float = 15.0):
        self.k = k

    @property
    def name(self):
        return f"Level 2 — Interior layer  k={self.k}"

    def exact_u(self, x):
        return torch.sin(np.pi * x) * torch.tanh(self.k * (x - 0.5))

    def source_f(self, x):
        k  = self.k
        s  = torch.sin(np.pi * x)
        c  = torch.cos(np.pi * x)
        th = torch.tanh(k * (x - 0.5))
        sc = 1.0 - th**2           # sech²(k(x-½))

        # u'' = -π²·s·th  +  2πk·c·sc  -  2k²·s·th·sc
        u_pp = -np.pi**2 * s * th + 2*np.pi*k * c * sc - 2*k**2 * s * th * sc
        return -u_pp


class OscPoisson1D(PDE1D):
    """
    Level 3 — High-frequency / oscillatory.
    u = sin(nπx),   f = (nπ)²·sin(nπx)

    For integer n, BC is satisfied exactly.
    Large n reveals spectral bias in both methods.
    """

    def __init__(self, n: int = 8):
        self.n = n

    @property
    def name(self):
        return f"Level 3 — Oscillatory  u=sin({self.n}πx)"

    def exact_u(self, x):
        return torch.sin(self.n * np.pi * x)

    def source_f(self, x):
        return (self.n * np.pi)**2 * torch.sin(self.n * np.pi * x)


# ════════════════════════════════════════════════════════════════
# 1-D Convection-Diffusion   -ε·u'' + β·u' = f  on [0,1],  u(0)=u(1)=0
# ════════════════════════════════════════════════════════════════

class ConvDiff1D(PDE1D):
    """
    Stationary convection-diffusion:   -ε·u'' + β·u' = f,  u(0)=u(1)=0

    Exact solution (satisfies both Dirichlet BCs exactly):
        u(x) = sin(πx) · (1 − e^((x−1)/ε))

    The exponential term creates a boundary layer of width ~ε at x=1.
    Away from the layer, u ≈ sin(πx) (outer solution).
    Péclet number  Pe = β/ε  governs the layer sharpness.

    Source term derived analytically:
        f = ε·π²·sin(πx)·(1−E) + β·π·cos(πx)·(1−E) + 2π·cos(πx)·E + (1−β)·sin(πx)·E/ε
        where  E = e^((x−1)/ε)
    """

    def __init__(self, eps: float = 0.1, beta: float = 1.0):
        self.eps  = eps
        self.beta = beta

    @property
    def name(self):
        return f"ConvDiff  ε={self.eps:.4g}  Pe={self.beta/self.eps:.0f}"

    def exact_u(self, x: torch.Tensor) -> torch.Tensor:
        E = torch.exp((x - 1.0) / self.eps)
        return torch.sin(np.pi * x) * (1.0 - E)

    def source_f(self, x: torch.Tensor) -> torch.Tensor:
        eps, beta = self.eps, self.beta
        E = torch.exp((x - 1.0) / eps)
        s = torch.sin(np.pi * x)
        c = torch.cos(np.pi * x)
        # u = s·(1-E),  u' = π·c·(1-E) - (s/ε)·E
        # u'' = -π²·s·(1-E) - 2π·c·E/ε - s·E/ε²
        # f = -ε·u'' + β·u'
        #   = ε·π²·s·(1-E) + 2π·c·E + (s/ε)·E  +  β·π·c·(1-E) - β·(s/ε)·E
        #   = (ε·π²·s + β·π·c)·(1-E)  +  (2π·c + (1-β)·s/ε)·E
        term_outer = eps * np.pi**2 * s + beta * np.pi * c
        term_layer = 2.0 * np.pi * c + (1.0 - beta) / eps * s
        return term_outer * (1.0 - E) + term_layer * E
