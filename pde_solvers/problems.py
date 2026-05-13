"""
PDE problem definitions.

All problems live on Ω = [-1, 1]² with homogeneous Dirichlet BC (u = 0 on ∂Ω).
This lets every solver use the same mollifier trick: û = (1-x₁²)(1-x₂²)·net(x).

Problems (increasing complexity):
  1. SmoothPoisson   – u = sin(λx₁)sin(λx₂),            λ = π
  2. OscPoisson      – same form, high frequency,          λ = 4π
  3. BumpPoisson     – u = (1-x₁²)(1-x₂²)·exp(-k·r²),   k = 20
"""

import numpy as np
import torch
from torch.autograd import Variable


class PDE2D:
    """Abstract 2-D PDE on [-1, 1]² with zero Dirichlet BC."""

    lb: float = -1.0
    ub: float =  1.0

    @property
    def name(self) -> str:
        raise NotImplementedError

    def exact_u(self, x: torch.Tensor) -> torch.Tensor:
        """Return exact solution at x  (N, 1)."""
        raise NotImplementedError

    def source_f(self, x: torch.Tensor) -> torch.Tensor:
        """Return RHS f = -Δu at x  (N, 1)."""
        raise NotImplementedError

    # ── shared helpers ────────────────────────────────────────────

    def interior_points(self, n: int) -> torch.Tensor:
        return torch.rand(n, 2) * (self.ub - self.lb) + self.lb

    def boundary_points(self, n_per_side: int = 100) -> tuple:
        """Return (x_bd, u_bd) — boundary points + exact values."""
        t  = torch.linspace(self.lb, self.ub, n_per_side)
        lb = torch.full((n_per_side,), self.lb)
        ub = torch.full((n_per_side,), self.ub)
        x_bd = torch.cat([
            torch.stack([t,  lb], dim=1),   # bottom
            torch.stack([t,  ub], dim=1),   # top
            torch.stack([lb, t],  dim=1),   # left
            torch.stack([ub, t],  dim=1),   # right
        ])
        return x_bd, self.exact_u(x_bd)

    def test_grid(self, n: int = 60) -> tuple:
        """Return (x_test, u_test) on an n×n uniform mesh."""
        t = torch.linspace(self.lb, self.ub, n)
        xx, yy = torch.meshgrid(t, t, indexing='ij')
        x = torch.stack([xx.flatten(), yy.flatten()], dim=1)
        return x, self.exact_u(x)

    def l2_rel(self, u_pred: torch.Tensor, u_true: torch.Tensor) -> float:
        return (torch.norm(u_pred - u_true) / torch.norm(u_true)).item()


# ──────────────────────────────────────────────────────────────────
# 1.  Smooth Poisson  (u = sin(λx₁)sin(λx₂),  λ = π)
# ──────────────────────────────────────────────────────────────────

class SmoothPoisson(PDE2D):
    """
    -Δu = f  on [-1,1]²,  u = 0 on ∂Ω
    Exact: u = sin(λx₁)sin(λx₂),  f = 2λ²u

    λ must be an integer multiple of π so u = 0 on x = ±1.
    """

    def __init__(self, freq: float = np.pi):
        assert abs(freq / np.pi - round(freq / np.pi)) < 1e-6, \
            "freq must be an integer multiple of π for zero BC on [-1,1]²"
        self.freq = freq

    @property
    def name(self):
        k = int(round(self.freq / np.pi))
        return f"Smooth Poisson  λ={k}π"

    def exact_u(self, x):
        lam = self.freq
        return torch.sin(lam * x[:, 0:1]) * torch.sin(lam * x[:, 1:2])

    def source_f(self, x):
        return 2.0 * self.freq**2 * self.exact_u(x)


# ──────────────────────────────────────────────────────────────────
# 2.  High-frequency Poisson  (λ = 4π  — spectral-bias stress test)
# ──────────────────────────────────────────────────────────────────

class OscPoisson(SmoothPoisson):
    """Same as SmoothPoisson but λ = 4π — oscillatory, harder for NNs."""

    def __init__(self):
        super().__init__(freq=4.0 * np.pi)

    @property
    def name(self):
        return "Oscillatory Poisson  λ=4π"


# ──────────────────────────────────────────────────────────────────
# 3.  Localised Gaussian Bump  (sharp peak at origin, zero BC)
# ──────────────────────────────────────────────────────────────────

class BumpPoisson(PDE2D):
    """
    -Δu = f  on [-1,1]²,  u = 0 on ∂Ω
    Exact: u = (1-x₁²)(1-x₂²)·exp(-k·(x₁²+x₂²))

    The mollifier (1-x₁²)(1-x₂²) enforces zero BC exactly.
    k controls the sharpness of the central peak.
    f = -Δu is computed analytically below.
    """

    def __init__(self, k: float = 20.0):
        self.k = k

    @property
    def name(self):
        return f"Bump Poisson  k={self.k}"

    def exact_u(self, x):
        x1, x2 = x[:, 0:1], x[:, 1:2]
        k = self.k
        return (1 - x1**2) * (1 - x2**2) * torch.exp(-k * (x1**2 + x2**2))

    def source_f(self, x):
        """
        Analytical f = -Δu for u = φ·g where φ=(1-x₁²)(1-x₂²), g=exp(-k·r²).

        Using the product rule for the Laplacian:
          Δ(φg) = (Δφ)g + 2∇φ·∇g + φ(Δg)
        """
        x1, x2 = x[:, 0:1], x[:, 1:2]
        k = self.k

        g    = torch.exp(-k * (x1**2 + x2**2))
        gx1  = -2*k*x1 * g
        gx2  = -2*k*x2 * g
        gx1x1 = (-2*k + 4*k**2*x1**2) * g
        gx2x2 = (-2*k + 4*k**2*x2**2) * g

        phi   = (1 - x1**2) * (1 - x2**2)
        phix1 = -2*x1 * (1 - x2**2)
        phix2 = -2*x2 * (1 - x1**2)
        lap_phi = -2*(1 - x2**2) - 2*(1 - x1**2)  # Δφ

        lap_u = lap_phi*g + 2*(phix1*gx1 + phix2*gx2) + phi*(gx1x1 + gx2x2)
        return -lap_u
