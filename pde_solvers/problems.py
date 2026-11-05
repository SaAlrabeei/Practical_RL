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

    dim:      int   = 1
    lb:       float = 0.0
    ub:       float = 1.0
    eps:      float = 1.0   # diffusion coefficient
    beta:     float = 0.0   # convection coefficient (0 = pure diffusion / Poisson)
    nonlinear: bool = False  # True → Deep Ritz uses LS energy

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

    def pde_residual(self, u_hat, grad, lap_u, x):
        """Pointwise PDE residual for collocation. Default: -eps·Δu + beta·∂u/∂x₁ - f."""
        eps  = getattr(self, 'eps',  1.0)
        beta = getattr(self, 'beta', 0.0)
        f    = self.source_f(x)
        res  = -eps * lap_u - f
        if beta != 0.0 and self.dim == 1:
            res = res + beta * grad[:, 0:1]
        return res

    def vpinn_lhs_1d(self, u_hat, du, w, v_vals, dv_dx1, eps):
        """VPINN 1D bilinear form LHS. Default: ε∫u'v' + β∫u'v."""
        beta = getattr(self, 'beta', 0.0)
        return torch.sum(w * (eps * dv_dx1 + beta * v_vals) * du, dim=-1)

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


# ════════════════════════════════════════════════════════════════
# 1-D Burgers   -ν·u'' + u·u' = f  on [0,1],  u(0)=u(1)=0
# ════════════════════════════════════════════════════════════════

class Burgers1D(PDE1D):
    """
    Stationary viscous Burgers:  -ν·u'' + u·u' = f,  u(0)=u(1)=0

    Same nonlinear convection structure as Navier-Stokes momentum equation.
    ν is the kinematic viscosity (analogue of 1/Re).

    Manufactured exact solution (satisfies both BCs exactly):
        u(x) = sin(πx)·(1 − e^((x−1)/ν))

    Boundary layer of width ~ν at x=1.

    Source term derived analytically:
        f = -ν·u'' + u·u'
    """

    nonlinear = True

    def __init__(self, nu: float = 0.1):
        self.eps  = nu      # diffusion coeff = viscosity
        self.beta = 0.0     # no linear convection term

    @property
    def name(self):
        return f"Burgers  ν={self.eps:.4g}  (1/Re≈{self.eps:.4g})"

    def exact_u(self, x: torch.Tensor) -> torch.Tensor:
        E = torch.exp((x - 1.0) / self.eps)
        return torch.sin(np.pi * x) * (1.0 - E)

    def source_f(self, x: torch.Tensor) -> torch.Tensor:
        nu = self.eps
        E  = torch.exp((x - 1.0) / nu)
        s  = torch.sin(np.pi * x)
        c  = torch.cos(np.pi * x)
        u  = s * (1.0 - E)
        up = np.pi * c * (1.0 - E) - (s / nu) * E
        # -ν·u'' = ν·π²·s·(1-E) + 2π·c·E + (s/ν)·E
        neg_nu_upp = nu * np.pi**2 * s * (1.0 - E) + 2 * np.pi * c * E + (s / nu) * E
        return neg_nu_upp + u * up

    def pde_residual(self, u_hat, grad, lap_u, x):
        nu = self.eps
        f  = self.source_f(x)
        return -nu * lap_u + u_hat * grad[:, 0:1] - f

    def vpinn_lhs_1d(self, u_hat, du, w, v_vals, dv_dx1, eps):
        # Weak form: ν∫u'v' - ½∫u²v'   (IBP of ∫uu'v = -½∫u²v')
        u_sq = u_hat.squeeze() ** 2
        return torch.sum(w * dv_dx1 * (eps * du - 0.5 * u_sq), dim=-1)


# ════════════════════════════════════════════════════════════════
# 2-D Lid-Driven Cavity   (parametric in Re)
# ════════════════════════════════════════════════════════════════

class LidDrivenCavity2D:
    """
    2D steady incompressible Navier-Stokes: lid-driven cavity.

    Domain : [0,1]²
    BCs    : top wall (y=1)   u=1, v=0   (moving lid)
             other walls      u=0, v=0   (no-slip)
    PDE    :
        continuity  : ∂u/∂x + ∂v/∂y = 0
        x-momentum  : u·∂u/∂x + v·∂u/∂y + ∂p/∂x − (1/Re)·(u_xx + u_yy) = 0
        y-momentum  : u·∂v/∂x + v·∂v/∂y + ∂p/∂y − (1/Re)·(v_xx + v_yy) = 0

    Parametric in Re: network input is (x, y, Re/Re_ref) with Re_ref=1000.
    Training samples Re uniformly from [Re_min, Re_max] each step, so a single
    network generalises across all Reynolds numbers in that range.

    Validation: Ghia et al. (1982), Computers & Fluids, 10(4), 339-348.
    Tabulated u-velocity on the vertical centreline x=0.5 for Re=100, 400, 1000.
    """

    name    = "Lid-Driven Cavity 2D (NS)"
    dim     = 2
    Re_ref  = 1000.0

    # Ghia et al. (1982) Table 1: vertical centreline x=0.5, u-velocity
    GHIA_Y = [1.0000, 0.9766, 0.9688, 0.9609, 0.9531,
              0.8516, 0.7344, 0.6172, 0.5000, 0.4531,
              0.2813, 0.1719, 0.1016, 0.0703, 0.0625, 0.0547, 0.0000]
    GHIA_U = {
        100:  [ 1.00000,  0.84123,  0.78871,  0.73722,  0.68717,
                0.23151,  0.00332, -0.13641, -0.20581, -0.21090,
               -0.15662, -0.10150, -0.06434, -0.04775, -0.04192, -0.03717,  0.00000],
        400:  [ 1.00000,  0.75837,  0.68439,  0.61756,  0.55892,
                0.29093,  0.16256,  0.02135, -0.11477, -0.17119,
               -0.32726, -0.24299, -0.14612, -0.10338, -0.09266, -0.08186,  0.00000],
        1000: [ 1.00000,  0.65928,  0.57492,  0.51117,  0.46604,
                0.33304,  0.18719,  0.05702, -0.06080, -0.10648,
               -0.27805, -0.38289, -0.29730, -0.22220, -0.20196, -0.18109,  0.00000],
    }

    def __init__(self, Re_min: float = 100.0, Re_max: float = 1000.0):
        self.Re_min = Re_min
        self.Re_max = Re_max

    def _pack(self, xy: np.ndarray, Re: float) -> torch.Tensor:
        Re_col = np.full((len(xy), 1), Re / self.Re_ref)
        return torch.tensor(np.hstack([xy, Re_col]), dtype=torch.float32)

    def sample_Re(self) -> float:
        return float(np.random.uniform(self.Re_min, self.Re_max))

    def interior_points(self, n: int, Re: float) -> torch.Tensor:
        """n random points strictly inside (0,1)²."""
        xy = np.random.rand(n, 2)
        return self._pack(xy, Re)

    def bc_points(self, n_per_side: int, Re: float) -> tuple:
        """
        Returns (x_bc [4n,3], u_bc [4n,1], v_bc [4n,1]).
        Sides: bottom y=0 (u=0,v=0), top y=1 (u=1,v=0),
               left x=0 (u=0,v=0), right x=1 (u=0,v=0).
        """
        t  = np.linspace(0, 1, n_per_side)
        z  = np.zeros(n_per_side)
        o  = np.ones(n_per_side)
        pts = np.vstack([
            np.stack([t, z], 1),   # bottom
            np.stack([t, o], 1),   # top  (lid)
            np.stack([z, t], 1),   # left
            np.stack([o, t], 1),   # right
        ])
        u_bc = np.concatenate([z, o, z, z])[:, None]  # u=1 on top only
        v_bc = np.zeros((4 * n_per_side, 1))
        return (self._pack(pts, Re),
                torch.tensor(u_bc, dtype=torch.float32),
                torch.tensor(v_bc, dtype=torch.float32))

    def pressure_ref_point(self, Re: float) -> torch.Tensor:
        """Single point at corner (0,0) used to fix pressure gauge."""
        return self._pack(np.array([[0.0, 0.0]]), Re)

    def test_grid(self, n: int = 50, Re: float = 100.0) -> torch.Tensor:
        """Regular n×n grid in [0,1]², returns (n²,3) tensor."""
        t  = np.linspace(0, 1, n)
        xx, yy = np.meshgrid(t, t, indexing='ij')
        xy = np.stack([xx.flatten(), yy.flatten()], axis=1)
        return self._pack(xy, Re)

    def ghia_centreline(self, Re: int) -> tuple:
        """(y_vals np.ndarray, u_vals np.ndarray) from Ghia et al. for Re in {100,400,1000}."""
        return np.array(self.GHIA_Y), np.array(self.GHIA_U[Re])
