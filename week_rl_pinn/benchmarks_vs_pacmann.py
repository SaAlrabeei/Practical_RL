"""
RL Collocation vs. Uniform / RAR / PACMANN
==========================================
Benchmarks: 1D Burgers and 1D Allen-Cahn equations.
Matches the PACMANN paper setup exactly (Visser et al. 2024, arXiv:2411.19632).

Key paper settings reproduced here:
  Network      : 4 × 64 tanh, Glorot normal init
  Training     : Adam lr=1e-3, equal loss weights (no w_bc/w_ic scaling)
  Collocation  : 2500 interior + 80 BC + 160 IC
  Metric       : L2 relative error  ‖u_pred − u_ref‖ / ‖u_ref‖

  PACMANN      : custom Adam (α=1e-5, β1=0.9, β2=0.999, ε=1e-7)
                 T=15 inner steps (Burgers) / T=5 (Allen-Cahn)
                 Period P=50 PINN epochs between moves
                 Out-of-domain points → replaced with uniform random samples
                 Moments reset at the start of each resampling event

  Allen-Cahn   : hard BC/IC via output_transform (matches paper exactly)
                   u(x,t) = x²cos(πx) + t·(1−x²)·u_NN(x,t)
  Burgers      : soft BC/IC, t ∈ [0, 0.99] (as in paper)

Equal-budget mode (default):
  All methods hold N=n_total points throughout.
  RAR:  replaces n_replace lowest-residual points with high-residual candidates.
  RL:   resamples all N points from the learned weight map each step.
  This is the fair comparison — no method has fewer points at any step.

Multi-seed: run K seeds, report mean ± std.
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import RegularGridInterpolator
from typing import Tuple, List, Optional


# ═══════════════════════════════════════════════════
# PART 1: REFERENCE SOLUTIONS
# ═══════════════════════════════════════════════════

def _burgers_reference(nu: float = 0.01 / np.pi,
                        nx: int = 512, nt_out: int = 101) -> Tuple:
    """RK45 FD reference on x∈[-1,1], t∈[0,0.99]."""
    dx    = 2.0 / (nx + 1)
    x_int = np.linspace(-1, 1, nx + 2)[1:-1]

    def rhs(t, u):
        uf = np.zeros(nx + 2)
        uf[1:-1] = u
        adv  = np.where(u >= 0,
                        u * (uf[1:-1] - uf[:-2]) / dx,
                        u * (uf[2:]   - uf[1:-1]) / dx)
        diff = nu * (uf[2:] - 2*uf[1:-1] + uf[:-2]) / dx**2
        return -adv + diff

    u0     = -np.sin(np.pi * x_int)
    t_eval = np.linspace(0, 0.99, nt_out)
    sol    = solve_ivp(rhs, [0, 0.99], u0, t_eval=t_eval,
                       method='RK45', rtol=1e-8, atol=1e-10)

    x_full = np.linspace(-1, 1, nx + 2)
    U = np.zeros((nt_out, nx + 2))
    U[:, 1:-1] = sol.y.T
    return x_full, t_eval, U


def _allen_cahn_reference(d: float = 0.001,
                           nx: int = 256, nt_out: int = 101) -> Tuple:
    """Radau FD reference on x∈[-1,1], t∈[0,1]."""
    dx    = 2.0 / (nx + 1)
    x_int = np.linspace(-1, 1, nx + 2)[1:-1]

    def rhs(t, u):
        uf = np.full(nx + 2, -1.0)
        uf[1:-1] = u
        diff  = d * (uf[2:] - 2*uf[1:-1] + uf[:-2]) / dx**2
        react = 5.0 * (u - u**3)
        return diff + react

    u0     = x_int**2 * np.cos(np.pi * x_int)
    t_eval = np.linspace(0, 1.0, nt_out)
    sol    = solve_ivp(rhs, [0, 1.0], u0, t_eval=t_eval,
                       method='Radau', rtol=1e-9, atol=1e-11)

    x_full = np.linspace(-1, 1, nx + 2)
    U = np.full((nt_out, nx + 2), -1.0)
    U[:, 1:-1] = sol.y.T
    return x_full, t_eval, U


# ═══════════════════════════════════════════════════
# PART 2: PDE CLASSES
# ═══════════════════════════════════════════════════

class Burgers1D:
    """
    u_t + u·u_x − ν·u_xx = 0   x∈[-1,1], t∈[0,0.99]
    IC: u(x,0)  = −sin(πx)
    BC: u(±1,t) = 0
    ν = 0.01/π
    """
    def __init__(self, nu: float = 0.01 / np.pi):
        self.nu      = nu
        self.x_range = (-1.0,  1.0)
        self.t_range = ( 0.0,  0.99)
        self._build_reference()

    def _build_reference(self):
        print("  Building Burgers reference (scipy RK45) …")
        xg, tg, U = _burgers_reference(self.nu)
        self._interp = RegularGridInterpolator(
            (tg, xg), U, method='linear', bounds_error=False, fill_value=None)

    def u_ref(self, x, t):
        pts = np.stack([t.detach().numpy(), x.detach().numpy()], axis=-1)
        return torch.tensor(self._interp(pts), dtype=torch.float32)

    def pde_residual(self, pinn, x, t):
        x = x.requires_grad_(True); t = t.requires_grad_(True)
        u    = pinn(x, t)
        u_t  = torch.autograd.grad(u, t, torch.ones_like(u),
                                    create_graph=True, retain_graph=True)[0]
        u_x  = torch.autograd.grad(u, x, torch.ones_like(u),
                                    create_graph=True, retain_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x),
                                    create_graph=True, retain_graph=True)[0]
        return u_t + u * u_x - self.nu * u_xx

    def sample_ic(self, n):
        x = torch.rand(n) * 2 - 1
        t = torch.zeros(n)
        return x, t, -torch.sin(torch.pi * x)

    def sample_bc(self, n_per_side):
        t   = torch.rand(n_per_side) * 0.99
        x_l = torch.full_like(t, -1.0)
        x_r = torch.full_like(t,  1.0)
        return (torch.cat([x_l, x_r]),
                torch.cat([t,   t]),
                torch.zeros(2 * n_per_side))


class AllenCahn1D:
    """
    u_t − d·u_xx − 5(u−u³) = 0   x∈[-1,1], t∈[0,1]
    IC: u(x,0)  = x²cos(πx)
    BC: u(±1,t) = −1
    d = 0.001

    Hard output_transform (paper exact):
        u(x,t) = x²cos(πx)  +  t·(1−x²)·u_NN(x,t)
    """
    def __init__(self, d: float = 0.001):
        self.d       = d
        self.x_range = (-1.0, 1.0)
        self.t_range = ( 0.0, 1.0)
        self._build_reference()

    def _build_reference(self):
        print("  Building Allen-Cahn reference (scipy Radau) …")
        xg, tg, U = _allen_cahn_reference(self.d)
        self._interp = RegularGridInterpolator(
            (tg, xg), U, method='linear', bounds_error=False, fill_value=None)

    def u_ref(self, x, t):
        pts = np.stack([t.detach().numpy(), x.detach().numpy()], axis=-1)
        return torch.tensor(self._interp(pts), dtype=torch.float32)

    @staticmethod
    def output_transform(x, t, u_nn):
        return x**2 * torch.cos(torch.pi * x) + t * (1 - x**2) * u_nn

    def pde_residual(self, pinn, x, t):
        x = x.requires_grad_(True); t = t.requires_grad_(True)
        u    = pinn(x, t)
        u_t  = torch.autograd.grad(u, t, torch.ones_like(u),
                                    create_graph=True, retain_graph=True)[0]
        u_x  = torch.autograd.grad(u, x, torch.ones_like(u),
                                    create_graph=True, retain_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x),
                                    create_graph=True, retain_graph=True)[0]
        return u_t - self.d * u_xx - 5.0 * (u - u**3)

    def sample_ic(self, n):
        x = torch.rand(n) * 2 - 1
        t = torch.zeros(n)
        u = x**2 * torch.cos(torch.pi * x)
        return x, t, u

    def sample_bc(self, n_per_side):
        t   = torch.rand(n_per_side)
        x_l = torch.full_like(t, -1.0)
        x_r = torch.full_like(t,  1.0)
        u_bc = torch.full((2 * n_per_side,), -1.0)
        return torch.cat([x_l, x_r]), torch.cat([t, t]), u_bc


# ═══════════════════════════════════════════════════
# PART 3: SPACE-TIME PINN
# ═══════════════════════════════════════════════════

class SpaceTimePINN(nn.Module):
    """4 × 64 tanh, Glorot normal. Optional output_transform for hard BC/IC."""
    def __init__(self, hidden_dim: int = 64, n_layers: int = 4,
                 output_transform=None):
        super().__init__()
        layers = [nn.Linear(2, hidden_dim), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers += [nn.Linear(hidden_dim, 1)]
        self.net = nn.Sequential(*layers)
        self.output_transform = output_transform

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        u_nn = self.net(torch.stack([x, t], dim=-1)).squeeze(-1)
        if self.output_transform is not None:
            return self.output_transform(x, t, u_nn)
        return u_nn

    def compute_l2_rel_error(self, pde, n: int = 5000) -> float:
        with torch.no_grad():
            x = torch.rand(n) * (pde.x_range[1] - pde.x_range[0]) + pde.x_range[0]
            t = torch.rand(n) * (pde.t_range[1] - pde.t_range[0]) + pde.t_range[0]
            u_pred  = self.forward(x, t)
            u_exact = pde.u_ref(x, t)
            num = torch.sqrt(torch.mean((u_pred - u_exact)**2))
            den = torch.sqrt(torch.mean(u_exact**2)) + 1e-8
            return (num / den).item()


def train_spacetime_step(pinn, optimizer, pde,
                          cx, ct, x_bc, t_bc, u_bc,
                          x_ic, t_ic, u_ic,
                          use_hard_constraints: bool = False):
    """Single Adam step. Equal loss weights (paper default)."""
    optimizer.zero_grad()
    L_pde = torch.mean(pde.pde_residual(pinn, cx, ct) ** 2)

    if use_hard_constraints:
        loss = L_pde
    else:
        L_bc  = torch.mean((pinn(x_bc, t_bc) - u_bc) ** 2)
        L_ic  = torch.mean((pinn(x_ic, t_ic) - u_ic) ** 2)
        loss  = L_pde + L_bc + L_ic

    loss.backward()
    optimizer.step()
    return L_pde.item()


# ═══════════════════════════════════════════════════
# PART 4: COLLOCATION STRATEGIES
# ═══════════════════════════════════════════════════

def _random_domain_pts(n, x_range, t_range):
    xl, xr = x_range; tl, tr = t_range
    x = torch.rand(n) * (xr - xl) + xl
    t = torch.rand(n) * (tr - tl) + tl
    return x, t


# ─────────────────────────────────────
# 4a. Uniform
# ─────────────────────────────────────

class UniformSpaceTime:
    """Fixed random set — sampled once, never changed."""
    def __init__(self, n_points, x_range=(-1., 1.), t_range=(0., .99)):
        self.x, self.t = _random_domain_pts(n_points, x_range, t_range)

    def get_points(self, pinn=None, pde=None):
        return self.x.detach(), self.t.detach()
    def update(self, *args): pass


# ─────────────────────────────────────
# 4b. RAR — equal-budget version
# ─────────────────────────────────────

class RARSpaceTime:
    """
    Residual Adaptive Refinement.

    Equal-budget mode (default):
      Starts with n_total points.  Each step, finds the n_replace lowest-
      residual existing points and swaps them for the highest-residual
      candidates from a fresh pool.  Total N stays constant.

    Growing mode (equal_budget=False):
      Original Lu et al. 2021 — starts with n_initial, adds n_replace each step.
    """
    def __init__(self, n_total: int, n_replace: int = 100,
                 n_candidates: int = 10_000,
                 x_range=(-1., 1.), t_range=(0., .99),
                 equal_budget: bool = True):
        self.n_replace    = n_replace
        self.n_cand       = n_candidates
        self.x_range      = x_range
        self.t_range      = t_range
        self.equal_budget = equal_budget
        self.x, self.t    = _random_domain_pts(n_total, x_range, t_range)

    def get_points(self, pinn=None, pde=None):
        return self.x.detach(), self.t.detach()

    def _residuals(self, pinn, pde, x, t):
        res = []
        for i in range(0, len(x), 500):
            xb, tb = x[i:i+500], t[i:i+500]
            res.append(pde.pde_residual(pinn, xb, tb).detach().abs())
        return torch.cat(res)

    def update(self, pinn, pde):
        if self.equal_budget:
            # Score existing points — drop lowest-residual n_replace
            res_existing = self._residuals(pinn, pde, self.x, self.t)
            _, worst_idx = torch.topk(res_existing, self.n_replace, largest=False)
            keep = torch.ones(len(self.x), dtype=torch.bool)
            keep[worst_idx] = False

            # Sample candidates, keep highest-residual n_replace
            xc, tc  = _random_domain_pts(self.n_cand, self.x_range, self.t_range)
            res_cand = self._residuals(pinn, pde, xc, tc)
            _, best_idx = torch.topk(res_cand, self.n_replace)

            self.x = torch.cat([self.x[keep], xc[best_idx]]).detach()
            self.t = torch.cat([self.t[keep], tc[best_idx]]).detach()
        else:
            # Original growing mode
            xc, tc  = _random_domain_pts(self.n_cand, self.x_range, self.t_range)
            res_cand = self._residuals(pinn, pde, xc, tc)
            _, best_idx = torch.topk(res_cand, self.n_replace)
            self.x = torch.cat([self.x, xc[best_idx]]).detach()
            self.t = torch.cat([self.t, tc[best_idx]]).detach()


# ─────────────────────────────────────
# 4c. PACMANN — exact paper implementation
# ─────────────────────────────────────

class PACMANNCollocation:
    """
    PACMANN (Visser et al. 2024) with exact paper hyperparameters.

    Custom Adam ascent on ‖r(x,t)‖², moments reset each resampling event.
    epsilon = 10e-8 = 1e-7 (literal value in paper code).
    OOD points replaced by uniform random samples.
    Period P=50: points move every 50 PINN epochs.
    T=15 for Burgers, T=5 for Allen-Cahn.
    """
    PERIOD = 50

    def __init__(self, n_points: int,
                 n_steps: int = 15,
                 lr: float = 1e-5,
                 x_range=(-1., 1.), t_range=(0., .99)):
        self.n_steps  = n_steps
        self.lr       = lr
        self.beta1    = 0.9
        self.beta2    = 0.999
        self.epsilon  = 10e-8
        self.x_range  = x_range
        self.t_range  = t_range
        self.x, self.t = _random_domain_pts(n_points, x_range, t_range)

    def get_points(self, pinn=None, pde=None):
        return self.x.detach(), self.t.detach()

    def update(self, pinn: SpaceTimePINN, pde):
        xl, xr = self.x_range
        tl, tr = self.t_range
        N = len(self.x)

        coords = np.stack([self.x.detach().numpy(),
                           self.t.detach().numpy()], axis=1)
        VdX = np.zeros((N, 2))
        SdX = np.zeros((N, 2))

        for n in range(self.n_steps):
            xt  = torch.tensor(coords, dtype=torch.float32, requires_grad=True)
            res = pde.pde_residual(pinn,
                                    xt[:, 0].requires_grad_(True),
                                    xt[:, 1].requires_grad_(True))
            torch.mean(res ** 2).backward()

            grad = xt.grad.detach().numpy()
            VdX  = self.beta1 * VdX + (1 - self.beta1) * grad
            SdX  = self.beta2 * SdX + (1 - self.beta2) * grad ** 2
            VdX_c = VdX / (1 - self.beta1 ** (n + 1))
            SdX_c = SdX / (1 - self.beta2 ** (n + 1))
            coords = coords + self.lr * VdX_c / (np.sqrt(SdX_c) + self.epsilon)

            oob = ((coords[:, 0] < xl) | (coords[:, 0] > xr) |
                   (coords[:, 1] < tl) | (coords[:, 1] > tr))
            if oob.any():
                n_oob = int(oob.sum())
                coords[oob, 0] = np.random.uniform(xl, xr, n_oob)
                coords[oob, 1] = np.random.uniform(tl, tr, n_oob)

        self.x = torch.tensor(coords[:, 0], dtype=torch.float32)
        self.t = torch.tensor(coords[:, 1], dtype=torch.float32)


# ─────────────────────────────────────
# 4d. RL (this work) — equal-budget version
# ─────────────────────────────────────

def _make_density_map_2d(x, t, G, x_range, t_range):
    xl, xr = x_range; tl, tr = t_range
    density = torch.zeros(G, G)
    ix = ((x - xl) / (xr - xl) * G).long().clamp(0, G-1)
    it = ((t - tl) / (tr - tl) * G).long().clamp(0, G-1)
    for i, j in zip(ix, it):
        density[i, j] += 1
    return density


def _entropy(d: torch.Tensor) -> float:
    p = d.flatten().float()
    p = p / (p.sum() + 1e-8)
    return -(p * (p + 1e-8).log()).sum().item()


class RLAgentSpaceTime(nn.Module):
    def __init__(self, G: int = 16):
        super().__init__()
        self.G = G
        sd, ad = G*G*3 + 3, G*G
        self.policy = nn.Sequential(
            nn.Linear(sd, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, ad),
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
        self.log_probs: List[torch.Tensor] = []
        self.rewards:   List[float]        = []

    def get_state(self, pinn, pde, x, t, l2, n_total, x_range, t_range):
        G = self.G; eps = 1e-8
        xl, xr = x_range; tl, tr = t_range
        gx = torch.linspace(xl, xr, G); gt = torch.linspace(tl, tr, G)
        XX, TT = torch.meshgrid(gx, gt, indexing='ij')
        xf, tf = XX.flatten(), TT.flatten()

        res_map = pde.pde_residual(pinn, xf, tf).detach().abs().reshape(G, G)

        xg, tg = xf.requires_grad_(True), tf.requires_grad_(True)
        u2 = pinn(xg, tg)
        ux = torch.autograd.grad(u2, xg, torch.ones_like(u2),
                                  retain_graph=True, create_graph=False)[0].detach()
        ut = torch.autograd.grad(u2, tg, torch.ones_like(u2),
                                  create_graph=False)[0].detach()
        grad_map    = (ux**2 + ut**2).sqrt().reshape(G, G)
        density_map = _make_density_map_2d(x, t, G, x_range, t_range)

        for m in (res_map, grad_map):
            m.div_(m.max() + eps)
        density_map = density_map / (density_map.max() + eps)

        scalars = torch.tensor([l2, len(x) / n_total, res_map.mean().item()])
        return torch.cat([res_map.flatten(), grad_map.flatten(),
                          density_map.flatten(), scalars])

    def act(self, state):
        w    = torch.softmax(self.policy(state), dim=-1)
        dist = torch.distributions.Categorical(probs=w)
        lp   = dist.log_prob(dist.sample((100,))).mean()
        return w, lp

    def sample_points(self, w, n, x_range, t_range):
        G = self.G
        xl, xr = x_range; tl, tr = t_range
        idx = torch.multinomial(w, n, replacement=True)
        ix  = idx // G;  it = idx % G
        sx  = (xr - xl) / G;  st = (tr - tl) / G
        x   = (ix.float() * sx + torch.rand(n) * sx + xl).clamp(xl, xr)
        t   = (it.float() * st + torch.rand(n) * st + tl).clamp(tl, tr)
        return x, t

    def update_policy(self, reward: float):
        self.rewards.append(reward)
        if not self.log_probs:
            self.rewards = []; return
        R = torch.tensor(self.rewards, dtype=torch.float32)
        if len(R) > 1:
            R = (R - R.mean()) / (R.std() + 1e-8)
        loss = torch.zeros(1, requires_grad=True)
        for lp, r in zip(self.log_probs, R):
            loss = loss - lp * r
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
        self.optimizer.step()
        self.log_probs = []; self.rewards = []


class RLSpaceTime:
    """
    RL collocation (this work).

    Equal-budget mode (default):
      Starts with n_total points. Each step, the agent observes the PDE state
      and resamples ALL n_total points from its learned weight map.
      This gives RL the same budget as Uniform/PACMANN at every step.

    Growing mode (equal_budget=False):
      Starts with n_initial points, adds n_add per step.
    """
    W1, W2 = 1.0, 0.1

    def __init__(self, n_total: int, n_add: int = 100, G: int = 16,
                 x_range=(-1., 1.), t_range=(0., .99),
                 equal_budget: bool = True):
        self.n_total      = n_total
        self.n_add        = n_add
        self.G            = G
        self.x_range      = x_range
        self.t_range      = t_range
        self.equal_budget = equal_budget
        self.agent        = RLAgentSpaceTime(G=G)
        self.x, self.t    = _random_domain_pts(n_total, x_range, t_range)
        self.prev_l2:      Optional[float]        = None
        self.prev_density: Optional[torch.Tensor] = None
        self.weight_history: List                 = []
        self._step = 0

    def get_points(self, pinn=None, pde=None):
        return self.x.detach(), self.t.detach()

    def _residuals(self, pinn, pde, x, t):
        res = []
        for i in range(0, len(x), 500):
            xb, tb = x[i:i+500].detach(), t[i:i+500].detach()
            res.append(pde.pde_residual(pinn, xb, tb).detach().abs())
        return torch.cat(res)

    def observe_and_act(self, pinn, pde, l2):
        self.prev_density = _make_density_map_2d(
            self.x, self.t, self.G, self.x_range, self.t_range)
        state = self.agent.get_state(
            pinn, pde, self.x, self.t, l2, self.n_total,
            self.x_range, self.t_range)
        w, lp = self.agent.act(state)
        self._step += 1
        if self._step in (1, 5, 10, 20):
            self.weight_history.append(
                (self._step, w.detach().reshape(self.G, self.G).clone()))
        self.agent.log_probs.append(lp)
        self.prev_l2 = l2

        if self.equal_budget:
            # Remove n_add lowest-residual existing points,
            # replace with n_add RL-sampled points — mirrors RAR but uses
            # the learned weight map for where to place new points.
            res_existing = self._residuals(pinn, pde, self.x, self.t)
            _, worst_idx = torch.topk(res_existing, self.n_add, largest=False)
            keep = torch.ones(len(self.x), dtype=torch.bool)
            keep[worst_idx] = False
            nx, nt = self.agent.sample_points(
                w.detach(), self.n_add, self.x_range, self.t_range)
            self.x = torch.cat([self.x[keep], nx]).detach()
            self.t = torch.cat([self.t[keep], nt]).detach()
        else:
            # Original: add n_add points
            nx, nt = self.agent.sample_points(
                w.detach(), self.n_add, self.x_range, self.t_range)
            self.x = torch.cat([self.x, nx]).detach()
            self.t = torch.cat([self.t, nt]).detach()

    def update_reward(self, current_l2: float) -> float:
        nd    = _make_density_map_2d(self.x, self.t, self.G,
                                      self.x_range, self.t_range)
        l2_r  = self.W1 * (self.prev_l2 - current_l2) / self.n_total * 1000
        ent_r = self.W2 * (_entropy(nd) - _entropy(self.prev_density))
        self.agent.update_policy(l2_r + ent_r)
        return l2_r + ent_r


# ═══════════════════════════════════════════════════
# PART 5: EXPERIMENT RUNNER
# ═══════════════════════════════════════════════════

def run_spacetime_experiment(
        strategy_name: str, strategy, pde,
        n_adapt_steps:   int  = 20,
        epochs_per_step: int  = 500,
        n_total:         int  = 2500,
        n_bc:            int  = 40,
        n_ic:            int  = 160,
        seed:            int  = 42,
        verbose:         bool = True) -> dict:

    torch.manual_seed(seed); np.random.seed(seed)

    use_hard = isinstance(pde, AllenCahn1D)

    pinn = SpaceTimePINN(
        hidden_dim=64, n_layers=4,
        output_transform=(AllenCahn1D.output_transform if use_hard else None)
    )
    optimizer = torch.optim.Adam(pinn.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=2000, gamma=0.5)

    x_bc, t_bc, u_bc = pde.sample_bc(n_per_side=n_bc)
    x_ic, t_ic, u_ic = pde.sample_ic(n_ic)

    l2_hist, nc_hist = [], []

    if verbose:
        print(f"\n{'='*50}\n{strategy_name}\n{'='*50}")

    cx, ct = strategy.get_points(pinn, pde)

    for step in range(n_adapt_steps + 1):

        for epoch in range(epochs_per_step):
            train_spacetime_step(pinn, optimizer, pde,
                                  cx, ct, x_bc, t_bc, u_bc,
                                  x_ic, t_ic, u_ic,
                                  use_hard_constraints=use_hard)
            scheduler.step()

            # PACMANN moves points every P=50 epochs
            if strategy_name == 'PACMANN' and (epoch + 1) % PACMANNCollocation.PERIOD == 0:
                strategy.update(pinn, pde)
                cx, ct = strategy.get_points()

        l2 = pinn.compute_l2_rel_error(pde)
        l2_hist.append(l2)
        nc_hist.append(len(cx))

        if verbose:
            print(f"  step {step:3d} | N={len(cx):5d} | L2_rel={l2:.5f}")
        if step == n_adapt_steps:
            break

        if strategy_name == 'RAR':
            strategy.update(pinn, pde)
        elif strategy_name == 'RL':
            if strategy.prev_l2 is not None:
                strategy.update_reward(l2)
            strategy.observe_and_act(pinn, pde, l2)

        cx, ct = strategy.get_points(pinn, pde)

    return {
        'name':     strategy_name,
        'l2':       l2_hist,
        'n_colloc': nc_hist,
        'final_l2': l2_hist[-1],
        'final_x':  cx.detach(),
        'final_t':  ct.detach(),
        'weight_history': getattr(strategy, 'weight_history', []),
    }


def _make_strategies(pde_name, pde, n_total, n_replace, equal_budget):
    """Create one fresh set of strategies for a single run."""
    xr = pde.x_range; tr = pde.t_range
    pacmann_T = 15 if pde_name == 'burgers' else 5
    return {
        'Uniform': UniformSpaceTime(n_total, xr, tr),
        'RAR':     RARSpaceTime(n_total, n_replace=n_replace,
                                x_range=xr, t_range=tr,
                                equal_budget=equal_budget),
        'PACMANN': PACMANNCollocation(n_total, n_steps=pacmann_T, lr=1e-5,
                                      x_range=xr, t_range=tr),
        'RL':      RLSpaceTime(n_total, n_add=n_replace, G=16,
                               x_range=xr, t_range=tr,
                               equal_budget=equal_budget),
    }


def run_comparison(pde_name: str = 'burgers',
                   n_adapt_steps:   int  = 20,
                   epochs_per_step: int  = 500,
                   n_total:         int  = 2500,
                   n_replace:       int  = 100,
                   equal_budget:    bool = True,
                   seed:            int  = 42,
                   verbose:         bool = True) -> dict:
    """Single-seed 4-way comparison."""
    pde = Burgers1D() if pde_name == 'burgers' else AllenCahn1D()
    strategies = _make_strategies(pde_name, pde, n_total, n_replace, equal_budget)

    kw = dict(n_adapt_steps=n_adapt_steps, epochs_per_step=epochs_per_step,
              n_total=n_total, seed=seed, verbose=verbose)
    results = {}
    for name, strat in strategies.items():
        results[name] = run_spacetime_experiment(name, strat, pde, **kw)
    return results


def run_multi_seed(pde_name: str,
                   seeds: List[int],
                   n_adapt_steps:   int  = 20,
                   epochs_per_step: int  = 500,
                   n_total:         int  = 2500,
                   n_replace:       int  = 100,
                   equal_budget:    bool = True) -> dict:
    """
    Run each seed independently, return aggregated mean ± std per method.
    Returns dict: method_name → {l2_mean, l2_std, final_l2_mean, final_l2_std, ...}
    """
    all_runs = []
    for i, seed in enumerate(seeds):
        print(f"\n{'─'*60}")
        print(f"  Seed {seed}  ({i+1}/{len(seeds)})")
        print(f"{'─'*60}")
        r = run_comparison(pde_name,
                           n_adapt_steps=n_adapt_steps,
                           epochs_per_step=epochs_per_step,
                           n_total=n_total,
                           n_replace=n_replace,
                           equal_budget=equal_budget,
                           seed=seed,
                           verbose=True)
        all_runs.append(r)

    methods = list(all_runs[0].keys())
    agg = {}
    for m in methods:
        l2_curves = np.array([r[m]['l2'] for r in all_runs])   # (n_seeds, n_steps+1)
        n_colloc  = all_runs[-1][m]['n_colloc']                  # same for all seeds
        agg[m] = {
            'name':           m,
            'l2_mean':        l2_curves.mean(axis=0),
            'l2_std':         l2_curves.std(axis=0),
            'final_l2_mean':  l2_curves[:, -1].mean(),
            'final_l2_std':   l2_curves[:, -1].std(),
            'n_colloc':       n_colloc,
            'final_x':        all_runs[-1][m]['final_x'],
            'final_t':        all_runs[-1][m]['final_t'],
            'weight_history': all_runs[-1][m]['weight_history'],
            'n_seeds':        len(seeds),
        }
    return agg


# ═══════════════════════════════════════════════════
# PART 6: VISUALISATION
# ═══════════════════════════════════════════════════

_COLORS = {'Uniform': '#e74c3c', 'RAR': '#f39c12',
           'PACMANN': '#3498db', 'RL':  '#2ecc71',
           'RL-PPO':  '#16a085', 'RL-PPO-transfer': '#1abc9c',
           'RL-fresh': '#95a5a6', 'RL-transfer(0)': '#27ae60',
           'RL-transfer(ft)': '#2ecc71'}

def _color(name: str) -> str:
    """Return a plot color, falling back to grey for unknown methods."""
    return _COLORS.get(name, '#7f8c8d')


def plot_multi_seed(agg: dict, pde_name: str, pde, n_seeds: int):
    """Plot with shaded std bands from multi-seed runs."""
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))

    # L2 convergence curve with std bands
    ax = axes[0]
    for name, res in agg.items():
        n  = np.array(res['n_colloc'])
        mu = res['l2_mean']
        sd = res['l2_std']
        ax.semilogy(n, mu, color=_color(name), lw=2, marker='o', ms=3, label=name)
        ax.fill_between(n, np.maximum(mu - sd, 1e-5), mu + sd,
                        color=_color(name), alpha=0.15)
    ax.set_xlabel('Collocation points')
    ax.set_ylabel('L2 relative error')
    ax.set_title(f'{pde_name} — L2 vs budget\n({n_seeds} seeds, mean ± std)')
    ax.legend(); ax.grid(alpha=0.3)

    # Final collocation scatter: RL variant vs PACMANN
    ax = axes[1]
    rl_key = next((k for k in agg if k.startswith('RL')), None)
    for name in ([rl_key] if rl_key else []) + ['PACMANN']:
        if name not in agg: continue
        r = agg[name]
        ax.scatter(r['final_x'].numpy(), r['final_t'].numpy(),
                   s=1, alpha=0.2, color=_color(name), label=name)
    ax.set_xlabel('x'); ax.set_ylabel('t')
    ax.set_title('Final collocation: RL vs PACMANN')
    ax.legend(markerscale=6); ax.grid(alpha=0.3)

    # Bar chart: mean ± std
    ax = axes[2]
    names = list(agg.keys())
    means = [agg[n]['final_l2_mean'] for n in names]
    stds  = [agg[n]['final_l2_std']  for n in names]
    bars  = ax.bar(names, means, color=[_color(n) for n in names],
                   edgecolor='white', yerr=stds, capsize=4)
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + s + 0.005,
                f'{m:.4f}\n±{s:.4f}', ha='center', va='bottom', fontsize=7)
    ax.set_ylabel('Final L2 relative error')
    ax.set_title(f'Final accuracy ({n_seeds} seeds)')
    ax.grid(axis='y', alpha=0.3)

    # Reference solution heatmap
    ax = axes[3]
    xg = np.linspace(*pde.x_range, 100); tg = np.linspace(*pde.t_range, 100)
    XX, TT = np.meshgrid(xg, tg)
    xt = torch.tensor(XX.flatten(), dtype=torch.float32)
    tt = torch.tensor(TT.flatten(), dtype=torch.float32)
    u_ref = pde.u_ref(xt, tt).numpy().reshape(100, 100)
    im = ax.imshow(u_ref, origin='lower', aspect='auto',
                   extent=[*pde.x_range, *pde.t_range], cmap='RdBu_r')
    plt.colorbar(im, ax=ax)
    ax.set_xlabel('x'); ax.set_ylabel('t')
    ax.set_title('Reference u(x,t)')

    fig.suptitle(f'RL vs Uniform / RAR / PACMANN — {pde_name}  '
                 f'[equal budget, {n_seeds} seeds]', fontsize=13)
    plt.tight_layout()
    fname = f'pacmann_comparison_{pde_name}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f"Saved {fname}")
    plt.close()


def print_multi_seed_summary(agg: dict, pde_name: str):
    n_seeds = list(agg.values())[0]['n_seeds']
    print(f"\n{'='*72}")
    print(f"  {pde_name.upper()} — Final L2 relative error  ({n_seeds} seeds)")
    print(f"{'='*72}")
    print(f"{'Method':<12} {'Mean ± Std':>18} {'vs Uniform':>12} "
          f"{'vs RAR':>10} {'vs PACMANN':>12}")
    print('-'*72)
    ru = agg['Uniform']['final_l2_mean']
    rr = agg['RAR']['final_l2_mean']
    rp = agg['PACMANN']['final_l2_mean']
    for name, res in agg.items():
        m, s = res['final_l2_mean'], res['final_l2_std']
        print(f"{name:<12} {m:>9.4f} ± {s:<6.4f}   "
              f"{ru/m:>9.2f}x   {rr/m:>7.2f}x   {rp/m:>9.2f}x")
    print('='*72)
    print("(>1x = lower error than that baseline)\n")


# ═══════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pde',    choices=['burgers', 'allen_cahn', 'both'],
                        default='burgers')
    parser.add_argument('--quick',  action='store_true',
                        help='6 steps × 200 epochs (smoke test)')
    parser.add_argument('--seeds',  type=int, default=5,
                        help='Number of random seeds (default 5)')
    parser.add_argument('--no-equal-budget', dest='equal_budget',
                        action='store_false',
                        help='Use growing budget (RAR/RL add points each step)')
    parser.set_defaults(equal_budget=True)
    args = parser.parse_args()

    if args.quick:
        kw = dict(n_adapt_steps=6, epochs_per_step=200,
                  n_total=450, n_replace=50)
    else:
        kw = dict(n_adapt_steps=20, epochs_per_step=500,
                  n_total=2500, n_replace=100)

    seeds = list(range(args.seeds))
    pdes  = ['burgers', 'allen_cahn'] if args.pde == 'both' else [args.pde]

    for pde_name in pdes:
        mode = 'quick' if args.quick else 'full'
        budget = 'equal-budget' if args.equal_budget else 'growing'
        print(f"\n{'#'*60}")
        print(f"#  {pde_name.upper()}  ({mode}, {budget}, {args.seeds} seeds)")
        print(f"{'#'*60}")

        agg = run_multi_seed(pde_name, seeds=seeds,
                             equal_budget=args.equal_budget, **kw)

        pde_obj = Burgers1D() if pde_name == 'burgers' else AllenCahn1D()
        plot_multi_seed(agg, pde_name, pde_obj, n_seeds=args.seeds)
        print_multi_seed_summary(agg, pde_name)
