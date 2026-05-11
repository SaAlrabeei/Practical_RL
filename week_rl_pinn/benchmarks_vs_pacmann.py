"""
RL Collocation vs. Uniform / RAR / PACMANN
==========================================
Benchmarks: 1D Burgers and 1D Allen-Cahn equations.
Matches the PACMANN paper setup (Visser et al. 2024, arXiv:2411.19632):
  - Network  : 4 hidden layers × 64 neurons, tanh
  - Budget   : 2500 interior + 80 BC + 160 IC points
  - Metric   : L2 relative error   ‖u_pred − u_ref‖ / ‖u_ref‖
  - Baselines: Uniform, RAR (Lu et al. 2021), PACMANN (Visser et al. 2024)
  - Proposed : RL collocation (REINFORCE, this work)

PACMANN core idea: freeze PINN weights → compute ∇_x r(x,t)² →
    move collocation points via Adam gradient ascent → unfreeze.
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
# PART 1: REFERENCE SOLUTIONS  (high-accuracy scipy)
# ═══════════════════════════════════════════════════

def _burgers_reference(nu: float = 0.01 / np.pi,
                        nx: int = 512,
                        nt_out: int = 101) -> Tuple:
    """
    Solve viscous Burgers on x∈[-1,1], t∈[0,1] via upwind FD + RK45.
    Returns (x_grid, t_grid, U) where U.shape = (nt_out, nx+2).
    """
    dx = 2.0 / (nx + 1)
    x_int = np.linspace(-1, 1, nx + 2)[1:-1]   # interior

    def rhs(t, u):
        u_full = np.zeros(nx + 2)          # u(-1,t)=u(1,t)=0
        u_full[1:-1] = u
        adv = np.where(u >= 0,
                       u * (u_full[1:-1] - u_full[:-2]) / dx,
                       u * (u_full[2:]   - u_full[1:-1]) / dx)
        diff = nu * (u_full[2:] - 2 * u_full[1:-1] + u_full[:-2]) / dx**2
        return -adv + diff

    u0 = -np.sin(np.pi * x_int)
    t_eval = np.linspace(0, 1, nt_out)
    sol = solve_ivp(rhs, [0, 1], u0, t_eval=t_eval,
                    method='RK45', rtol=1e-8, atol=1e-10)

    x_full = np.linspace(-1, 1, nx + 2)
    U = np.zeros((nt_out, nx + 2))
    U[:, 1:-1] = sol.y.T
    return x_full, t_eval, U


def _allen_cahn_reference(d: float = 0.001,
                           nx: int = 256,
                           nt_out: int = 101) -> Tuple:
    """
    Solve Allen-Cahn on x∈[-1,1], t∈[0,1] via central FD + Radau.
    u_t = d·u_xx + 5(u − u³),  u(±1,t)=-1,  u(x,0)=x²cos(πx)
    Returns (x_grid, t_grid, U) where U.shape = (nt_out, nx+2).
    """
    dx = 2.0 / (nx + 1)
    x_int = np.linspace(-1, 1, nx + 2)[1:-1]

    def rhs(t, u):
        u_full = np.full(nx + 2, -1.0)    # u(±1,t) = -1
        u_full[1:-1] = u
        diff  = d * (u_full[2:] - 2*u_full[1:-1] + u_full[:-2]) / dx**2
        react = 5.0 * (u - u**3)
        return diff + react

    u0 = x_int**2 * np.cos(np.pi * x_int)
    t_eval = np.linspace(0, 1, nt_out)
    sol = solve_ivp(rhs, [0, 1], u0, t_eval=t_eval,
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
    1D viscous Burgers: u_t + u·u_x = ν·u_xx
    x∈[-1,1], t∈[0,1], ν = 0.01/π
    IC: u(x,0) = -sin(πx)    BC: u(±1,t) = 0
    """
    def __init__(self, nu: float = 0.01 / np.pi):
        self.nu = nu
        self.x_range = (-1.0, 1.0)
        self.t_range  = (0.0,  1.0)
        self._build_reference()

    def _build_reference(self):
        print("  Building Burgers reference solution (scipy RK45) …")
        xg, tg, U = _burgers_reference(nu=self.nu)
        self._interp = RegularGridInterpolator(
            (tg, xg), U, method='linear', bounds_error=False, fill_value=None)
        self._t_grid = tg
        self._x_grid = xg
        self._U = U

    def u_ref(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        pts = np.stack([t.detach().numpy(), x.detach().numpy()], axis=-1)
        return torch.tensor(self._interp(pts), dtype=torch.float32)

    def pde_residual(self, pinn, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """r = u_t + u·u_x − ν·u_xx"""
        x = x.requires_grad_(True)
        t = t.requires_grad_(True)
        u = pinn(x, t)
        u_t  = torch.autograd.grad(u, t, torch.ones_like(u),
                                    create_graph=True, retain_graph=True)[0]
        u_x  = torch.autograd.grad(u, x, torch.ones_like(u),
                                    create_graph=True, retain_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x),
                                    create_graph=True, retain_graph=True)[0]
        return u_t + u * u_x - self.nu * u_xx

    def sample_ic(self, n: int) -> Tuple:
        x  = torch.rand(n) * 2 - 1        # x ∈ [-1, 1]
        t  = torch.zeros(n)
        u  = -torch.sin(torch.pi * x)
        return x, t, u

    def sample_bc(self, n_per_side: int) -> Tuple:
        t  = torch.rand(n_per_side)
        x_l = torch.full_like(t, -1.0)
        x_r = torch.full_like(t,  1.0)
        x_bc = torch.cat([x_l, x_r])
        t_bc = torch.cat([t,   t])
        u_bc = torch.zeros(2 * n_per_side)
        return x_bc, t_bc, u_bc


class AllenCahn1D:
    """
    1D Allen-Cahn: u_t − d·u_xx − 5(u−u³) = 0
    x∈[-1,1], t∈[0,1], d = 0.001
    IC: u(x,0) = x²cos(πx)    BC: u(±1,t) = -1
    """
    def __init__(self, d: float = 0.001):
        self.d = d
        self.x_range = (-1.0, 1.0)
        self.t_range  = (0.0,  1.0)
        self._build_reference()

    def _build_reference(self):
        print("  Building Allen-Cahn reference solution (scipy Radau) …")
        xg, tg, U = _allen_cahn_reference(d=self.d)
        self._interp = RegularGridInterpolator(
            (tg, xg), U, method='linear', bounds_error=False, fill_value=None)

    def u_ref(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        pts = np.stack([t.detach().numpy(), x.detach().numpy()], axis=-1)
        return torch.tensor(self._interp(pts), dtype=torch.float32)

    def pde_residual(self, pinn, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """r = u_t − d·u_xx − 5(u − u³)"""
        x = x.requires_grad_(True)
        t = t.requires_grad_(True)
        u    = pinn(x, t)
        u_t  = torch.autograd.grad(u, t, torch.ones_like(u),
                                    create_graph=True, retain_graph=True)[0]
        u_x  = torch.autograd.grad(u, x, torch.ones_like(u),
                                    create_graph=True, retain_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x),
                                    create_graph=True, retain_graph=True)[0]
        return u_t - self.d * u_xx - 5.0 * (u - u**3)

    def sample_ic(self, n: int) -> Tuple:
        x = torch.rand(n) * 2 - 1
        t = torch.zeros(n)
        u = x**2 * torch.cos(torch.pi * x)
        return x, t, u

    def sample_bc(self, n_per_side: int) -> Tuple:
        t    = torch.rand(n_per_side)
        x_l  = torch.full_like(t, -1.0)
        x_r  = torch.full_like(t,  1.0)
        x_bc = torch.cat([x_l, x_r])
        t_bc = torch.cat([t,   t])
        u_bc = torch.full((2 * n_per_side,), -1.0)
        return x_bc, t_bc, u_bc


# ═══════════════════════════════════════════════════
# PART 3: SPACE-TIME PINN
# ═══════════════════════════════════════════════════

class SpaceTimePINN(nn.Module):
    """
    PINN for 1D space-time problems.
    Input: (x, t)  →  Output: u(x,t)
    Architecture: 4 × 64 tanh  (PACMANN paper spec, Section 4)
    """
    def __init__(self, hidden_dim: int = 64, n_layers: int = 4):
        super().__init__()
        layers = [nn.Linear(2, hidden_dim), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers += [nn.Linear(hidden_dim, 1)]
        self.net = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.stack([x, t], dim=-1)).squeeze(-1)

    def compute_l2_rel_error(self, pde, n: int = 5000) -> float:
        """L2 relative error on n random test points (PACMANN metric)."""
        with torch.no_grad():
            x = torch.rand(n) * 2 - 1
            t = torch.rand(n)
            u_pred  = self.forward(x, t)
            u_exact = pde.u_ref(x, t)
            num = torch.sqrt(torch.mean((u_pred - u_exact)**2))
            den = torch.sqrt(torch.mean(u_exact**2))
            return (num / (den + 1e-8)).item()


def train_spacetime_step(pinn: SpaceTimePINN,
                          optimizer,
                          pde,
                          cx: torch.Tensor, ct: torch.Tensor,
                          x_bc: torch.Tensor, t_bc: torch.Tensor, u_bc: torch.Tensor,
                          x_ic: torch.Tensor, t_ic: torch.Tensor, u_ic: torch.Tensor,
                          w_pde: float = 1.0,
                          w_bc:  float = 10.0,
                          w_ic:  float = 10.0):
    """Single Adam step with PDE + BC + IC losses."""
    optimizer.zero_grad()

    L_pde = torch.mean(pde.pde_residual(pinn, cx, ct) ** 2)
    L_bc  = torch.mean((pinn(x_bc, t_bc) - u_bc) ** 2)
    L_ic  = torch.mean((pinn(x_ic, t_ic) - u_ic) ** 2)

    (w_pde * L_pde + w_bc * L_bc + w_ic * L_ic).backward()
    optimizer.step()
    return L_pde.item(), L_bc.item(), L_ic.item()


# ═══════════════════════════════════════════════════
# PART 4: COLLOCATION STRATEGIES
# ═══════════════════════════════════════════════════

# ─────────────────────────────────────
# 4a. Uniform
# ─────────────────────────────────────

class UniformSpaceTime:
    """Fixed random collocation in (x,t) space — sampled once."""
    def __init__(self, n_points: int, x_range=(-1., 1.), t_range=(0., 1.)):
        xl, xr = x_range
        tl, tr = t_range
        self.x = torch.rand(n_points) * (xr - xl) + xl
        self.t = torch.rand(n_points) * (tr - tl) + tl

    def get_points(self, pinn=None, pde=None):
        return self.x, self.t

    def update(self, *args):
        pass


# ─────────────────────────────────────
# 4b. RAR
# ─────────────────────────────────────

class RARSpaceTime:
    """Residual-Adaptive Refinement for (x,t) domain."""
    def __init__(self, n_initial: int, n_add: int,
                 n_candidates: int = 10_000,
                 x_range=(-1., 1.), t_range=(0., 1.)):
        xl, xr = x_range
        tl, tr = t_range
        self.n_add  = n_add
        self.n_cand = n_candidates
        self.x_range = x_range
        self.t_range  = t_range
        self.x = torch.rand(n_initial) * (xr - xl) + xl
        self.t = torch.rand(n_initial) * (tr - tl) + tl

    def get_points(self, pinn=None, pde=None):
        return self.x, self.t

    def update(self, pinn: SpaceTimePINN, pde):
        xl, xr = self.x_range
        tl, tr = self.t_range
        xc = torch.rand(self.n_cand) * (xr - xl) + xl
        tc = torch.rand(self.n_cand) * (tr - tl) + tl
        res_vals = []
        for i in range(0, self.n_cand, 500):
            xb = xc[i:i+500]; tb = tc[i:i+500]
            res_vals.append(pde.pde_residual(pinn, xb, tb).detach().abs())
        res_vals = torch.cat(res_vals)
        _, idx = torch.topk(res_vals, self.n_add)
        self.x = torch.cat([self.x, xc[idx]])
        self.t = torch.cat([self.t, tc[idx]])


# ─────────────────────────────────────
# 4c. PACMANN  (Visser et al. 2024)
# ─────────────────────────────────────

class PACMANNCollocation:
    """
    Point Adaptive Collocation Method for ANNs.
    Moves collocation points via gradient ascent on r(x,t)²
    with PINN weights frozen  (Section 3, Visser et al. 2024).

    Hyperparameters:
        n_steps  T  — gradient-ascent steps per resampling event
        lr       α  — Adam learning rate for point movement
    """
    def __init__(self, n_points: int,
                 n_steps: int = 20,
                 lr: float = 1e-2,
                 x_range=(-1., 1.), t_range=(0., 1.)):
        xl, xr = x_range
        tl, tr = t_range
        self.n_steps = n_steps
        self.lr      = lr
        self.x_range = x_range
        self.t_range  = t_range
        self.x = torch.rand(n_points) * (xr - xl) + xl
        self.t = torch.rand(n_points) * (tr - tl) + tl

    def get_points(self, pinn=None, pde=None):
        return self.x, self.t

    def update(self, pinn: SpaceTimePINN, pde):
        """
        Freeze PINN → Adam-ascent on ‖r‖² w.r.t. (x,t) → unfreeze.
        Points that leave the domain are clipped back inside.
        """
        xl, xr = self.x_range
        tl, tr = self.t_range

        # Freeze PINN weights
        for p in pinn.parameters():
            p.requires_grad_(False)

        x = self.x.detach().clone().requires_grad_(True)
        t = self.t.detach().clone().requires_grad_(True)

        # Maximise ‖r‖² — use Adam with negated loss (gradient ascent)
        opt = torch.optim.Adam([x, t], lr=self.lr)

        for _ in range(self.n_steps):
            opt.zero_grad()
            res  = pde.pde_residual(pinn, x, t)
            loss = -torch.mean(res ** 2)   # negate → maximise
            loss.backward()
            opt.step()
            with torch.no_grad():
                x.clamp_(xl, xr)
                t.clamp_(tl, tr)

        self.x = x.detach()
        self.t = t.detach()

        # Unfreeze PINN weights
        for p in pinn.parameters():
            p.requires_grad_(True)


# ─────────────────────────────────────
# 4d. RL Collocation (this work)
# ─────────────────────────────────────

def _make_density_map_2d(x, t, G, x_range, t_range):
    """G×G histogram of (x,t) point positions."""
    xl, xr = x_range
    tl, tr = t_range
    density = torch.zeros(G, G)
    ix = ((x - xl) / (xr - xl) * G).long().clamp(0, G - 1)
    it = ((t - tl) / (tr - tl) * G).long().clamp(0, G - 1)
    for i, j in zip(ix, it):
        density[i, j] += 1
    return density


def _entropy(density: torch.Tensor) -> float:
    p = density.flatten().float()
    p = p / (p.sum() + 1e-8)
    return -(p * (p + 1e-8).log()).sum().item()


class RLAgentSpaceTime(nn.Module):
    """REINFORCE agent for (x,t) collocation — G×G grid."""
    def __init__(self, G: int = 16):
        super().__init__()
        self.G = G
        state_dim  = G * G * 3 + 3
        action_dim = G * G

        self.policy = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256),       nn.ReLU(),
            nn.Linear(256, action_dim),
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
        self.log_probs: List[torch.Tensor] = []
        self.rewards:   List[float]        = []

    def get_state(self, pinn, pde, x, t, l2_rel, n_total,
                   x_range, t_range) -> torch.Tensor:
        G   = self.G
        eps = 1e-8
        xl, xr = x_range
        tl, tr  = t_range

        gx = torch.linspace(xl, xr, G)
        gt = torch.linspace(tl, tr, G)
        XX, TT = torch.meshgrid(gx, gt, indexing='ij')
        xf, tf = XX.flatten(), TT.flatten()

        res_map = pde.pde_residual(pinn, xf, tf).detach().abs().reshape(G, G)

        xg, tg = xf.requires_grad_(True), tf.requires_grad_(True)
        u2 = pinn(xg, tg)
        ux = torch.autograd.grad(u2, xg, torch.ones_like(u2),
                                  retain_graph=True, create_graph=False)[0].detach()
        ut = torch.autograd.grad(u2, tg, torch.ones_like(u2),
                                  create_graph=False)[0].detach()
        grad_map = (ux**2 + ut**2).sqrt().reshape(G, G)

        density_map = _make_density_map_2d(x, t, G, x_range, t_range)
        density_map = density_map / (density_map.max() + eps)
        res_map     = res_map     / (res_map.max()     + eps)
        grad_map    = grad_map    / (grad_map.max()    + eps)

        scalars = torch.tensor([l2_rel, len(x) / n_total, res_map.mean().item()])
        return torch.cat([res_map.flatten(), grad_map.flatten(),
                          density_map.flatten(), scalars])

    def act(self, state):
        weights  = torch.softmax(self.policy(state), dim=-1)
        dist     = torch.distributions.Categorical(probs=weights)
        idx      = dist.sample((100,))
        log_prob = dist.log_prob(idx).mean()
        return weights, log_prob

    def sample_points(self, weights, n, x_range, t_range):
        G  = self.G
        xl, xr = x_range
        tl, tr  = t_range
        cell_idx = torch.multinomial(weights, n, replacement=True)
        ix = cell_idx // G;  it = cell_idx % G
        sx = (xr - xl) / G;  st = (tr - tl) / G
        x  = (ix.float() * sx + torch.rand(n) * sx + xl).clamp(xl, xr)
        t  = (it.float() * st + torch.rand(n) * st + tl).clamp(tl, tr)
        return x, t

    def update_policy(self, reward: float):
        self.rewards.append(reward)
        if len(self.log_probs) == 0:
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
    """RL collocation wrapper for (x,t) space-time domains."""
    W1, W2 = 1.0, 0.1

    def __init__(self, n_initial, n_add, G=16,
                 x_range=(-1., 1.), t_range=(0., 1.)):
        self.n_add   = n_add
        self.G       = G
        self.x_range = x_range
        self.t_range  = t_range
        self.agent   = RLAgentSpaceTime(G=G)
        xl, xr = x_range; tl, tr = t_range
        self.x = torch.rand(n_initial) * (xr - xl) + xl
        self.t = torch.rand(n_initial) * (tr - tl) + tl
        self.prev_l2: Optional[float]        = None
        self.prev_density: Optional[torch.Tensor] = None
        self.weight_history: List            = []
        self._step = 0

    def get_points(self, pinn=None, pde=None):
        return self.x, self.t

    def observe_and_act(self, pinn, pde, l2_rel, n_total):
        self.prev_density = _make_density_map_2d(
            self.x, self.t, self.G, self.x_range, self.t_range)
        state = self.agent.get_state(
            pinn, pde, self.x, self.t, l2_rel, n_total,
            self.x_range, self.t_range)
        weights, log_prob = self.agent.act(state)
        self._step += 1
        if self._step in (1, 5, 10, 20):
            self.weight_history.append(
                (self._step, weights.detach().reshape(self.G, self.G).clone()))
        self.agent.log_probs.append(log_prob)
        self.prev_l2 = l2_rel
        nx, nt = self.agent.sample_points(
            weights.detach(), self.n_add, self.x_range, self.t_range)
        self.x = torch.cat([self.x, nx])
        self.t = torch.cat([self.t, nt])

    def update(self, current_l2: float) -> float:
        new_density  = _make_density_map_2d(
            self.x, self.t, self.G, self.x_range, self.t_range)
        l2_reward    = self.W1 * (self.prev_l2 - current_l2) / self.n_add * 1000
        entropy_gain = _entropy(new_density) - _entropy(self.prev_density)
        reward       = l2_reward + self.W2 * entropy_gain
        self.agent.update_policy(reward)
        return reward


# ═══════════════════════════════════════════════════
# PART 5: EXPERIMENT RUNNER
# ═══════════════════════════════════════════════════

def run_spacetime_experiment(
        strategy_name: str, strategy, pde,
        n_adapt_steps:   int = 20,
        epochs_per_step: int = 500,
        n_initial:       int = 500,
        n_add:           int = 100,
        n_bc:            int = 40,    # per side → 80 total
        n_ic:            int = 160,
        seed:            int = 42,
        verbose:         bool = True) -> dict:

    torch.manual_seed(seed)
    np.random.seed(seed)

    pinn      = SpaceTimePINN(hidden_dim=64, n_layers=4)
    optimizer = torch.optim.Adam(pinn.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=2000, gamma=0.5)

    x_bc, t_bc, u_bc = pde.sample_bc(n_per_side=n_bc)
    x_ic, t_ic, u_ic = pde.sample_ic(n_ic)

    n_total = n_initial + n_adapt_steps * n_add
    l2_hist, nc_hist, res_hist = [], [], []

    if verbose:
        print(f"\n{'='*50}\n{strategy_name}\n{'='*50}")

    cx, ct = strategy.get_points(pinn, pde)

    for step in range(n_adapt_steps + 1):
        for _ in range(epochs_per_step):
            train_spacetime_step(pinn, optimizer, pde,
                                  cx, ct, x_bc, t_bc, u_bc,
                                  x_ic, t_ic, u_ic)
            scheduler.step()

        l2  = pinn.compute_l2_rel_error(pde)
        l2_hist.append(l2)
        nc_hist.append(len(cx))
        gx, gt = torch.linspace(*pde.x_range, 16), torch.linspace(*pde.t_range, 16)
        XX, TT = torch.meshgrid(gx, gt, indexing='ij')
        res_hist.append(
            pde.pde_residual(pinn, XX.flatten(), TT.flatten())
            .detach().abs().reshape(16, 16)
        )

        if verbose:
            print(f"  step {step:3d} | N={len(cx):5d} | L2_rel={l2:.5f}")
        if step == n_adapt_steps:
            break

        if strategy_name == 'RL':
            if strategy.prev_l2 is not None:
                strategy.update(l2)
            strategy.observe_and_act(pinn, pde, l2, n_total)
        else:
            strategy.update(pinn, pde)

        cx, ct = strategy.get_points(pinn, pde)

    return {
        'name':     strategy_name,
        'l2':       l2_hist,
        'n_colloc': nc_hist,
        'final_l2': l2_hist[-1],
        'final_x':  cx.detach(),
        'final_t':  ct.detach(),
        'res_hist': res_hist,
        'weight_history': getattr(strategy, 'weight_history', []),
    }


def run_comparison(pde_name: str = 'burgers',
                    n_adapt_steps:   int = 20,
                    epochs_per_step: int = 500,
                    n_initial:       int = 500,
                    n_add:           int = 100) -> dict:
    """
    4-way comparison: Uniform / RAR / PACMANN / RL
    on either 'burgers' or 'allen_cahn'.
    """
    if pde_name == 'burgers':
        pde = Burgers1D()
    elif pde_name == 'allen_cahn':
        pde = AllenCahn1D()
    else:
        raise ValueError(f"Unknown PDE: {pde_name}")

    xr = pde.x_range
    tr = pde.t_range
    n_total = n_initial + n_adapt_steps * n_add

    strategies = {
        'Uniform': UniformSpaceTime(n_total, xr, tr),
        'RAR':     RARSpaceTime(n_initial, n_add, x_range=xr, t_range=tr),
        # PACMANN starts with the full budget and moves points — never adds.
        # This matches Visser et al. 2024 where N_colloc is fixed throughout.
        'PACMANN': PACMANNCollocation(n_total, n_steps=20, lr=1e-2,
                                       x_range=xr, t_range=tr),
        'RL':      RLSpaceTime(n_initial, n_add, G=16, x_range=xr, t_range=tr),
    }

    results = {}
    kw = dict(n_adapt_steps=n_adapt_steps, epochs_per_step=epochs_per_step,
              n_initial=n_initial, n_add=n_add)
    for name, strat in strategies.items():
        results[name] = run_spacetime_experiment(name, strat, pde, **kw)

    return results


# ═══════════════════════════════════════════════════
# PART 6: VISUALISATION
# ═══════════════════════════════════════════════════

_COLORS = {'Uniform': '#e74c3c', 'RAR': '#f39c12',
           'PACMANN': '#3498db',  'RL':  '#2ecc71'}


def plot_spacetime_comparison(results: dict, pde_name: str, pde):
    """
    4-panel figure:
      1. L2 relative error vs. collocation budget
      2. Final collocation scatter (x,t) for RL vs PACMANN
      3. Final L2 bar chart
      4. Reference solution heatmap
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Panel 1: L2 rel error curves
    ax = axes[0]
    for name, res in results.items():
        ax.semilogy(res['n_colloc'], res['l2'],
                    color=_COLORS[name], lw=2, marker='o', ms=4, label=name)
    ax.set_xlabel('Collocation points'); ax.set_ylabel('L2 relative error')
    ax.set_title(f'{pde_name} — accuracy vs. budget'); ax.legend(); ax.grid(alpha=0.3)

    # Panel 2: Scatter of final colloc points
    ax = axes[1]
    for name in ('RL', 'PACMANN'):
        if name not in results: continue
        res = results[name]
        ax.scatter(res['final_x'].numpy(), res['final_t'].numpy(),
                   s=1, alpha=0.25, color=_COLORS[name], label=name)
    ax.set_xlabel('x'); ax.set_ylabel('t')
    ax.set_title('Final colloc: RL vs PACMANN')
    ax.legend(markerscale=6); ax.grid(alpha=0.3)

    # Panel 3: Final L2 bar chart
    ax = axes[2]
    names = list(results.keys())
    errs  = [results[n]['final_l2'] for n in names]
    bars  = ax.bar(names, errs, color=[_COLORS[n] for n in names], edgecolor='white')
    for bar, e in zip(bars, errs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.05,
                f'{e:.4f}', ha='center', va='bottom', fontsize=8)
    ax.set_ylabel('Final L2 relative error')
    ax.set_title('Final accuracy comparison'); ax.grid(axis='y', alpha=0.3)

    # Panel 4: Reference solution heatmap
    ax = axes[3]
    nx_plot, nt_plot = 100, 100
    xg = np.linspace(*pde.x_range, nx_plot)
    tg = np.linspace(*pde.t_range, nt_plot)
    XX, TT = np.meshgrid(xg, tg)
    xt = torch.tensor(XX.flatten(), dtype=torch.float32)
    tt = torch.tensor(TT.flatten(), dtype=torch.float32)
    u_ref = pde.u_ref(xt, tt).numpy().reshape(nt_plot, nx_plot)
    im = ax.imshow(u_ref, origin='lower', aspect='auto',
                   extent=[*pde.x_range, *pde.t_range], cmap='RdBu_r')
    plt.colorbar(im, ax=ax)
    ax.set_xlabel('x'); ax.set_ylabel('t')
    ax.set_title('Reference solution u(x,t)')

    fig.suptitle(f'RL Collocation vs. Uniform / RAR / PACMANN — {pde_name}',
                 fontsize=13)
    plt.tight_layout()
    fname = f'pacmann_comparison_{pde_name}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f"Saved {fname}")
    plt.show()


def print_summary(results: dict, pde_name: str):
    """Table comparing all four methods."""
    print(f"\n{'='*65}")
    print(f"{'Method':<12} {'Final L2 rel':>14} {'vs Uniform':>12} "
          f"{'vs RAR':>10} {'vs PACMANN':>12}")
    print('='*65)
    ref_u = results['Uniform']['final_l2']
    ref_r = results['RAR']['final_l2']
    ref_p = results['PACMANN']['final_l2']
    for name, res in results.items():
        l2 = res['final_l2']
        print(f"{name:<12} {l2:>14.6f} {ref_u/l2:>11.2f}x "
              f"{ref_r/l2:>9.2f}x {ref_p/l2:>11.2f}x")
    print('='*65)
    print("(>1x means the method achieves lower error for same budget)\n")


# ═══════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='RL vs Uniform/RAR/PACMANN on Burgers & Allen-Cahn')
    parser.add_argument('--pde', choices=['burgers', 'allen_cahn', 'both'],
                        default='burgers')
    parser.add_argument('--quick', action='store_true',
                        help='Fast run: 5 steps × 200 epochs')
    args = parser.parse_args()

    kw = dict(n_adapt_steps=5, epochs_per_step=200,
               n_initial=200, n_add=50) if args.quick else \
         dict(n_adapt_steps=20, epochs_per_step=500,
               n_initial=500, n_add=100)

    pdes = (['burgers', 'allen_cahn'] if args.pde == 'both'
            else [args.pde])

    for pde_name in pdes:
        print(f"\n{'#'*60}")
        print(f"#  {pde_name.upper()}  ({'quick' if args.quick else 'full'})")
        print(f"{'#'*60}")

        results = run_comparison(pde_name=pde_name, **kw)

        # Rebuild pde object for reference plot
        pde = Burgers1D() if pde_name == 'burgers' else AllenCahn1D()
        plot_spacetime_comparison(results, pde_name, pde)
        print_summary(results, pde_name)
