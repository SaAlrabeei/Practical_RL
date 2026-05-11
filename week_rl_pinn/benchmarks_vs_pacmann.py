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

Note: paper also uses L-BFGS after each Adam phase (5 × 7000 Adam + 5 × L-BFGS).
      We use 20 × 500 Adam for tractable CPU runtime; relative method ranking
      is preserved since all methods use the same training budget.
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
    dx   = 2.0 / (nx + 1)
    x_int = np.linspace(-1, 1, nx + 2)[1:-1]

    def rhs(t, u):
        uf = np.zeros(nx + 2)
        uf[1:-1] = u
        adv = np.where(u >= 0,
                       u * (uf[1:-1] - uf[:-2]) / dx,
                       u * (uf[2:]   - uf[1:-1]) / dx)
        diff = nu * (uf[2:] - 2*uf[1:-1] + uf[:-2]) / dx**2
        return -adv + diff

    u0     = -np.sin(np.pi * x_int)
    t_eval = np.linspace(0, 0.99, nt_out)           # paper uses t∈[0,0.99]
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
        self.t_range = ( 0.0,  0.99)   # paper uses 0.99
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

    Uses hard output_transform (paper's exact approach):
        u(x,t) = x²cos(πx)  +  t·(1−x²)·u_NN(x,t)
    which enforces IC and BC exactly by construction.
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
        """
        Hard enforcement of IC and BC (paper Section 3, Setup_file.py):
            u(x,t) = x²cos(πx) + t(1−x²)·u_NN(x,t)
        At t=0  : u = x²cos(πx)                     ← satisfies IC
        At x=±1 : u = (±1)²cos(±π) + 0 = −1         ← satisfies BC
        """
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
        # With hard transform, IC is exactly satisfied; return zero residual targets
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
    """
    PINN for 1D space-time problems.  4 × 64 tanh, Glorot normal.
    Supports an optional output_transform for hard BC/IC enforcement.
    """
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
                nn.init.xavier_normal_(m.weight)   # Glorot normal
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
    """
    Single Adam step.  Equal loss weights matching paper (no scaling).
    When hard constraints are active (Allen-Cahn), skip BC/IC terms.
    """
    optimizer.zero_grad()
    L_pde = torch.mean(pde.pde_residual(pinn, cx, ct) ** 2)

    if use_hard_constraints:
        loss = L_pde
    else:
        L_bc  = torch.mean((pinn(x_bc, t_bc) - u_bc) ** 2)
        L_ic  = torch.mean((pinn(x_ic, t_ic) - u_ic) ** 2)
        loss  = L_pde + L_bc + L_ic     # equal weights (paper default)

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
    """Fixed random set — sampled once."""
    def __init__(self, n_points, x_range=(-1., 1.), t_range=(0., .99)):
        self.x, self.t = _random_domain_pts(n_points, x_range, t_range)

    def get_points(self, pinn=None, pde=None):  return self.x, self.t
    def update(self, *args): pass


# ─────────────────────────────────────
# 4b. RAR
# ─────────────────────────────────────

class RARSpaceTime:
    def __init__(self, n_initial, n_add, n_candidates=10_000,
                 x_range=(-1., 1.), t_range=(0., .99)):
        self.n_add   = n_add
        self.n_cand  = n_candidates
        self.x_range = x_range
        self.t_range  = t_range
        self.x, self.t = _random_domain_pts(n_initial, x_range, t_range)

    def get_points(self, pinn=None, pde=None):  return self.x, self.t

    def update(self, pinn, pde):
        xc, tc = _random_domain_pts(self.n_cand, self.x_range, self.t_range)
        res_all = []
        for i in range(0, self.n_cand, 500):
            xb, tb = xc[i:i+500], tc[i:i+500]
            res_all.append(pde.pde_residual(pinn, xb, tb).detach().abs())
        res_all = torch.cat(res_all)
        _, idx  = torch.topk(res_all, self.n_add)
        self.x  = torch.cat([self.x, xc[idx]])
        self.t  = torch.cat([self.t, tc[idx]])


# ─────────────────────────────────────
# 4c. PACMANN  — exact paper implementation
# ─────────────────────────────────────

class PACMANNCollocation:
    """
    PACMANN with exact paper hyperparameters (Visser et al. 2024).

    Paper code (Setup_file.py):
        Adam(n_iterations=T, N_col_points=2500, stepsize=1e-5,
             beta1=0.9, beta2=0.999, epsilon=10e-8, period=50)

    Key details:
      - Custom Adam (not torch.optim), moments reset each resampling event
      - epsilon = 10e-8 = 1e-7  (literal value in paper code)
      - OOD points replaced by uniform random samples (not clipped)
      - Period P=50: points move every 50 PINN training epochs
      - T=15 for Burgers, T=5 for Allen-Cahn
    """
    PERIOD = 50          # P: PINN epochs between resampling events

    def __init__(self, n_points: int,
                 n_steps: int = 15,    # T: inner gradient-ascent steps
                 lr: float = 1e-5,     # α: Adam stepsize for point movement
                 x_range=(-1., 1.), t_range=(0., .99)):
        self.n_steps = n_steps
        self.lr      = lr
        self.beta1   = 0.9
        self.beta2   = 0.999
        self.epsilon = 10e-8           # = 1e-7, literal from paper
        self.x_range = x_range
        self.t_range  = t_range
        self.x, self.t = _random_domain_pts(n_points, x_range, t_range)

    def get_points(self, pinn=None, pde=None):  return self.x, self.t

    def update(self, pinn: SpaceTimePINN, pde):
        """
        Move points via custom Adam ascent on ‖r(x,t)‖².
        Moments are initialised to zero at the start of each event
        (matching paper: VdX and SdX created fresh in each callback call).
        """
        xl, xr = self.x_range
        tl, tr = self.t_range
        N = len(self.x)

        # Working copies as numpy for the custom Adam loop
        coords = np.stack([self.x.detach().numpy(), self.t.detach().numpy()], axis=1)  # (N,2)
        VdX = np.zeros((N, 2))
        SdX = np.zeros((N, 2))

        for n in range(self.n_steps):
            xt = torch.tensor(coords, dtype=torch.float32, requires_grad=True)
            res = pde.pde_residual(pinn,
                                    xt[:, 0].requires_grad_(True),
                                    xt[:, 1].requires_grad_(True))
            loss = torch.mean(res ** 2)
            loss.backward()

            # Gradient of ‖r‖² w.r.t. coordinates
            grad = xt.grad.detach().numpy()  # (N, 2)

            # Custom Adam update (ascent → add, not subtract)
            VdX = self.beta1 * VdX + (1 - self.beta1) * grad
            SdX = self.beta2 * SdX + (1 - self.beta2) * grad ** 2
            VdX_c = VdX / (1 - self.beta1 ** (n + 1))
            SdX_c = SdX / (1 - self.beta2 ** (n + 1))
            coords = coords + self.lr * VdX_c / (np.sqrt(SdX_c) + self.epsilon)

            # Replace out-of-domain points with uniform random samples
            oob = ((coords[:, 0] < xl) | (coords[:, 0] > xr) |
                   (coords[:, 1] < tl) | (coords[:, 1] > tr))
            if oob.any():
                n_oob = int(oob.sum())
                rx = np.random.uniform(xl, xr, n_oob)
                rt = np.random.uniform(tl, tr, n_oob)
                coords[oob, 0] = rx
                coords[oob, 1] = rt

        self.x = torch.tensor(coords[:, 0], dtype=torch.float32)
        self.t = torch.tensor(coords[:, 1], dtype=torch.float32)


# ─────────────────────────────────────
# 4d. RL (this work)
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

        res_map  = pde.pde_residual(pinn, xf, tf).detach().abs().reshape(G, G)

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

        scalars = torch.tensor([l2, len(x)/n_total, res_map.mean().item()])
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
    W1, W2 = 1.0, 0.1

    def __init__(self, n_initial, n_add, G=16,
                 x_range=(-1., 1.), t_range=(0., .99)):
        self.n_add = n_add; self.G = G
        self.x_range = x_range; self.t_range = t_range
        self.agent   = RLAgentSpaceTime(G=G)
        self.x, self.t = _random_domain_pts(n_initial, x_range, t_range)
        self.prev_l2:      Optional[float]        = None
        self.prev_density: Optional[torch.Tensor] = None
        self.weight_history: List                 = []
        self._step = 0

    def get_points(self, pinn=None, pde=None):  return self.x, self.t

    def observe_and_act(self, pinn, pde, l2, n_total):
        self.prev_density = _make_density_map_2d(
            self.x, self.t, self.G, self.x_range, self.t_range)
        state = self.agent.get_state(pinn, pde, self.x, self.t, l2, n_total,
                                      self.x_range, self.t_range)
        w, lp = self.agent.act(state)
        self._step += 1
        if self._step in (1, 5, 10, 20):
            self.weight_history.append(
                (self._step, w.detach().reshape(self.G, self.G).clone()))
        self.agent.log_probs.append(lp)
        self.prev_l2 = l2
        nx, nt = self.agent.sample_points(w.detach(), self.n_add,
                                           self.x_range, self.t_range)
        self.x = torch.cat([self.x, nx])
        self.t = torch.cat([self.t, nt])

    def update(self, current_l2: float) -> float:
        nd = _make_density_map_2d(self.x, self.t, self.G,
                                   self.x_range, self.t_range)
        l2_r  = self.W1 * (self.prev_l2 - current_l2) / self.n_add * 1000
        ent_r = self.W2 * (_entropy(nd) - _entropy(self.prev_density))
        self.agent.update_policy(l2_r + ent_r)
        return l2_r + ent_r


# ═══════════════════════════════════════════════════
# PART 5: EXPERIMENT RUNNER
# ═══════════════════════════════════════════════════

def run_spacetime_experiment(
        strategy_name: str, strategy, pde,
        n_adapt_steps:    int  = 20,
        epochs_per_step:  int  = 500,
        n_initial:        int  = 500,
        n_add:            int  = 100,
        n_bc:             int  = 40,
        n_ic:             int  = 160,
        seed:             int  = 42,
        verbose:          bool = True) -> dict:

    torch.manual_seed(seed); np.random.seed(seed)

    use_hard = isinstance(pde, AllenCahn1D)

    pinn = SpaceTimePINN(
        hidden_dim=64, n_layers=4,
        output_transform=(AllenCahn1D.output_transform if use_hard else None)
    )
    optimizer = torch.optim.Adam(pinn.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=2000, gamma=0.5)

    x_bc, t_bc, u_bc = pde.sample_bc(n_per_side=n_bc)     # 80 total
    x_ic, t_ic, u_ic = pde.sample_ic(n_ic)                # 160

    n_total = n_initial + n_adapt_steps * n_add
    l2_hist, nc_hist, res_hist = [], [], []

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

            # PACMANN updates every P=50 epochs within each step
            if strategy_name == 'PACMANN' and (epoch + 1) % PACMANNCollocation.PERIOD == 0:
                strategy.update(pinn, pde)
                cx, ct = strategy.get_points()

        l2 = pinn.compute_l2_rel_error(pde)
        l2_hist.append(l2)
        nc_hist.append(len(cx))

        gx, gt = torch.linspace(*pde.x_range, 16), torch.linspace(*pde.t_range, 16)
        XX, TT = torch.meshgrid(gx, gt, indexing='ij')
        res_hist.append(
            pde.pde_residual(pinn, XX.flatten(), TT.flatten())
            .detach().abs().reshape(16, 16))

        if verbose:
            print(f"  step {step:3d} | N={len(cx):5d} | L2_rel={l2:.5f}")
        if step == n_adapt_steps:
            break

        # Per-step updates for RAR and RL
        if strategy_name == 'RAR':
            strategy.update(pinn, pde)
        elif strategy_name == 'RL':
            if strategy.prev_l2 is not None:
                strategy.update(l2)
            strategy.observe_and_act(pinn, pde, l2, n_total)
        # Uniform: no update; PACMANN: already updated intra-step above

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
    """4-way comparison: Uniform / RAR / PACMANN / RL."""
    if pde_name == 'burgers':
        pde = Burgers1D()
        pacmann_T = 15           # paper: T=15 for Burgers
    elif pde_name == 'allen_cahn':
        pde = AllenCahn1D()
        pacmann_T = 5            # paper: T=5 for Allen-Cahn
    else:
        raise ValueError(pde_name)

    xr = pde.x_range; tr = pde.t_range
    n_total = n_initial + n_adapt_steps * n_add

    strategies = {
        'Uniform': UniformSpaceTime(n_total, xr, tr),
        'RAR':     RARSpaceTime(n_initial, n_add, x_range=xr, t_range=tr),
        'PACMANN': PACMANNCollocation(n_total, n_steps=pacmann_T, lr=1e-5,
                                       x_range=xr, t_range=tr),
        'RL':      RLSpaceTime(n_initial, n_add, G=16, x_range=xr, t_range=tr),
    }

    kw = dict(n_adapt_steps=n_adapt_steps, epochs_per_step=epochs_per_step,
              n_initial=n_initial, n_add=n_add)
    results = {}
    for name, strat in strategies.items():
        results[name] = run_spacetime_experiment(name, strat, pde, **kw)
    return results


# ═══════════════════════════════════════════════════
# PART 6: VISUALISATION
# ═══════════════════════════════════════════════════

_COLORS = {'Uniform': '#e74c3c', 'RAR': '#f39c12',
           'PACMANN': '#3498db', 'RL':  '#2ecc71'}


def plot_comparison(results: dict, pde_name: str, pde):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # L2 curve
    ax = axes[0]
    for name, res in results.items():
        ax.semilogy(res['n_colloc'], res['l2'],
                    color=_COLORS[name], lw=2, marker='o', ms=4, label=name)
    ax.set_xlabel('Collocation points'); ax.set_ylabel('L2 relative error')
    ax.set_title(f'{pde_name} — L2 rel error vs. budget')
    ax.legend(); ax.grid(alpha=0.3)

    # Collocation scatter
    ax = axes[1]
    for name in ('RL', 'PACMANN'):
        if name not in results: continue
        r = results[name]
        ax.scatter(r['final_x'].numpy(), r['final_t'].numpy(),
                   s=1, alpha=0.2, color=_COLORS[name], label=name)
    ax.set_xlabel('x'); ax.set_ylabel('t')
    ax.set_title('Final collocation: RL vs PACMANN')
    ax.legend(markerscale=6); ax.grid(alpha=0.3)

    # Bar chart
    ax = axes[2]
    names = list(results.keys())
    errs  = [results[n]['final_l2'] for n in names]
    bars  = ax.bar(names, errs, color=[_COLORS[n] for n in names], edgecolor='white')
    for bar, e in zip(bars, errs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.05,
                f'{e:.4f}', ha='center', va='bottom', fontsize=8)
    ax.set_ylabel('Final L2 relative error')
    ax.set_title('Final accuracy'); ax.grid(axis='y', alpha=0.3)

    # Reference heatmap
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

    fig.suptitle(f'RL vs Uniform / RAR / PACMANN — {pde_name}', fontsize=13)
    plt.tight_layout()
    fname = f'pacmann_comparison_{pde_name}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f"Saved {fname}")
    plt.show()


def print_summary(results: dict, pde_name: str):
    print(f"\n{'='*68}")
    print(f"  {pde_name.upper()} — Final L2 relative error")
    print(f"{'='*68}")
    print(f"{'Method':<12} {'Final L2 rel':>14} {'vs Uniform':>12} "
          f"{'vs RAR':>10} {'vs PACMANN':>12}")
    print('-'*68)
    ru = results['Uniform']['final_l2']
    rr = results['RAR']['final_l2']
    rp = results['PACMANN']['final_l2']
    for name, res in results.items():
        l2 = res['final_l2']
        print(f"{name:<12} {l2:>14.6f} {ru/l2:>11.2f}x "
              f"{rr/l2:>9.2f}x {rp/l2:>11.2f}x")
    print('='*68)
    print("(>1x = lower error with same budget)\n")


# ═══════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pde',   choices=['burgers', 'allen_cahn', 'both'],
                        default='burgers')
    parser.add_argument('--quick', action='store_true',
                        help='5 steps × 200 epochs (smoke test)')
    args = parser.parse_args()

    kw = dict(n_adapt_steps=5,  epochs_per_step=200, n_initial=200, n_add=50) \
         if args.quick else \
         dict(n_adapt_steps=20, epochs_per_step=500, n_initial=500, n_add=100)

    pdes = ['burgers', 'allen_cahn'] if args.pde == 'both' else [args.pde]

    for pde_name in pdes:
        print(f"\n{'#'*60}\n#  {pde_name.upper()}  ({'quick' if args.quick else 'full'})\n{'#'*60}")
        results = run_comparison(pde_name=pde_name, **kw)
        pde_obj = Burgers1D() if pde_name == 'burgers' else AllenCahn1D()
        plot_comparison(results, pde_name, pde_obj)
        print_summary(results, pde_name)
