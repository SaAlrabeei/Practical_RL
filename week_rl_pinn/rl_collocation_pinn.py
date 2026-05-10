"""
Reinforcement Learning for Adaptive Collocation in PINNs
=========================================================
Implements the MDP framework from the paper draft:
  State  : residual map + gradient map + density map + scalar features (G×G grid)
  Action : softmax sampling weights over G×G grid cells
  Reward : w1·ΔL2/n_new + w2·ΔH(density) − w3·n_new  (Section 3.2)

Three strategies are compared:
  1. Uniform   — fixed random set (baseline)
  2. RAR        — residual-adaptive refinement  (Lu et al. 2021)
  3. RL         — REINFORCE agent (this work)

Experiments run on the sharp Poisson equation (Section 5.1) with
sharpness k ∈ {5, 20, 50}.  Multi-seed aggregation and four
visualisation modes match Sections 5.3–5.6.
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')          # headless — no display required
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Optional, List, Tuple


# ═══════════════════════════════════════════════════
# PART 1: THE PDE AND ANALYTICAL SOLUTION
# ═══════════════════════════════════════════════════

class SharpPoisson:
    """
    2D Poisson equation with sharp internal layer.

    −∇²u = f(x,y)   on [0,1]²
    u = g(x,y)        on boundary

    Analytical solution:
        u(x,y) = tanh(k·(x−0.5)) · tanh(k·(y−0.5))

    Source term f derived analytically from −∇²u.
    """

    def __init__(self, k: float = 20.0):
        self.k = k

    def u_exact(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        k = self.k
        return torch.tanh(k * (x - 0.5)) * torch.tanh(k * (y - 0.5))

    def f_source(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """RHS: f = −∇²u, derived analytically."""
        k = self.k
        th_x    = torch.tanh(k * (x - 0.5))
        th_y    = torch.tanh(k * (y - 0.5))
        sech2_x = 1.0 - th_x ** 2
        sech2_y = 1.0 - th_y ** 2
        d2u_dx2 = -2.0 * k**2 * th_x * sech2_x * th_y
        d2u_dy2 = -2.0 * k**2 * th_y * sech2_y * th_x
        return -(d2u_dx2 + d2u_dy2)

    def u_boundary(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.u_exact(x, y)

    def sample_boundary(self, n_per_side: int) -> Tuple:
        """Sample n_per_side points on each of the 4 boundary edges."""
        t   = torch.linspace(0, 1, n_per_side)
        z   = torch.zeros(n_per_side)
        o   = torch.ones(n_per_side)
        x_bc = torch.cat([t, t, z, o])   # bottom, top, left, right
        y_bc = torch.cat([z, o, t, t])
        return x_bc, y_bc, self.u_boundary(x_bc, y_bc)


# ═══════════════════════════════════════════════════
# PART 2: THE PINN
# ═══════════════════════════════════════════════════

class PoissonPINN(nn.Module):
    """
    PINN for 2D Poisson equation.
    Input  : (x, y)
    Output : u(x, y)
    Architecture: n_layers hidden layers of width hidden_dim, Tanh activations.
    """

    def __init__(self, hidden_dim: int = 64, n_layers: int = 5):
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

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.net(torch.stack([x, y], dim=-1)).squeeze(-1)

    def pde_residual(self, x: torch.Tensor, y: torch.Tensor,
                     pde: SharpPoisson) -> torch.Tensor:
        """Compute −∇²u − f via autograd."""
        x = x.requires_grad_(True)
        y = y.requires_grad_(True)
        u = self.forward(x, y)

        u_x = torch.autograd.grad(
            u, x, torch.ones_like(u), create_graph=True, retain_graph=True)[0]
        u_y = torch.autograd.grad(
            u, y, torch.ones_like(u), create_graph=True, retain_graph=True)[0]
        u_xx = torch.autograd.grad(
            u_x, x, torch.ones_like(u_x), create_graph=True, retain_graph=True)[0]
        u_yy = torch.autograd.grad(
            u_y, y, torch.ones_like(u_y), create_graph=True, retain_graph=True)[0]

        return -(u_xx + u_yy) - pde.f_source(x, y)

    def compute_l2_error(self, pde: SharpPoisson, n: int = 10_000) -> float:
        """L2 relative error on n uniform test points."""
        with torch.no_grad():
            x, y    = torch.rand(n), torch.rand(n)
            u_pred  = self.forward(x, y)
            u_exact = pde.u_exact(x, y)
            return torch.sqrt(torch.mean((u_pred - u_exact) ** 2)).item()


def train_pinn_step(pinn: PoissonPINN,
                    optimizer: torch.optim.Optimizer,
                    pde: SharpPoisson,
                    colloc_x: torch.Tensor,
                    colloc_y: torch.Tensor,
                    x_bc: torch.Tensor,
                    y_bc: torch.Tensor,
                    u_bc: torch.Tensor,
                    w_pde: float = 1.0,
                    w_bc:  float = 10.0) -> Tuple[float, float]:
    """Single Adam gradient step."""
    optimizer.zero_grad()
    L_pde = torch.mean(pinn.pde_residual(colloc_x, colloc_y, pde) ** 2)
    L_bc  = torch.mean((pinn.forward(x_bc, y_bc) - u_bc) ** 2)
    (w_pde * L_pde + w_bc * L_bc).backward()
    optimizer.step()
    return L_pde.item(), L_bc.item()


# ═══════════════════════════════════════════════════
# PART 3: SHARED SPATIAL HELPERS
# ═══════════════════════════════════════════════════

def _make_density_map(x: torch.Tensor, y: torch.Tensor,
                      G: int) -> torch.Tensor:
    """Build a G×G histogram of collocation point positions."""
    density = torch.zeros(G, G)
    ix = (x * G).long().clamp(0, G - 1)
    iy = (y * G).long().clamp(0, G - 1)
    for i, j in zip(ix, iy):
        density[i, j] += 1
    return density


def _density_entropy(density: torch.Tensor) -> float:
    """Shannon entropy H(D) of a spatial density map (Section 3.2)."""
    p = density.flatten().float()
    p = p / (p.sum() + 1e-8)
    return -(p * (p + 1e-8).log()).sum().item()


def compute_residual_map(pinn: PoissonPINN, pde: SharpPoisson,
                         G: int = 16) -> torch.Tensor:
    """Return a detached G×G absolute PDE residual map."""
    gx, gy   = torch.linspace(0, 1, G), torch.linspace(0, 1, G)
    xx, yy   = torch.meshgrid(gx, gy, indexing='ij')
    xf, yf   = xx.flatten(), yy.flatten()
    res      = pinn.pde_residual(xf.requires_grad_(True),
                                 yf.requires_grad_(True), pde)
    return res.detach().abs().reshape(G, G)


# ═══════════════════════════════════════════════════
# PART 4: THE THREE COLLOCATION STRATEGIES
# ═══════════════════════════════════════════════════

# ─────────────────────────────────────
# Strategy 1 — Uniform (baseline)
# ─────────────────────────────────────

class UniformCollocation:
    """Fixed uniform random collocation — sampled once at construction."""

    def __init__(self, n_points: int):
        self.x = torch.rand(n_points)
        self.y = torch.rand(n_points)

    def get_points(self, pinn=None, pde=None):
        return self.x, self.y

    def update(self, *args):
        pass


# ─────────────────────────────────────
# Strategy 2 — RAR  (Lu et al. 2021)
# ─────────────────────────────────────

class RARCollocation:
    """
    Residual-Adaptive Refinement.
    Periodically adds n_add points at the highest-residual locations
    among n_candidates random candidates.
    """

    def __init__(self, n_initial: int, n_add_per_step: int,
                 n_candidates: int = 10_000):
        self.n_add        = n_add_per_step
        self.n_candidates = n_candidates
        self.x            = torch.rand(n_initial)
        self.y            = torch.rand(n_initial)

    def get_points(self, pinn=None, pde=None):
        return self.x, self.y

    def update(self, pinn: PoissonPINN, pde: SharpPoisson):
        x_cand = torch.rand(self.n_candidates)
        y_cand = torch.rand(self.n_candidates)

        res_vals = []
        batch = 1000
        for i in range(0, self.n_candidates, batch):
            xb = x_cand[i:i + batch].requires_grad_(True)
            yb = y_cand[i:i + batch].requires_grad_(True)
            res_vals.append(pinn.pde_residual(xb, yb, pde).detach().abs())

        res_vals = torch.cat(res_vals)
        _, top_idx = torch.topk(res_vals, self.n_add)
        self.x = torch.cat([self.x, x_cand[top_idx]])
        self.y = torch.cat([self.y, y_cand[top_idx]])


# ─────────────────────────────────────
# Strategy 3 — RL Agent
# ─────────────────────────────────────

class RLCollocationAgent(nn.Module):
    """
    Policy and value networks for the MDP defined in Section 3.2.

    State  s_t = [R(t), G(t), D(t), φ(t)]  ∈ ℝ^{3G²+3}
    Action a_t = softmax(logits)             ∈ Δ^{G²}
    """

    def __init__(self, G: int = 16):
        super().__init__()
        self.G          = G
        self.state_dim  = G * G * 3 + 3   # 771 for G=16
        self.action_dim = G * G            # 256 for G=16

        self.policy = nn.Sequential(
            nn.Linear(self.state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256),            nn.ReLU(),
            nn.Linear(256, self.action_dim),
        )
        self.value = nn.Sequential(
            nn.Linear(self.state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256),            nn.ReLU(),
            nn.Linear(256, 1),
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)

        # Experience buffer — cleared after every policy update
        self.log_probs: List[torch.Tensor] = []
        self.rewards:   List[float]        = []

    def get_state(self, pinn: PoissonPINN, pde: SharpPoisson,
                  colloc_x: torch.Tensor, colloc_y: torch.Tensor,
                  l2_error: float, n_total: int) -> torch.Tensor:
        """Build state vector s_t from current PINN quality maps."""
        G   = self.G
        eps = 1e-8

        # Evaluation grid
        gx, gy   = torch.linspace(0, 1, G), torch.linspace(0, 1, G)
        xx, yy   = torch.meshgrid(gx, gy, indexing='ij')
        xf, yf   = xx.flatten(), yy.flatten()

        # R(t): PDE residual magnitude
        res_map  = pinn.pde_residual(
            xf.requires_grad_(True), yf.requires_grad_(True), pde
        ).detach().abs().reshape(G, G)

        # G(t): solution gradient magnitude
        xg, yg = xf.requires_grad_(True), yf.requires_grad_(True)
        u2     = pinn.forward(xg, yg)
        # retain_graph=True on the first grad call so the graph survives for uy
        ux = torch.autograd.grad(u2, xg, torch.ones_like(u2),
                                  create_graph=False, retain_graph=True)[0].detach()
        uy = torch.autograd.grad(u2, yg, torch.ones_like(u2),
                                  create_graph=False)[0].detach()
        grad_map = (ux ** 2 + uy ** 2).sqrt().reshape(G, G)

        # D(t): collocation density
        density_map = _make_density_map(colloc_x, colloc_y, G)
        density_map = density_map / (density_map.max() + eps)

        # Normalise spatial maps to [0, 1]
        res_map  = res_map  / (res_map.max()  + eps)
        grad_map = grad_map / (grad_map.max() + eps)

        # φ(t): scalar features
        scalars = torch.tensor([l2_error,
                                 len(colloc_x) / n_total,
                                 res_map.mean().item()])

        return torch.cat([res_map.flatten(), grad_map.flatten(),
                          density_map.flatten(), scalars])

    def act(self, state: torch.Tensor) -> Tuple:
        """Return (weights, log_prob, value) for the given state."""
        weights  = torch.softmax(self.policy(state), dim=-1)
        dist     = torch.distributions.Categorical(probs=weights)
        # Sample 100 cells and average log-prob (reduces variance)
        idx      = dist.sample((100,))
        log_prob = dist.log_prob(idx).mean()
        value    = self.value(state).squeeze()
        return weights, log_prob, value

    def sample_points_from_weights(self, weights: torch.Tensor,
                                    n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Draw n (x, y) points by sampling cells ∝ weights + uniform cell jitter."""
        G         = self.G
        cell_idx  = torch.multinomial(weights, n, replacement=True)
        cell_size = 1.0 / G
        x = (cell_idx // G).float() * cell_size + torch.rand(n) * cell_size
        y = (cell_idx %  G).float() * cell_size + torch.rand(n) * cell_size
        return x.clamp(0, 1), y.clamp(0, 1)

    def update_policy(self, reward: float):
        """REINFORCE update (Section 4.2, step 6)."""
        # Append reward FIRST so the guard never silently drops it.
        self.rewards.append(reward)

        if len(self.log_probs) == 0:
            self.rewards = []
            return

        returns = torch.tensor(self.rewards, dtype=torch.float32)
        # Normalise only when we have more than one sample (std=0 otherwise).
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        loss = torch.zeros(1, requires_grad=True)
        for log_p, R in zip(self.log_probs, returns):
            loss = loss - log_p * R

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)
        self.optimizer.step()

        self.log_probs = []
        self.rewards   = []


class RLCollocation:
    """
    Manages the RL agent + growing collocation set.

    Reward (Section 3.2):
        r_t = w1 · ΔL2 / n_new  +  w2 · ΔH(D)  −  w3 · n_new
    where w1=1.0, w2=0.1, w3=0 (n_new is fixed → constant, irrelevant).
    """

    W1 = 1.0    # L2 improvement weight
    W2 = 0.1    # coverage entropy bonus weight

    def __init__(self, n_initial: int, n_add_per_step: int, G: int = 16):
        self.n_add          = n_add_per_step
        self.G              = G
        self.agent          = RLCollocationAgent(G=G)

        self.x = torch.rand(n_initial)
        self.y = torch.rand(n_initial)

        self.prev_l2:      Optional[float]         = None
        self.prev_density: Optional[torch.Tensor]  = None

        # For visualisation: weight maps captured at key RL steps
        self.weight_history: List[Tuple[int, torch.Tensor]] = []
        self._step_count = 0

    def get_points(self, pinn=None, pde=None):
        return self.x, self.y

    def observe_and_act(self, pinn: PoissonPINN, pde: SharpPoisson,
                         l2_error: float, n_total: int):
        """
        Observe PINN state → sample action → add new collocation points.
        Stores previous density for the entropy bonus computed in update().
        """
        # Capture density BEFORE placing new points (needed for ΔH in update)
        self.prev_density = _make_density_map(self.x, self.y, self.G)

        state            = self.agent.get_state(pinn, pde, self.x, self.y,
                                                l2_error, n_total)
        weights, log_prob, _ = self.agent.act(state)

        self._step_count += 1
        # Capture weight maps at key steps for Section 5.6 visualisation
        if self._step_count in (1, 5, 10, 20):
            self.weight_history.append(
                (self._step_count, weights.detach().reshape(self.G, self.G).clone())
            )

        self.agent.log_probs.append(log_prob)
        self.prev_l2 = l2_error

        new_x, new_y = self.agent.sample_points_from_weights(
            weights.detach(), self.n_add
        )
        self.x = torch.cat([self.x, new_x])
        self.y = torch.cat([self.y, new_y])

    def update(self, current_l2: float) -> float:
        """
        Compute reward from L2 improvement + coverage entropy gain,
        then update the REINFORCE policy.

        Must be called AFTER PINN training on the previously placed points
        and BEFORE the next observe_and_act (Section 4.2 step order).
        """
        # w1 term: accuracy improvement per point
        l2_reward = self.W1 * (self.prev_l2 - current_l2) / self.n_add * 1000

        # w2 term: coverage entropy bonus (encourages domain exploration)
        new_density  = _make_density_map(self.x, self.y, self.G)
        entropy_gain = _density_entropy(new_density) - _density_entropy(self.prev_density)
        entropy_reward = self.W2 * entropy_gain

        reward = l2_reward + entropy_reward
        self.agent.update_policy(reward)
        return reward


# ═══════════════════════════════════════════════════
# PART 5: COMPARISON EXPERIMENT
# ═══════════════════════════════════════════════════

def run_experiment(strategy_name: str,
                   strategy,
                   pde: SharpPoisson,
                   n_adapt_steps:  int = 20,
                   epochs_per_step: int = 500,
                   n_initial:       int = 500,
                   n_add:           int = 100,
                   seed:            int = 42,
                   G:               int = 16,
                   verbose:         bool = True) -> dict:
    """
    Train PINN with the given collocation strategy.
    Returns a results dict containing L2 history, colloc-count history,
    residual-map snapshots, and (for RL) weight-map history.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    pinn      = PoissonPINN(hidden_dim=64, n_layers=5)
    optimizer = torch.optim.Adam(pinn.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)

    x_bc, y_bc, u_bc = pde.sample_boundary(n_per_side=100)

    n_total_budget = n_initial + n_adapt_steps * n_add

    l2_history       = []
    colloc_history   = []
    residual_history = []   # G×G residual maps at each eval step

    if verbose:
        print(f"\n{'='*52}\nStrategy: {strategy_name}  (seed={seed})\n{'='*52}")

    colloc_x, colloc_y = strategy.get_points(pinn, pde)

    for step in range(n_adapt_steps + 1):

        # ── Train PINN for K epochs ──────────────────
        for _ in range(epochs_per_step):
            train_pinn_step(pinn, optimizer, pde, colloc_x, colloc_y,
                            x_bc, y_bc, u_bc)
            scheduler.step()

        # ── Evaluate ────────────────────────────────
        l2 = pinn.compute_l2_error(pde)
        l2_history.append(l2)
        colloc_history.append(len(colloc_x))
        residual_history.append(compute_residual_map(pinn, pde, G=G))

        if verbose:
            print(f"  step {step:3d} | N={len(colloc_x):5d} | L2={l2:.6f}")

        if step == n_adapt_steps:
            break

        # ── Update collocation strategy ──────────────
        # For RL the correct order is:
        #   (1) reward for previous action (training just happened on those points)
        #   (2) observe new state and act  (choose next batch of points)
        if strategy_name == 'RL':
            if strategy.prev_l2 is not None:
                strategy.update(current_l2=l2)
            strategy.observe_and_act(pinn, pde, l2, n_total_budget)
        else:
            strategy.update(pinn, pde)

        colloc_x, colloc_y = strategy.get_points(pinn, pde)

    return {
        'name':             strategy_name,
        'l2':               l2_history,
        'n_colloc':         colloc_history,
        'final_l2':         l2_history[-1],
        'final_x':          colloc_x.detach(),
        'final_y':          colloc_y.detach(),
        'residual_history': residual_history,
        # Only populated for RL strategy
        'weight_history':   getattr(strategy, 'weight_history', []),
    }


def run_multi_seed(strategy_name: str,
                   strategy_factory,
                   pde: SharpPoisson,
                   n_seeds:         int = 5,
                   **experiment_kwargs) -> dict:
    """
    Run the same experiment with n_seeds independent seeds and aggregate.
    Returns mean ± std L2 curves (Section 5.3: 10 seeds recommended).
    """
    seed_results = []
    for seed in range(n_seeds):
        strategy = strategy_factory()
        r = run_experiment(strategy_name, strategy, pde,
                           seed=seed * 17 + 3, verbose=False,
                           **experiment_kwargs)
        seed_results.append(r)
        print(f"  [{strategy_name}] seed {seed+1}/{n_seeds}  "
              f"final L2={r['final_l2']:.6f}")

    l2_arr = np.array([r['l2'] for r in seed_results])   # (n_seeds, n_steps+1)
    return {
        'name':          strategy_name,
        'l2_mean':       l2_arr.mean(axis=0),
        'l2_std':        l2_arr.std(axis=0),
        'n_colloc':      seed_results[0]['n_colloc'],
        'final_l2_mean': l2_arr[:, -1].mean(),
        'final_l2_std':  l2_arr[:, -1].std(),
        'seed_results':  seed_results,
        # Use the last seed's spatial data for qualitative plots
        'final_x':       seed_results[-1]['final_x'],
        'final_y':       seed_results[-1]['final_y'],
        'residual_history': seed_results[-1]['residual_history'],
        'weight_history':   seed_results[-1]['weight_history'],
    }


def run_all_comparisons(k: float = 20.0,
                        n_seeds: int = 1,
                        n_adapt_steps:  int = 20,
                        epochs_per_step: int = 500,
                        n_initial: int = 500,
                        n_add:     int = 100) -> dict:
    """Run all three strategies for a given k and return results dict."""
    pde = SharpPoisson(k=k)
    kwargs = dict(n_adapt_steps=n_adapt_steps,
                  epochs_per_step=epochs_per_step,
                  n_initial=n_initial, n_add=n_add)

    def make_uniform():
        return UniformCollocation(
            n_points=n_initial + n_adapt_steps * n_add)

    def make_rar():
        return RARCollocation(n_initial=n_initial,
                              n_add_per_step=n_add)

    def make_rl():
        return RLCollocation(n_initial=n_initial,
                             n_add_per_step=n_add, G=16)

    results = {}
    for name, factory in [('Uniform', make_uniform),
                           ('RAR',     make_rar),
                           ('RL',      make_rl)]:
        print(f"\n{'='*52}\nStrategy: {name}  k={k}\n{'='*52}")
        if n_seeds == 1:
            strategy = factory()
            results[name] = run_experiment(name, strategy, pde, **kwargs)
        else:
            results[name] = run_multi_seed(name, factory, pde,
                                           n_seeds=n_seeds, **kwargs)

    return results


# ═══════════════════════════════════════════════════
# PART 6: VISUALISATION  (Section 5.6)
# ═══════════════════════════════════════════════════

_COLORS = {'Uniform': '#e74c3c', 'RAR': '#f39c12', 'RL': '#2ecc71'}


def plot_comparison(results: dict, k: float, multi_seed: bool = False):
    """
    Three-panel figure:
      Left   — L2 error vs. collocation budget (efficiency curve)
      Centre — final collocation scatter overlaid on analytical solution
      Right  — final L2 bar chart
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # ── Panel 1: L2 efficiency curve ─────────────
    ax = axes[0]
    for name, res in results.items():
        nc = res['n_colloc']
        if multi_seed:
            mu, sd = res['l2_mean'], res['l2_std']
            ax.semilogy(nc, mu, color=_COLORS[name], lw=2,
                        marker='o', ms=4, label=name)
            ax.fill_between(nc, mu - sd, mu + sd,
                            color=_COLORS[name], alpha=0.15)
        else:
            ax.semilogy(nc, res['l2'], color=_COLORS[name], lw=2,
                        marker='o', ms=4, label=name)
    ax.set_xlabel('Number of collocation points', fontsize=12)
    ax.set_ylabel('L2 error (log scale)', fontsize=12)
    ax.set_title(f'Accuracy vs. budget  (k={k})', fontsize=12)
    ax.legend(); ax.grid(alpha=0.3)

    # ── Panel 2: Final collocation on exact-solution heatmap ────
    ax = axes[1]
    pde = SharpPoisson(k=k)
    g   = torch.linspace(0, 1, 200)
    xx, yy = torch.meshgrid(g, g, indexing='ij')
    with torch.no_grad():
        u_map = pde.u_exact(xx.flatten(), yy.flatten()).reshape(200, 200)
    ax.imshow(u_map.numpy().T, origin='lower', extent=[0, 1, 0, 1],
              cmap='RdBu_r', aspect='equal', alpha=0.8)

    for name in ('Uniform', 'RL'):
        if name not in results:
            continue
        res = results[name]
        x_np = res['final_x'].numpy()
        y_np = res['final_y'].numpy()
        ax.scatter(x_np, y_np, s=1, alpha=0.25,
                   color=_COLORS[name], label=name)

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title('Final collocation  (RL vs Uniform)\non analytical solution', fontsize=11)
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.legend(markerscale=6)

    # ── Panel 3: Final L2 bar chart ───────────────
    ax = axes[2]
    names = list(results.keys())
    if multi_seed:
        errors = [results[n]['final_l2_mean'] for n in names]
        errs   = [results[n]['final_l2_std']  for n in names]
    else:
        errors = [results[n]['final_l2'] for n in names]
        errs   = None

    bars = ax.bar(names, errors,
                  color=[_COLORS[n] for n in names],
                  yerr=errs, capsize=5, edgecolor='white')
    for bar, e in zip(bars, errors):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.05, f'{e:.5f}',
                ha='center', va='bottom', fontsize=9)
    ax.set_ylabel('Final L2 error', fontsize=12)
    ax.set_title('Final accuracy comparison', fontsize=12)
    ax.grid(axis='y', alpha=0.3)

    fig.suptitle(f'RL Collocation vs. Baselines — Sharp Poisson  k={k}',
                 fontsize=14)
    plt.tight_layout()
    fname = f'comparison_k{int(k)}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f"Saved {fname}")
    plt.show()


def plot_rl_weight_maps(rl_strategy: RLCollocation,
                         pde: SharpPoisson, k: float):
    """
    Show RL agent weight maps at RL steps 1, 5, 10, 20 side-by-side
    with the analytical solution gradient magnitude (Section 5.6).
    """
    history = rl_strategy.weight_history
    if not history:
        print("No weight-map history recorded.")
        return

    G   = rl_strategy.G
    n   = len(history)
    fig, axes = plt.subplots(1, n + 1, figsize=(4 * (n + 1), 4))

    # Ground-truth gradient magnitude
    gx, gy = torch.linspace(0, 1, G), torch.linspace(0, 1, G)
    xx, yy = torch.meshgrid(gx, gy, indexing='ij')
    xf, yf = xx.flatten().requires_grad_(True), yy.flatten().requires_grad_(True)
    u  = pde.u_exact(xf, yf)
    ux = torch.autograd.grad(u, xf, torch.ones_like(u),
                              create_graph=False, retain_graph=True)[0].detach()
    uy = torch.autograd.grad(u, yf, torch.ones_like(u),
                              create_graph=False)[0].detach()
    grad_gt = (ux ** 2 + uy ** 2).sqrt().reshape(G, G).numpy()

    axes[0].imshow(grad_gt.T, origin='lower', cmap='hot', aspect='equal')
    axes[0].set_title('Analytical |∇u|', fontsize=11)
    axes[0].axis('off')

    for ax, (step, wmap) in zip(axes[1:], history):
        ax.imshow(wmap.numpy().T, origin='lower', cmap='viridis', aspect='equal')
        ax.set_title(f'RL weights\nstep {step}', fontsize=11)
        ax.axis('off')

    fig.suptitle(f'RL agent sampling weights  (k={k})', fontsize=13)
    plt.tight_layout()
    fname = f'rl_weights_k{int(k)}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f"Saved {fname}")
    plt.show()


def plot_residual_evolution(results: dict, k: float,
                             display_steps: Tuple[int, ...] = (0, 5, 10, 20)):
    """
    Grid of residual maps at selected adaptation steps for each method
    (Section 5.6: residual map evolution).
    """
    methods = list(results.keys())
    n_show  = min(len(display_steps),
                  min(len(r['residual_history']) for r in results.values()))

    fig, axes = plt.subplots(len(methods), n_show,
                              figsize=(4 * n_show, 3.5 * len(methods)))
    if len(methods) == 1:
        axes = axes[np.newaxis, :]

    for row, name in enumerate(methods):
        rh = results[name]['residual_history']
        for col, s in enumerate(display_steps[:n_show]):
            ax  = axes[row, col]
            idx = min(s, len(rh) - 1)
            ax.imshow(rh[idx].numpy().T, origin='lower',
                      cmap='hot', aspect='equal', vmin=0)
            if col == 0:
                ax.set_ylabel(name, fontsize=12, fontweight='bold')
            ax.set_title(f'step {s}', fontsize=10)
            ax.axis('off')

    fig.suptitle(f'PDE residual magnitude evolution  (k={k})', fontsize=13)
    plt.tight_layout()
    fname = f'residual_evolution_k{int(k)}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f"Saved {fname}")
    plt.show()


def print_summary(results: dict, multi_seed: bool = False):
    """Tabular summary — reproduces the paper's comparison table."""
    print(f"\n{'='*60}")
    header = f"{'Method':<12} {'Final L2':>12} {'vs Uniform':>12} {'vs RAR':>12}"
    if multi_seed:
        header += f"  {'± std':>8}"
    print(header)
    print('=' * 60)

    ref_u = (results['Uniform']['final_l2_mean'] if multi_seed
             else results['Uniform']['final_l2'])
    ref_r = (results['RAR']['final_l2_mean'] if multi_seed
             else results['RAR']['final_l2'])

    for name, res in results.items():
        l2 = res['final_l2_mean'] if multi_seed else res['final_l2']
        row = (f"{name:<12} {l2:>12.6f} "
               f"{ref_u/l2:>11.2f}x {ref_r/l2:>11.2f}x")
        if multi_seed:
            row += f"  {res['final_l2_std']:>8.6f}"
        print(row)

    print('=' * 60)
    print("(ratio > 1 means RL achieves lower error for same budget)\n")


# ═══════════════════════════════════════════════════
# QUICK DEMO  (fast, reduced parameters)
# ═══════════════════════════════════════════════════

def quick_demo(k: float = 20.0):
    """
    Run a fast comparison (k=20 by default) with reduced parameters
    so results appear in a few minutes on CPU.

    Paper-scale experiment: set n_adapt_steps=20, epochs_per_step=500.
    """
    print(f"\n{'#'*60}")
    print(f"#  QUICK DEMO  k={k}")
    print(f"#  5 adaptation steps × 200 PINN epochs")
    print(f"{'#'*60}")

    res = run_all_comparisons(
        k               = k,
        n_seeds         = 1,
        n_adapt_steps   = 5,
        epochs_per_step = 200,
        n_initial       = 200,
        n_add           = 50,
    )

    plot_comparison(res, k=k, multi_seed=False)
    plot_residual_evolution(res, k=k, display_steps=(0, 2, 4, 5))
    print_summary(res, multi_seed=False)

    rl_dummy = RLCollocation(1, 1, G=16)
    rl_dummy.weight_history = res['RL']['weight_history']
    plot_rl_weight_maps(rl_dummy, SharpPoisson(k=k), k=k)

    return res


# ═══════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='RL Collocation PINN — benchmark comparison')
    parser.add_argument('--quick', action='store_true',
                        help='Run quick demo (5 steps, 200 epochs) instead of full experiment')
    parser.add_argument('--k', type=float, nargs='+', default=[5.0, 20.0, 50.0],
                        help='Sharpness values to test (default: 5 20 50)')
    parser.add_argument('--seeds', type=int, default=1,
                        help='Number of independent seeds (default: 1)')
    args = parser.parse_args()

    if args.quick:
        # Fast path — single k, reduced parameters
        quick_demo(k=args.k[0])

    else:
        # ── Full experiment (Section 5.3) ────────────
        N_INITIAL       = 500
        N_ADD_PER_STEP  = 100
        N_ADAPT_STEPS   = 20
        EPOCHS_PER_STEP = 500

        all_results = {}

        for k in args.k:
            print(f"\n{'#'*60}")
            print(f"#  k = {k}  (sharpness parameter)")
            print(f"{'#'*60}")

            res = run_all_comparisons(
                k               = k,
                n_seeds         = args.seeds,
                n_adapt_steps   = N_ADAPT_STEPS,
                epochs_per_step = EPOCHS_PER_STEP,
                n_initial       = N_INITIAL,
                n_add           = N_ADD_PER_STEP,
            )
            all_results[k] = res

            multi = args.seeds > 1
            plot_comparison(res, k=k, multi_seed=multi)
            plot_residual_evolution(res, k=k)
            print_summary(res, multi_seed=multi)

            if args.seeds == 1:
                rl_dummy = RLCollocation(1, 1, G=16)
                rl_dummy.weight_history = res['RL']['weight_history']
                plot_rl_weight_maps(rl_dummy, SharpPoisson(k=k), k=k)

        # ── Sharpness sensitivity table (Section 5.4) ─
        if len(all_results) > 1:
            print(f"\n{'='*55}")
            print("Sharpness sensitivity  (RL advantage over Uniform)")
            print(f"{'='*55}")
            print(f"{'k':>4}  {'RL / Uniform':>14}  {'RL / RAR':>10}")
            for k in args.k:
                res   = all_results[k]
                key   = 'final_l2_mean' if args.seeds > 1 else 'final_l2'
                gap_u = res['Uniform'][key] / res['RL'][key]
                gap_r = res['RAR'][key]     / res['RL'][key]
                print(f"{k:>4.0f}  {gap_u:>13.2f}x  {gap_r:>9.2f}x")
            print(f"{'='*55}")
            print("Hypothesis: RL advantage grows monotonically with k.")
