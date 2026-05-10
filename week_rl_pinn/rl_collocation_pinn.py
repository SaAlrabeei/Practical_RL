import numpy as np
import torch
import torch.nn as nn
from collections import deque
import matplotlib.pyplot as plt


# ═══════════════════════════════════════════════════
# PART 1: THE PDE AND ANALYTICAL SOLUTION
# ═══════════════════════════════════════════════════

class SharpPoisson:
    """
    2D Poisson equation with sharp internal layer.

    -∇²u = f(x,y)    on [0,1]²
    u = g(x,y)        on boundary

    Analytical solution:
        u(x,y) = tanh(k*(x-0.5)) * tanh(k*(y-0.5))

    Source term f derived from applying -∇² to u.
    """

    def __init__(self, k: float = 20.0):
        self.k = k

    def u_exact(self, x, y):
        """Analytical solution."""
        k = self.k
        return (torch.tanh(k * (x - 0.5)) *
                torch.tanh(k * (y - 0.5)))

    def f_source(self, x, y):
        """
        Right-hand side: f = -∇²u
        Derived analytically from u_exact.
        """
        k = self.k
        th_x = torch.tanh(k * (x - 0.5))
        th_y = torch.tanh(k * (y - 0.5))
        # d²/dx²[tanh(k(x-0.5))] = -2k²tanh·sech²
        sech2_x = 1 - th_x**2
        sech2_y = 1 - th_y**2
        d2u_dx2 = -2 * k**2 * th_x * sech2_x * th_y
        d2u_dy2 = -2 * k**2 * th_y * sech2_y * th_x
        return -(d2u_dx2 + d2u_dy2)

    def u_boundary(self, x, y):
        """Boundary condition (exact solution on boundary)."""
        return self.u_exact(x, y)

    def sample_boundary(self, n_per_side: int):
        """Sample n points on each of 4 boundary sides."""
        t = torch.linspace(0, 1, n_per_side)
        z = torch.zeros(n_per_side)
        o = torch.ones(n_per_side)

        x_bc = torch.cat([t, t, z, o])   # bottom, top, left, right
        y_bc = torch.cat([z, o, t, t])

        u_bc = self.u_boundary(x_bc, y_bc)
        return x_bc, y_bc, u_bc


# ═══════════════════════════════════════════════════
# PART 2: THE PINN
# ═══════════════════════════════════════════════════

class PoissonPINN(nn.Module):
    """
    PINN for 2D Poisson equation.
    Input:  (x, y)
    Output: u(x, y)
    """

    def __init__(self, hidden_dim=64, n_layers=5):
        super().__init__()

        layers = [nn.Linear(2, hidden_dim), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim),
                       nn.Tanh()]
        layers += [nn.Linear(hidden_dim, 1)]
        self.net = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, y):
        xy  = torch.stack([x, y], dim=-1)
        return self.net(xy).squeeze(-1)

    def pde_residual(self, x, y, pde: SharpPoisson):
        """
        Compute PDE residual: -∇²u - f = 0
        Uses autograd for spatial derivatives.
        """
        x = x.requires_grad_(True)
        y = y.requires_grad_(True)

        u = self.forward(x, y)

        u_x = torch.autograd.grad(
            u, x, torch.ones_like(u),
            create_graph=True, retain_graph=True
        )[0]
        u_y = torch.autograd.grad(
            u, y, torch.ones_like(u),
            create_graph=True, retain_graph=True
        )[0]

        u_xx = torch.autograd.grad(
            u_x, x, torch.ones_like(u_x),
            create_graph=True, retain_graph=True
        )[0]
        u_yy = torch.autograd.grad(
            u_y, y, torch.ones_like(u_y),
            create_graph=True, retain_graph=True
        )[0]

        f        = pde.f_source(x, y)
        residual = -(u_xx + u_yy) - f

        return residual

    def compute_l2_error(self, pde: SharpPoisson,
                          n: int = 10000) -> float:
        """L2 error against analytical solution."""
        with torch.no_grad():
            x  = torch.rand(n)
            y  = torch.rand(n)
            u_pred  = self.forward(x, y)
            u_exact = pde.u_exact(x, y)
            l2 = torch.sqrt(
                torch.mean((u_pred - u_exact)**2)
            ).item()
        return l2


def train_pinn_step(pinn, optimizer, pde,
                    colloc_x, colloc_y,
                    x_bc, y_bc, u_bc,
                    w_pde=1.0, w_bc=10.0):
    """Single PINN gradient step."""
    optimizer.zero_grad()

    res     = pinn.pde_residual(colloc_x, colloc_y, pde)
    L_pde   = torch.mean(res**2)

    u_pred  = pinn.forward(x_bc, y_bc)
    L_bc    = torch.mean((u_pred - u_bc)**2)

    loss    = w_pde * L_pde + w_bc * L_bc
    loss.backward()
    optimizer.step()

    return L_pde.item(), L_bc.item()


# ═══════════════════════════════════════════════════
# PART 3: THE THREE COLLOCATION STRATEGIES
# ═══════════════════════════════════════════════════

# ───────────────────────────────────────────────────
# Strategy 1: Uniform (baseline)
# ───────────────────────────────────────────────────

class UniformCollocation:
    """Fixed uniform random collocation. Sampled once at init."""

    def __init__(self, n_points: int):
        self.n_points = n_points
        # Fix: sample once at construction so the set truly never changes.
        self.x = torch.rand(n_points)
        self.y = torch.rand(n_points)

    def get_points(self, pinn=None, pde=None):
        return self.x, self.y

    def update(self, *args):
        pass


# ───────────────────────────────────────────────────
# Strategy 2: Residual Adaptive Refinement (RAR)
# Strong classical baseline from literature
# Lu et al. 2021, DeepXDE
# ───────────────────────────────────────────────────

class RARCollocation:
    """
    Residual-Adaptive Refinement.
    Periodically adds points where |residual| is largest.
    """

    def __init__(self, n_initial: int, n_add_per_step: int,
                 n_candidates: int = 10000):
        self.n_initial      = n_initial
        self.n_add          = n_add_per_step
        self.n_candidates   = n_candidates

        self.x = torch.rand(n_initial)
        self.y = torch.rand(n_initial)

    def get_points(self, pinn=None, pde=None):
        return self.x, self.y

    def update(self, pinn: PoissonPINN, pde: SharpPoisson):
        """Add n_add points at highest residual locations."""
        x_cand = torch.rand(self.n_candidates)
        y_cand = torch.rand(self.n_candidates)

        with torch.no_grad():
            res_vals = []
            batch = 1000
            for i in range(0, self.n_candidates, batch):
                xb = x_cand[i:i+batch].requires_grad_(True)
                yb = y_cand[i:i+batch].requires_grad_(True)
                res = pinn.pde_residual(xb, yb, pde)
                res_vals.append(res.detach().abs())

        res_vals = torch.cat(res_vals)

        _, top_idx = torch.topk(res_vals, self.n_add)

        self.x = torch.cat([self.x, x_cand[top_idx]])
        self.y = torch.cat([self.y, y_cand[top_idx]])


# ───────────────────────────────────────────────────
# Strategy 3: RL Collocation Agent (proposed)
# ───────────────────────────────────────────────────

class RLCollocationAgent(nn.Module):
    """
    RL agent that outputs a spatial sampling weight map.

    State  : PINN quality features on G×G grid
    Action : sampling weights on G×G grid (softmax)
    Reward : PINN L2 error improvement per point spent
    """

    def __init__(self, G: int = 16):
        super().__init__()
        self.G         = G
        self.state_dim = G * G * 3 + 3   # residual + grad + density + scalars
        self.action_dim= G * G

        self.policy = nn.Sequential(
            nn.Linear(self.state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.action_dim),
        )

        self.value = nn.Sequential(
            nn.Linear(self.state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        self.optimizer  = torch.optim.Adam(
            self.parameters(), lr=3e-4
        )

        self.states     = []
        self.actions    = []
        self.rewards    = []
        self.log_probs  = []
        self.values     = []

    def get_state(self, pinn: PoissonPINN,
                  pde: SharpPoisson,
                  colloc_x, colloc_y,
                  l2_error: float,
                  n_total: int) -> torch.Tensor:
        """Build state vector from current PINN quality."""
        G   = self.G
        eps = 1e-8

        gx  = torch.linspace(0, 1, G)
        gy  = torch.linspace(0, 1, G)
        xx, yy = torch.meshgrid(gx, gy, indexing='ij')
        xf  = xx.flatten()
        yf  = yy.flatten()

        xr  = xf.requires_grad_(True)
        yr  = yf.requires_grad_(True)
        res = pinn.pde_residual(xr, yr, pde).detach().abs()
        res_map = res.reshape(G, G)

        with torch.no_grad():
            u   = pinn.forward(xf, yf)
        xg  = xf.requires_grad_(True)
        yg  = yf.requires_grad_(True)
        u2  = pinn.forward(xg, yg)
        ux  = torch.autograd.grad(
            u2, xg, torch.ones_like(u2),
            create_graph=False
        )[0].detach()
        uy  = torch.autograd.grad(
            u2, yg, torch.ones_like(u2),
            create_graph=False
        )[0].detach()
        grad_map = (ux**2 + uy**2).sqrt().reshape(G, G)

        density_map = torch.zeros(G, G)
        ix = (colloc_x * G).long().clamp(0, G-1)
        iy = (colloc_y * G).long().clamp(0, G-1)
        for i, j in zip(ix, iy):
            density_map[i, j] += 1
        density_map = density_map / (density_map.max() + eps)

        res_map   = res_map   / (res_map.max()   + eps)
        grad_map  = grad_map  / (grad_map.max()  + eps)

        scalars = torch.tensor([
            l2_error,
            len(colloc_x) / n_total,
            res_map.mean(),
        ])

        state = torch.cat([
            res_map.flatten(),
            grad_map.flatten(),
            density_map.flatten(),
            scalars,
        ])

        return state

    def act(self, state: torch.Tensor):
        """
        Sample action (sampling weight map) from policy.
        Returns weights, action_indices, log_prob, value.
        """
        logits     = self.policy(state)
        weights    = torch.softmax(logits, dim=-1)

        dist       = torch.distributions.Categorical(probs=weights)
        action_idx = dist.sample((100,))
        log_prob   = dist.log_prob(action_idx).mean()
        value      = self.value(state)

        return weights, action_idx, log_prob, value.squeeze()

    def sample_points_from_weights(self,
                                    weights: torch.Tensor,
                                    n: int) -> tuple:
        """Given G×G weight map, sample n (x,y) points."""
        G   = self.G

        cell_idx  = torch.multinomial(weights, n, replacement=True)
        cell_i    = cell_idx // G
        cell_j    = cell_idx % G

        cell_size = 1.0 / G
        x = (cell_i.float() + torch.rand(n)) * cell_size
        y = (cell_j.float() + torch.rand(n)) * cell_size
        x = x.clamp(0, 1)
        y = y.clamp(0, 1)

        return x, y

    def update_policy(self, reward: float):
        """Simple REINFORCE update."""
        # Fix: append reward BEFORE the guard so it is never silently dropped.
        self.rewards.append(reward)

        if len(self.log_probs) == 0:
            self.rewards = []
            return

        returns = torch.tensor(self.rewards, dtype=torch.float32)
        # Fix: std() is NaN for a single-element tensor; skip normalisation then.
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        loss = torch.zeros(1, requires_grad=True)
        # Zip stops at the shorter sequence, keeping rewards/log_probs in sync.
        for log_p, R in zip(self.log_probs, returns):
            loss = loss - log_p * R

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
        self.optimizer.step()

        self.states    = []
        self.actions   = []
        self.rewards   = []
        self.log_probs = []
        self.values    = []


class RLCollocation:
    """Wrapper that manages RL agent + collocation set."""

    def __init__(self, n_initial: int,
                 n_add_per_step: int, G: int = 16):
        self.n_initial      = n_initial
        self.n_add          = n_add_per_step
        self.G              = G
        self.agent          = RLCollocationAgent(G=G)

        self.x = torch.rand(n_initial)
        self.y = torch.rand(n_initial)

        self.prev_l2        = None
        self.current_state  = None
        self.current_weights= None
        self.current_logp   = None

    def get_points(self, pinn=None, pde=None):
        return self.x, self.y

    def observe_and_act(self, pinn: PoissonPINN,
                         pde: SharpPoisson,
                         l2_error: float,
                         n_total: int):
        """Observe current PINN state and add the next batch of points."""
        state = self.agent.get_state(
            pinn, pde, self.x, self.y, l2_error, n_total
        )

        weights, _, log_prob, value = self.agent.act(state)

        self.current_state   = state
        self.current_weights = weights.detach()
        self.current_logp    = log_prob
        self.agent.log_probs.append(log_prob)
        self.prev_l2         = l2_error

        new_x, new_y = self.agent.sample_points_from_weights(
            weights.detach(), self.n_add
        )

        self.x = torch.cat([self.x, new_x])
        self.y = torch.cat([self.y, new_y])

    def update(self, current_l2: float):
        """
        Compute reward from L2 improvement since last observe_and_act
        and update the policy.

        Fix: accepts the already-computed l2 instead of re-evaluating the
        PINN, and is called BEFORE observe_and_act so that the reward
        reflects genuine training improvement on the previously placed points.
        """
        improvement = self.prev_l2 - current_l2      # positive = better
        efficiency  = improvement / self.n_add
        reward      = efficiency * 1000

        self.agent.update_policy(reward)
        return reward


# ═══════════════════════════════════════════════════
# PART 4: COMPARISON EXPERIMENT
# ═══════════════════════════════════════════════════

def run_experiment(strategy_name: str,
                   strategy,
                   pde: SharpPoisson,
                   n_adapt_steps: int = 20,
                   epochs_per_step: int = 500,
                   n_initial: int = 500,
                   n_add: int = 100,
                   seed: int = 42) -> dict:
    """
    Train PINN with given collocation strategy.
    Returns L2 error history vs. collocation count.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    pinn      = PoissonPINN(hidden_dim=64, n_layers=5)
    optimizer = torch.optim.Adam(pinn.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=2000, gamma=0.5
    )

    x_bc, y_bc, u_bc = pde.sample_boundary(n_per_side=100)

    l2_history      = []
    colloc_history  = []
    n_total_budget  = n_initial + n_adapt_steps * n_add

    print(f"\n{'='*50}")
    print(f"Strategy: {strategy_name}")
    print(f"{'='*50}")

    colloc_x, colloc_y = strategy.get_points(pinn, pde)

    for step in range(n_adapt_steps + 1):

        for epoch in range(epochs_per_step):
            train_pinn_step(
                pinn, optimizer, pde,
                colloc_x, colloc_y,
                x_bc, y_bc, u_bc,
            )
            scheduler.step()

        l2       = pinn.compute_l2_error(pde)
        n_colloc = len(colloc_x)
        l2_history.append(l2)
        colloc_history.append(n_colloc)

        print(f"  Step {step:3d} | "
              f"N_colloc: {n_colloc:5d} | "
              f"L2 error: {l2:.6f}")

        if step == n_adapt_steps:
            break

        if strategy_name == 'RL':
            # Fix: compute reward FIRST (PINN has now trained on the points
            # placed in the previous observe_and_act call), then pick the
            # next placement.  Original code did this in reverse so the
            # reward was always ≈ 0 and the agent never learned.
            if strategy.prev_l2 is not None:
                strategy.update(current_l2=l2)
            strategy.observe_and_act(pinn, pde, l2, n_total_budget)
        else:
            strategy.update(pinn, pde)

        colloc_x, colloc_y = strategy.get_points(pinn, pde)

    return {
        'name':     strategy_name,
        'l2':       l2_history,
        'n_colloc': colloc_history,
        'final_l2': l2_history[-1],
        'final_x':  colloc_x,
        'final_y':  colloc_y,
    }


def run_all_comparisons(k: float = 20.0):
    """Run all three strategies and compare."""
    pde = SharpPoisson(k=k)

    N_INITIAL       = 500
    N_ADD_PER_STEP  = 100
    N_ADAPT_STEPS   = 20
    EPOCHS_PER_STEP = 500

    strategies = {
        'Uniform': UniformCollocation(
            n_points=N_INITIAL + N_ADAPT_STEPS * N_ADD_PER_STEP
        ),
        'RAR': RARCollocation(
            n_initial=N_INITIAL,
            n_add_per_step=N_ADD_PER_STEP
        ),
        'RL': RLCollocation(
            n_initial=N_INITIAL,
            n_add_per_step=N_ADD_PER_STEP,
            G=16
        ),
    }

    results = {}
    for name, strategy in strategies.items():
        results[name] = run_experiment(
            strategy_name   = name,
            strategy        = strategy,
            pde             = pde,
            n_adapt_steps   = N_ADAPT_STEPS,
            epochs_per_step = EPOCHS_PER_STEP,
            n_initial       = N_INITIAL,
            n_add           = N_ADD_PER_STEP,
        )

    plot_comparison(results, k=k)
    print_summary(results)
    return results


# ═══════════════════════════════════════════════════
# PART 5: RESULTS VISUALIZATION
# ═══════════════════════════════════════════════════

def plot_comparison(results: dict, k: float):
    """Reproduce paper-style comparison figures."""

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = {
        'Uniform': '#e74c3c',
        'RAR':     '#f39c12',
        'RL':      '#2ecc71',
    }

    # ── Plot 1: L2 error vs collocation points ────
    ax = axes[0]
    for name, res in results.items():
        ax.semilogy(
            res['n_colloc'], res['l2'],
            color=colors[name],
            linewidth=2,
            marker='o', markersize=4,
            label=name,
        )
    ax.set_xlabel('Number of Collocation Points', fontsize=12)
    ax.set_ylabel('L2 Error (log scale)', fontsize=12)
    ax.set_title(f'Accuracy vs. Budget\n(k={k})', fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)

    # ── Plot 2: Final collocation distributions ───
    ax = axes[1]
    for name in ('Uniform', 'RL'):
        if name not in results:
            continue
        res = results[name]
        x = res['final_x'].detach().numpy()
        y = res['final_y'].detach().numpy()
        ax.scatter(x, y, s=1, alpha=0.3,
                   color=colors[name], label=name)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('Final Collocation Distribution\n(RL vs Uniform)',
                 fontsize=12)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend(markerscale=6)
    ax.grid(alpha=0.3)

    # ── Plot 3: Bar chart — final L2 errors ───────
    ax = axes[2]
    names  = list(results.keys())
    errors = [results[n]['final_l2'] for n in names]
    bars   = ax.bar(names, errors,
                    color=[colors[n] for n in names],
                    edgecolor='white')
    for bar, err in zip(bars, errors):
        ax.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() * 1.05,
            f'{err:.5f}',
            ha='center', va='bottom', fontsize=10
        )
    ax.set_ylabel('Final L2 Error', fontsize=12)
    ax.set_title('Final Accuracy Comparison', fontsize=12)
    ax.grid(axis='y', alpha=0.3)

    fig.suptitle(
        f'RL Collocation vs. Baselines — '
        f'Sharp Poisson (k={k})',
        fontsize=14
    )
    plt.tight_layout()
    plt.savefig(f'comparison_k{int(k)}.png',
                dpi=150, bbox_inches='tight')
    plt.show()


def print_summary(results: dict):
    """Print comparison table."""
    print(f"\n{'='*55}")
    print(f"{'Method':<12} {'Final L2':>12} "
          f"{'vs Uniform':>12} {'vs RAR':>12}")
    print(f"{'='*55}")

    ref_uniform = results['Uniform']['final_l2']
    ref_rar     = results['RAR']['final_l2']

    for name, res in results.items():
        l2          = res['final_l2']
        vs_uniform  = ref_uniform / l2
        vs_rar      = ref_rar     / l2
        print(
            f"{name:<12} {l2:>12.6f} "
            f"{vs_uniform:>11.2f}x {vs_rar:>11.2f}x"
        )
    print(f"{'='*55}")
    print("(>1x means RL achieves lower error "
          "with same budget)")


# ═══════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════

if __name__ == '__main__':

    print("Experiment 1: Moderate sharpness (k=20)")
    results_20 = run_all_comparisons(k=20.0)

    print("\nExperiment 2: High sharpness (k=50)")
    results_50 = run_all_comparisons(k=50.0)

    # Key hypothesis:
    # Gap between RL and baselines should be
    # LARGER for k=50 than k=20
    # Because sharper layers make placement MORE critical
    print("\nSharpness sensitivity:")
    for k, res in [(20, results_20), (50, results_50)]:
        gap = res['Uniform']['final_l2'] / res['RL']['final_l2']
        print(f"  k={k:2d}: RL is {gap:.2f}x better than Uniform")
