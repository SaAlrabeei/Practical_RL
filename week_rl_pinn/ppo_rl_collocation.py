"""
PPO-based RL Collocation with Pre-training and Transfer
=======================================================
Replaces the REINFORCE agent in benchmarks_vs_pacmann.py with:
  - PPOAgentSpaceTime: actor-critic with GAE advantages (lower variance)
  - pretrain_rl_agent(): train on a family of PDEs, save weights
  - run_transfer_experiment(): zero-shot / fine-tune transfer demo

Key improvements over REINFORCE:
  1. PPO clipped objective — much lower gradient variance
  2. Critic value baseline (actor-critic, GAE)
  3. Episode-level PPO update (4 epochs over full trajectory)
  4. Shared trunk — efficient feature reuse across actor and critic
  5. Pre-training: policy trained on PDE family, reused at test time

Usage:
  python ppo_rl_collocation.py --pretrain               # pre-train on Burgers family
  python ppo_rl_collocation.py --transfer               # zero-shot to Allen-Cahn
  python ppo_rl_collocation.py --compare --pde burgers  # compare PPO vs baselines
  python ppo_rl_collocation.py --compare --pde both --seeds 5
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import List, Optional, Callable

# Import shared components from the benchmark file
import sys
sys.path.insert(0, os.path.dirname(__file__))
from benchmarks_vs_pacmann import (
    Burgers1D, AllenCahn1D,
    SpaceTimePINN, train_spacetime_step,
    UniformSpaceTime, RARSpaceTime, PACMANNCollocation,
    _random_domain_pts, _make_density_map_2d, _entropy,
    _make_strategies, run_multi_seed, plot_multi_seed, print_multi_seed_summary,
    _COLORS,
)


# ═══════════════════════════════════════════════════
# PPO AGENT
# ═══════════════════════════════════════════════════

class PPOAgentSpaceTime(nn.Module):
    """
    Actor-Critic PPO agent for collocation point selection.

    Architecture: shared trunk → separate actor/critic heads.
    Update: PPO-clip with GAE advantages, run at end of each episode.

    Designed to be PRE-TRAINED on a family of PDEs and then reused
    (frozen or fine-tuned) on new problems — this is the key advantage
    over PACMANN which always starts from scratch.
    """

    def __init__(self, G: int = 16,
                 lr: float = 3e-4,
                 clip_eps: float = 0.2,
                 vf_coef: float = 0.5,
                 ent_coef: float = 0.01,
                 gae_lambda: float = 0.95,
                 gamma: float = 0.99,
                 n_ppo_epochs: int = 4):
        super().__init__()
        self.G           = G
        self.clip_eps    = clip_eps
        self.vf_coef     = vf_coef
        self.ent_coef    = ent_coef
        self.gae_lambda  = gae_lambda
        self.gamma       = gamma
        self.n_ppo_epochs = n_ppo_epochs

        sd = G * G * 3 + 3   # state dim
        ad = G * G            # action dim (grid cells)

        # Shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(sd, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
        )
        self.actor_head  = nn.Linear(256, ad)
        self.critic_head = nn.Linear(256, 1)

        # Xavier init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        # Trajectory buffer (filled during one episode, cleared after update)
        self._states:    List[torch.Tensor] = []
        self._log_probs: List[torch.Tensor] = []
        self._rewards:   List[float]        = []
        self._values:    List[float]        = []

    # ── Forward ──────────────────────────────────────

    def _forward(self, state: torch.Tensor):
        h      = self.trunk(state)
        logits = self.actor_head(h)
        value  = self.critic_head(h).squeeze(-1)
        return logits, value

    # ── State builder (same as REINFORCE version) ────

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

    # ── Act ──────────────────────────────────────────

    def act(self, state: torch.Tensor):
        """Return (weights, log_prob, value) for one step."""
        logits, value = self._forward(state)
        dist     = torch.distributions.Categorical(logits=logits)
        samples  = dist.sample((100,))
        log_prob = dist.log_prob(samples).mean()
        weights  = dist.probs
        return weights, log_prob, value.item()

    def sample_points(self, w, n, x_range, t_range):
        G = self.G
        xl, xr = x_range; tl, tr = t_range
        idx = torch.multinomial(w, n, replacement=True)
        ix  = idx // G;  it = idx % G
        sx  = (xr - xl) / G;  st = (tr - tl) / G
        x   = (ix.float() * sx + torch.rand(n) * sx + xl).clamp(xl, xr)
        t   = (it.float() * st + torch.rand(n) * st + tl).clamp(tl, tr)
        return x, t

    # ── Buffer ───────────────────────────────────────

    def store(self, state: torch.Tensor, log_prob: torch.Tensor,
              reward: float, value: float):
        self._states.append(state.detach())
        self._log_probs.append(log_prob.detach())
        self._rewards.append(reward)
        self._values.append(value)

    # ── PPO update ───────────────────────────────────

    def update(self, last_value: float = 0.0):
        """
        Run PPO-clip update over the stored episode trajectory.
        Called once at the END of each episode (not per-step).
        """
        T = len(self._states)
        if T < 2:
            self._clear(); return

        # GAE advantage estimation
        values_ext = self._values + [last_value]
        gae = 0.0
        advantages = []
        for t in reversed(range(T)):
            delta = self._rewards[t] + self.gamma * values_ext[t + 1] - values_ext[t]
            gae   = delta + self.gamma * self.gae_lambda * gae
            advantages.insert(0, gae)

        adv = torch.tensor(advantages, dtype=torch.float32)
        ret = adv + torch.tensor(self._values, dtype=torch.float32)

        # Normalize advantages
        if T > 1:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        states    = torch.stack(self._states)           # (T, sd)
        old_lps   = torch.stack(self._log_probs)        # (T,)

        # PPO epochs
        for _ in range(self.n_ppo_epochs):
            new_lps, new_vals, new_ents = [], [], []
            for s in states:
                logits, v = self._forward(s)
                dist = torch.distributions.Categorical(logits=logits)
                smp  = dist.sample((100,))
                new_lps.append(dist.log_prob(smp).mean())
                new_vals.append(v)
                new_ents.append(dist.entropy())

            new_lps  = torch.stack(new_lps)
            new_vals = torch.stack(new_vals)
            entropy  = torch.stack(new_ents).mean()

            ratio  = torch.exp(new_lps - old_lps)
            surr1  = ratio * adv
            surr2  = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv

            actor_loss  = -torch.min(surr1, surr2).mean()
            critic_loss = F.mse_loss(new_vals, ret)
            loss        = actor_loss + self.vf_coef * critic_loss \
                          - self.ent_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 0.5)
            self.optimizer.step()

        self._clear()

    def _clear(self):
        self._states    = []
        self._log_probs = []
        self._rewards   = []
        self._values    = []

    # ── Save / load ──────────────────────────────────

    def save(self, path: str):
        torch.save(self.state_dict(), path)
        print(f"  Saved PPO agent → {path}")

    def load(self, path: str):
        self.load_state_dict(torch.load(path, map_location='cpu'))
        print(f"  Loaded PPO agent ← {path}")


# ═══════════════════════════════════════════════════
# RL STRATEGY USING PPO
# ═══════════════════════════════════════════════════

class RLSpaceTimePPO:
    """
    Equal-budget RL collocation using PPO.

    Each step: remove n_add lowest-residual points, add n_add PPO-sampled points.
    PPO update runs at END of episode (after all n_adapt_steps).

    Accepts a pre-trained agent via `agent=` for transfer experiments.
    Set `frozen=True` to evaluate without further training (zero-shot).
    """
    W1, W2 = 1.0, 0.1

    def __init__(self, n_total: int, n_add: int = 100, G: int = 16,
                 x_range=(-1., 1.), t_range=(0., .99),
                 agent: Optional[PPOAgentSpaceTime] = None,
                 frozen: bool = False):
        self.n_total  = n_total
        self.n_add    = n_add
        self.G        = G
        self.x_range  = x_range
        self.t_range  = t_range
        self.frozen   = frozen

        self.agent = agent if agent is not None else PPOAgentSpaceTime(G=G)
        self.x, self.t = _random_domain_pts(n_total, x_range, t_range)

        self.prev_l2:      Optional[float]        = None
        self.prev_density: Optional[torch.Tensor] = None
        self.weight_history: List                 = []
        self._step     = 0
        self._state_buf: Optional[torch.Tensor]   = None

    def get_points(self, pinn=None, pde=None):
        return self.x.detach(), self.t.detach()

    def _residuals(self, pinn, pde, x, t):
        res = []
        for i in range(0, len(x), 500):
            xb, tb = x[i:i+500].detach(), t[i:i+500].detach()
            res.append(pde.pde_residual(pinn, xb, tb).detach().abs())
        return torch.cat(res)

    def observe_and_act(self, pinn, pde, l2: float):
        """Observe state, store transition if prev reward available, act."""
        self.prev_density = _make_density_map_2d(
            self.x, self.t, self.G, self.x_range, self.t_range)

        state = self.agent.get_state(
            pinn, pde, self.x, self.t, l2, self.n_total,
            self.x_range, self.t_range)

        w, log_prob, value = self.agent.act(state)
        self._step += 1

        if self._step in (1, 5, 10, 20):
            self.weight_history.append(
                (self._step, w.detach().reshape(self.G, self.G).clone()))

        # Store state and log_prob; reward stored in update_reward()
        self._state_buf  = state
        self._lp_buf     = log_prob
        self._val_buf    = value
        self.prev_l2     = l2

        # Replace n_add lowest-residual points with PPO-sampled points
        res_existing = self._residuals(pinn, pde, self.x, self.t)
        _, worst_idx = torch.topk(res_existing, self.n_add, largest=False)
        keep = torch.ones(len(self.x), dtype=torch.bool)
        keep[worst_idx] = False
        nx, nt = self.agent.sample_points(
            w.detach(), self.n_add, self.x_range, self.t_range)
        self.x = torch.cat([self.x[keep], nx]).detach()
        self.t = torch.cat([self.t[keep], nt]).detach()

    def store_reward(self, current_l2: float) -> float:
        """Compute reward for previous action and store in agent buffer."""
        nd    = _make_density_map_2d(self.x, self.t, self.G,
                                      self.x_range, self.t_range)
        l2_r  = self.W1 * (self.prev_l2 - current_l2) / self.n_total * 1000
        ent_r = self.W2 * (_entropy(nd) - _entropy(self.prev_density))
        reward = l2_r + ent_r

        if not self.frozen:
            self.agent.store(self._state_buf, self._lp_buf, reward, self._val_buf)
        return reward

    def finalize_episode(self, last_l2: float = 0.0):
        """Trigger PPO update at end of episode."""
        if not self.frozen:
            self.agent.update(last_value=last_l2)


# ═══════════════════════════════════════════════════
# EXPERIMENT RUNNER (PPO version)
# ═══════════════════════════════════════════════════

def run_ppo_experiment(strategy_name: str, strategy, pde,
                       n_adapt_steps:   int  = 20,
                       epochs_per_step: int  = 500,
                       n_total:         int  = 2500,
                       n_bc:            int  = 40,
                       n_ic:            int  = 160,
                       seed:            int  = 42,
                       verbose:         bool = True) -> dict:
    """Run one PINN training episode with the given strategy."""
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
    is_ppo = isinstance(strategy, RLSpaceTimePPO)

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
        elif is_ppo:
            if strategy.prev_l2 is not None:
                strategy.store_reward(l2)
            strategy.observe_and_act(pinn, pde, l2)

        cx, ct = strategy.get_points(pinn, pde)

    # PPO update at end of episode
    if is_ppo:
        strategy.finalize_episode(last_l2=l2_hist[-1])

    return {
        'name':     strategy_name,
        'l2':       l2_hist,
        'n_colloc': nc_hist,
        'final_l2': l2_hist[-1],
        'final_x':  cx.detach(),
        'final_t':  ct.detach(),
        'weight_history': getattr(strategy, 'weight_history', []),
    }


# ═══════════════════════════════════════════════════
# PRE-TRAINING
# ═══════════════════════════════════════════════════

def pretrain_rl_agent(agent: PPOAgentSpaceTime,
                       pde_factories: List[Callable],
                       n_episodes:      int = 5,
                       n_steps:         int = 10,
                       epochs_per_step: int = 200,
                       n_total:         int = 2500,
                       n_replace:       int = 100,
                       save_path:       str = None,
                       verbose:         bool = True) -> PPOAgentSpaceTime:
    """
    Pre-train the PPO agent on a family of PDE instances.

    Args:
        pde_factories : list of callables returning PDE instances
        n_episodes    : number of full episodes PER factory
        n_steps       : adapt steps per episode (shorter = faster pre-training)
        epochs_per_step: PINN epochs per step
        save_path     : if set, save agent weights here after training

    The agent accumulates experience across all PDEs and variants,
    learning a policy that generalises across the family.
    """
    total = n_episodes * len(pde_factories)
    run_idx = 0

    for ep in range(n_episodes):
        for fi, factory in enumerate(pde_factories):
            run_idx += 1
            pde = factory()
            if verbose:
                pde_name = pde.__class__.__name__
                print(f"\n[Pre-train {run_idx}/{total}] {pde_name}  ep={ep}")

            strategy = RLSpaceTimePPO(
                n_total, n_add=n_replace, G=agent.G,
                x_range=pde.x_range, t_range=pde.t_range,
                agent=agent,      # ← shared agent
                frozen=False,
            )
            run_ppo_experiment(
                'RL', strategy, pde,
                n_adapt_steps=n_steps,
                epochs_per_step=epochs_per_step,
                n_total=n_total,
                seed=ep * 1000 + fi,
                verbose=verbose,
            )

    if save_path:
        agent.save(save_path)
    return agent


# ═══════════════════════════════════════════════════
# COMPARISON: PPO vs baselines (multi-seed)
# ═══════════════════════════════════════════════════

def run_ppo_comparison(pde_name: str = 'burgers',
                        n_adapt_steps:   int  = 20,
                        epochs_per_step: int  = 500,
                        n_total:         int  = 2500,
                        n_replace:       int  = 100,
                        seed:            int  = 42,
                        pretrained_path: str  = None,
                        frozen:          bool = False,
                        verbose:         bool = True) -> dict:
    """Single-seed comparison: Uniform / RAR / PACMANN / RL-PPO."""
    pde = Burgers1D() if pde_name == 'burgers' else AllenCahn1D()
    xr, tr = pde.x_range, pde.t_range
    pacmann_T = 15 if pde_name == 'burgers' else 5

    # Build fresh PPO agent (or load pre-trained)
    agent = PPOAgentSpaceTime(G=16)
    if pretrained_path and os.path.exists(pretrained_path):
        agent.load(pretrained_path)
    if frozen:
        for p in agent.parameters():
            p.requires_grad_(False)

    rl_label = 'RL-PPO' + ('-transfer' if frozen else '')

    strategies = {
        'Uniform': UniformSpaceTime(n_total, xr, tr),
        'RAR':     RARSpaceTime(n_total, n_replace=n_replace,
                                x_range=xr, t_range=tr),
        'PACMANN': PACMANNCollocation(n_total, n_steps=pacmann_T, lr=1e-5,
                                      x_range=xr, t_range=tr),
        rl_label:  RLSpaceTimePPO(n_total, n_add=n_replace, G=16,
                                   x_range=xr, t_range=tr,
                                   agent=agent, frozen=frozen),
    }

    kw = dict(n_adapt_steps=n_adapt_steps, epochs_per_step=epochs_per_step,
              n_total=n_total, seed=seed, verbose=verbose)

    results = {}
    for name, strat in strategies.items():
        if isinstance(strat, RLSpaceTimePPO):
            results[name] = run_ppo_experiment(name, strat, pde, **kw)
        else:
            results[name] = run_ppo_experiment(name, strat, pde, **kw)
    return results


def run_ppo_multi_seed(pde_name: str,
                        seeds: List[int],
                        n_adapt_steps:   int  = 20,
                        epochs_per_step: int  = 500,
                        n_total:         int  = 2500,
                        n_replace:       int  = 100,
                        pretrained_path: str  = None,
                        frozen:          bool = False) -> dict:
    all_runs = []
    for i, seed in enumerate(seeds):
        print(f"\n{'─'*60}")
        print(f"  Seed {seed}  ({i+1}/{len(seeds)})")
        print(f"{'─'*60}")
        r = run_ppo_comparison(
            pde_name,
            n_adapt_steps=n_adapt_steps,
            epochs_per_step=epochs_per_step,
            n_total=n_total,
            n_replace=n_replace,
            seed=seed,
            pretrained_path=pretrained_path,
            frozen=frozen,
        )
        all_runs.append(r)

    methods = list(all_runs[0].keys())
    agg = {}
    for m in methods:
        l2_curves = np.array([r[m]['l2'] for r in all_runs])
        n_colloc  = all_runs[-1][m]['n_colloc']
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
# TRANSFER EXPERIMENT
# ═══════════════════════════════════════════════════

def run_transfer_experiment(
        source_pde:  str = 'burgers',
        target_pde:  str = 'allen_cahn',
        n_pretrain_episodes: int = 5,
        pretrain_steps:      int = 10,
        pretrain_epochs:     int = 200,
        eval_steps:          int = 20,
        eval_epochs:         int = 500,
        n_total:             int = 2500,
        n_replace:           int = 100,
        seeds:               List[int] = None,
        agent_path:          str = 'ppo_pretrained.pt',
        verbose:             bool = True):
    """
    1. Pre-train PPO on a family of `source_pde` variants.
    2. Zero-shot transfer to `target_pde` (no gradient updates to policy).
    3. Compare: PACMANN / Fresh-RL-PPO / Transfer-RL-PPO on target_pde.

    This demonstrates RL's key advantage: a reusable, transferable policy.
    PACMANN always runs from scratch; our RL agent improves with experience.
    """
    if seeds is None:
        seeds = list(range(3))

    # ── Build PDE family for pre-training ─────────────
    if source_pde == 'burgers':
        source_factories = [
            lambda: Burgers1D(nu=0.005 / np.pi),
            lambda: Burgers1D(nu=0.01  / np.pi),
            lambda: Burgers1D(nu=0.02  / np.pi),
            lambda: Burgers1D(nu=0.05  / np.pi),
        ]
        source_label = 'Burgers (ν varied)'
    else:
        source_factories = [
            lambda: AllenCahn1D(d=0.0005),
            lambda: AllenCahn1D(d=0.001),
            lambda: AllenCahn1D(d=0.002),
        ]
        source_label = 'Allen-Cahn (d varied)'

    # ── Pre-train ─────────────────────────────────────
    print(f"\n{'#'*60}")
    print(f"#  PRE-TRAINING on {source_label}")
    print(f"#  {n_pretrain_episodes} eps × {len(source_factories)} variants "
          f"× {pretrain_steps} steps × {pretrain_epochs} epochs")
    print(f"{'#'*60}")

    agent = PPOAgentSpaceTime(G=16)

    if os.path.exists(agent_path):
        print(f"  Found cached weights at {agent_path}, skipping pre-train.")
        agent.load(agent_path)
    else:
        pretrain_rl_agent(
            agent, source_factories,
            n_episodes=n_pretrain_episodes,
            n_steps=pretrain_steps,
            epochs_per_step=pretrain_epochs,
            n_total=n_total,
            n_replace=n_replace,
            save_path=agent_path,
            verbose=verbose,
        )

    # ── Evaluate on target PDE ────────────────────────
    print(f"\n{'#'*60}")
    print(f"#  TRANSFER EVALUATION on {target_pde.upper()}  ({len(seeds)} seeds)")
    print(f"{'#'*60}")

    # Build results for all strategies
    results_per_seed = []

    for i, seed in enumerate(seeds):
        print(f"\n{'─'*60}  Seed {seed}  ({i+1}/{len(seeds)})")
        target_pde_obj = Burgers1D() if target_pde == 'burgers' else AllenCahn1D()
        xr, tr = target_pde_obj.x_range, target_pde_obj.t_range
        pacmann_T = 15 if target_pde == 'burgers' else 5

        # Fresh PPO (no pre-training) — baseline
        fresh_agent = PPOAgentSpaceTime(G=16)

        # Pre-trained PPO — zero-shot (frozen)
        frozen_agent = PPOAgentSpaceTime(G=16)
        frozen_agent.load_state_dict(agent.state_dict())

        # Pre-trained PPO — fine-tune
        finetune_agent = PPOAgentSpaceTime(G=16)
        finetune_agent.load_state_dict(agent.state_dict())

        strategies = {
            'PACMANN':        PACMANNCollocation(n_total, n_steps=pacmann_T,
                                                  lr=1e-5, x_range=xr, t_range=tr),
            'RL-fresh':       RLSpaceTimePPO(n_total, n_add=n_replace, G=16,
                                              x_range=xr, t_range=tr,
                                              agent=fresh_agent, frozen=False),
            'RL-transfer(0)': RLSpaceTimePPO(n_total, n_add=n_replace, G=16,
                                              x_range=xr, t_range=tr,
                                              agent=frozen_agent, frozen=True),
            'RL-transfer(ft)':RLSpaceTimePPO(n_total, n_add=n_replace, G=16,
                                              x_range=xr, t_range=tr,
                                              agent=finetune_agent, frozen=False),
        }

        kw = dict(n_adapt_steps=eval_steps, epochs_per_step=eval_epochs,
                  n_total=n_total, seed=seed, verbose=verbose)

        seed_results = {}
        for name, strat in strategies.items():
            seed_results[name] = run_ppo_experiment(name, strat,
                                                     target_pde_obj, **kw)
        results_per_seed.append(seed_results)

    # ── Aggregate ─────────────────────────────────────
    methods = list(results_per_seed[0].keys())
    colors  = {
        'PACMANN':         '#3498db',
        'RL-fresh':        '#95a5a6',
        'RL-transfer(0)':  '#2ecc71',
        'RL-transfer(ft)': '#27ae60',
    }
    agg = {}
    for m in methods:
        curves = np.array([r[m]['l2'] for r in results_per_seed])
        n_col  = results_per_seed[-1][m]['n_colloc']
        agg[m] = {
            'l2_mean':       curves.mean(0),
            'l2_std':        curves.std(0),
            'final_l2_mean': curves[:, -1].mean(),
            'final_l2_std':  curves[:, -1].std(),
            'n_colloc':      n_col,
            'n_seeds':       len(seeds),
        }

    # ── Plot ──────────────────────────────────────────
    _plot_transfer(agg, target_pde, colors, len(seeds))
    _print_transfer_summary(agg, target_pde, source_label)
    return agg


def _plot_transfer(agg, pde_name, colors, n_seeds):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    for name, res in agg.items():
        c   = colors.get(name, '#333333')
        n   = np.array(res['n_colloc'])
        mu  = res['l2_mean']
        sd  = res['l2_std']
        ls  = '--' if name == 'RL-fresh' else '-'
        ax.semilogy(n, mu, color=c, lw=2, ls=ls, label=name)
        ax.fill_between(n, np.maximum(mu - sd, 1e-5), mu + sd,
                        color=c, alpha=0.15)
    ax.set_xlabel('Collocation points')
    ax.set_ylabel('L2 relative error')
    ax.set_title(f'Transfer to {pde_name}  ({n_seeds} seeds)')
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    names = list(agg.keys())
    means = [agg[n]['final_l2_mean'] for n in names]
    stds  = [agg[n]['final_l2_std']  for n in names]
    c_list = [colors.get(n, '#333333') for n in names]
    ax.bar(names, means, color=c_list, yerr=stds, capsize=5, edgecolor='white')
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.005, f'{m:.4f}', ha='center', fontsize=8)
    ax.set_ylabel('Final L2 relative error')
    ax.set_title(f'Zero-shot vs fine-tune vs fresh vs PACMANN')
    ax.grid(axis='y', alpha=0.3)

    fig.suptitle(f'RL Transfer Experiment — target: {pde_name}', fontsize=13)
    plt.tight_layout()
    fname = f'transfer_{pde_name}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f"Saved {fname}")
    plt.close()


def _print_transfer_summary(agg, pde_name, source_label):
    n_seeds = list(agg.values())[0]['n_seeds']
    print(f"\n{'='*68}")
    print(f"  TRANSFER → {pde_name.upper()}  (pre-trained on {source_label})")
    print(f"  {n_seeds} seeds")
    print(f"{'='*68}")
    print(f"{'Method':<20} {'Mean ± Std':>18}")
    print('-'*40)
    for name, res in agg.items():
        m, s = res['final_l2_mean'], res['final_l2_std']
        print(f"{name:<20} {m:>9.4f} ± {s:<6.4f}")
    print('='*68)


# ═══════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['compare', 'pretrain', 'transfer'],
                        default='compare')
    parser.add_argument('--pde',    choices=['burgers', 'allen_cahn', 'both'],
                        default='burgers')
    parser.add_argument('--quick',  action='store_true',
                        help='Short run for smoke testing')
    parser.add_argument('--seeds',  type=int, default=5)
    parser.add_argument('--agent',  default='week_rl_pinn/ppo_pretrained.pt',
                        help='Path to save/load pre-trained agent weights')
    # Pre-train params
    parser.add_argument('--pretrain-episodes', type=int, default=5)
    parser.add_argument('--pretrain-steps',    type=int, default=10)
    parser.add_argument('--pretrain-epochs',   type=int, default=200)
    # Transfer params
    parser.add_argument('--source-pde', default='burgers',
                        choices=['burgers', 'allen_cahn'])
    parser.add_argument('--target-pde', default='allen_cahn',
                        choices=['burgers', 'allen_cahn'])
    parser.add_argument('--frozen', action='store_true',
                        help='Zero-shot transfer (no fine-tuning)')
    args = parser.parse_args()

    if args.quick:
        eval_kw    = dict(n_adapt_steps=6,  epochs_per_step=150, n_total=450,  n_replace=50)
        pretrain_kw = dict(n_episodes=2, n_steps=5, epochs_per_step=100, n_total=450, n_replace=50)
    else:
        eval_kw    = dict(n_adapt_steps=20, epochs_per_step=500, n_total=2500, n_replace=100)
        pretrain_kw = dict(n_episodes=args.pretrain_episodes,
                           n_steps=args.pretrain_steps,
                           epochs_per_step=args.pretrain_epochs,
                           n_total=2500, n_replace=100)

    if args.mode == 'compare':
        # Multi-seed comparison: PPO vs Uniform/RAR/PACMANN
        seeds = list(range(args.seeds))
        pdes  = ['burgers', 'allen_cahn'] if args.pde == 'both' else [args.pde]
        mode_str = 'quick' if args.quick else 'full'

        for pde_name in pdes:
            print(f"\n{'#'*60}")
            print(f"#  PPO COMPARISON — {pde_name.upper()}  ({mode_str}, {args.seeds} seeds)")
            print(f"{'#'*60}")
            agg = run_ppo_multi_seed(
                pde_name, seeds=seeds,
                pretrained_path=args.agent if os.path.exists(args.agent) else None,
                frozen=args.frozen,
                **eval_kw,
            )
            pde_obj = Burgers1D() if pde_name == 'burgers' else AllenCahn1D()

            # Reuse plot/summary from benchmarks
            ppo_colors = {**_COLORS, 'RL-PPO': '#16a085', 'RL-PPO-transfer': '#1abc9c'}
            plot_multi_seed(agg, pde_name + '_ppo', pde_obj, n_seeds=args.seeds)
            print_multi_seed_summary(agg, pde_name)

    elif args.mode == 'pretrain':
        # Pre-train on PDE family and save weights
        if args.source_pde == 'burgers':
            factories = [
                lambda: Burgers1D(nu=0.005 / np.pi),
                lambda: Burgers1D(nu=0.01  / np.pi),
                lambda: Burgers1D(nu=0.02  / np.pi),
                lambda: Burgers1D(nu=0.05  / np.pi),
            ]
        else:
            factories = [
                lambda: AllenCahn1D(d=0.0005),
                lambda: AllenCahn1D(d=0.001),
                lambda: AllenCahn1D(d=0.002),
            ]

        agent = PPOAgentSpaceTime(G=16)
        pretrain_rl_agent(agent, factories,
                          save_path=args.agent,
                          **pretrain_kw)

    elif args.mode == 'transfer':
        run_transfer_experiment(
            source_pde=args.source_pde,
            target_pde=args.target_pde,
            n_pretrain_episodes=args.pretrain_episodes,
            pretrain_steps=args.pretrain_steps,
            pretrain_epochs=args.pretrain_epochs,
            eval_steps=eval_kw['n_adapt_steps'],
            eval_epochs=eval_kw['epochs_per_step'],
            n_total=eval_kw['n_total'],
            n_replace=eval_kw['n_replace'],
            seeds=list(range(args.seeds)),
            agent_path=args.agent,
        )
