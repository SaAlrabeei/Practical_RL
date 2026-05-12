"""
2D Geometry Transfer Experiment
================================
Core claim: An RL agent trained on a family of 2D Poisson problems
(sharp source at different locations) learns a *spatial* collocation
strategy that immediately focuses near complex regions on unseen
geometries — without rerunning any optimisation from scratch.

PDE :  -Δu = f(x,y)  on [0,1]²,   u = 0 on ∂Ω
f    :  sharp Gaussian at (μx, μy), σ = 0.05
BC   :  enforced exactly via output transform
           u(x,y) = x(1-x)y(1-y) · u_NN(x,y)

Train source locations (seen by RL):
    (0.2,0.5), (0.5,0.8), (0.8,0.3), (0.3,0.2), (0.7,0.7)

Test source locations (UNSEEN):
    (0.4,0.6), (0.7,0.2), (0.5,0.5)

Metric: PINN gradient steps to reach L2 < threshold.

Usage:
  python geometry_transfer.py --quick          # smoke test
  python geometry_transfer.py                  # full run
  python geometry_transfer.py --skip-pretrain  # reuse cached agent
"""

import os, sys, copy, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.sparse import diags, kron, eye
from scipy.sparse.linalg import spsolve
from typing import List, Optional, Tuple

# ── Paths ─────────────────────────────────────────────────────
DIR        = os.path.dirname(__file__)
AGENT_PATH = os.path.join(DIR, 'ppo_geometry_agent.pt')

# ── Problem constants ─────────────────────────────────────────
SIGMA      = 0.05     # source sharpness
AMPLITUDE  = 10.0     # source amplitude
N_TOTAL    = 2000     # collocation points
N_REPLACE  = 100      # points replaced per step
G          = 16       # RL grid resolution

TRAIN_LOCS = [(0.2,0.5),(0.5,0.8),(0.8,0.3),(0.3,0.2),(0.7,0.7)]
TEST_LOCS  = [(0.4,0.6),(0.7,0.2),(0.5,0.5)]

# ── Colours ───────────────────────────────────────────────────
COLORS = {'Uniform':'#e74c3c','RAR':'#f39c12',
          'PACMANN':'#3498db','RL-pretrained':'#27ae60'}


# ═══════════════════════════════════════════════════════════════
# 1. REFERENCE SOLUTION  (2D FD, sparse Poisson solver)
# ═══════════════════════════════════════════════════════════════

def _fd_poisson_2d(f_vals: np.ndarray, n: int = 64) -> np.ndarray:
    """
    Solve -Δu = f on [0,1]² with u=0 on boundary.
    f_vals: (n,n) array of f evaluated on interior grid.
    Returns (n+2,n+2) array including boundary zeros.
    """
    h  = 1.0 / (n + 1)
    N  = n * n
    d0 = np.full(N,  4.0 / h**2)
    d1 = np.full(N - 1, -1.0 / h**2)
    dn = np.full(N - n, -1.0 / h**2)
    # Remove coupling across row boundaries
    for i in range(1, n):
        d1[i*n - 1] = 0.0
    L  = diags([d0, d1, d1, dn, dn], [0, 1, -1, n, -n], format='csr')
    rhs = f_vals.flatten()
    u_int = spsolve(L, rhs).reshape(n, n)
    U = np.zeros((n+2, n+2))
    U[1:-1, 1:-1] = u_int
    return U


class SharpPoisson2D:
    """
    -Δu = f   on [0,1]²,  u=0 on ∂Ω
    f(x,y) = A·exp(-[(x-μx)²+(y-μy)²]/(2σ²))
    """
    def __init__(self, src_x: float, src_y: float,
                 sigma: float = SIGMA, amplitude: float = AMPLITUDE,
                 fd_n: int = 128):
        self.src_x, self.src_y = src_x, src_y
        self.sigma   = sigma
        self.amp     = amplitude
        self.x_range = (0., 1.)
        self.y_range = (0., 1.)
        self._build_reference(fd_n)

    def f(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.amp * torch.exp(
            -((x - self.src_x)**2 + (y - self.src_y)**2) / (2 * self.sigma**2))

    def _build_reference(self, n: int):
        xi = np.linspace(0, 1, n + 2)[1:-1]
        XX, YY = np.meshgrid(xi, xi, indexing='ij')
        F = self.amp * np.exp(
            -((XX - self.src_x)**2 + (YY - self.src_y)**2) / (2 * self.sigma**2))
        U  = _fd_poisson_2d(F, n)
        xg = np.linspace(0, 1, n + 2)
        from scipy.interpolate import RegularGridInterpolator
        self._interp = RegularGridInterpolator(
            (xg, xg), U, method='linear',
            bounds_error=False, fill_value=0.0)

    def u_ref(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pts = np.stack([x.detach().numpy(), y.detach().numpy()], axis=-1)
        return torch.tensor(self._interp(pts), dtype=torch.float32)

    def pde_residual(self, pinn, x: torch.Tensor,
                     y: torch.Tensor) -> torch.Tensor:
        x = x.requires_grad_(True); y = y.requires_grad_(True)
        u    = pinn(x, y)
        u_x  = torch.autograd.grad(u, x, torch.ones_like(u),
                                    create_graph=True, retain_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x),
                                    create_graph=True, retain_graph=True)[0]
        u_y  = torch.autograd.grad(u, y, torch.ones_like(u),
                                    create_graph=True, retain_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, y, torch.ones_like(u_y),
                                    create_graph=True, retain_graph=True)[0]
        return -(u_xx + u_yy) - self.f(x, y)


# ═══════════════════════════════════════════════════════════════
# 2. SPATIAL PINN  (hard BC)
# ═══════════════════════════════════════════════════════════════

class SpatialPINN(nn.Module):
    """
    4 × 64 tanh, Glorot.
    Hard BC: u(x,y) = x(1-x)y(1-y)·u_NN  → u=0 on all 4 walls.
    """
    def __init__(self, hidden: int = 64, n_layers: int = 4):
        super().__init__()
        layers = [nn.Linear(2, hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        u_nn = self.net(torch.stack([x, y], dim=-1)).squeeze(-1)
        return x * (1 - x) * y * (1 - y) * u_nn

    def compute_l2_rel(self, pde: SharpPoisson2D, n: int = 4000) -> float:
        with torch.no_grad():
            # Avoid boundary where solution is trivially 0
            x = torch.rand(n) * 0.98 + 0.01
            y = torch.rand(n) * 0.98 + 0.01
            u_pred  = self(x, y)
            u_exact = pde.u_ref(x, y)
            num = torch.sqrt(torch.mean((u_pred - u_exact)**2))
            den = torch.sqrt(torch.mean(u_exact**2)) + 1e-8
            return (num / den).item()


def train_step(pinn, opt, pde, cx, cy):
    opt.zero_grad()
    loss = torch.mean(pde.pde_residual(pinn, cx, cy)**2)
    loss.backward()
    opt.step()
    return loss.item()


# ═══════════════════════════════════════════════════════════════
# 3. COLLOCATION STRATEGIES
# ═══════════════════════════════════════════════════════════════

def _rand_pts(n):
    """Uniform random in (0,1)², avoid boundary."""
    return (torch.rand(n) * 0.98 + 0.01,
            torch.rand(n) * 0.98 + 0.01)


class UniformSpatial:
    def __init__(self, n):
        self.x, self.y = _rand_pts(n)
    def get_points(self, *a):  return self.x.detach(), self.y.detach()
    def update(self, *a): pass


class RARSpatial:
    def __init__(self, n, n_replace=N_REPLACE, n_cand=10_000):
        self.n_replace = n_replace; self.n_cand = n_cand
        self.x, self.y = _rand_pts(n)

    def get_points(self, *a):  return self.x.detach(), self.y.detach()

    def _res(self, pinn, pde, x, y):
        r = []
        for i in range(0, len(x), 500):
            r.append(pde.pde_residual(pinn, x[i:i+500].detach(),
                                       y[i:i+500].detach()).detach().abs())
        return torch.cat(r)

    def update(self, pinn, pde):
        res = self._res(pinn, pde, self.x, self.y)
        _, worst = torch.topk(res, self.n_replace, largest=False)
        keep = torch.ones(len(self.x), dtype=torch.bool); keep[worst] = False
        xc, yc  = _rand_pts(self.n_cand)
        rc       = self._res(pinn, pde, xc, yc)
        _, best  = torch.topk(rc, self.n_replace)
        self.x   = torch.cat([self.x[keep], xc[best]]).detach()
        self.y   = torch.cat([self.y[keep], yc[best]]).detach()


class PACMANNSpatial:
    PERIOD = 50
    def __init__(self, n, n_steps=15, lr=1e-5):
        self.n_steps = n_steps; self.lr = lr
        self.b1=0.9; self.b2=0.999; self.eps=10e-8
        self.x, self.y = _rand_pts(n)

    def get_points(self, *a):  return self.x.detach(), self.y.detach()

    def update(self, pinn, pde):
        N   = len(self.x)
        pts = np.stack([self.x.detach().numpy(),
                        self.y.detach().numpy()], axis=1)
        V = np.zeros((N,2)); S = np.zeros((N,2))
        for n in range(self.n_steps):
            t  = torch.tensor(pts, dtype=torch.float32, requires_grad=True)
            r  = pde.pde_residual(pinn, t[:,0].requires_grad_(True),
                                         t[:,1].requires_grad_(True))
            torch.mean(r**2).backward()
            g  = t.grad.detach().numpy()
            V  = self.b1*V + (1-self.b1)*g
            S  = self.b2*S + (1-self.b2)*g**2
            Vc = V / (1 - self.b1**(n+1))
            Sc = S / (1 - self.b2**(n+1))
            pts = pts + self.lr * Vc / (np.sqrt(Sc) + self.eps)
            pts = np.clip(pts, 0.01, 0.99)   # stay inside domain
        self.x = torch.tensor(pts[:,0], dtype=torch.float32)
        self.y = torch.tensor(pts[:,1], dtype=torch.float32)


# ═══════════════════════════════════════════════════════════════
# 4. PPO AGENT (2D spatial)
# ═══════════════════════════════════════════════════════════════

def _density_map(x, y, G):
    d = torch.zeros(G, G)
    ix = (x * G).long().clamp(0, G-1)
    iy = (y * G).long().clamp(0, G-1)
    for i, j in zip(ix, iy):
        d[i, j] += 1
    return d


class PPOSpatialAgent(nn.Module):
    def __init__(self, G=G, lr=3e-4, clip=0.2, vf=0.5,
                 ent=0.01, lam=0.95, gamma=0.99, n_epochs=4):
        super().__init__()
        self.G=G; self.clip=clip; self.vf=vf; self.ent=ent
        self.lam=lam; self.gamma=gamma; self.n_epochs=n_epochs

        sd = G*G*3 + 2    # res_map + grad_map + density + [l2, mean_res]
        ad = G*G
        self.trunk  = nn.Sequential(nn.Linear(sd,256),nn.ReLU(),
                                     nn.Linear(256,256),nn.ReLU())
        self.actor  = nn.Linear(256, ad)
        self.critic = nn.Linear(256, 1)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
        self.opt = torch.optim.Adam(self.parameters(), lr=lr)

        self._s: List[torch.Tensor] = []
        self._lp: List[torch.Tensor] = []
        self._r: List[float]  = []
        self._v: List[float]  = []

    def get_state(self, pinn, pde, x, y, l2):
        G = self.G; eps = 1e-8
        gx = torch.linspace(0, 1, G)
        XX, YY = torch.meshgrid(gx, gx, indexing='ij')
        xf, yf = XX.flatten(), YY.flatten()

        res_map = pde.pde_residual(pinn, xf, yf).detach().abs().reshape(G,G)

        xg = xf.requires_grad_(True); yg = yf.requires_grad_(True)
        u2  = pinn(xg, yg)
        ux  = torch.autograd.grad(u2, xg, torch.ones_like(u2),
                                   retain_graph=True, create_graph=False)[0].detach()
        uy  = torch.autograd.grad(u2, yg, torch.ones_like(u2),
                                   create_graph=False)[0].detach()
        grad_map = (ux**2 + uy**2).sqrt().reshape(G, G)
        dens_map = _density_map(x, y, G)

        for m in (res_map, grad_map):
            m.div_(m.max() + eps)
        dens_map = dens_map / (dens_map.max() + eps)

        scalars = torch.tensor([l2, res_map.mean().item()])
        return torch.cat([res_map.flatten(), grad_map.flatten(),
                          dens_map.flatten(), scalars])

    def _fwd(self, s):
        h = self.trunk(s)
        return self.actor(h), self.critic(h).squeeze(-1)

    def act(self, state):
        logits, val = self._fwd(state)
        dist = torch.distributions.Categorical(logits=logits)
        lp   = dist.log_prob(dist.sample((100,))).mean()
        return dist.probs, lp, val.item()

    def sample_points(self, w, n):
        G = self.G
        idx = torch.multinomial(w, n, replacement=True)
        ix  = idx // G; iy = idx % G
        s   = 1.0 / G
        x   = (ix.float()*s + torch.rand(n)*s).clamp(0.01, 0.99)
        y   = (iy.float()*s + torch.rand(n)*s).clamp(0.01, 0.99)
        return x, y

    def store(self, s, lp, r, v):
        self._s.append(s.detach()); self._lp.append(lp.detach())
        self._r.append(r); self._v.append(v)

    def update(self, last_v=0.0):
        T = len(self._s)
        if T < 2: self._clear(); return
        vals_ext = self._v + [last_v]
        gae = 0.0; advs = []
        for t in reversed(range(T)):
            delta = self._r[t] + self.gamma*vals_ext[t+1] - vals_ext[t]
            gae   = delta + self.gamma*self.lam*gae
            advs.insert(0, gae)
        adv = torch.tensor(advs, dtype=torch.float32)
        ret = adv + torch.tensor(self._v, dtype=torch.float32)
        if T > 1: adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        states  = torch.stack(self._s)
        old_lps = torch.stack(self._lp)
        for _ in range(self.n_epochs):
            nlps, nvals, nents = [], [], []
            for s in states:
                logits, v = self._fwd(s)
                d = torch.distributions.Categorical(logits=logits)
                nlps.append(d.log_prob(d.sample((100,))).mean())
                nvals.append(v); nents.append(d.entropy())
            nlps = torch.stack(nlps); nvals = torch.stack(nvals)
            ent  = torch.stack(nents).mean()
            ratio = torch.exp(nlps - old_lps)
            loss  = (-torch.min(ratio*adv,
                                torch.clamp(ratio,1-self.clip,1+self.clip)*adv).mean()
                     + self.vf * F.mse_loss(nvals, ret)
                     - self.ent * ent)
            self.opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 0.5)
            self.opt.step()
        self._clear()

    def _clear(self):
        self._s=[]; self._lp=[]; self._r=[]; self._v=[]

    def save(self, p): torch.save(self.state_dict(), p); print(f"  Saved → {p}")
    def load(self, p): self.load_state_dict(torch.load(p, map_location='cpu')); print(f"  Loaded ← {p}")


class RLSpatialPPO:
    W1, W2 = 1.0, 0.1

    def __init__(self, n, n_replace=N_REPLACE,
                 agent: Optional[PPOSpatialAgent] = None,
                 frozen: bool = False):
        self.n       = n
        self.n_rep   = n_replace
        self.frozen  = frozen
        self.agent   = agent if agent else PPOSpatialAgent()
        self.x, self.y = _rand_pts(n)
        self.prev_l2   = None
        self.prev_dens = None
        self._sbuf = self._lpbuf = self._vbuf = None
        self.weight_history = []
        self._step = 0

    def get_points(self, *a):  return self.x.detach(), self.y.detach()

    def _res(self, pinn, pde, x, y):
        r = []
        for i in range(0, len(x), 500):
            r.append(pde.pde_residual(pinn, x[i:i+500].detach(),
                                       y[i:i+500].detach()).detach().abs())
        return torch.cat(r)

    def _entropy(self, d):
        p = d.flatten().float(); p = p/(p.sum()+1e-8)
        return -(p*(p+1e-8).log()).sum().item()

    def observe_and_act(self, pinn, pde, l2):
        self.prev_dens = _density_map(self.x, self.y, self.agent.G)
        state = self.agent.get_state(pinn, pde, self.x, self.y, l2)
        w, lp, v = self.agent.act(state)
        self._step += 1
        if self._step in (1,5,10,20):
            self.weight_history.append(
                (self._step, w.detach().reshape(self.agent.G, self.agent.G).clone()))
        self._sbuf=state; self._lpbuf=lp; self._vbuf=v
        self.prev_l2 = l2

        res = self._res(pinn, pde, self.x, self.y)
        _, worst = torch.topk(res, self.n_rep, largest=False)
        keep = torch.ones(len(self.x), dtype=torch.bool); keep[worst]=False
        nx, ny = self.agent.sample_points(w.detach(), self.n_rep)
        self.x = torch.cat([self.x[keep], nx]).detach()
        self.y = torch.cat([self.y[keep], ny]).detach()

    def store_reward(self, l2):
        nd   = _density_map(self.x, self.y, self.agent.G)
        r    = (self.W1*(self.prev_l2-l2)/self.n*1000
                + self.W2*(self._entropy(nd)-self._entropy(self.prev_dens)))
        if not self.frozen:
            self.agent.store(self._sbuf, self._lpbuf, r, self._vbuf)
        return r

    def finalize(self, last_l2=0.0):
        if not self.frozen:
            self.agent.update(last_v=last_l2)


# ═══════════════════════════════════════════════════════════════
# 5. EXPERIMENT RUNNER
# ═══════════════════════════════════════════════════════════════

def run_experiment(name, strategy, pde,
                   n_steps=20, epochs=500, seed=42, verbose=True):
    torch.manual_seed(seed); np.random.seed(seed)
    pinn = SpatialPINN()
    opt  = torch.optim.Adam(pinn.parameters(), lr=1e-3)
    sch  = torch.optim.lr_scheduler.StepLR(opt, step_size=2000, gamma=0.5)

    if verbose: print(f"\n{'='*46}\n{name}\n{'='*46}")

    cx, cy = strategy.get_points(pinn, pde)
    l2_hist = []

    for step in range(n_steps + 1):
        for ep in range(epochs):
            train_step(pinn, opt, pde, cx, cy)
            sch.step()
            if name=='PACMANN' and (ep+1)%PACMANNSpatial.PERIOD==0:
                strategy.update(pinn, pde)
                cx, cy = strategy.get_points()

        l2 = pinn.compute_l2_rel(pde)
        l2_hist.append(l2)
        if verbose: print(f"  step {step:3d} | L2={l2:.5f}")
        if step == n_steps: break

        if name == 'RAR':
            strategy.update(pinn, pde)
        elif isinstance(strategy, RLSpatialPPO):
            if strategy.prev_l2 is not None:
                strategy.store_reward(l2)
            strategy.observe_and_act(pinn, pde, l2)

        cx, cy = strategy.get_points(pinn, pde)

    if isinstance(strategy, RLSpatialPPO):
        strategy.finalize(l2_hist[-1])

    return {'name':name, 'l2':l2_hist,
            'final_l2':l2_hist[-1],
            'final_x':cx.detach(), 'final_y':cy.detach(),
            'weight_history':getattr(strategy,'weight_history',[])}


# ═══════════════════════════════════════════════════════════════
# 6. PRE-TRAINING
# ═══════════════════════════════════════════════════════════════

def pretrain(agent, locs, n_ep=3, n_steps=12, epochs=300, verbose=True):
    total = n_ep * len(locs)
    run   = 0
    for ep in range(n_ep):
        for fi, (sx, sy) in enumerate(locs):
            run += 1
            pde = SharpPoisson2D(sx, sy)
            if verbose:
                print(f"\n[Pre-train {run}/{total}]  src=({sx},{sy})  ep={ep}")
            strat = RLSpatialPPO(N_TOTAL, n_replace=N_REPLACE,
                                  agent=agent, frozen=False)
            run_experiment('RL-pretrained', strat, pde,
                           n_steps=n_steps, epochs=epochs,
                           seed=ep*100+fi, verbose=verbose)
    return agent


# ═══════════════════════════════════════════════════════════════
# 7. TRANSFER EVALUATION
# ═══════════════════════════════════════════════════════════════

def steps_to_threshold(l2_hist, thr, epochs):
    idx = next((i for i,v in enumerate(l2_hist) if v<thr), None)
    return np.inf if idx is None else (idx+1)*epochs


def run_transfer(agent, test_locs, n_steps, epochs, seeds, thr, verbose):
    results = {loc: [] for loc in test_locs}

    for si, seed in enumerate(seeds):
        print(f"\n{'─'*56}  seed {seed}  ({si+1}/{len(seeds)})")
        for loc in test_locs:
            sx, sy = loc
            print(f"\n  src=({sx},{sy})")
            pde  = SharpPoisson2D(sx, sy)
            frozen = copy.deepcopy(agent)
            for p in frozen.parameters(): p.requires_grad_(False)

            # Seed before creating strategies so initial collocation is reproducible
            torch.manual_seed(seed + 1000)
            np.random.seed(seed + 1000)
            strats = {
                'Uniform': UniformSpatial(N_TOTAL),
                'RAR':     RARSpatial(N_TOTAL, N_REPLACE),
                'PACMANN': PACMANNSpatial(N_TOTAL),
                'RL-pretrained': RLSpatialPPO(N_TOTAL, N_REPLACE,
                                               agent=frozen, frozen=True),
            }
            seed_res = {}
            for name, strat in strats.items():
                r = run_experiment(name, strat, pde,
                                   n_steps=n_steps, epochs=epochs,
                                   seed=seed, verbose=verbose)
                r['steps_to_thr'] = steps_to_threshold(r['l2'], thr, epochs)
                seed_res[name] = r
            results[loc].append(seed_res)
    return results


# ═══════════════════════════════════════════════════════════════
# 8. SUMMARY + PLOTS
# ═══════════════════════════════════════════════════════════════

def print_summary(results, thr, epochs, n_steps):
    methods  = list(next(iter(results.values()))[0].keys())
    max_s    = (n_steps + 1) * epochs
    locs     = list(results.keys())

    print(f"\n{'='*70}")
    print(f"  Gradient steps to L2 < {thr}  (median, {len(next(iter(results.values())))} seeds)")
    print(f"{'='*70}")
    print(f"{'Method':<18}", end='')
    for sx,sy in locs:
        print(f"  src=({sx},{sy})", end='')
    print()
    print('-'*70)

    for m in methods:
        print(f"{m:<18}", end='')
        for loc in locs:
            vals = [r[m]['steps_to_thr'] for r in results[loc]]
            med  = np.median(vals)
            tag  = 'never' if np.isinf(med) else f'{int(med):,}'
            print(f"  {tag:>12}", end='')
        print()

    print()
    print(f"{'Speedup RL/PAC':<18}", end='')
    for loc in locs:
        p_vals = [r['PACMANN']['steps_to_thr']      for r in results[loc]]
        r_vals = [r['RL-pretrained']['steps_to_thr'] for r in results[loc]]
        p_med  = np.median(p_vals); r_med = np.median(r_vals)
        tag    = '—' if (np.isinf(p_med) or np.isinf(r_med)) else f'{p_med/r_med:.2f}×'
        print(f"  {tag:>12}", end='')
    print(f"\n{'='*70}\n")


def plot_results(results, thr, epochs, save_prefix=os.path.join(DIR, 'geometry_transfer')):
    methods = list(next(iter(results.values()))[0].keys())
    locs    = list(results.keys())
    n_locs  = len(locs)

    fig, axes = plt.subplots(1, n_locs, figsize=(6*n_locs, 5), sharey=True)
    if n_locs == 1: axes = [axes]

    for col, loc in enumerate(locs):
        ax = axes[col]
        sx, sy = loc
        seed_results = results[loc]

        for m in methods:
            c    = COLORS.get(m,'#7f8c8d')
            lw   = 2.5 if m=='RL-pretrained' else 1.8
            curves = np.array([r[m]['l2'] for r in seed_results])
            gs   = np.array([(s+1)*epochs for s in range(curves.shape[1])])
            mu   = curves.mean(0); sd = curves.std(0)
            ax.semilogy(gs, mu, color=c, lw=lw,
                        label=m if col==0 else '_')
            ax.fill_between(gs, np.maximum(mu-sd,1e-5), mu+sd,
                            color=c, alpha=0.15)

        ax.axhline(thr, ls='--', color='#555', lw=1.2, alpha=0.8)
        ax.text(gs[0], thr*1.12, f'τ={thr}', fontsize=8, color='#555')
        ax.set_title(f'src = ({sx}, {sy})  [unseen]', fontsize=11)
        ax.set_xlabel('PINN gradient steps')
        if col==0: ax.set_ylabel('L2 relative error')
        ax.grid(alpha=0.25)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(
            lambda x,_: f'{int(x/1000)}k' if x>=1000 else str(int(x))))

    axes[0].legend(fontsize=9)
    fig.suptitle('2D Poisson — RL geometry transfer vs from-scratch methods\n'
                 '(pre-trained on different source locations, tested on unseen ones)',
                 fontsize=12, y=1.01)
    plt.tight_layout()
    fname = f'{save_prefix}_convergence.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f"Saved {fname}")
    plt.close()

    # Weight map for RL at first vs last step on each test location
    _plot_weight_maps(results, save_prefix)


def _plot_weight_maps(results, prefix):
    """Show what the RL agent attends to on each unseen geometry."""
    locs  = list(results.keys())
    fig, axes = plt.subplots(2, len(locs), figsize=(5*len(locs), 8))
    if len(locs)==1: axes = axes.reshape(2,1)

    for col, loc in enumerate(locs):
        sx, sy = loc
        wh = results[loc][0]['RL-pretrained']['weight_history']
        if len(wh) < 2:
            continue
        for row, (step_i, wmap) in enumerate(wh[:2]):
            ax = axes[row][col]
            im = ax.imshow(wmap.numpy().T, origin='lower',
                           extent=[0,1,0,1], cmap='hot', aspect='auto')
            ax.scatter([sx],[sy], c='cyan', s=80, marker='*',
                       zorder=5, label='true src')
            ax.set_title(f'src=({sx},{sy})\nstep {step_i}', fontsize=9)
            ax.set_xlabel('x'); ax.set_ylabel('y')
            plt.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle('RL attention map on unseen geometries\n'
                 '(★ = true source location)', fontsize=12)
    plt.tight_layout()
    fname = f'{prefix}_weight_maps.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f"Saved {fname}")
    plt.close()


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--skip-pretrain', action='store_true')
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--agent', default=AGENT_PATH)
    parser.add_argument('--threshold', type=float, default=0.20)
    parser.add_argument('--n-steps', type=int, default=None,
                        help='Override eval steps (default: 6 quick / 12 full)')
    parser.add_argument('--n-epochs', type=int, default=None,
                        help='Override epochs per step (default: 150 quick / 300 full)')
    args = parser.parse_args()

    if args.quick:
        pre_kw  = dict(n_ep=1, n_steps=5, epochs=100)
        eval_kw = dict(n_steps=6, epochs=150)
        test_locs = TEST_LOCS[:2]
    else:
        pre_kw  = dict(n_ep=3, n_steps=12, epochs=300)
        eval_kw = dict(n_steps=12, epochs=300)
        test_locs = TEST_LOCS

    if args.n_steps  is not None: eval_kw['n_steps']  = args.n_steps
    if args.n_epochs is not None: eval_kw['epochs']   = args.n_epochs

    seeds = list(range(args.seeds))

    # ── Pre-train ─────────────────────────────────────────────
    agent = PPOSpatialAgent(G=G)
    if args.skip_pretrain and os.path.exists(args.agent):
        agent.load(args.agent)
    elif not args.skip_pretrain:
        if os.path.exists(args.agent):
            print(f"  Cached agent found, skipping pre-train.")
            agent.load(args.agent)
        else:
            print(f"\n{'#'*60}")
            print(f"#  PRE-TRAINING on {len(TRAIN_LOCS)} source locations")
            print(f"{'#'*60}")
            pretrain(agent, TRAIN_LOCS, **pre_kw)
            agent.save(args.agent)

    # ── Transfer evaluation ───────────────────────────────────
    print(f"\n{'#'*60}")
    print(f"#  TRANSFER EVALUATION  ({len(test_locs)} geometries, {args.seeds} seeds)")
    print(f"{'#'*60}")

    results = run_transfer(agent, test_locs,
                           n_steps=eval_kw['n_steps'],
                           epochs=eval_kw['epochs'],
                           seeds=seeds,
                           thr=args.threshold,
                           verbose=True)

    print_summary(results, args.threshold,
                  eval_kw['epochs'], eval_kw['n_steps'])
    plot_results(results, args.threshold,
                 eval_kw['epochs'])

    # Save raw
    np.savez(os.path.join(DIR, 'geometry_transfer_results.npz'),
             test_locs=np.array(test_locs),
             train_locs=np.array(TRAIN_LOCS),
             threshold=args.threshold)
    print("Done.")
