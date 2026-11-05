"""
NS-PINN  —  Strong-form collocation for 2D steady incompressible NS
====================================================================
Network: (x, y, Re_norm) → (u, v, p)   [3 inputs, 3 outputs]

PDE (non-dimensionalised on [0,1]²):
    continuity  : ∂u/∂x + ∂v/∂y = 0
    x-momentum  : u·u_x + v·u_y + p_x − (1/Re)·(u_xx + u_yy) = 0
    y-momentum  : u·v_x + v·v_y + p_y − (1/Re)·(v_xx + v_yy) = 0

Training strategy (parametric surrogate):
    Re is sampled uniformly from [Re_min, Re_max] at EVERY step.
    One network generalises across the full Re range — no retraining needed
    for a new operating condition.

Loss:
    L = λ_pde  · mean(cont² + xmom² + ymom²)
      + λ_bc   · mean((u−u_bc)² + (v−v_bc)²)
      + λ_pref · p(x_ref)²          ← fixes pressure gauge

Architecture options:
    standard : FCNet(3 → [128,128,128,128] → 3)   tanh activations
    fourier  : FourierFCNet(fourier_dim=64, sigma)  — better for high Re
"""

import time
import numpy as np
import torch
from torch.autograd import Variable

from ..network import FCNet, FourierFCNet


class NSCollocation:

    name = "NS-Collocation"

    def __init__(self,
                 n_interior:   int   = 2000,
                 n_bc:         int   = 100,    # per side
                 hidden:       list  = None,
                 activation:   str   = 'tanh',
                 fourier:       bool  = False,
                 fourier_dim:   int   = 64,
                 sigma:         float = 1.0,
                 lr:            float = 1e-3,
                 step_size:     int   = 2000,
                 gamma:         float = 0.5,
                 lam_pde:       float = 1.0,
                 lam_bc:        float = 10.0,
                 lam_pref:      float = 1.0):
        self.n_interior  = n_interior
        self.n_bc        = n_bc
        self.hidden      = hidden or [128, 128, 128, 128]
        self.activation  = activation
        self.fourier     = fourier
        self.fourier_dim = fourier_dim
        self.sigma       = sigma
        self.lr          = lr
        self.step_size   = step_size
        self.gamma       = gamma
        self.lam_pde     = lam_pde
        self.lam_bc      = lam_bc
        self.lam_pref    = lam_pref
        self.net         = None

    # ── network factory ──────────────────────────────────────────

    def _make_net(self):
        if self.fourier:
            return FourierFCNet(in_dim=3, out_dim=3,
                                fourier_dim=self.fourier_dim,
                                sigma=self.sigma,
                                hidden=self.hidden,
                                activation=self.activation)
        return FCNet([3] + self.hidden + [3], self.activation)

    # ── derivative helper ────────────────────────────────────────

    @staticmethod
    def _derivatives(net, xy_var: torch.Tensor, Re_col: torch.Tensor):
        """
        Forward pass + all first and second spatial derivatives.
        xy_var : (N,2)  with requires_grad=True
        Re_col : (N,1)  no grad needed
        Returns dict of tensors, each (N,1).
        """
        x_in = torch.cat([xy_var, Re_col], dim=1)
        out  = net(x_in)             # (N, 3)
        u, v, p = out[:, 0:1], out[:, 1:2], out[:, 2:3]

        def _g1(scalar):
            return torch.autograd.grad(
                scalar, xy_var, torch.ones_like(scalar),
                create_graph=True)[0]              # (N,2)

        gu = _g1(u);  u_x, u_y = gu[:, 0:1], gu[:, 1:2]
        gv = _g1(v);  v_x, v_y = gv[:, 0:1], gv[:, 1:2]
        gp = _g1(p);  p_x, p_y = gp[:, 0:1], gp[:, 1:2]

        u_xx = _g1(u_x)[:, 0:1];  u_yy = _g1(u_y)[:, 1:2]
        v_xx = _g1(v_x)[:, 0:1];  v_yy = _g1(v_y)[:, 1:2]

        return dict(u=u, v=v, p=p,
                    u_x=u_x, u_y=u_y, v_x=v_x, v_y=v_y,
                    p_x=p_x, p_y=p_y,
                    lap_u=u_xx + u_yy, lap_v=v_xx + v_yy)

    # ── public API ───────────────────────────────────────────────

    def solve(self, problem, epochs: int = 10000,
              log_every: int = 1000) -> dict:

        net       = self._make_net()
        optimizer = torch.optim.Adam(net.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.step_size, gamma=self.gamma)

        history = dict(epoch=[], loss_pde=[], loss_bc=[], time=[])
        t0 = time.time()
        arch = "Fourier" if self.fourier else "standard"
        print(f"    arch={arch}  Re=[{problem.Re_min:.0f},{problem.Re_max:.0f}]")

        for epoch in range(1, epochs + 1):
            net.train()
            Re = problem.sample_Re()
            inv_Re = 1.0 / Re

            # ── interior points ───────────────────────────────
            x_int   = problem.interior_points(self.n_interior, Re)
            xy_var  = Variable(x_int[:, :2], requires_grad=True)
            Re_col  = x_int[:, 2:3].detach()

            d = self._derivatives(net, xy_var, Re_col)
            u, v     = d['u'], d['v']
            u_x, u_y = d['u_x'], d['u_y']
            v_x, v_y = d['v_x'], d['v_y']
            p_x, p_y = d['p_x'], d['p_y']
            lap_u, lap_v = d['lap_u'], d['lap_v']

            cont  = u_x + v_y
            xmom  = u * u_x + v * u_y + p_x - inv_Re * lap_u
            ymom  = u * v_x + v * v_y + p_y - inv_Re * lap_v

            loss_pde = torch.mean(cont**2 + xmom**2 + ymom**2)

            # ── boundary points ───────────────────────────────
            x_bc, u_bc, v_bc = problem.bc_points(self.n_bc, Re)
            with torch.no_grad():
                x_in_bc = x_bc
            out_bc  = net(x_in_bc)
            u_pred_bc = out_bc[:, 0:1]
            v_pred_bc = out_bc[:, 1:2]
            loss_bc = torch.mean((u_pred_bc - u_bc)**2 +
                                 (v_pred_bc - v_bc)**2)

            # ── pressure reference (gauge fix at corner) ──────
            x_ref  = problem.pressure_ref_point(Re)
            p_ref  = net(x_ref)[:, 2:3]
            loss_p = p_ref**2

            loss = (self.lam_pde  * loss_pde
                  + self.lam_bc   * loss_bc
                  + self.lam_pref * loss_p)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if epoch % 200 == 0:
                history['epoch'].append(epoch)
                history['loss_pde'].append(loss_pde.item())
                history['loss_bc'].append(loss_bc.item())
                history['time'].append(time.time() - t0)

                if epoch % log_every == 0:
                    lr_now = optimizer.param_groups[0]['lr']
                    print(f"    epoch {epoch:6d} | Re={Re:6.0f}"
                          f" | pde={loss_pde.item():.3e}"
                          f" | bc={loss_bc.item():.3e}"
                          f" | lr={lr_now:.1e}")

        self.net     = net
        self.problem = problem
        return history

    # ── inference ────────────────────────────────────────────────

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N,3) = (x, y, Re_norm)  →  (N,3) = (u, v, p)."""
        self.net.eval()
        with torch.no_grad():
            return self.net(x)

    def predict_Re(self, xy: np.ndarray, Re: float) -> np.ndarray:
        """
        Convenience: xy (N,2) numpy + scalar Re → (N,3) numpy (u,v,p).
        Fast inference at any Re without retraining.
        """
        Re_col = np.full((len(xy), 1), Re / self.problem.Re_ref)
        x_in   = torch.tensor(np.hstack([xy, Re_col]), dtype=torch.float32)
        return self.predict(x_in).numpy()
