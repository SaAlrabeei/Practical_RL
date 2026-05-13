"""
Deep Ritz Method
================
Minimises the energy functional of  -Δu = f  with zero Dirichlet BC:

    E[u] = ½ ∫_Ω |∇u|² dx  -  ∫_Ω f·u dx

The minimiser satisfies the Euler–Lagrange equation -Δu = f.
Only first-order autograd is needed (same cost as VPINN).
û = mollifier(x)·net(x) enforces the BC exactly.

Integration: Gauss-Legendre quadrature.
  1-D  domain [0,1]:      n_quad points mapped from [-1,1]
  2-D  domain [-1,1]²:    tensor-product GL quadrature
"""

import time
import numpy as np
import torch
from torch.autograd import Variable
from numpy.polynomial.legendre import leggauss

from ..network import FCNet


class DeepRitz:

    name = "Deep Ritz"

    def __init__(self,
                 n_quad:    int   = 50,
                 activation: str  = 'tanh',
                 lr:        float = 1e-3,
                 step_size: int   = 1000,
                 gamma:     float = 0.5):
        self.n_quad    = n_quad
        self.activation = activation
        self.lr        = lr
        self.step_size = step_size
        self.gamma     = gamma
        self.net       = None

    # ── quadrature setup ─────────────────────────────────────────

    def _setup_1d(self):
        pts, wts = leggauss(self.n_quad)
        xq = (pts + 1.0) / 2.0    # map [-1,1] → [0,1]
        wq = wts / 2.0             # Jacobian of the mapping

        self.x_quad = torch.tensor(xq[:, None], dtype=torch.float32)
        self.w_quad = torch.tensor(wq,           dtype=torch.float32)

    def _setup_2d(self):
        pts, wts = leggauss(self.n_quad)
        xx, yy   = np.meshgrid(pts, pts, indexing='ij')
        wx, wy   = np.meshgrid(wts, wts, indexing='ij')

        self.x_quad = torch.tensor(
            np.stack([xx.flatten(), yy.flatten()], axis=1), dtype=torch.float32)
        self.w_quad = torch.tensor((wx * wy).flatten(), dtype=torch.float32)

    # ── public API ───────────────────────────────────────────────

    def solve(self, problem, epochs: int = 5000,
              log_every: int = 500) -> dict:

        if problem.dim == 1:
            self._setup_1d()
        else:
            self._setup_2d()

        layers    = problem.default_layers()
        net       = FCNet(layers, self.activation)
        optimizer = torch.optim.Adam(net.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.step_size, gamma=self.gamma)

        x_test, u_test = problem.test_grid()

        with torch.no_grad():
            f_quad = problem.source_f(self.x_quad).squeeze()   # (Nq,)

        history = dict(epoch=[], loss_pde=[], l2_err=[], time=[])
        t0 = time.time()

        for epoch in range(1, epochs + 1):
            net.train()

            x_var = Variable(self.x_quad, requires_grad=True)
            u_hat = problem.mollifier(x_var) * net(x_var)      # (Nq, 1)

            u_grad = torch.autograd.grad(
                u_hat, x_var, torch.ones_like(u_hat), create_graph=True)[0]

            w = self.w_quad                                    # (Nq,)

            if problem.dim == 1:
                grad_sq = u_grad[:, 0] ** 2                   # (Nq,)
            else:
                grad_sq = u_grad[:, 0] ** 2 + u_grad[:, 1] ** 2

            # E[u] = ½ ∫|∇u|² - ∫ f·u
            energy_term = 0.5 * torch.sum(w * grad_sq)
            source_term = torch.sum(w * f_quad * u_hat.squeeze())
            loss = energy_term - source_term

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if epoch % 100 == 0:
                net.eval()
                with torch.no_grad():
                    u_p = problem.mollifier(x_test) * net(x_test)
                    l2  = problem.l2_rel(u_p, u_test)
                history['epoch'].append(epoch)
                history['loss_pde'].append(loss.item())
                history['l2_err'].append(l2)
                history['time'].append(time.time() - t0)

                if epoch % log_every == 0:
                    lr_now = optimizer.param_groups[0]['lr']
                    print(f"    epoch {epoch:5d} | energy={loss.item():.3e}"
                          f" | l2={l2:.4e} | lr={lr_now:.1e}")

        self.net     = net
        self.problem = problem
        return history

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.net.eval()
        with torch.no_grad():
            return self.problem.mollifier(x) * self.net(x)
