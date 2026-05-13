"""
Deep Ritz Method
================
For symmetric operators (β=0, pure diffusion):
    Minimises  E[u] = ½ ∫_Ω |∇u|² dx  −  ∫_Ω f·u dx
    Only first-order autograd needed (same cost as VPINN).

For non-symmetric operators (β≠0, convection-diffusion):
    Standard energy ignores the skew-symmetric convection term,
    so we fall back to a least-squares energy:
        E_LS[u] = ½ ∫_Ω (−ε·u'' + β·u' − f)² dx
    This requires second-order autograd (same cost as collocation)
    but uses deterministic GL quadrature instead of random sampling.

û = mollifier(x)·net(x) enforces the BC exactly in both cases.
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
        xq = (pts + 1.0) / 2.0
        wq = wts / 2.0
        self.x_quad = torch.tensor(xq[:, None], dtype=torch.float32)
        self.w_quad = torch.tensor(wq,           dtype=torch.float32)

    def _setup_2d(self):
        pts, wts = leggauss(self.n_quad)
        xx, yy   = np.meshgrid(pts, pts, indexing='ij')
        wx, wy   = np.meshgrid(wts, wts, indexing='ij')
        self.x_quad = torch.tensor(
            np.stack([xx.flatten(), yy.flatten()], axis=1), dtype=torch.float32)
        self.w_quad = torch.tensor((wx * wy).flatten(), dtype=torch.float32)

    # ── second-order derivatives (for LS mode) ───────────────────

    @staticmethod
    def _grad_and_laplacian(u, x_var):
        """Returns (∇u [N,dim],  Δu [N,1]) via two autograd passes."""
        g = torch.autograd.grad(
            u, x_var, torch.ones_like(u), create_graph=True)[0]
        lap = torch.zeros_like(u)
        for i in range(x_var.shape[1]):
            gi  = g[:, i:i+1]
            uii = torch.autograd.grad(
                gi, x_var, torch.ones_like(gi), create_graph=True)[0][:, i:i+1]
            lap = lap + uii
        return g, lap

    # ── public API ───────────────────────────────────────────────

    def solve(self, problem, epochs: int = 5000,
              log_every: int = 500) -> dict:

        if problem.dim == 1:
            self._setup_1d()
        else:
            self._setup_2d()

        eps  = getattr(problem, 'eps',  1.0)
        beta = getattr(problem, 'beta', 0.0)
        use_ls = (beta != 0.0)   # least-squares mode for non-symmetric operators

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

        mode_str = "LS" if use_ls else "energy"

        for epoch in range(1, epochs + 1):
            net.train()

            x_var = Variable(self.x_quad, requires_grad=True)
            u_hat = problem.mollifier(x_var) * net(x_var)      # (Nq, 1)
            w     = self.w_quad                                 # (Nq,)

            if use_ls:
                # Least-squares: ½ ∫(-ε·u'' + β·u' − f)² dx  (needs u'')
                g, lap_u = self._grad_and_laplacian(u_hat, x_var)
                conv = beta * g[:, 0:1] if problem.dim == 1 else 0.0
                res  = -eps * lap_u + conv - f_quad.unsqueeze(1)  # (Nq,1)
                loss = 0.5 * torch.sum(w * res.squeeze() ** 2)
            else:
                # Standard energy: ½ ∫|∇u|² − ∫f·u
                u_grad = torch.autograd.grad(
                    u_hat, x_var, torch.ones_like(u_hat), create_graph=True)[0]
                if problem.dim == 1:
                    grad_sq = u_grad[:, 0] ** 2
                else:
                    grad_sq = u_grad[:, 0] ** 2 + u_grad[:, 1] ** 2
                loss = (0.5 * torch.sum(w * grad_sq)
                        - torch.sum(w * f_quad * u_hat.squeeze()))

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
                    print(f"    epoch {epoch:5d} | {mode_str}={loss.item():.3e}"
                          f" | l2={l2:.4e} | lr={lr_now:.1e}")

        self.net     = net
        self.problem = problem
        return history

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.net.eval()
        with torch.no_grad():
            return self.problem.mollifier(x) * self.net(x)
