"""
LS-VPINN  (Least-Squares VPINN with SUPG test functions)
=========================================================
Projects the strong-form residual R(û) = Lû − f onto SUPG-modified
test functions and minimises the sum-of-squares of those projections:

    Loss = mean_k  ( ∫_Ω R(û) · v_k^*(x) dx )²

    v_k^*(x) = v_k(x) + τ · β · v_k'(x)   (SUPG modification)

Properties vs. other solvers:

    Collocation PINN : strong residual, random pts, no test fns
    VPINN            : weak form (IBP), GL quad, Galerkin test fns
    SUPG-VPINN       : weak form (IBP), GL quad, SUPG test fns
    Deep Ritz LS     : strong residual, GL quad, no test fns
    LS-VPINN         : strong residual, GL quad, SUPG test fns  ← here

Advantages:
  - GL quadrature → deterministic, no random-sampling noise
  - SUPG test fns → upwind bias suppresses Galerkin oscillations
  - Loss = sum-of-squares of projections → well-conditioned gradient
  - Works directly for nonlinear operators (Burgers) via pde_residual
  - Does NOT require IBP, so no integration-by-parts boundary terms

Cost: two autograd passes (needs u'' like collocation), but deterministic
quadrature usually compensates vs. random resampling at each step.
"""

import time
import numpy as np
import torch
from torch.autograd import Variable
from numpy.polynomial.legendre import leggauss, legval, legder

from ..network import FCNet


class LSVPINN:

    name = "LS-VPINN"

    def __init__(self,
                 n_quad:     int   = 100,
                 n_test:     int   = 20,
                 activation: str   = 'tanh',
                 lr:         float = 1e-3,
                 step_size:  int   = 1000,
                 gamma:      float = 0.5,
                 tau:        float = None):
        self.n_quad     = n_quad
        self.n_test     = n_test
        self.activation = activation
        self.lr         = lr
        self.step_size  = step_size
        self.gamma      = gamma
        self.tau_user   = tau

    # ── quadrature + SUPG test-function setup ────────────────────

    def _setup_1d(self, eps: float, beta: float):
        """
        GL quadrature on [0,1] and precomputed SUPG test-function values
        v_k^*(x_q) = v_k(x_q) + τ·β·v_k'(x_q).

        Only v and v' are needed (R already contains u'').
        """
        pts, wts = leggauss(self.n_quad)
        xq = (pts + 1.0) / 2.0
        wq = wts / 2.0

        self.x_quad = torch.tensor(xq[:, None], dtype=torch.float32)
        self.w_quad = torch.tensor(wq,           dtype=torch.float32)

        z    = 2.0 * xq - 1.0
        phi  = xq * (1.0 - xq)
        dphi = 1.0 - 2.0 * xq

        Nq, Nt = len(xq), self.n_test
        v  = np.zeros((Nt, Nq))
        dv = np.zeros((Nt, Nq))

        for k in range(Nt):
            deg = k + 1
            c   = np.zeros(deg + 1); c[deg] = 1.0
            dc  = legder(c)
            Lk  = legval(z, c)
            dLk = legval(z, dc)
            v[k]  = phi  * Lk
            dv[k] = dphi * Lk + phi * dLk * 2.0

        # SUPG stabilisation parameter τ (same formula as SUPG-VPINN)
        if self.tau_user is not None:
            self.tau = self.tau_user
        else:
            h    = 1.0 / max(self.n_test, 1)
            babs = abs(beta)
            if babs < 1e-12:
                self.tau = 0.0
            else:
                Pe_h = babs * h / (2.0 * max(eps, 1e-12))
                if Pe_h < 1e-8:
                    xi = Pe_h / 3.0
                else:
                    xi = 1.0 / np.tanh(Pe_h) - 1.0 / Pe_h
                self.tau = h / (2.0 * babs) * xi

        tb = self.tau * beta
        self.v_star = torch.tensor(v + tb * dv, dtype=torch.float32)  # (Nt, Nq)

    # ── second-order derivatives ──────────────────────────────────

    @staticmethod
    def _grad_and_laplacian(u, x_var):
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

        assert problem.dim == 1, "LS-VPINN currently supports 1-D only."

        eps  = getattr(problem, 'eps',  1.0)
        beta = getattr(problem, 'beta', 0.0)
        # Nonlinear problems: use characteristic velocity for τ
        if getattr(problem, 'nonlinear', False) and beta == 0.0:
            beta_eff = 0.5
        else:
            beta_eff = beta

        self._setup_1d(eps, beta_eff)

        layers    = problem.default_layers()
        net       = FCNet(layers, self.activation)
        optimizer = torch.optim.Adam(net.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.step_size, gamma=self.gamma)

        x_test, u_test = problem.test_grid()

        history = dict(epoch=[], loss_pde=[], l2_err=[], time=[])
        t0 = time.time()

        if log_every <= epochs:
            tau = self.tau
            print(f"    τ = {tau:.3e}  (Pe_h = {abs(beta_eff)/(self.n_test * max(eps,1e-12)):.2f})")

        for epoch in range(1, epochs + 1):
            net.train()

            x_var = Variable(self.x_quad, requires_grad=True)
            u_hat = problem.mollifier(x_var) * net(x_var)   # (Nq, 1)

            grad, lap_u = self._grad_and_laplacian(u_hat, x_var)

            # Strong-form residual R(û) at every quadrature point: (Nq, 1)
            residual = problem.pde_residual(u_hat, grad, lap_u, x_var)

            # Project onto SUPG test functions: r_k = ∑_q w_q R_q v_k^*(x_q)
            w = self.w_quad                                    # (Nq,)
            R = residual.squeeze()                             # (Nq,)
            r = torch.sum(w * R * self.v_star, dim=-1)        # (Nt,)

            loss = torch.mean(r ** 2)

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
                    print(f"    epoch {epoch:5d} | ls={loss.item():.3e}"
                          f" | l2={l2:.4e} | lr={lr_now:.1e}")

        self.net     = net
        self.problem = problem
        return history

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.net.eval()
        with torch.no_grad():
            return self.problem.mollifier(x) * self.net(x)
