"""
Variational PINN  (weak / Galerkin form)
=========================================
Weak form of  -Δu = f  with zero Dirichlet BC:

    ∫_Ω  ∇û · ∇v  dx  =  ∫_Ω  f · v  dx       for all test functions v

Only first derivatives of û are needed (cheaper autograd than collocation).
û = mollifier(x) · net(x)  satisfies the BC exactly.

Test functions  v_k  are chosen to also vanish on ∂Ω:

  1-D  (domain [0,1]):
      v_k(x) = x(1-x) · L_{k+1}(2x-1)         k = 1…N_test
      The factor x(1-x) ensures v_k(0)=v_k(1)=0.
      L_{k+1} are Legendre polynomials on [-1,1]; 2x-1 maps [0,1]→[-1,1].

  2-D  (domain [-1,1]²):
      v_{pq}(x) = (1-x₁²)(1-x₂²) · L_{p+1}(x₁) · L_{q+1}(x₂)

Integration: Gauss-Legendre quadrature (exact for polynomials, spectrally
accurate for smooth functions).
"""

import time
import numpy as np
import torch
from torch.autograd import Variable
from numpy.polynomial.legendre import leggauss, legval, legder

from ..network import FCNet
from ..problems import PDE1D, PDE2D


class VPINN:

    name = "VPINN"

    def __init__(self,
                 n_quad:    int   = 50,    # GL points (1-D) or per axis (2-D)
                 n_test:    int   = 20,    # test functions (1-D) or per axis (2-D)
                 activation: str  = 'tanh',
                 lr:        float = 1e-3,
                 step_size: int   = 1000,
                 gamma:     float = 0.5):
        self.n_quad    = n_quad
        self.n_test    = n_test
        self.activation = activation
        self.lr        = lr
        self.step_size = step_size
        self.gamma     = gamma
        self.net       = None

    # ── quadrature + test-function setup ─────────────────────────

    def _setup_1d(self):
        """
        1-D Gauss-Legendre quadrature on [0,1] and precomputed
        v_k, dv_k/dx at every quadrature point.

        v_k(x) = x(1-x) · L_{k+1}(2x-1)
        v_k'(x) = (1-2x)·L_{k+1}(z) + 2·x(1-x)·L'_{k+1}(z)·2
                  where z = 2x-1 and the extra factor 2 is the chain-rule.
        """
        pts, wts = leggauss(self.n_quad)          # points in [-1,1]
        xq = (pts + 1.0) / 2.0                   # map to [0,1]
        wq = wts / 2.0                            # adjusted weights

        self.x_quad = torch.tensor(xq[:, None], dtype=torch.float32)  # (Nq,1)
        self.w_quad = torch.tensor(wq,           dtype=torch.float32)  # (Nq,)

        z    = 2.0 * xq - 1.0    # quadrature points in [-1,1] for Legendre eval
        phi  = xq * (1.0 - xq)   # x(1-x)
        dphi = 1.0 - 2.0 * xq    # d/dx [x(1-x)]

        Nq   = len(xq)
        Nt   = self.n_test
        v    = np.zeros((Nt, Nq))
        dv   = np.zeros((Nt, Nq))

        for k in range(Nt):
            deg = k + 1
            c   = np.zeros(deg + 1); c[deg] = 1.0
            dc  = legder(c)
            Lk  = legval(z, c)
            dLk = legval(z, dc)       # L'(z) w.r.t. z
            # v_k = phi * Lk
            v[k]  = phi * Lk
            # dv/dx = dphi*Lk + phi*(dLk * 2)   [chain rule: dz/dx = 2]
            dv[k] = dphi * Lk + phi * dLk * 2.0

        self.v_vals  = torch.tensor(v,  dtype=torch.float32)   # (Nt, Nq)
        self.dv_dx1  = torch.tensor(dv, dtype=torch.float32)   # (Nt, Nq) ← only x₁

    def _setup_2d(self):
        """
        2-D tensor-product GL quadrature on [-1,1]² and precomputed
        v_{pq}, ∂v/∂x₁, ∂v/∂x₂ at every quadrature point.

        v_{pq}(x) = (1-x₁²)(1-x₂²) · L_{p+1}(x₁) · L_{q+1}(x₂)
        """
        pts, wts = leggauss(self.n_quad)
        xx, yy   = np.meshgrid(pts, pts, indexing='ij')
        wx, wy   = np.meshgrid(wts, wts, indexing='ij')

        self.x_quad = torch.tensor(
            np.stack([xx.flatten(), yy.flatten()], axis=1), dtype=torch.float32)
        self.w_quad = torch.tensor(
            (wx * wy).flatten(), dtype=torch.float32)

        xq, yq = xx.flatten(), yy.flatten()
        Nq = len(xq)
        Nt = self.n_test

        # 1-D Legendre bases along each axis
        Lx  = np.zeros((Nt, Nq)); dLx = np.zeros((Nt, Nq))
        Ly  = np.zeros((Nt, Nq)); dLy = np.zeros((Nt, Nq))
        for k in range(Nt):
            deg = k + 1
            c   = np.zeros(deg + 1); c[deg] = 1.0
            dc  = legder(c)
            Lx[k]  = legval(xq, c);   dLx[k] = legval(xq, dc)
            Ly[k]  = legval(yq, c);   dLy[k] = legval(yq, dc)

        phi_x  = 1.0 - xq**2;  dphi_x = -2.0 * xq
        phi_y  = 1.0 - yq**2;  dphi_y = -2.0 * yq

        Nk = Nt * Nt
        v   = np.zeros((Nk, Nq))
        dvx = np.zeros((Nk, Nq))
        dvy = np.zeros((Nk, Nq))

        for p in range(Nt):
            for q in range(Nt):
                k  = p * Nt + q
                Ax = phi_x * Lx[p];  dAx = dphi_x * Lx[p] + phi_x * dLx[p]
                Ay = phi_y * Ly[q];  dAy = dphi_y * Ly[q] + phi_y * dLy[q]
                v[k]   = Ax * Ay
                dvx[k] = dAx * Ay
                dvy[k] = Ax  * dAy

        self.v_vals = torch.tensor(v,   dtype=torch.float32)   # (Nk, Nq)
        self.dv_dx1 = torch.tensor(dvx, dtype=torch.float32)
        self.dv_dx2 = torch.tensor(dvy, dtype=torch.float32)

    # ── public API ───────────────────────────────────────────────

    def solve(self, problem, epochs: int = 5000,
              log_every: int = 500) -> dict:

        # Build quadrature + test functions for the right dimension
        if problem.dim == 1:
            self._setup_1d()
        else:
            self._setup_2d()

        layers    = problem.default_layers()
        net       = FCNet(layers, self.activation)
        optimizer = torch.optim.Adam(net.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.step_size, gamma=self.gamma)

        eps  = getattr(problem, 'eps',  1.0)
        beta = getattr(problem, 'beta', 0.0)

        x_test, u_test = problem.test_grid()

        # Pre-compute f at quadrature points (fixed throughout training)
        with torch.no_grad():
            f_quad = problem.source_f(self.x_quad).squeeze()   # (Nq,)

        history = dict(epoch=[], loss_pde=[], l2_err=[], time=[])
        t0 = time.time()

        for epoch in range(1, epochs + 1):
            net.train()

            x_var = Variable(self.x_quad, requires_grad=True)
            u_hat = problem.mollifier(x_var) * net(x_var)      # (Nq, 1)

            # ∇û via one autograd pass
            u_grad = torch.autograd.grad(
                u_hat, x_var, torch.ones_like(u_hat), create_graph=True)[0]

            w = self.w_quad   # (Nq,)

            if problem.dim == 1:
                du = u_grad[:, 0]                              # (Nq,)
                # a(û,v_k) = ∫ ε·û'·v_k' dx + ∫ β·û'·v_k dx
                # Pure diffusion (β=0): just ε·∫û'·v_k'
                lhs = torch.sum(
                    w * (eps * self.dv_dx1 + beta * self.v_vals) * du,
                    dim=-1)                                        # (Nt,)
                rhs = torch.sum(w * self.v_vals * f_quad, dim=-1)  # (Nt,)
            else:
                du1 = u_grad[:, 0]; du2 = u_grad[:, 1]
                lhs = torch.sum(
                    w * eps * (self.dv_dx1 * du1 + self.dv_dx2 * du2), dim=-1)
                rhs = torch.sum(w * self.v_vals * f_quad, dim=-1)

            loss = torch.mean((lhs - rhs) ** 2)

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
                    print(f"    epoch {epoch:5d} | weak={loss.item():.3e}"
                          f" | l2={l2:.4e} | lr={lr_now:.1e}")

        self.net     = net
        self.problem = problem
        return history

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.net.eval()
        with torch.no_grad():
            return self.problem.mollifier(x) * self.net(x)
