"""
Variational PINN  (weak / Galerkin form)
=========================================
Instead of enforcing  -Δu = f  pointwise, multiply both sides by a test
function v and integrate by parts over Ω = [-1,1]²:

    ∫_Ω ∇u · ∇v dx  =  ∫_Ω f·v dx          (*)

This holds for all v ∈ H₀¹(Ω).  (*) only needs first derivatives of u
(cheaper autograd) and yields a smoother loss landscape.

Approximation choices
─────────────────────
u_θ  = (1-x₁²)(1-x₂²)·net(x)         [mollifier → zero BC exactly]

Test functions:
  v_{pq}(x) = (1-x₁²)(1-x₂²)·L_{p+1}(x₁)·L_{q+1}(x₂)
  where L_k is the Legendre polynomial of degree k,  p,q = 0…N_test-1.
  The (1-xᵢ²) factor makes v_{pq} ∈ H₀¹ (vanishes on ∂Ω).

Integration: 2-D Gauss-Legendre quadrature with N_quad points per axis.

Loss:
  L = Σ_{p,q}  ( Σ_i w_i ∇u_θ(x_i)·∇v_{pq}(x_i)  −  Σ_i w_i f(x_i)v_{pq}(x_i) )²
"""

import time
import numpy as np
import torch
from torch.autograd import Variable
from numpy.polynomial.legendre import leggauss, legval, legder

from ..network import FCNet
from ..problems import PDE2D


class VPINN:

    name = "VPINN"

    def __init__(self,
                 n_quad:      int   = 30,    # GL points per axis (30² = 900 total)
                 n_test:      int   = 12,    # test-fn degree per axis (12² = 144 fns)
                 layer_sizes: tuple = (2, 64, 64, 64, 64, 1),
                 activation:  str   = 'tanh',
                 lr:          float = 1e-3,
                 step_size:   int   = 1000,
                 gamma:       float = 0.5):
        self.n_quad      = n_quad
        self.n_test      = n_test
        self.layer_sizes = layer_sizes
        self.activation  = activation
        self.lr          = lr
        self.step_size   = step_size
        self.gamma       = gamma
        self.net         = None

        self._build_quadrature()
        self._build_test_functions()

    # ── quadrature setup ─────────────────────────────────────────

    def _build_quadrature(self):
        """2-D Gauss-Legendre rule on [-1,1]² via tensor product."""
        pts, wts = leggauss(self.n_quad)           # 1-D rule
        xx, yy   = np.meshgrid(pts, pts, indexing='ij')
        wx, wy   = np.meshgrid(wts, wts, indexing='ij')

        self.x_quad = torch.tensor(
            np.stack([xx.flatten(), yy.flatten()], axis=1), dtype=torch.float32)
        self.w_quad = torch.tensor(
            (wx * wy).flatten(), dtype=torch.float32)            # (N_q,)

    # ── test-function setup ──────────────────────────────────────

    def _build_test_functions(self):
        """
        Precompute v_{pq} and ∇v_{pq} at all quadrature points.

        v_{pq} = φ(x₁)·L_{p+1}(x₁) · φ(x₂)·L_{q+1}(x₂)
        where  φ(t) = 1 - t²,  φ'(t) = -2t

        ∂v_{pq}/∂x₁ = [φ'(x₁)·L_{p+1}(x₁) + φ(x₁)·L'_{p+1}(x₁)] · φ(x₂)·L_{q+1}(x₂)
        ∂v_{pq}/∂x₂ = φ(x₁)·L_{p+1}(x₁) · [φ'(x₂)·L_{q+1}(x₂) + φ(x₂)·L'_{q+1}(x₂)]

        Stored tensors (float32):
          v_vals : (N_k, N_q)
          dv_dx  : (N_k, N_q)
          dv_dy  : (N_k, N_q)
        where N_k = n_test².
        """
        xq = self.x_quad[:, 0].numpy()
        yq = self.x_quad[:, 1].numpy()
        N_q = len(xq)
        N_k = self.n_test ** 2

        # Legendre poly values and derivatives at quadrature points
        Lx  = np.zeros((self.n_test, N_q))
        dLx = np.zeros((self.n_test, N_q))
        Ly  = np.zeros((self.n_test, N_q))
        dLy = np.zeros((self.n_test, N_q))

        for k in range(self.n_test):
            deg = k + 1                          # degrees 1 … n_test
            c   = np.zeros(deg + 1); c[deg] = 1.0
            dc  = legder(c)
            Lx[k]  = legval(xq, c);   dLx[k] = legval(xq, dc)
            Ly[k]  = legval(yq, c);   dLy[k] = legval(yq, dc)

        # Mollifier  φ(t) = 1-t²  and its derivative
        phi_x  = 1.0 - xq**2;   dphi_x = -2.0 * xq
        phi_y  = 1.0 - yq**2;   dphi_y = -2.0 * yq

        v_vals = np.zeros((N_k, N_q))
        dv_dx  = np.zeros((N_k, N_q))
        dv_dy  = np.zeros((N_k, N_q))

        for p in range(self.n_test):
            for q in range(self.n_test):
                k = p * self.n_test + q
                Ax  = phi_x  * Lx[p]                         # φ(x₁)·L_{p+1}(x₁)
                dAx = dphi_x * Lx[p] + phi_x  * dLx[p]      # d/dx₁
                Ay  = phi_y  * Ly[q]
                dAy = dphi_y * Ly[q] + phi_y  * dLy[q]

                v_vals[k] = Ax * Ay
                dv_dx[k]  = dAx * Ay
                dv_dy[k]  = Ax  * dAy

        self.v_vals = torch.tensor(v_vals, dtype=torch.float32)  # (N_k, N_q)
        self.dv_dx  = torch.tensor(dv_dx,  dtype=torch.float32)
        self.dv_dy  = torch.tensor(dv_dy,  dtype=torch.float32)

    # ── internal helpers ─────────────────────────────────────────

    @staticmethod
    def _mollifier(x: torch.Tensor) -> torch.Tensor:
        return (1.0 - x[:, 0:1]**2) * (1.0 - x[:, 1:2]**2)

    def _u_hat(self, net, x: torch.Tensor) -> torch.Tensor:
        return self._mollifier(x) * net(x)

    # ── public API ───────────────────────────────────────────────

    def solve(self, problem: PDE2D, epochs: int = 5000,
              log_every: int = 500) -> dict:
        """
        Train on `problem` for `epochs` epochs.
        Returns history dict with keys: epoch, loss_pde, l2_err, time.
        """
        net       = FCNet(list(self.layer_sizes), self.activation)
        optimizer = torch.optim.Adam(net.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.step_size, gamma=self.gamma)

        x_test, u_test = problem.test_grid(n=60)

        # Pre-compute f at quadrature points (fixed for the whole run)
        with torch.no_grad():
            f_quad = problem.source_f(self.x_quad).squeeze()   # (N_q,)

        history = dict(epoch=[], loss_pde=[], l2_err=[], time=[])
        t0 = time.time()

        for epoch in range(1, epochs + 1):
            net.train()

            # u_hat at quadrature points — mollifier enforces zero BC
            x_var = Variable(self.x_quad, requires_grad=True)
            u_hat = self._u_hat(net, x_var)               # (N_q, 1)

            # ∇u_hat via a single autograd pass
            u_grad = torch.autograd.grad(
                u_hat, x_var, torch.ones_like(u_hat), create_graph=True)[0]
            du_dx = u_grad[:, 0]                           # (N_q,)
            du_dy = u_grad[:, 1]

            # Bilinear form  a(u,v) = ∫ ∇u·∇v dx
            # Vectorised over all N_k test functions at once:
            #   lhs[k] = Σ_q  w_q · (du_dx[q]·dv_dx[k,q] + du_dy[q]·dv_dy[k,q])
            w = self.w_quad                                # (N_q,)
            lhs = torch.sum(
                w * (self.dv_dx * du_dx + self.dv_dy * du_dy), dim=-1)   # (N_k,)

            # Linear functional  L(v) = ∫ f·v dx
            rhs = torch.sum(w * self.v_vals * f_quad, dim=-1)             # (N_k,)

            loss_pde = torch.mean((lhs - rhs) ** 2)

            optimizer.zero_grad()
            loss_pde.backward()
            optimizer.step()
            scheduler.step()

            if epoch % 100 == 0:
                net.eval()
                with torch.no_grad():
                    u_p = self._u_hat(net, x_test)
                    l2  = problem.l2_rel(u_p, u_test)
                history['epoch'].append(epoch)
                history['loss_pde'].append(loss_pde.item())
                history['l2_err'].append(l2)
                history['time'].append(time.time() - t0)

                if epoch % log_every == 0:
                    lr_now = optimizer.param_groups[0]['lr']
                    print(f"    epoch {epoch:5d} | weak={loss_pde.item():.3e} "
                          f"| l2={l2:.4e} | lr={lr_now:.1e}")

        self.net = net
        return history

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.net.eval()
        with torch.no_grad():
            return self._u_hat(self.net, x)
