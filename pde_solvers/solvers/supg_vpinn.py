"""
SUPG-VPINN  (Streamline Upwind Petrov-Galerkin)
================================================
Fixes Galerkin instability of standard VPINN for convection-dominated flows.

Standard VPINN uses test functions v_k that are symmetric — they treat upstream
and downstream equally.  For large Péclet number (β/ε >> 1) this causes the
same spurious-oscillation instability seen in classical Galerkin FEM.

SUPG remedy: perturb each test function by a small upwind bias:

    v_k^*(x) = v_k(x) + τ · β · v_k'(x)

where τ is the SUPG stabilisation parameter (units: time).  The extra term
adds weight *upstream* of each test-function support, which mirrors how
information flows in convection-dominated problems.

Modified bilinear form (1-D, -ε·u'' + β·u' = f):
    a(û, v_k^*) = ∫ ε·û'·(v_k' + τβ·v_k'') dx
                + ∫ β·û'·(v_k  + τβ·v_k' ) dx

Modified RHS:
    l(v_k^*) = ∫ f · (v_k + τβ·v_k') dx

v_k'' is precomputed analytically; only first-order autograd of û is needed.
For β=0 (Poisson) the scheme reduces exactly to standard VPINN.

Stabilisation parameter (classical 1-D formula):
    Pe_h = β · h / (2ε),   h = 1/n_test  (effective mode width)
    ξ(Pe) = coth(Pe) − 1/Pe   (Langevin / upwinding function)
    τ = h / (2β) · ξ(Pe_h)

    Large Pe:  ξ → 1  →  τ ≈ h/(2β)   (full upwinding)
    Small Pe:  ξ ≈ Pe/3  →  τ ≈ h²/(6ε)  (diffusion-dominated, small)
"""

import time
import numpy as np
import torch
from torch.autograd import Variable
from numpy.polynomial.legendre import leggauss, legval, legder

from ..network import FCNet


class SUPG_VPINN:

    name = "SUPG-VPINN"

    def __init__(self,
                 n_quad:     int   = 100,
                 n_test:     int   = 20,
                 activation: str   = 'tanh',
                 lr:         float = 1e-3,
                 step_size:  int   = 1000,
                 gamma:      float = 0.5,
                 tau:        float = None):   # None = auto from problem params
        self.n_quad    = n_quad
        self.n_test    = n_test
        self.activation = activation
        self.lr        = lr
        self.step_size = step_size
        self.gamma     = gamma
        self.tau_user  = tau   # user override; None = auto

    # ── quadrature + test-function setup ─────────────────────────

    def _setup_1d(self, eps: float, beta: float):
        """
        GL quadrature on [0,1] and precomputed v_k, v_k', v_k'' at
        every quadrature point.

        v_k = φ·L_{k+1}(z),    φ = x(1-x),  z = 2x-1
        v_k'  = φ'·L + 2φ·L'
        v_k'' = -2·L + 4φ'·L' + 4φ·L''      (φ'' = -2)
        """
        pts, wts = leggauss(self.n_quad)
        xq = (pts + 1.0) / 2.0
        wq = wts / 2.0

        self.x_quad = torch.tensor(xq[:, None], dtype=torch.float32)
        self.w_quad = torch.tensor(wq,           dtype=torch.float32)

        z    = 2.0 * xq - 1.0
        phi  = xq * (1.0 - xq)
        dphi = 1.0 - 2.0 * xq       # φ'
        # φ'' = -2  (constant)

        Nq, Nt = len(xq), self.n_test
        v   = np.zeros((Nt, Nq))
        dv  = np.zeros((Nt, Nq))
        d2v = np.zeros((Nt, Nq))

        for k in range(Nt):
            deg  = k + 1
            c    = np.zeros(deg + 1);  c[deg]  = 1.0
            dc   = legder(c)
            ddc  = legder(dc)
            Lk   = legval(z, c)
            dLk  = legval(z, dc)    # dL/dz
            d2Lk = legval(z, ddc)   # d²L/dz²

            v[k]   = phi  * Lk
            dv[k]  = dphi * Lk + phi * dLk * 2.0
            d2v[k] = (-2.0) * Lk + 4.0 * dphi * dLk + 4.0 * phi * d2Lk

        self.v_vals  = torch.tensor(v,   dtype=torch.float32)   # (Nt, Nq)
        self.dv_dx   = torch.tensor(dv,  dtype=torch.float32)   # (Nt, Nq)
        self.d2v_dx2 = torch.tensor(d2v, dtype=torch.float32)   # (Nt, Nq)

        # ── SUPG stabilisation parameter τ ────────────────────────
        if self.tau_user is not None:
            self.tau = self.tau_user
        else:
            h    = 1.0 / max(self.n_test, 1)
            babs = abs(beta)
            if babs < 1e-12:
                self.tau = 0.0          # pure diffusion → standard VPINN
            else:
                Pe_h = babs * h / (2.0 * max(eps, 1e-12))
                if Pe_h < 1e-8:
                    xi = Pe_h / 3.0     # small-Pe limit: ξ ≈ Pe/3
                else:
                    xi = 1.0 / np.tanh(Pe_h) - 1.0 / Pe_h   # coth(Pe) - 1/Pe
                self.tau = h / (2.0 * babs) * xi

    # ── public API ───────────────────────────────────────────────

    def solve(self, problem, epochs: int = 5000,
              log_every: int = 500) -> dict:

        assert problem.dim == 1, "SUPG-VPINN currently supports 1-D only."

        eps  = getattr(problem, 'eps',  1.0)
        beta = getattr(problem, 'beta', 0.0)

        self._setup_1d(eps, beta)
        tau = self.tau

        layers    = problem.default_layers()
        net       = FCNet(layers, self.activation)
        optimizer = torch.optim.Adam(net.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.step_size, gamma=self.gamma)

        x_test, u_test = problem.test_grid()

        with torch.no_grad():
            f_quad = problem.source_f(self.x_quad).squeeze()   # (Nq,)

        # Precompute SUPG-modified test-function coefficients (fixed)
        # v^* = v + τβ·v'      →  coefficients for RHS
        # (v^*)' = v' + τβ·v'' →  coefficients for LHS diffusion term
        tb = tau * beta
        v_star    = self.v_vals + tb * self.dv_dx        # (Nt, Nq)
        dv_star   = self.dv_dx  + tb * self.d2v_dx2     # (Nt, Nq)

        # LHS coefficient combined:  ε·(v^*)' + β·v^*
        lhs_coeff = eps * dv_star + beta * v_star        # (Nt, Nq)

        history = dict(epoch=[], loss_pde=[], l2_err=[], time=[])
        t0 = time.time()

        if log_every <= epochs:
            print(f"    τ = {tau:.3e}  (Pe_h = {abs(beta)/(self.n_test * max(eps,1e-12)):.2f})")

        for epoch in range(1, epochs + 1):
            net.train()

            x_var = Variable(self.x_quad, requires_grad=True)
            u_hat = problem.mollifier(x_var) * net(x_var)   # (Nq, 1)

            u_grad = torch.autograd.grad(
                u_hat, x_var, torch.ones_like(u_hat), create_graph=True)[0]

            du = u_grad[:, 0]       # (Nq,)
            w  = self.w_quad        # (Nq,)

            # a(û, v^*) = ∫ (ε·(v^*)' + β·v^*) · û' dx
            lhs = torch.sum(w * lhs_coeff * du, dim=-1)   # (Nt,)
            # l(v^*) = ∫ f · v^* dx
            rhs = torch.sum(w * v_star * f_quad, dim=-1)  # (Nt,)

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
                    print(f"    epoch {epoch:5d} | supg={loss.item():.3e}"
                          f" | l2={l2:.4e} | lr={lr_now:.1e}")

        self.net     = net
        self.problem = problem
        return history

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.net.eval()
        with torch.no_grad():
            return self.problem.mollifier(x) * self.net(x)
