"""
Collocation PINN  (strong form)
================================
Enforces  -ε·Δu + β·∂u/∂x₁ = f  pointwise at random interior points.

Loss = mean( residual² )

û = mollifier(x) · net(x)  — automatically satisfies u=0 on ∂Ω.

Works for both 1-D and 2-D.  Pure diffusion (Poisson) is the special case
ε=1, β=0.  Problem attributes  .eps  and  .beta  configure the operator.
"""

import time
import torch
from torch.autograd import Variable

from ..network import FCNet


class CollocationPINN:

    name = "Collocation PINN"

    def __init__(self,
                 activation:  str   = 'tanh',
                 lr:          float = 1e-3,
                 step_size:   int   = 1000,
                 gamma:       float = 0.5,
                 n_interior:  int   = 2000):
        self.activation  = activation
        self.lr          = lr
        self.step_size   = step_size
        self.gamma       = gamma
        self.n_interior  = n_interior
        self.net         = None

    # ── derivatives ──────────────────────────────────────────────

    @staticmethod
    def _grad_and_laplacian(u: torch.Tensor,
                             x_var: Variable) -> tuple:
        """Returns (∇u  [N,dim],  Δu  [N,1])  via two autograd passes."""
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
        layers    = problem.default_layers()
        net       = FCNet(layers, self.activation)
        optimizer = torch.optim.Adam(net.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.step_size, gamma=self.gamma)

        x_test, u_test = problem.test_grid()
        history = dict(epoch=[], loss_pde=[], l2_err=[], time=[])
        t0 = time.time()

        for epoch in range(1, epochs + 1):
            net.train()

            x_in  = Variable(problem.interior_points(self.n_interior),
                             requires_grad=True)
            u_hat = problem.mollifier(x_in) * net(x_in)
            grad, lap_u = self._grad_and_laplacian(u_hat, x_in)
            residual = problem.pde_residual(u_hat, grad, lap_u, x_in)
            loss = torch.mean(residual ** 2)

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
                    print(f"    epoch {epoch:5d} | loss={loss.item():.3e}"
                          f" | l2={l2:.4e} | lr={lr_now:.1e}")

        self.net     = net
        self.problem = problem
        return history

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.net.eval()
        with torch.no_grad():
            return self.problem.mollifier(x) * self.net(x)
