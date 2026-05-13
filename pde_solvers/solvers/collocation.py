"""
Collocation PINN  (strong form)
================================
Enforces  -Δu = f  pointwise at random interior collocation points.

Loss = mean( (-Δû - f)² )  +  w_bc · mean( (û|∂Ω)² )

û = (1-x₁²)(1-x₂²)·net(x)  — mollifier automatically satisfies u=0 on ∂[-1,1]².
The BC term is kept as a sanity monitor; it should be near-zero from epoch 1.
"""

import time
import torch
from torch.autograd import Variable

from ..network import FCNet
from ..problems import PDE2D


class CollocationPINN:

    name = "Collocation PINN"

    def __init__(self,
                 layer_sizes: tuple = (2, 64, 64, 64, 64, 1),
                 activation:  str   = 'tanh',
                 lr:          float = 1e-3,
                 step_size:   int   = 1000,
                 gamma:       float = 0.5,
                 n_interior:  int   = 2000,
                 w_bc:        float = 10.0):
        self.layer_sizes = layer_sizes
        self.activation  = activation
        self.lr          = lr
        self.step_size   = step_size
        self.gamma       = gamma
        self.n_interior  = n_interior
        self.w_bc        = w_bc
        self.net         = None

    # ── internal helpers ─────────────────────────────────────────

    @staticmethod
    def _mollifier(x: torch.Tensor) -> torch.Tensor:
        """(1-x₁²)(1-x₂²) — zero on all four edges of [-1,1]²."""
        return (1.0 - x[:, 0:1]**2) * (1.0 - x[:, 1:2]**2)

    def _u_hat(self, net, x: torch.Tensor) -> torch.Tensor:
        return self._mollifier(x) * net(x)

    def _laplacian(self, u: torch.Tensor, x_var: Variable) -> torch.Tensor:
        """Δu at x_var via two autograd passes."""
        g = torch.autograd.grad(u, x_var, torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(
            g[:, 0:1], x_var, torch.ones_like(g[:, 0:1]), create_graph=True)[0][:, 0:1]
        u_yy = torch.autograd.grad(
            g[:, 1:2], x_var, torch.ones_like(g[:, 1:2]), create_graph=True)[0][:, 1:2]
        return u_xx + u_yy

    # ── public API ───────────────────────────────────────────────

    def solve(self, problem: PDE2D, epochs: int = 5000,
              log_every: int = 500) -> dict:
        """
        Train on `problem` for `epochs` epochs.
        Returns history dict with keys: epoch, loss_pde, loss_bc, l2_err, time.
        """
        net       = FCNet(list(self.layer_sizes), self.activation)
        optimizer = torch.optim.Adam(net.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.step_size, gamma=self.gamma)

        x_bd, u_bd   = problem.boundary_points(n_per_side=100)
        x_test, u_test = problem.test_grid(n=60)

        history = dict(epoch=[], loss_pde=[], loss_bc=[], l2_err=[], time=[])
        t0 = time.time()

        for epoch in range(1, epochs + 1):
            net.train()

            x_in  = Variable(problem.interior_points(self.n_interior),
                             requires_grad=True)
            u_hat = self._u_hat(net, x_in)
            lap_u = self._laplacian(u_hat, x_in)
            f_val = problem.source_f(x_in)
            loss_pde = torch.mean((-lap_u - f_val) ** 2)

            # BC residual (should be ~0 thanks to mollifier, but we track it)
            u_bc  = self._u_hat(net, x_bd)
            loss_bc  = torch.mean((u_bc - u_bd) ** 2)

            loss = loss_pde + self.w_bc * loss_bc
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if epoch % 100 == 0:
                net.eval()
                with torch.no_grad():
                    u_p = self._u_hat(net, x_test)
                    l2  = problem.l2_rel(u_p, u_test)
                history['epoch'].append(epoch)
                history['loss_pde'].append(loss_pde.item())
                history['loss_bc'].append(loss_bc.item())
                history['l2_err'].append(l2)
                history['time'].append(time.time() - t0)

                if epoch % log_every == 0:
                    lr_now = optimizer.param_groups[0]['lr']
                    print(f"    epoch {epoch:5d} | pde={loss_pde.item():.3e} "
                          f"| bc={loss_bc.item():.3e} | l2={l2:.4e} | lr={lr_now:.1e}")

        self.net = net
        return history

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.net.eval()
        with torch.no_grad():
            return self._u_hat(self.net, x)
