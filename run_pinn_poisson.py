"""
Classical PINN for 2D Poisson equation.

PDE:  -Δu = f  on Ω = [-1,1]²
BC:    u  = 0  on ∂Ω
Exact: u(x) = sin(λx₁)sin(λx₂),  λ = 4π
RHS:   f(x) = 2λ²sin(λx₁)sin(λx₂)
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── reproducibility ──────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device('cpu')
dtype  = torch.float32

# ── hyper-parameters ─────────────────────────────────────────────
LAMBDA      = 1.0 * np.pi   # λ=π: smooth target, ideal for demonstrating PINN
N_INTERIOR  = 2000
N_BD_EACH   = 200
N_TEST_MESH = 60            # 60×60 test grid
EPOCHS      = 5000
LR          = 1e-3
STEP_SIZE   = 1000
GAMMA       = 0.5
W_BC        = 20.0          # boundary condition weight

# ── network ──────────────────────────────────────────────────────
from Networks.FCNet import FCNet

net = FCNet([2, 64, 64, 64, 64, 1], activation='Tanh_Sin', dtype=dtype).to(device)

# ── PDE helpers ──────────────────────────────────────────────────
def exact_u(x: torch.Tensor) -> torch.Tensor:
    return torch.sin(LAMBDA * x[:, 0:1]) * torch.sin(LAMBDA * x[:, 1:2])

def source_f(x: torch.Tensor) -> torch.Tensor:
    return 2.0 * LAMBDA**2 * torch.sin(LAMBDA * x[:, 0:1]) * torch.sin(LAMBDA * x[:, 1:2])

def laplacian(u: torch.Tensor, x: Variable) -> torch.Tensor:
    """Compute Δu via autograd (2-D)."""
    grads = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                create_graph=True)[0]
    u_xx = torch.autograd.grad(grads[:, 0:1], x,
                               grad_outputs=torch.ones_like(grads[:, 0:1]),
                               create_graph=True)[0][:, 0:1]
    u_yy = torch.autograd.grad(grads[:, 1:2], x,
                               grad_outputs=torch.ones_like(grads[:, 1:2]),
                               create_graph=True)[0][:, 1:2]
    return u_xx + u_yy

# ── point generation ─────────────────────────────────────────────
def rand_interior(n):
    x = torch.rand(n, 2, dtype=dtype) * 2.0 - 1.0   # uniform on [-1,1]²
    return x.to(device)

def boundary_points(n_each):
    """4 edges of [-1,1]², n_each points each."""
    t = torch.linspace(-1, 1, n_each, dtype=dtype)
    ones = torch.ones(n_each, dtype=dtype)
    edges = [
        torch.stack([-ones, t], dim=1),   # left
        torch.stack([ ones, t], dim=1),   # right
        torch.stack([t, -ones], dim=1),   # bottom
        torch.stack([t,  ones], dim=1),   # top
    ]
    return torch.cat(edges, dim=0).to(device)

def test_mesh(n):
    t = torch.linspace(-1, 1, n, dtype=dtype)
    xx, yy = torch.meshgrid(t, t, indexing='ij')
    x = torch.stack([xx.flatten(), yy.flatten()], dim=1)
    return x.to(device)

x_bd  = boundary_points(N_BD_EACH)    # (4*N_BD_EACH, 2)  – fixed
x_test = test_mesh(N_TEST_MESH)        # (N²,2) – fixed

# ── optimizer & scheduler ─────────────────────────────────────────
optimizer = torch.optim.Adam(net.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

# ── training loop ─────────────────────────────────────────────────
history = {'epoch': [], 'loss_pde': [], 'loss_bc': [], 'loss': [], 'l2_err': []}

print(f"Training PINN on 2-D Poisson  |  epochs={EPOCHS}  |  device={device}")
print(f"Network: FCNet [2,64,64,64,1] + Tanh_Sin  |  λ={LAMBDA:.4f}")
print("-" * 65)

for epoch in range(1, EPOCHS + 1):
    net.train()

    # fresh interior collocation each epoch (encourages broader coverage)
    x_in = Variable(rand_interior(N_INTERIOR), requires_grad=True)

    u_pred = net(x_in)
    lap_u  = laplacian(u_pred, x_in)
    f_vals = source_f(x_in)
    loss_pde = torch.mean((- lap_u - f_vals) ** 2)

    # boundary: u = 0
    u_bd   = net(x_bd)
    loss_bc = torch.mean(u_bd ** 2)

    loss = loss_pde + W_BC * loss_bc

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    if epoch % 100 == 0:
        net.eval()
        with torch.no_grad():
            u_pred_t = net(x_test)
            u_true_t = exact_u(x_test)
            l2_err   = (torch.norm(u_pred_t - u_true_t) /
                        torch.norm(u_true_t)).item()

        history['epoch'].append(epoch)
        history['loss_pde'].append(loss_pde.item())
        history['loss_bc'].append(loss_bc.item())
        history['loss'].append(loss.item())
        history['l2_err'].append(l2_err)

        lr_now = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:4d} | loss={loss.item():.4e} "
              f"| pde={loss_pde.item():.4e} | bc={loss_bc.item():.4e} "
              f"| l2_err={l2_err:.4e} | lr={lr_now:.1e}")

# ── final evaluation ─────────────────────────────────────────────
net.eval()
with torch.no_grad():
    u_pred_np = net(x_test).cpu().numpy().reshape(N_TEST_MESH, N_TEST_MESH)
u_true_np = exact_u(x_test).cpu().numpy().reshape(N_TEST_MESH, N_TEST_MESH)
err_np    = np.abs(u_pred_np - u_true_np)

final_l2  = np.linalg.norm(u_pred_np - u_true_np) / np.linalg.norm(u_true_np)
print(f"\nFinal relative L2 error: {final_l2:.4e}")

t = np.linspace(-1, 1, N_TEST_MESH)
XX, YY = np.meshgrid(t, t, indexing='ij')

# ── plots ────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 10))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

vmin, vmax = u_true_np.min(), u_true_np.max()

# 1 – true solution
ax = fig.add_subplot(gs[0, 0])
im = ax.contourf(XX, YY, u_true_np, levels=50, cmap='RdBu_r')
fig.colorbar(im, ax=ax)
ax.set_title('Exact  $u$')
ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')

# 2 – predicted solution
ax = fig.add_subplot(gs[0, 1])
im = ax.contourf(XX, YY, u_pred_np, levels=50, cmap='RdBu_r',
                 vmin=vmin, vmax=vmax)
fig.colorbar(im, ax=ax)
ax.set_title('PINN  $\\hat u$')
ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')

# 3 – point-wise error
ax = fig.add_subplot(gs[0, 2])
im = ax.contourf(XX, YY, err_np, levels=50, cmap='hot_r')
fig.colorbar(im, ax=ax)
ax.set_title(f'|error|  (L2 rel = {final_l2:.2e})')
ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')

# 4 – total loss
ax = fig.add_subplot(gs[1, 0])
ax.semilogy(history['epoch'], history['loss'],    label='total')
ax.semilogy(history['epoch'], history['loss_pde'], label='PDE residual')
ax.semilogy(history['epoch'], history['loss_bc'],  label='BC')
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
ax.set_title('Training losses')
ax.legend()
ax.grid(True, which='both', ls='--', alpha=0.5)

# 5 – L2 error curve
ax = fig.add_subplot(gs[1, 1])
ax.semilogy(history['epoch'], history['l2_err'], color='green')
ax.set_xlabel('Epoch'); ax.set_ylabel('Relative L2 error')
ax.set_title('Test error vs epoch')
ax.grid(True, which='both', ls='--', alpha=0.5)

# 6 – 1-D cross-section at x₂ = 0
ax = fig.add_subplot(gs[1, 2])
mid = N_TEST_MESH // 2
ax.plot(t, u_true_np[:, mid], 'k-',  lw=2,   label='Exact')
ax.plot(t, u_pred_np[:, mid], 'r--', lw=1.5, label='PINN')
ax.set_xlabel('$x_1$'); ax.set_ylabel('$u(x_1, 0)$')
ax.set_title('Slice at $x_2=0$')
ax.legend(); ax.grid(True, ls='--', alpha=0.5)

fig.suptitle('Physics-Informed Neural Network  –  2D Poisson  '
             f'($\\lambda = \\pi$,  {EPOCHS} epochs)', fontsize=13)

out = 'pinn_poisson_results.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
print(f"\nResults saved to {out}")
