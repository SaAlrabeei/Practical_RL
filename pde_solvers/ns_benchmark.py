"""
2D Lid-Driven Cavity  —  Parametric NS surrogate benchmark.

Smooth-lid problem: u_lid(x) = 16x²(1-x)²  (removes corner singularity).
Reference solution: finite-difference stream function-vorticity solver.
Surrogate: single NS-PINN trained once on Re ∈ [100, 1000].

The key story:
  - FD solver: ~4–10 s per Re value  (must re-run for every new Re)
  - PINN surrogate: ~10 min training ONCE, then 0.26 ms per new Re

Usage:
    python -m pde_solvers.ns_benchmark            # 15 000 epochs
    python -m pde_solvers.ns_benchmark --quick    # 2 000 epochs, smoke test
    python -m pde_solvers.ns_benchmark --fourier  # use Fourier feature network
"""

import argparse
import time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from .problems import LidDrivenCavity2D
from .solvers.ns_pinn import NSCollocation
from .solvers.fd_cavity import solve_cavity_fd, centreline_u


# ════════════════════════════════════════════════════════════════
# Reference solution (FD)
# ════════════════════════════════════════════════════════════════

def compute_fd_references(Re_list, smooth_lid=True):
    """Run FD solver at each Re; return dict Re -> {'y': ..., 'u': ..., 'elapsed': ...}."""
    refs = {}
    print(f"\n  Computing FD reference solutions...")
    for Re in Re_list:
        r = solve_cavity_fd(Re=Re, n=65, smooth_lid=smooth_lid,
                            max_iter=25000, tol=1e-7, verbose=True)
        y_cl, u_cl = centreline_u(r)
        refs[Re] = dict(y=y_cl, u=u_cl, elapsed=r['elapsed'],
                        converged=r['converged'], full=r)
    return refs


# ════════════════════════════════════════════════════════════════
# Validation helpers
# ════════════════════════════════════════════════════════════════

def print_fd_errors(solver, fd_refs):
    """Print L2 error vs FD reference for each Re."""
    print(f"\n{'═'*60}")
    print(f"  {'Re':>6}   L2 err vs FD   FD time   PINN time")
    print(f"{'─'*60}")
    for Re, ref in fd_refs.items():
        xy = np.stack([np.full(len(ref['y']), 0.5), ref['y']], axis=1)
        t0 = time.time()
        uvp = solver.predict_Re(xy, Re)
        t_pinn = (time.time() - t0) * 1000
        err = (np.linalg.norm(uvp[:, 0] - ref['u']) /
               max(np.linalg.norm(ref['u']), 1e-12))
        print(f"  {Re:>6}   {err:.4e}   {ref['elapsed']:5.1f}s     {t_pinn:.2f} ms")
    print(f"{'═'*60}")

    # Interpolation at Re=600 (unseen)
    y_line = np.linspace(0, 1, 200)
    xy_line = np.stack([np.full(200, 0.5), y_line], axis=1)
    t0 = time.time()
    _ = solver.predict_Re(xy_line, 600.0)
    ms = (time.time() - t0) * 1000
    print(f"\n  Inference at Re=600 (interpolated): {ms:.2f} ms")


# ════════════════════════════════════════════════════════════════
# Plotting
# ════════════════════════════════════════════════════════════════

def plot_results(solver, problem, history, fd_refs,
                 save_path='ns_cavity_benchmark.png'):
    """
    3-row figure:
      Row 0: training convergence
      Row 1: centreline u vs FD reference  (Re=100, 400, 1000)
      Row 2: 2D speed fields               (Re=100, Re=600, Re=1000)
    """
    Re_vals  = list(fd_refs.keys())       # [100, 400, 1000]
    colors   = ['#e74c3c', '#2ecc71', '#9b59b6']
    fig = plt.figure(figsize=(18, 14))
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.38)

    # ── Row 0: convergence ────────────────────────────────────────
    ax = fig.add_subplot(gs[0, :])
    ep = history['epoch']
    ax.semilogy(ep, history['loss_pde'], color='#e74c3c', lw=2,
                label='PDE residual')
    ax.semilogy(ep, history['loss_bc'],  color='#3498db', lw=2, ls='--',
                label='BC loss')
    ax.set_xlabel('Epoch');  ax.set_ylabel('Loss')
    ax.set_title('Training convergence  — parametric Re ∈ [100, 1000]  '
                 '(train ONCE, infer at any Re)', fontsize=10)
    ax.legend(fontsize=9);  ax.grid(True, which='both', ls='--', alpha=0.35)

    # ── Row 1: centreline vs FD ───────────────────────────────────
    for col, (Re, color) in enumerate(zip(Re_vals, colors)):
        ax  = fig.add_subplot(gs[1, col])
        ref = fd_refs[Re]

        y_line = np.linspace(0, 1, 200)
        xy_in  = np.stack([np.full(200, 0.5), y_line], axis=1)
        t0     = time.time()
        uvp    = solver.predict_Re(xy_in, Re)
        t_ms   = (time.time() - t0) * 1000

        err = (np.linalg.norm(
                   np.interp(ref['y'], y_line, uvp[:, 0]) - ref['u']) /
               max(np.linalg.norm(ref['u']), 1e-12))

        ax.plot(uvp[:, 0], y_line, color=color, lw=2.5,
                label=f'PINN  (Re={Re})')
        ax.plot(ref['u'], ref['y'], 'k--', lw=1.5, label='FD reference')
        ax.set_xlabel('u');  ax.set_ylabel('y')
        ax.set_title(f'Centreline u  Re={Re}\nL2 err={err:.2%}  '
                     f'| PINN: {t_ms:.2f} ms  FD: {ref["elapsed"]:.1f} s',
                     fontsize=9)
        ax.legend(fontsize=7);  ax.grid(True, ls='--', alpha=0.35)

    # ── Row 2: 2D speed fields ────────────────────────────────────
    for col, Re in enumerate([100, 600, 1000]):
        ax = fig.add_subplot(gs[2, col])
        n  = 40
        t  = np.linspace(0, 1, n)
        xx, yy = np.meshgrid(t, t, indexing='ij')
        xy_grid = np.stack([xx.flatten(), yy.flatten()], axis=1)
        uvp  = solver.predict_Re(xy_grid, Re)
        U    = uvp[:, 0].reshape(n, n)
        V    = uvp[:, 1].reshape(n, n)
        speed = np.sqrt(U**2 + V**2)
        cm   = ax.contourf(xx, yy, speed, levels=20, cmap='viridis')
        ax.streamplot(t, t, U.T, V.T, color='white', linewidth=0.6,
                      density=1.2, arrowsize=0.8)
        plt.colorbar(cm, ax=ax, fraction=0.046, pad=0.04)
        label = f'Re={Re}' + (' (interpolated)' if Re == 600 else '')
        ax.set_title(f'Speed |u|  {label}', fontsize=9)
        ax.set_xlabel('x');  ax.set_ylabel('y')

    # Timing annotation
    total_fd = sum(r['elapsed'] for r in fd_refs.values())
    y_inf    = np.linspace(0, 1, 200)
    xy_inf   = np.stack([np.full(200, 0.5), y_inf], axis=1)
    t0       = time.time()
    _        = solver.predict_Re(xy_inf, 600.0)
    t_pinn_ms = (time.time() - t0) * 1000

    fig.suptitle(
        '2D Lid-Driven Cavity  —  Parametric NS-PINN surrogate\n'
        f'Train once: {total_fd:.0f}s FD reference at 3 Re values | '
        f'PINN inference: {t_pinn_ms:.2f} ms at ANY Re  '
        f'(smooth lid: u_lid = 16x²(1−x)²)',
        fontsize=11, y=1.01)

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved → {save_path}")


# ════════════════════════════════════════════════════════════════
# Entry point
# ════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick',   action='store_true',
                        help='2 000 epochs — fast smoke test')
    parser.add_argument('--fourier', action='store_true',
                        help='Use Fourier feature network')
    args = parser.parse_args()

    epochs     = 2000  if args.quick else 15000
    log_every  = 500   if args.quick else 1000
    n_pretrain = 500   if args.quick else 2000

    Re_list = [100, 400, 1000]
    problem = LidDrivenCavity2D(Re_min=100.0, Re_max=1000.0, smooth_lid=True)

    # ── 1. FD reference solutions ─────────────────────────────────
    fd_refs = compute_fd_references(Re_list, smooth_lid=True)

    # ── 2. Train PINN surrogate ───────────────────────────────────
    solver = NSCollocation(
        n_interior = 2000,
        n_bc       = 100,
        fourier    = args.fourier,
        fourier_dim= 64,
        sigma      = 2.0,
        lr         = 1e-3,
        step_size  = 3000,
        gamma      = 0.5,
        lam_pde    = 1.0,
        lam_bc     = 100.0,
        lam_pref   = 1.0,
    )

    print(f"\n{'#'*60}")
    print(f"#  2D Lid-Driven Cavity NS Benchmark   epochs={epochs}")
    print(f"#  Network: {'Fourier' if args.fourier else 'Standard FCNet'}")
    print(f"{'#'*60}\n")

    t0 = time.time()
    history = solver.solve(problem, epochs=epochs, log_every=log_every,
                           n_pretrain=n_pretrain)
    elapsed = time.time() - t0
    print(f"\n  Total training time: {elapsed:.1f}s")

    # ── 3. Evaluate ───────────────────────────────────────────────
    print_fd_errors(solver, fd_refs)
    plot_results(solver, problem, history, fd_refs,
                 save_path='ns_cavity_benchmark.png')


if __name__ == '__main__':
    main()
