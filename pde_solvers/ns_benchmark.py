"""
2D Lid-Driven Cavity  —  Parametric NS surrogate benchmark.

Trains a single NS-PINN that takes (x, y, Re) as input and outputs (u, v, p).
Validates against Ghia et al. (1982) benchmark data at Re=100, 400, 1000.
Tests interpolation at Re=600 (unseen during training).

Usage:
    python -m pde_solvers.ns_benchmark            # 10 000 epochs
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


# ════════════════════════════════════════════════════════════════
# Plotting helpers
# ════════════════════════════════════════════════════════════════

def plot_results(solver, problem, history, save_path='ns_cavity_benchmark.png'):
    """
    3-row figure:
      Row 0: convergence (pde + bc loss)
      Row 1-3: centreline u-velocity vs Ghia for Re=100, 400, 1000
      Last row: 2D flow field for Re=100 and Re=1000 side-by-side
    """
    Re_vals = [100, 400, 1000]
    n_rows  = 3   # convergence | centreline x3 | flow fields
    fig = plt.figure(figsize=(18, 14))
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.38)

    # ── (row 0) convergence ───────────────────────────────────
    ax = fig.add_subplot(gs[0, :])
    ep = history['epoch']
    ax.semilogy(ep, history['loss_pde'], label='PDE residual', color='#e74c3c', lw=2)
    ax.semilogy(ep, history['loss_bc'],  label='BC loss',      color='#3498db', lw=2, ls='--')
    ax.set_xlabel('Epoch');  ax.set_ylabel('Loss')
    ax.set_title('Training convergence  (parametric Re ∈ [100, 1000])', fontsize=10)
    ax.legend(fontsize=9);  ax.grid(True, which='both', ls='--', alpha=0.35)

    # ── (row 1) centreline profiles vs Ghia ──────────────────
    colors = ['#e74c3c', '#2ecc71', '#9b59b6']
    for col, Re in enumerate(Re_vals):
        ax = fig.add_subplot(gs[1, col])
        y_ghia, u_ghia = problem.ghia_centreline(Re)

        # Predict on vertical centreline x=0.5
        y_line = np.linspace(0, 1, 200)
        xy_line = np.stack([np.full(200, 0.5), y_line], axis=1)
        t0 = time.time()
        uvp = solver.predict_Re(xy_line, Re)
        t_inf = (time.time() - t0) * 1000   # ms

        ax.plot(uvp[:, 0], y_line,     color=colors[col], lw=2,
                label=f'PINN  (Re={Re})')
        ax.scatter(u_ghia, y_ghia, marker='o', s=30, color='k',
                   zorder=5, label='Ghia et al.')
        ax.set_xlabel('u');  ax.set_ylabel('y')
        ax.set_title(f'Centreline u  Re={Re}\n({t_inf:.1f} ms inference)', fontsize=9)
        ax.legend(fontsize=7);  ax.grid(True, ls='--', alpha=0.35)

    # ── (row 2) 2D flow field: Re=100 | Re=600 | Re=1000 ─────
    for col, Re in enumerate([100, 600, 1000]):
        ax = fig.add_subplot(gs[2, col])
        n  = 40
        t  = np.linspace(0, 1, n)
        xx, yy = np.meshgrid(t, t, indexing='ij')
        xy_grid = np.stack([xx.flatten(), yy.flatten()], axis=1)
        uvp = solver.predict_Re(xy_grid, Re)
        U = uvp[:, 0].reshape(n, n)
        V = uvp[:, 1].reshape(n, n)
        speed = np.sqrt(U**2 + V**2)
        cm = ax.contourf(xx, yy, speed, levels=20, cmap='viridis')
        ax.streamplot(t, t, U.T, V.T, color='white', linewidth=0.6,
                      density=1.2, arrowsize=0.8)
        plt.colorbar(cm, ax=ax, fraction=0.046, pad=0.04)
        label = f'Re={Re}' + (' (interpolated)' if Re == 600 else '')
        ax.set_title(f'Speed field  {label}', fontsize=9)
        ax.set_xlabel('x');  ax.set_ylabel('y')

    fig.suptitle('2D Lid-Driven Cavity  —  Parametric NS-PINN surrogate\n'
                 'Trained on Re∈[100,1000], validated vs Ghia et al. (1982)',
                 fontsize=12, y=1.01)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved → {save_path}")


def print_ghia_errors(solver, problem):
    """Print L2 error vs Ghia benchmark for Re=100, 400, 1000."""
    print(f"\n{'═'*55}")
    print(f"  {'Re':>6}   L2 error vs Ghia centreline")
    print(f"{'─'*55}")
    for Re in [100, 400, 1000]:
        y_ghia, u_ghia = problem.ghia_centreline(Re)
        xy_pts = np.stack([np.full(len(y_ghia), 0.5), y_ghia], axis=1)
        uvp    = solver.predict_Re(xy_pts, Re)
        u_pred = uvp[:, 0]
        err = np.linalg.norm(u_pred - u_ghia) / np.linalg.norm(u_ghia)
        print(f"  {Re:>6}   {err:.4e}")
    print(f"{'═'*55}")

    # Interpolation at Re=600 (no reference — just show it runs)
    y_line = np.linspace(0, 1, 200)
    xy_line = np.stack([np.full(200, 0.5), y_line], axis=1)
    t0  = time.time()
    _   = solver.predict_Re(xy_line, 600.0)
    ms  = (time.time() - t0) * 1000
    print(f"\n  Inference at Re=600 (interpolated): {ms:.2f} ms")


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

    epochs    = 2000  if args.quick else 10000
    log_every = 500   if args.quick else 1000

    problem = LidDrivenCavity2D(Re_min=100.0, Re_max=1000.0)

    solver = NSCollocation(
        n_interior = 2000,
        n_bc       = 100,
        fourier    = args.fourier,
        fourier_dim= 64,
        sigma      = 2.0,
        lr         = 1e-3,
        step_size  = 2000,
        gamma      = 0.5,
        lam_pde    = 1.0,
        lam_bc     = 10.0,
        lam_pref   = 1.0,
    )

    print(f"\n{'#'*60}")
    print(f"#  2D Lid-Driven Cavity NS Benchmark   epochs={epochs}")
    print(f"#  Network: {'Fourier' if args.fourier else 'Standard FCNet'}")
    print(f"{'#'*60}\n")

    t0 = time.time()
    history = solver.solve(problem, epochs=epochs, log_every=log_every)
    elapsed = time.time() - t0
    print(f"\n  Total training time: {elapsed:.1f}s")

    print_ghia_errors(solver, problem)
    plot_results(solver, problem, history,
                 save_path='ns_cavity_benchmark.png')


if __name__ == '__main__':
    main()
