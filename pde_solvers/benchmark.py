"""
Benchmark runner — compare all registered solvers on all problems.

Usage:
    python -m pde_solvers.benchmark          # full run
    python -m pde_solvers.benchmark --quick  # fewer epochs, fast check

Output:
    benchmark_results.png  — convergence + solution + bar charts
    (printed table to stdout)
"""

import argparse
import time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from .problems import SmoothPoisson, OscPoisson, BumpPoisson
from .solvers.collocation import CollocationPINN
from .solvers.vpinn import VPINN


# ── colour / style registry (extend as new solvers are added) ────
_STYLE = {
    'Collocation PINN': dict(color='#e74c3c', marker='o', ls='-'),
    'VPINN':            dict(color='#3498db', marker='s', ls='-'),
    'Deep Ritz':        dict(color='#2ecc71', marker='^', ls='-'),
}


# ════════════════════════════════════════════════════════════════
# Core runner
# ════════════════════════════════════════════════════════════════

def run_benchmark(problems: list, solvers: list,
                  epochs: int = 5000,
                  log_every: int = 500) -> dict:
    """
    Run every solver on every problem.
    Returns  results[problem.name][solver.name] = history dict.
    """
    all_results: dict = {}

    for problem in problems:
        print(f"\n{'═'*62}")
        print(f"  Problem : {problem.name}")
        print(f"{'═'*62}")
        all_results[problem.name] = {}

        for solver in solvers:
            print(f"\n  ▶  Solver : {solver.name}")
            t0      = time.time()
            history = solver.solve(problem, epochs=epochs, log_every=log_every)
            elapsed = time.time() - t0

            history['total_time'] = elapsed
            history['final_l2']   = history['l2_err'][-1]
            all_results[problem.name][solver.name] = history

            print(f"     ✓  done in {elapsed:.1f}s  |  final L2 = {history['final_l2']:.4e}")

    return all_results


# ════════════════════════════════════════════════════════════════
# Visualisation
# ════════════════════════════════════════════════════════════════

def plot_benchmark(all_results: dict, problems: list, solvers: list,
                   save_path: str = 'benchmark_results.png'):
    """
    Grid figure:
      Rows   = problems
      Col 0  = L2 convergence curve
      Col 1  = exact solution heatmap
      Col 2  = best-solver prediction heatmap
      Col 3  = point-wise |error| heatmap
      Col 4  = final L2 bar chart
    """
    n_rows = len(problems)
    fig    = plt.figure(figsize=(20, 4.5 * n_rows))
    gs     = gridspec.GridSpec(n_rows, 5, figure=fig,
                               hspace=0.50, wspace=0.38)

    for row, problem in enumerate(problems):
        results  = all_results[problem.name]
        x_test, u_test = problem.test_grid(n=60)
        n  = 60
        t  = np.linspace(problem.lb, problem.ub, n)
        XX, YY = np.meshgrid(t, t, indexing='ij')
        u_true_np = u_test.reshape(n, n).numpy()

        # ── col 0 : convergence ───────────────────────────────
        ax = fig.add_subplot(gs[row, 0])
        for solver in solvers:
            h  = results[solver.name]
            st = _STYLE.get(solver.name, dict(color='gray', marker='o', ls='-'))
            ax.semilogy(h['epoch'], h['l2_err'],
                        label=solver.name,
                        color=st['color'], marker=st['marker'],
                        markevery=max(1, len(h['epoch'])//8),
                        ms=5, lw=2, ls=st['ls'])
        ax.set_xlabel('Epoch');  ax.set_ylabel('Rel. L2 error')
        ax.set_title(f'{problem.name}\nConvergence', fontsize=9)
        ax.legend(fontsize=7);  ax.grid(True, which='both', ls='--', alpha=0.35)

        # Pick the best solver (lowest final L2) for solution plots
        best_solver = min(solvers,
                          key=lambda s: results[s.name]['final_l2'])
        u_pred_np   = best_solver.predict(x_test).reshape(n, n).numpy()
        err_np      = np.abs(u_pred_np - u_true_np)
        vmin, vmax  = u_true_np.min(), u_true_np.max()

        # ── col 1 : exact solution ────────────────────────────
        ax = fig.add_subplot(gs[row, 1])
        im = ax.contourf(XX, YY, u_true_np, levels=40, cmap='RdBu_r')
        fig.colorbar(im, ax=ax, shrink=0.85)
        ax.set_title('Exact $u$', fontsize=9)
        ax.set_xlabel('$x_1$');  ax.set_ylabel('$x_2$')

        # ── col 2 : best prediction ───────────────────────────
        ax = fig.add_subplot(gs[row, 2])
        im = ax.contourf(XX, YY, u_pred_np, levels=40, cmap='RdBu_r',
                         vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=ax, shrink=0.85)
        ax.set_title(f'{best_solver.name}\nprediction', fontsize=9)
        ax.set_xlabel('$x_1$');  ax.set_ylabel('$x_2$')

        # ── col 3 : point-wise error ──────────────────────────
        ax = fig.add_subplot(gs[row, 3])
        final_l2 = results[best_solver.name]['final_l2']
        im = ax.contourf(XX, YY, err_np, levels=40, cmap='hot_r')
        fig.colorbar(im, ax=ax, shrink=0.85)
        ax.set_title(f'|error|  (L2={final_l2:.2e})', fontsize=9)
        ax.set_xlabel('$x_1$');  ax.set_ylabel('$x_2$')

        # ── col 4 : final L2 bar chart ────────────────────────
        ax   = fig.add_subplot(gs[row, 4])
        names = [s.name for s in solvers]
        vals  = [results[n]['final_l2'] for n in names]
        cols  = [_STYLE.get(n, {}).get('color', 'gray') for n in names]
        bars  = ax.bar(names, vals, color=cols, edgecolor='white', width=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.08, f'{v:.2e}',
                    ha='center', va='bottom', fontsize=7)
        ax.set_ylabel('Final rel. L2');  ax.set_title('Accuracy', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=15, fontsize=7)

    fig.suptitle('PDE Solver Benchmark', fontsize=14, y=1.01)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved → {save_path}")


def print_table(all_results: dict, problems: list, solvers: list):
    """Print a compact summary table."""
    W = 62
    print(f"\n{'═'*W}")
    print(f"  {'Problem':<26} {'Solver':<20} {'L2':>8}  {'Time':>6}")
    print(f"{'─'*W}")
    for problem in problems:
        for solver in solvers:
            h = all_results[problem.name][solver.name]
            print(f"  {problem.name:<26} {solver.name:<20} "
                  f"{h['final_l2']:>8.4e}  {h['total_time']:>5.1f}s")
        print(f"{'─'*W}")
    print(f"{'═'*W}")

    # Speed-up table
    if len(solvers) == 2:
        baseline = solvers[0].name
        other    = solvers[1].name
        print(f"\n  Relative accuracy  ({other} L2) / ({baseline} L2)")
        print(f"  {'Problem':<30}  {'ratio':>8}")
        print(f"  {'─'*40}")
        for problem in problems:
            r0 = all_results[problem.name][baseline]['final_l2']
            r1 = all_results[problem.name][other]['final_l2']
            note = '↑ better' if r1 < r0 else '↓ worse'
            print(f"  {problem.name:<30}  {r1/r0:>8.3f}  {note}")


# ════════════════════════════════════════════════════════════════
# Entry point
# ════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='PDE solver benchmark')
    parser.add_argument('--quick', action='store_true',
                        help='Fast run: fewer epochs (500) for smoke-testing')
    args = parser.parse_args()

    epochs    = 500  if args.quick else 5000
    log_every = 100  if args.quick else 500

    problems = [
        SmoothPoisson(freq=np.pi),      # easy   — smooth, λ=π
        OscPoisson(),                    # medium — oscillatory, λ=4π
        BumpPoisson(k=20.0),             # hard   — sharp Gaussian peak
    ]

    solvers = [
        CollocationPINN(),               # strong form, pointwise residual
        VPINN(),                         # weak form, Galerkin test functions
    ]

    print(f"\n{'#'*62}")
    print(f"#  PDE Solver Benchmark   (epochs={epochs})")
    print(f"#  Problems : {[p.name for p in problems]}")
    print(f"#  Solvers  : {[s.name for s in solvers]}")
    print(f"{'#'*62}")

    all_results = run_benchmark(problems, solvers,
                                epochs=epochs, log_every=log_every)

    plot_benchmark(all_results, problems, solvers,
                   save_path='benchmark_results.png')
    print_table(all_results, problems, solvers)


if __name__ == '__main__':
    main()
