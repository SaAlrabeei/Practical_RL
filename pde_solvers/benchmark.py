"""
Benchmark runner  —  1-D Poisson comparison.

Runs all registered solvers on the three difficulty levels and produces:
  poisson1d_benchmark.png  — one row per problem, four panels each:
      (a) convergence curve        (b) solution comparison
      (c) point-wise |error|       (d) final L2 bar chart

Usage:
    python -m pde_solvers.benchmark            # full 5 000-epoch run
    python -m pde_solvers.benchmark --quick    # 500 epochs, smoke test
"""

import argparse
import time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from .problems import SmoothPoisson1D, LayerPoisson1D, OscPoisson1D
from .solvers.collocation import CollocationPINN
from .solvers.vpinn import VPINN

# ── colour registry (extend as new solvers are added) ────────────
_STYLE = {
    'Collocation PINN': dict(color='#e74c3c', marker='o', ls='-',  lw=2),
    'VPINN':            dict(color='#3498db', marker='s', ls='--', lw=2),
    'Deep Ritz':        dict(color='#2ecc71', marker='^', ls=':',  lw=2),
}


# ════════════════════════════════════════════════════════════════
# Core runner
# ════════════════════════════════════════════════════════════════

def run_benchmark(problems: list, solvers: list,
                  epochs: int = 5000, log_every: int = 500) -> dict:
    all_results: dict = {}
    for problem in problems:
        print(f"\n{'═'*60}\n  {problem.name}\n{'═'*60}")
        all_results[problem.name] = {}
        for solver in solvers:
            print(f"\n  ▶  {solver.name}")
            t0      = time.time()
            history = solver.solve(problem, epochs=epochs, log_every=log_every)
            elapsed = time.time() - t0
            history['total_time'] = elapsed
            history['final_l2']   = history['l2_err'][-1]
            all_results[problem.name][solver.name] = history
            print(f"     ✓ {elapsed:.1f}s  |  final L2 = {history['final_l2']:.4e}")
    return all_results


# ════════════════════════════════════════════════════════════════
# Plotting  (1-D layout)
# ════════════════════════════════════════════════════════════════

def plot_1d_benchmark(all_results: dict, problems: list, solvers: list,
                      save_path: str = 'poisson1d_benchmark.png'):
    """
    Grid: rows = problems,  columns = [convergence | solution | |error| | bar]
    """
    n_rows = len(problems)
    fig    = plt.figure(figsize=(18, 4.2 * n_rows))
    gs     = gridspec.GridSpec(n_rows, 4, figure=fig,
                               hspace=0.55, wspace=0.38)

    for row, problem in enumerate(problems):
        results        = all_results[problem.name]
        x_test, u_true = problem.test_grid(n=400)
        x_np           = x_test[:, 0].numpy()
        u_np           = u_true[:, 0].numpy()

        # ── (a) convergence ───────────────────────────────────────
        ax = fig.add_subplot(gs[row, 0])
        for solver in solvers:
            h  = results[solver.name]
            st = _STYLE.get(solver.name, {})
            ax.semilogy(h['epoch'], h['l2_err'],
                        label=solver.name,
                        color=st.get('color','gray'),
                        ls=st.get('ls','-'), lw=st.get('lw',1.5),
                        marker=st.get('marker','o'),
                        markevery=max(1, len(h['epoch'])//6), ms=4)
        ax.set_xlabel('Epoch');  ax.set_ylabel('Rel. L2 error')
        ax.set_title(f'{problem.name}\nConvergence', fontsize=9)
        ax.legend(fontsize=7);  ax.grid(True, which='both', ls='--', alpha=0.35)

        # ── (b) solution comparison ───────────────────────────────
        ax = fig.add_subplot(gs[row, 1])
        ax.plot(x_np, u_np, 'k-', lw=2, label='Exact', zorder=5)
        for solver in solvers:
            st    = _STYLE.get(solver.name, {})
            u_pred = solver.predict(x_test)[:, 0].numpy()
            ax.plot(x_np, u_pred,
                    color=st.get('color','gray'),
                    ls=st.get('ls','--'), lw=1.5,
                    label=solver.name)
        ax.set_xlabel('x');  ax.set_ylabel('u(x)')
        ax.set_title('Solution', fontsize=9)
        ax.legend(fontsize=7);  ax.grid(True, ls='--', alpha=0.35)

        # ── (c) point-wise |error| ────────────────────────────────
        ax = fig.add_subplot(gs[row, 2])
        for solver in solvers:
            st     = _STYLE.get(solver.name, {})
            u_pred = solver.predict(x_test)[:, 0].numpy()
            err    = np.abs(u_pred - u_np)
            ax.semilogy(x_np, err + 1e-16,
                        color=st.get('color','gray'),
                        ls=st.get('ls','-'), lw=1.5,
                        label=solver.name)
        ax.set_xlabel('x');  ax.set_ylabel('|error|  (log)')
        ax.set_title('Point-wise error', fontsize=9)
        ax.legend(fontsize=7);  ax.grid(True, which='both', ls='--', alpha=0.35)

        # ── (d) final L2 bar chart ────────────────────────────────
        ax    = fig.add_subplot(gs[row, 3])
        names = [s.name for s in solvers]
        vals  = [results[n]['final_l2'] for n in names]
        times = [results[n]['total_time'] for n in names]
        cols  = [_STYLE.get(n,{}).get('color','gray') for n in names]
        bars  = ax.bar(names, vals, color=cols, edgecolor='white', width=0.5)
        for bar, v, t in zip(bars, vals, times):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() * 1.12,
                    f'{v:.2e}\n({t:.0f}s)',
                    ha='center', va='bottom', fontsize=7)
        ax.set_yscale('log')
        ax.set_ylabel('Final rel. L2');  ax.set_title('Accuracy', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=20, ha='right', fontsize=7)

    fig.suptitle('1-D Poisson Benchmark  —  Collocation PINN vs VPINN',
                 fontsize=13, y=1.01)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved → {save_path}")


# ════════════════════════════════════════════════════════════════
# Summary table
# ════════════════════════════════════════════════════════════════

def print_table(all_results: dict, problems: list, solvers: list):
    W = 70
    print(f"\n{'═'*W}")
    print(f"  {'Problem':<36} {'Solver':<20} {'L2':>8}  {'Time':>6}")
    print(f"{'─'*W}")
    for problem in problems:
        for solver in solvers:
            h = all_results[problem.name][solver.name]
            print(f"  {problem.name:<36} {solver.name:<20}"
                  f" {h['final_l2']:>8.4e}  {h['total_time']:>5.1f}s")
        print(f"{'─'*W}")
    print(f"{'═'*W}")

    # Per-problem winner
    print(f"\n  {'Problem':<36}  Winner            L2 ratio")
    print(f"  {'─'*60}")
    for problem in problems:
        l2s = {s.name: all_results[problem.name][s.name]['final_l2']
               for s in solvers}
        winner = min(l2s, key=l2s.get)
        loser  = max(l2s, key=l2s.get)
        ratio  = l2s[loser] / l2s[winner]
        print(f"  {problem.name:<36}  {winner:<18}  {ratio:.1f}× better")


# ════════════════════════════════════════════════════════════════
# Entry point
# ════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true',
                        help='500 epochs — fast smoke test')
    args = parser.parse_args()

    epochs    = 500  if args.quick else 5000
    log_every = 100  if args.quick else 500

    problems = [
        SmoothPoisson1D(),          # Level 1 — smooth
        LayerPoisson1D(k=15),       # Level 2 — interior layer
        OscPoisson1D(n=8),          # Level 3 — oscillatory
    ]

    solvers = [
        CollocationPINN(),
        VPINN(),
    ]

    print(f"\n{'#'*60}")
    print(f"#  1-D Poisson Benchmark   epochs={epochs}")
    print(f"#  Problems : {[p.name for p in problems]}")
    print(f"#  Solvers  : {[s.name for s in solvers]}")
    print(f"{'#'*60}")

    results = run_benchmark(problems, solvers, epochs=epochs, log_every=log_every)
    plot_1d_benchmark(results, problems, solvers, save_path='poisson1d_benchmark.png')
    print_table(results, problems, solvers)


if __name__ == '__main__':
    main()
