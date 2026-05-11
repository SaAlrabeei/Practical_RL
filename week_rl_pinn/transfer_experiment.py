"""
Parametric Family Transfer Experiment
======================================
Core claim: An RL agent pre-trained on a family of Burgers equations
(varying viscosity ν) learns a transferable collocation strategy.
On unseen test scenarios (new ν values), it converges faster than
PACMANN — which always starts from scratch.

Metric: PINN gradient steps to reach L2 < threshold (convergence speed).
This mirrors the industrial use case: thousands of similar scenarios
where per-scenario training cost must be minimised.

Train family : ν ∈ {0.005/π, 0.01/π, 0.02/π, 0.05/π}  (4 values)
Test family  : ν ∈ {0.007/π, 0.015/π, 0.03/π}           (3 unseen values)

Usage:
  python transfer_experiment.py --quick          # smoke test (~5 min)
  python transfer_experiment.py                  # full run (~4-8 h)
  python transfer_experiment.py --skip-pretrain  # reuse saved agent
"""

import os
import copy
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Import shared components
import sys
sys.path.insert(0, os.path.dirname(__file__))

from benchmarks_vs_pacmann import (
    Burgers1D,
    UniformSpaceTime, RARSpaceTime, PACMANNCollocation,
)
from ppo_rl_collocation import (
    PPOAgentSpaceTime, RLSpaceTimePPO,
    pretrain_rl_agent, run_ppo_experiment,
)

# ── Colour map ────────────────────────────────────────────────
COLORS = {
    'Uniform':       '#e74c3c',
    'RAR':           '#f39c12',
    'PACMANN':       '#3498db',
    'RL-pretrained': '#27ae60',
}

# ── Hyperparameters ───────────────────────────────────────────
TRAIN_NUS = [
    0.005 / np.pi,
    0.01  / np.pi,
    0.02  / np.pi,
    0.05  / np.pi,
]
TEST_NUS = [
    0.007 / np.pi,
    0.015 / np.pi,
    0.03  / np.pi,
]

N_TOTAL   = 2500
N_REPLACE = 100
L2_THRESHOLD = 0.10      # convergence target

# Pre-train
PRETRAIN_EPISODES     = 3    # per ν variant
PRETRAIN_STEPS        = 15   # adapt steps per episode
PRETRAIN_EPOCHS       = 300  # PINN epochs per step

# Evaluation
EVAL_STEPS  = 20
EVAL_EPOCHS = 500
EVAL_SEEDS  = 3

AGENT_PATH = os.path.join(os.path.dirname(__file__),
                           'ppo_pretrained_burgers.pt')


# ═══════════════════════════════════════════════════════════
# PRE-TRAINING
# ═══════════════════════════════════════════════════════════

def pretrain_agent(n_episodes=PRETRAIN_EPISODES,
                   n_steps=PRETRAIN_STEPS,
                   epochs=PRETRAIN_EPOCHS,
                   agent_path=AGENT_PATH,
                   verbose=True) -> PPOAgentSpaceTime:
    """Load cached agent or pre-train from scratch."""
    agent = PPOAgentSpaceTime(G=16)

    if os.path.exists(agent_path):
        print(f"\n  Cached agent found: {agent_path}")
        agent.load(agent_path)
        return agent

    # lambda nu=nu avoids the Python closure capture trap
    factories = [lambda nu=nu: Burgers1D(nu=nu) for nu in TRAIN_NUS]

    total = n_episodes * len(factories)
    print(f"\n{'#'*60}")
    print(f"#  PRE-TRAINING  ({total} episodes, "
          f"{n_steps} steps × {epochs} epochs each)")
    print(f"#  ν ∈ {[f'{nu:.5f}' for nu in TRAIN_NUS]}")
    print(f"{'#'*60}")

    pretrain_rl_agent(
        agent, factories,
        n_episodes=n_episodes,
        n_steps=n_steps,
        epochs_per_step=epochs,
        n_total=N_TOTAL,
        n_replace=N_REPLACE,
        save_path=agent_path,
        verbose=verbose,
    )
    return agent


# ═══════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════

def run_one_eval(pde, seed: int,
                 pretrained_agent: PPOAgentSpaceTime,
                 n_adapt_steps: int = EVAL_STEPS,
                 epochs_per_step: int = EVAL_EPOCHS,
                 verbose: bool = True) -> dict:
    """
    Run all 4 methods on one PDE instance with one seed.
    RL uses a deep-copied frozen agent (zero-shot transfer).
    Returns {method_name: result_dict}.
    """
    xr, tr = pde.x_range, pde.t_range

    # Deep-copy to avoid any shared mutable state across seeds
    frozen = copy.deepcopy(pretrained_agent)
    for p in frozen.parameters():
        p.requires_grad_(False)

    strategies = {
        'Uniform': UniformSpaceTime(N_TOTAL, xr, tr),
        'RAR':     RARSpaceTime(N_TOTAL, n_replace=N_REPLACE,
                                x_range=xr, t_range=tr,
                                equal_budget=True),
        'PACMANN': PACMANNCollocation(N_TOTAL, n_steps=15, lr=1e-5,
                                      x_range=xr, t_range=tr),
        'RL-pretrained': RLSpaceTimePPO(N_TOTAL, n_add=N_REPLACE, G=16,
                                        x_range=xr, t_range=tr,
                                        agent=frozen, frozen=True),
    }

    kw = dict(n_adapt_steps=n_adapt_steps,
              epochs_per_step=epochs_per_step,
              n_total=N_TOTAL, seed=seed, verbose=verbose)

    results = {}
    for name, strat in strategies.items():
        results[name] = run_ppo_experiment(name, strat, pde, **kw)
    return results


def aggregate_seeds(per_seed: list, epochs_per_step: int) -> dict:
    """
    Aggregate a list of run_one_eval outputs (one per seed).
    Returns {method: {l2_mean, l2_std, grad_steps, n_seeds}}.
    X-axis is cumulative PINN gradient steps, not collocation budget.
    """
    methods = list(per_seed[0].keys())
    agg = {}
    for m in methods:
        curves = np.array([r[m]['l2'] for r in per_seed])   # (seeds, steps+1)
        n_pts  = curves.shape[1]
        grad_steps = np.array([(s + 1) * epochs_per_step
                                for s in range(n_pts)])
        agg[m] = {
            'l2_mean':   curves.mean(axis=0),
            'l2_std':    curves.std(axis=0),
            'grad_steps': grad_steps,
            'n_seeds':   len(per_seed),
            'all_curves': curves,
        }
    return agg


def steps_to_threshold(per_seed: list, method: str,
                        threshold: float,
                        epochs_per_step: int) -> np.ndarray:
    """
    For each seed, return the gradient-step count at which L2 first
    drops below threshold.  Returns max possible if never reached.
    """
    results = []
    max_steps = None
    for r in per_seed:
        l2_hist = r[method]['l2']
        if max_steps is None:
            max_steps = len(l2_hist) * epochs_per_step
        crossed = next((i for i, v in enumerate(l2_hist) if v < threshold),
                       None)
        if crossed is None:
            results.append(np.inf)      # never reached — distinct from max
        else:
            results.append((crossed + 1) * epochs_per_step)
    return np.array(results, dtype=float)


# ═══════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════

def plot_convergence(all_agg: dict, test_nus: list,
                     threshold: float, save_path: str):
    """
    3-panel figure: one panel per test ν.
    X: cumulative PINN gradient steps.
    Y: L2 relative error (log scale).
    Dashed line at threshold.
    """
    n_panels = len(test_nus)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5),
                              sharey=True)
    if n_panels == 1:
        axes = [axes]

    methods = list(next(iter(all_agg.values())).keys())

    for col, (nu, agg) in enumerate(zip(test_nus, all_agg.values())):
        ax = axes[col]
        for m in methods:
            c   = COLORS.get(m, '#7f8c8d')
            gs  = agg[m]['grad_steps']
            mu  = agg[m]['l2_mean']
            sd  = agg[m]['l2_std']
            lw  = 2.5 if m == 'RL-pretrained' else 1.8
            ls  = '-'
            ax.semilogy(gs, mu, color=c, lw=lw, ls=ls,
                        label=m if col == 0 else '_nolegend_')
            ax.fill_between(gs,
                            np.maximum(mu - sd, 1e-5),
                            mu + sd,
                            color=c, alpha=0.15)

        ax.axhline(threshold, ls='--', color='#555', lw=1.2, alpha=0.7)
        ax.text(gs[0] * 1.02, threshold * 1.1,
                f'τ={threshold}', fontsize=8, color='#555')

        nu_pi = nu * np.pi
        ax.set_title(f'ν = {nu_pi:.3f}/π  (unseen)', fontsize=11)
        ax.set_xlabel('PINN gradient steps')
        if col == 0:
            ax.set_ylabel('L2 relative error')
        ax.grid(alpha=0.25)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(
            lambda x, _: f'{int(x/1000)}k' if x >= 1000 else str(int(x))))

    axes[0].legend(loc='upper right', fontsize=9)
    fig.suptitle('Convergence speed: pre-trained RL vs from-scratch methods\n'
                 'Burgers equation — unseen viscosity values',
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved {save_path}")
    plt.close()


def plot_speedup_table(speedup_data: dict, test_nus: list,
                       threshold: float, max_steps: int,
                       save_path: str):
    """
    Matplotlib table: rows=methods, cols=test ν values.
    Shows grad-steps to threshold + speedup vs PACMANN.
    """
    methods  = list(next(iter(speedup_data.values())).keys())
    nu_labels = [f'ν={nu*np.pi:.3f}/π' for nu in test_nus]

    # Build cell text
    def fmt(v):
        return f'never' if np.isinf(v) else f'{int(v):,}'

    rows = []
    for m in methods:
        row = [m]
        for nu_label in nu_labels:
            med = np.median(speedup_data[nu_label][m])
            row.append(fmt(med))
        rows.append(row)

    # Speedup row
    speedup_row = ['Speedup\nvs PACMANN']
    for nu_label in nu_labels:
        p_med = np.median(speedup_data[nu_label]['PACMANN'])
        r_med = np.median(speedup_data[nu_label]['RL-pretrained'])
        if r_med >= max_steps:
            speedup_row.append('—')
        else:
            speedup_row.append(f'{p_med / r_med:.2f}×')
    rows.append(speedup_row)

    col_labels = ['Method'] + nu_labels

    fig, ax = plt.subplots(figsize=(4 + 2.5 * len(test_nus), 2.8))
    ax.axis('off')

    cell_colors = []
    for i, row in enumerate(rows):
        r_cols = []
        for j, val in enumerate(row):
            if j == 0:
                r_cols.append('#f0f0f0')
            elif i == len(rows) - 1:          # speedup row
                try:
                    v = float(val.replace('×', ''))
                    r_cols.append('#d5f5e3' if v >= 1.0 else '#fadbd8')
                except Exception:
                    r_cols.append('#f0f0f0')
            elif row[0] == 'RL-pretrained':
                r_cols.append('#d5f5e3')
            else:
                r_cols.append('white')
        cell_colors.append(r_cols)

    tbl = ax.table(cellText=rows, colLabels=col_labels,
                   cellColours=cell_colors,
                   cellLoc='center', loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 2.0)

    fig.suptitle(f'Gradient steps to reach L2 < {threshold}  '
                 f'(median over {list(speedup_data.values())[0]["RL-pretrained"].size} seeds)',
                 fontsize=11, y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved {save_path}")
    plt.close()


def print_latex_table(speedup_data: dict, test_nus: list,
                       threshold: float, max_steps: int):
    """Print a LaTeX tabular for direct use in the paper."""
    nu_labels = [f'ν={nu*np.pi:.3f}/π' for nu in test_nus]
    methods   = list(next(iter(speedup_data.values())).keys())

    def fmt(v, bold=False):
        s = f'>{max_steps}' if v >= max_steps else f'{int(v):,}'
        return f'\\textbf{{{s}}}' if bold else s

    print('\n% ── LaTeX table ──────────────────────────────────────')
    print('\\begin{tabular}{l' + 'r' * len(test_nus) + '}')
    print('\\toprule')
    header = 'Method & ' + ' & '.join(nu_labels) + ' \\\\'
    print(header)
    print('\\midrule')

    for m in methods:
        row_vals = []
        for nu_label in nu_labels:
            med = np.median(speedup_data[nu_label][m])
            bold = (m == 'RL-pretrained')
            row_vals.append(fmt(med, bold))
        print(f'{m} & ' + ' & '.join(row_vals) + ' \\\\')

    print('\\midrule')
    speedup_vals = []
    for nu_label in nu_labels:
        p = np.median(speedup_data[nu_label]['PACMANN'])
        r = np.median(speedup_data[nu_label]['RL-pretrained'])
        speedup_vals.append('—' if np.isinf(r) or np.isinf(p)
                             else f'\\textbf{{{p/r:.2f}\\times}}')
    print('Speedup vs PACMANN & ' + ' & '.join(speedup_vals) + ' \\\\')
    print('\\bottomrule')
    print('\\end{tabular}')
    print(f'% threshold τ = {threshold}; median over seeds')
    print('% ─────────────────────────────────────────────────────\n')


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true',
                        help='Smoke test: short pretrain + 6-step eval')
    parser.add_argument('--skip-pretrain', action='store_true',
                        help='Load cached agent, skip pre-training')
    parser.add_argument('--seeds',  type=int, default=EVAL_SEEDS)
    parser.add_argument('--agent',  default=AGENT_PATH)
    parser.add_argument('--threshold', type=float, default=L2_THRESHOLD)
    args = parser.parse_args()

    if args.quick:
        pre_kw  = dict(n_episodes=1, n_steps=5, epochs=100)
        eval_kw = dict(n_adapt_steps=6, epochs_per_step=200)
        test_nus = TEST_NUS[:2]          # only 2 panels for speed
    else:
        pre_kw  = dict(n_episodes=PRETRAIN_EPISODES,
                       n_steps=PRETRAIN_STEPS,
                       epochs=PRETRAIN_EPOCHS)
        eval_kw = dict(n_adapt_steps=EVAL_STEPS,
                       epochs_per_step=EVAL_EPOCHS)
        test_nus = TEST_NUS

    seeds = list(range(args.seeds))
    max_steps = (eval_kw['n_adapt_steps'] + 1) * eval_kw['epochs_per_step']

    # ── 1. Pre-train ─────────────────────────────────────
    if args.skip_pretrain:
        agent = PPOAgentSpaceTime(G=16)
        agent.load(args.agent)
    else:
        agent = pretrain_agent(agent_path=args.agent, **pre_kw)

    # ── 2. Evaluate on each test ν ────────────────────────
    # Build PDE objects once — reused across seeds to save reference solves
    test_pdes = [Burgers1D(nu=nu) for nu in test_nus]

    all_per_seed  = {}   # {nu_idx: [seed0_results, seed1_results, ...]}
    all_agg       = {}   # {nu_idx: aggregated}
    speedup_data  = {}   # {nu_label: {method: steps_array}}

    for ni, (nu, pde) in enumerate(zip(test_nus, test_pdes)):
        nu_label = f'ν={nu*np.pi:.3f}/π'
        print(f"\n{'#'*60}")
        print(f"#  TEST  {nu_label}  ({ni+1}/{len(test_nus)})")
        print(f"{'#'*60}")

        per_seed = []
        for si, seed in enumerate(seeds):
            print(f"\n  ── seed {seed}  ({si+1}/{len(seeds)}) ──")
            result = run_one_eval(pde, seed, agent,
                                  verbose=True, **eval_kw)
            per_seed.append(result)

        all_per_seed[ni] = per_seed
        all_agg[ni]      = aggregate_seeds(per_seed,
                                            eval_kw['epochs_per_step'])

        speedup_data[nu_label] = {
            m: steps_to_threshold(per_seed, m, args.threshold,
                                   eval_kw['epochs_per_step'])
            for m in per_seed[0].keys()
        }

    # ── 3. Print per-ν summary ────────────────────────────
    methods = list(all_per_seed[0][0].keys())
    print(f"\n{'='*68}")
    print(f"  Gradient steps to L2 < {args.threshold}  (median ± std, {len(seeds)} seeds)")
    print(f"{'='*68}")
    print(f"{'Method':<18}", end='')
    for nu in test_nus:
        print(f"  ν={nu*np.pi:.3f}/π", end='')
    print()
    print('-' * 68)
    for m in methods:
        print(f"{m:<18}", end='')
        for ni, nu in enumerate(test_nus):
            nu_label = f'ν={nu*np.pi:.3f}/π'
            arr = speedup_data[nu_label][m]
            med, std = np.median(arr), arr.std()
            tag = 'never' if np.isinf(med) else f'{int(med):5,}'
            print(f"  {tag:>8} ±{std:4.0f}", end='')
        print()
    print('='*68)

    # Speedup row
    print(f"{'Speedup(RL/PAC)':<18}", end='')
    for nu in test_nus:
        nu_label = f'ν={nu*np.pi:.3f}/π'
        p = np.median(speedup_data[nu_label]['PACMANN'])
        r = np.median(speedup_data[nu_label]['RL-pretrained'])
        sp = '  —' if (np.isinf(r) or np.isinf(p)) else f'  {p/r:6.2f}×'
        print(f"  {sp:>12}", end='')
    print('\n')

    # ── 4. Plots ──────────────────────────────────────────
    plot_convergence(
        {i: all_agg[i] for i in range(len(test_nus))},
        test_nus, args.threshold,
        save_path='burgers_transfer_convergence.png',
    )
    plot_speedup_table(
        speedup_data, test_nus, args.threshold, max_steps,
        save_path='burgers_transfer_speedup.png',
    )
    print_latex_table(speedup_data, test_nus, args.threshold, max_steps)

    # ── 5. Save raw results ───────────────────────────────
    np.savez('transfer_results.npz',
             test_nus=np.array(test_nus),
             train_nus=np.array(TRAIN_NUS),
             threshold=args.threshold,
             **{f'l2_mean_{ni}_{m.replace("-","_").replace("(","").replace(")","")}'
                : all_agg[ni][m]['l2_mean']
                for ni in range(len(test_nus)) for m in methods},
             **{f'grad_steps_{ni}': all_agg[ni][methods[0]]['grad_steps']
                for ni in range(len(test_nus))})
    print("Saved transfer_results.npz")


if __name__ == '__main__':
    main()
