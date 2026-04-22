"""
Empirical timing analysis of LSM+CV-LLH computational complexity.

Usage (from project root):
    cd src && python timing_analysis.py

Produces:
    figs/fig_timing_combined.png    - 1x2: stage breakdown + cost-precision scatter
    figs/fig_scaling_all.png        - 1x3: wall time vs N, M, P
"""

import sys, os, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import priceModels as pm
import amerPrice as ap

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

STYLE = {
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 1.5,
    'lines.markersize': 5,
}

# Table 1 parameters
PARAMS = dict(r=0.01, kappa=5, nu=0.2, lam=0.9, eta=0.01, rho=-0.2,
              sigma0=0.15, theta0=0.18)
S0, K_STRIKE, T = 100.0, 100.0, 1.0
N_DEFAULT, M_DEFAULT, P_DEFAULT, K_RK4 = 10_000, 52, 513, 256
SEED = 42

# theta-driver used by every simulate_prices call in this module.
# 'bm'  matches the LLH formula's PDE derivation (priceModels default)
#       and the constant used in generate_plots.py.
# 'gbm' matches the paper's original simulation.
THETA_DRIVER = 'bm'


def _savefig(fig, name):
    """Save figure to the output directory and close it."""
    fig.savefig(os.path.join(OUTPUT_DIR, name))
    plt.close(fig)
    print(f"  Saved {name}")


def _time_stages(n_paths, n_steps_mc, n_phi, n_steps_rk4, seed=SEED):
    """Time each stage: simulation, shared European precomp, plain backward, CV-LLH backward."""
    model = pm.ImprovedSteinStein(**PARAMS, seed=seed)
    ode_kw = dict(phi_max=300.0, n_phi=n_phi, n_steps_rk4=n_steps_rk4)

    # 1. Simulation
    t0 = time.perf_counter()
    sim = model.simulate_prices(S0=S0, T=T, n_steps_mc=n_steps_mc, n_paths=n_paths,
                                theta_driver=THETA_DRIVER)
    t_sim = time.perf_counter() - t0

    # 2. Shared European precomputation (ODE + European prices)
    t0 = time.perf_counter()
    pre = ap.precompute_european(model, sim, K=K_STRIKE, **ode_kw)
    t_precomp = time.perf_counter() - t0

    # 3. Plain LSM backward (with precomputed European grid)
    t0 = time.perf_counter()
    res_plain = ap.price_american_put_lsm_llh(
        model, sim, K=K_STRIKE, use_cv=False, precomputed=pre)
    t_plain_backward = time.perf_counter() - t0

    # 4. CV-LLH backward (with precomputed European grid)
    t0 = time.perf_counter()
    result = ap.price_american_put_lsm_llh(
        model, sim, K=K_STRIKE, use_cv=True, euro_method='llh',
        precomputed=pre)
    t_llh_backward = time.perf_counter() - t0

    return {
        'Simulation': t_sim,
        'Precompute': t_precomp,
        'Plain_backward': t_plain_backward,
        'LLH_backward': t_llh_backward,
        'Plain_total': t_sim + t_precomp + t_plain_backward,
        'LLH_total': t_sim + t_precomp + t_llh_backward,
        'Total': t_sim + t_precomp + t_plain_backward + t_llh_backward,
        'price_imp': result.get('price_imp', result['price']),
        'std_err_imp': result.get('std_err_imp', result['std_err']),
    }


def _time_method(method, n_paths, n_steps_mc, seed=SEED, **kw):
    """Time a single pricing method end-to-end (sim + precomp + pricer)."""
    model = pm.ImprovedSteinStein(**PARAMS, seed=seed)
    sim = model.simulate_prices(S0=S0, T=T, n_steps_mc=n_steps_mc, n_paths=n_paths,
                                theta_driver=THETA_DRIVER)
    ode_kw = dict(phi_max=300.0, n_phi=P_DEFAULT, n_steps_rk4=K_RK4)

    t0 = time.perf_counter()
    if method == 'plain':
        pre = ap.precompute_european(model, sim, K=K_STRIKE, **ode_kw)
        res = ap.price_american_put_lsm_llh(model, sim, K=K_STRIKE,
                                              use_cv=False, precomputed=pre)
    elif method == 'bs':
        res = ap.price_american_put_lsm_llh(model, sim, K=K_STRIKE,
                                              use_cv=True, euro_method='bs')
    elif method == 'llh':
        pre = ap.precompute_european(model, sim, K=K_STRIKE, **ode_kw)
        res = ap.price_american_put_lsm_llh(model, sim, K=K_STRIKE,
                                              use_cv=True, euro_method='llh',
                                              precomputed=pre)
    return time.perf_counter() - t0, res


# ===================================================================
# Plot 1: Stage breakdown (horizontal bar chart)
# ===================================================================

def plot_timing_breakdown():
    """Single-panel figure: stage breakdown horizontal bar chart."""
    s = _time_stages(N_DEFAULT, M_DEFAULT, P_DEFAULT, K_RK4)

    fig, ax = plt.subplots(figsize=(8, 3.5))

    labels = [
        'Simulation',
        'European precomp\n(shared)',
        'Plain LSM\nBackward loop',
        'CV-LLH\nBackward loop',
    ]
    sizes = [
        s['Simulation'], s['Precompute'],
        s['Plain_backward'], s['LLH_backward'],
    ]
    colors = ['#1f77b4', '#ff7f0e', '#d62728', '#d62728']

    y = np.arange(4, dtype=float)
    bars = ax.barh(y, sizes, color=colors, alpha=0.85, height=0.5)

    max_val = max(sizes)
    for bar, val in zip(bars, sizes):
        ax.text(bar.get_width() + max_val * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f'{val:.2f}s', va='center', fontsize=9)

    ax.set_yticks(np.arange(4))
    ax.set_yticklabels(labels)
    ax.set_xlabel('Computational time (s)')
    ax.set_xlim(0, max_val * 1.3)
    ax.invert_yaxis()

    fig.suptitle(
        f'Computational cost breakdown: Plain LSM vs LSM+CV-LLH\n'
        f'$N={N_DEFAULT:,}$, $M={M_DEFAULT}$, $P={P_DEFAULT}$',
        fontsize=13, y=1.02)
    fig.tight_layout()
    _savefig(fig, 'fig_timing_breakdown.png')


def plot_timing_combined():
    """1x2 figure: (left) stage breakdown bar chart, (right) cost-precision scatter."""
    s = _time_stages(N_DEFAULT, M_DEFAULT, P_DEFAULT, K_RK4)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- Left panel: stage breakdown (shared precomp + per-method backward) ---
    labels = [
        'Simulation',
        'European precomp\n(shared)',
        'Plain LSM\nBackward loop',
        'CV-LLH\nBackward loop',
    ]
    sizes = [
        s['Simulation'], s['Precompute'],
        s['Plain_backward'], s['LLH_backward'],
    ]
    colors = ['#1f77b4', '#ff7f0e', '#d62728', '#d62728']

    y = np.arange(4, dtype=float)
    bars = ax1.barh(y, sizes, color=colors, alpha=0.85, height=0.5)

    max_val = max(sizes)
    for bar, val in zip(bars, sizes):
        ax1.text(bar.get_width() + max_val * 0.01,
                 bar.get_y() + bar.get_height() / 2,
                 f'{val:.2f}s', va='center', fontsize=9)

    ax1.set_yticks(np.arange(4))
    ax1.set_yticklabels(labels)
    ax1.set_xlabel('Computational time (s)')
    ax1.set_xlim(0, max_val * 1.3)
    ax1.invert_yaxis()
    ax1.set_title('Computational cost breakdown')

    # --- Right panel: cost-precision scatter ---
    methods = [('plain', 'Plain LSM', N_DEFAULT),
               ('llh',   'CV-LLH',    N_DEFAULT)]
    mcolors = {'Plain LSM': '#1f77b4', 'CV-LLH': '#d62728'}
    markers = {'Plain LSM': 'o', 'CV-LLH': '^'}

    for method, label, n in methods:
        kw = dict(phi_max=300.0, n_phi=P_DEFAULT, n_steps_rk4=K_RK4) if method == 'llh' else {}
        elapsed, res = _time_method(method, n, M_DEFAULT, **kw)
        se = res.get('std_err_imp', res['std_err'])
        print(f"  {label}: {elapsed:.3f}s, SE={se:.6f}, N={n}")

        ax2.scatter(elapsed, se, s=120, marker=markers[label],
                    color=mcolors[label], zorder=5, label=label)
        ax2.annotate(f"{label}\nSE={se:.4f}\n{elapsed:.2f}s",
                     xy=(elapsed, se),
                     xytext=(15, 10), textcoords='offset points',
                     fontsize=9, color=mcolors[label],
                     arrowprops=dict(arrowstyle='->', color=mcolors[label], lw=0.8))

    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Computational time (s)')
    ax2.set_ylabel('Standard error')
    ax2.set_title('Cost--precision tradeoff')
    ax2.legend(fontsize=10)

    fig.suptitle(
        f'Plain LSM vs LSM+CV-LLH ($N={N_DEFAULT:,}$, $M={M_DEFAULT}$, '
        f'$S_0={S0:.0f}$, $K={K_STRIKE:.0f}$, $T={T}$, Table~1)',
        fontsize=13, y=1.03)
    fig.tight_layout()
    _savefig(fig, 'fig_timing_combined.png')


# ===================================================================
# Plots 2-4: Scaling experiments
# ===================================================================

def _scaling_plot(param_name, param_sym, values, fixed_label,
                  color, marker, fname, vary_fn):
    """Generate a scaling plot: vary one parameter and measure total wall time."""
    times = []
    for v in values:
        s = vary_fn(v)
        times.append(s['LLH_total'])
        print(f"  {param_name}={v}: {s['LLH_total']:.3f}s")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(values, times, f'{marker}-', color=color)
    ax.set_xlabel(param_sym)
    ax.set_ylabel('Computational time (s)')
    fig.suptitle(
        f'LSM+CV-LLH computational scaling with {param_sym}\n{fixed_label}',
        fontsize=13, y=1.02)
    fig.tight_layout()
    _savefig(fig, fname)


def plot_scaling_N():
    """Plot wall-time scaling with number of MC paths N."""
    _scaling_plot('N', '$N$ (paths)', [1000, 2000, 5000, 10000],
                  f'$M={M_DEFAULT}$, $P={P_DEFAULT}$',
                  '#d62728', 'o', 'fig_scaling_N.png',
                  lambda n: _time_stages(n, M_DEFAULT, P_DEFAULT, K_RK4))


def plot_scaling_M():
    """Plot wall-time scaling with number of exercise dates M."""
    _scaling_plot('M', '$M$ (exercise dates)', [12, 22, 52, 104, 252],
                  f'$N={N_DEFAULT:,}$, $P={P_DEFAULT}$',
                  '#1f77b4', 's', 'fig_scaling_M.png',
                  lambda m: _time_stages(N_DEFAULT, m, P_DEFAULT, K_RK4))


def plot_scaling_P():
    """Plot wall-time scaling with number of quadrature nodes P."""
    _scaling_plot('P', '$P$ (quadrature nodes)', [65, 129, 257, 513, 1025],
                  f'$N={N_DEFAULT:,}$, $M={M_DEFAULT}$',
                  '#2ca02c', '^', 'fig_scaling_P.png',
                  lambda p: _time_stages(N_DEFAULT, M_DEFAULT, p, K_RK4))


# ===================================================================
# Plot 5: Method comparison (cost vs precision)
# ===================================================================

# ===================================================================
# Scaling experiments (1x3 horizontal)
# ===================================================================

def plot_scaling_all():
    """Generate combined 1x3 figure showing wall-time scaling with N, M, and P."""
    configs = [
        ('N', '$N$ (paths)', [1000, 2000, 5000, 10000],
         f'$M={M_DEFAULT}$, $P={P_DEFAULT}$', '#d62728', 'o',
         lambda v: _time_stages(v, M_DEFAULT, P_DEFAULT, K_RK4)),
        ('M', '$M$ (exercise dates)', [12, 22, 52, 104, 252],
         f'$N={N_DEFAULT:,}$, $P={P_DEFAULT}$', '#1f77b4', 's',
         lambda v: _time_stages(N_DEFAULT, v, P_DEFAULT, K_RK4)),
        ('P', '$P$ (quadrature nodes)', [65, 129, 257, 513, 1025],
         f'$N={N_DEFAULT:,}$, $M={M_DEFAULT}$', '#2ca02c', '^',
         lambda v: _time_stages(N_DEFAULT, M_DEFAULT, v, K_RK4)),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    for ax, (name, sym, values, fixed, color, marker, vary_fn) in zip(axes, configs):
        times = []
        for v in values:
            s = vary_fn(v)
            times.append(s['LLH_total'])
            print(f"  {name}={v}: {s['LLH_total']:.3f}s")
        ax.plot(values, times, f'{marker}-', color=color)
        ax.set_xlabel(sym)
        ax.set_title(fixed, fontsize=10)

    axes[0].set_ylabel('Computational time (s)')
    fig.suptitle('LSM+CV-LLH computational scaling', fontsize=13, y=1.02)
    fig.tight_layout()
    _savefig(fig, 'fig_scaling_all.png')


# ===================================================================

def main():
    """Entry point: generate all timing analysis figures."""
    plt.rcParams.update(STYLE)

    print("=== Timing breakdown ===")
    plot_timing_breakdown()
    print("\n=== Scaling (combined) ===")
    plot_scaling_all()
    print("\nDone.")


if __name__ == '__main__':
    main()
