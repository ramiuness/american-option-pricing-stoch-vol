"""
Empirical timing analysis of LSM+CV-LLH computational complexity.

Usage (from project root):
    cd src && python timing_analysis.py

Produces:
    figs/fig_timing_breakdown.png   - stage-wise wall time (horizontal bar)
    figs/fig_scaling_N.png          - wall time vs N (paths)
    figs/fig_scaling_M.png          - wall time vs M (exercise dates)
    figs/fig_scaling_P.png          - wall time vs P (quadrature nodes)
    figs/fig_timing_comparison.png  - Plain vs CV-BS vs CV-LLH cost-precision
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
N_DEFAULT, M_DEFAULT, P_DEFAULT, K_RK4 = 10_000, 52, 513, 128
SEED = 42


def _savefig(fig, name):
    """Save figure to the output directory and close it."""
    fig.savefig(os.path.join(OUTPUT_DIR, name))
    plt.close(fig)
    print(f"  Saved {name}")


def _time_stages(n_paths, n_steps_mc, n_phi, n_steps_rk4, seed=SEED):
    """Time each stage (simulation, ODE precomputation, backward loop) of LSM+CV-LLH."""
    model = pm.ImprovedSteinStein(**PARAMS, seed=seed)

    # 1. Simulation
    t0 = time.perf_counter()
    sim = model.simulate_prices(S0=S0, T=T, n_steps_mc=n_steps_mc, n_paths=n_paths)
    t_sim = time.perf_counter() - t0

    # 2. ODE precomputation
    dt = T / n_steps_mc
    t0 = time.perf_counter()
    tau_cache = {}
    for j in range(1, n_steps_mc):
        sr = n_steps_mc - j
        if sr not in tau_cache:
            tau_cache[sr] = model.llh_precompute_tau(
                sr * dt, phi_max=300.0, n_phi=n_phi, n_steps_ode=n_steps_rk4)
    t_ode = time.perf_counter() - t0

    # 3. Full pricer
    t0 = time.perf_counter()
    result = ap.price_american_put_lsm_llh(
        model, sim, K=K_STRIKE, use_cv=True, euro_method='llh',
        phi_max=300.0, n_phi=n_phi, n_steps_rk4=n_steps_rk4)
    t_pricer = time.perf_counter() - t0

    t_backward = max(0, t_pricer - t_ode)

    # 4. Plain LSM (reuse same sim)
    t0 = time.perf_counter()
    res_plain = ap.price_american_put_lsm_llh(
        model, sim, K=K_STRIKE, use_cv=False)
    t_plain_backward = time.perf_counter() - t0

    return {
        'Simulation': t_sim,
        'ODE precomp': t_ode,
        'Backward loop': t_backward,
        'Total': t_sim + t_pricer,
        'Plain_sim': t_sim,
        'Plain_backward': t_plain_backward,
        'Plain_total': t_sim + t_plain_backward,
        'price_imp': result.get('price_imp', result['price']),
        'std_err_imp': result.get('std_err_imp', result['std_err']),
    }


def _time_method(method, n_paths, n_steps_mc, seed=SEED, **kw):
    """Time a single pricing method ('plain', 'bs', or 'llh') end-to-end."""
    model = pm.ImprovedSteinStein(**PARAMS, seed=seed)
    sim = model.simulate_prices(S0=S0, T=T, n_steps_mc=n_steps_mc, n_paths=n_paths)
    t0 = time.perf_counter()
    if method == 'plain':
        res = ap.price_american_put_lsm_llh(model, sim, K=K_STRIKE,
                                              use_cv=False)
    elif method == 'bs':
        res = ap.price_american_put_lsm_llh(model, sim, K=K_STRIKE,
                                              use_cv=True, euro_method='bs')
    elif method == 'llh':
        res = ap.price_american_put_lsm_llh(model, sim, K=K_STRIKE,
                                              use_cv=True, euro_method='llh',
                                              **kw)
    return time.perf_counter() - t0, res


# ===================================================================
# Plot 1: Stage breakdown (horizontal bar chart)
# ===================================================================

def plot_stage_breakdown():
    """Generate horizontal bar chart comparing Plain LSM and CV-LLH wall-time breakdown."""
    s = _time_stages(N_DEFAULT, M_DEFAULT, P_DEFAULT, K_RK4)

    # Two groups with a gap: Plain LSM (top), CV-LLH (bottom)
    labels = [
        'Plain LSM\nSimulation',
        'Plain LSM\nBackward loop',
        '',  # gap
        'CV-LLH\nSimulation',
        'CV-LLH\nODE precomputation',
        'CV-LLH\nBackward loop',
    ]
    sizes = [
        s['Plain_sim'], s['Plain_backward'],
        0,
        s['Simulation'], s['ODE precomp'], s['Backward loop'],
    ]
    colors = ['#1f77b4', '#d62728',
              'white',
              '#1f77b4', '#ff7f0e', '#d62728']
    totals = [s['Plain_total'], s['Plain_total'],
              0,
              s['Total'], s['Total'], s['Total']]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    y = np.array([0, 1, 2, 3, 4, 5], dtype=float)
    bars = ax.barh(y, sizes, color=colors, alpha=0.85, height=0.5)

    max_val = s['Total']
    for bar, val, tot in zip(bars, sizes, totals):
        if val > 0:
            pct = 100 * val / tot
            ax.text(bar.get_width() + max_val * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f'{val:.2f}s ({pct:.0f}%)', va='center', fontsize=9)

    ax.set_yticks([i for i in range(6) if i != 2])
    ax.set_yticklabels([l for l in labels if l])
    ax.set_xlabel('Wall time (s)')
    ax.set_xlim(0, max_val * 1.3)
    ax.invert_yaxis()
    fig.suptitle(
        f'Wall-time breakdown: Plain LSM vs LSM+CV-LLH\n'
        f'$N={N_DEFAULT:,}$, $M={M_DEFAULT}$, $P={P_DEFAULT}$',
        fontsize=13, y=1.02)
    fig.tight_layout()
    _savefig(fig, 'fig_timing_breakdown.png')


# ===================================================================
# Plots 2-4: Scaling experiments
# ===================================================================

def _scaling_plot(param_name, param_sym, values, fixed_label,
                  color, marker, fname, vary_fn):
    """Generate a scaling plot: vary one parameter and measure total wall time."""
    times = []
    for v in values:
        s = vary_fn(v)
        times.append(s['Total'])
        print(f"  {param_name}={v}: {s['Total']:.3f}s")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(values, times, f'{marker}-', color=color)
    ax.set_xlabel(param_sym)
    ax.set_ylabel('Wall time (s)')
    fig.suptitle(
        f'LSM+CV-LLH scaling with {param_sym}\n{fixed_label}',
        fontsize=13, y=1.02)
    fig.tight_layout()
    _savefig(fig, fname)


def plot_scaling_N():
    """Plot wall-time scaling with number of MC paths N."""
    _scaling_plot('N', '$N$ (paths)', [1000, 2000, 5000, 10000, 20000, 50000],
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

def plot_method_comparison():
    """Generate grouped bar chart comparing wall time and SE for Plain, CV-BS, and CV-LLH."""
    methods = [('plain', 'Plain LSM', N_DEFAULT),
               ('bs',    'CV-BS',     N_DEFAULT),
               ('llh',   'CV-LLH',    5000)]
    labels, times, ses = [], [], []
    colors = ['#1f77b4', '#ff7f0e', '#d62728']

    for method, label, n in methods:
        kw = dict(phi_max=300.0, n_phi=P_DEFAULT, n_steps_rk4=K_RK4) if method == 'llh' else {}
        elapsed, res = _time_method(method, n, M_DEFAULT, **kw)
        se = res.get('std_err_imp', res['std_err'])
        labels.append(label)
        times.append(elapsed)
        ses.append(se)
        print(f"  {label}: {elapsed:.3f}s, SE={se:.6f}, N={n}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(labels))

    ax1.bar(x, times, color=colors, alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel('Wall time (s)')
    ax1.set_title('Computation time')

    ax2.bar(x, ses, color=colors, alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel('Standard error')
    ax2.set_title('Estimator precision')

    fig.suptitle(
        f'Cost--precision tradeoff\n'
        f'$S_0={S0:.0f}$, $K={K_STRIKE:.0f}$, $T={T}$, Table~1 params',
        fontsize=13, y=1.02)
    fig.tight_layout()
    _savefig(fig, 'fig_timing_comparison.png')


# ===================================================================
# Plot 6: Combined scaling (1x3 horizontal)
# ===================================================================

def plot_scaling_all():
    """Generate combined 1x3 figure showing wall-time scaling with N, M, and P."""
    configs = [
        ('N', '$N$ (paths)', [1000, 2000, 5000, 10000, 20000, 50000],
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
            times.append(s['Total'])
            print(f"  {name}={v}: {s['Total']:.3f}s")
        ax.plot(values, times, f'{marker}-', color=color)
        ax.set_xlabel(sym)
        ax.set_title(fixed, fontsize=10)

    axes[0].set_ylabel('Wall time (s)')
    fig.suptitle('LSM+CV-LLH wall-time scaling', fontsize=13, y=1.02)
    fig.tight_layout()
    _savefig(fig, 'fig_scaling_all.png')


# ===================================================================

def main():
    """Entry point: generate all timing analysis figures."""
    plt.rcParams.update(STYLE)

    print("=== Stage breakdown ===")
    plot_stage_breakdown()
    print("\n=== Scaling (combined) ===")
    plot_scaling_all()
    print("\n=== Method comparison ===")
    plot_method_comparison()
    print("\nDone.")


if __name__ == '__main__':
    main()
