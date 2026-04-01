"""
Publication-quality PNG plots for European pricing under the LLH model.

Usage (from project root):
    cd src && python generate_plots.py           # all param sets
    cd src && python generate_plots.py T1        # just Table 1
    cd src && python generate_plots.py T1 T2     # specific sets
"""

import sys
import os
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import priceModels as pm
import amOptPricer as aop

# ── Style ──
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

TAU = 1.0

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figs')

# ── Parameter sets ──

PARAM_SETS = {
    'T1': dict(r=0.01, rho=-0.2, kappa=5, nu=0.2,
               sigma0=0.15, theta0=0.18, lam=0.9, eta=0.01),
    'T2': dict(r=0.01, rho=0.1691, kappa=4.9394, nu=0.3943,
               sigma0=0.2924, theta0=0.1319, lam=0.3115, eta=0.4112),
    'stress_eta': dict(r=0.01, rho=-0.3, kappa=5.0, nu=0.2,
                       sigma0=0.35, theta0=0.35, lam=0.5, eta=0.8),
    'stress_both': dict(r=0.01, rho=-0.3, kappa=5.0, nu=0.2,
                        sigma0=0.35, theta0=0.35, lam=1.2, eta=1.0),
}

PARAM_LABELS = {
    'T1': 'Table 1',
    'T2': 'Table 2',
    'stress_eta': r'Stress ($\eta=0.8$)',
    'stress_both': r'Stress ($\lambda=1.2,\,\eta=1.0$)',
}

# Per-set MC base seeds (avoid pathological single-path outliers in
# heavy-tailed regimes; see discussion in the report).
MC_BASE_SEEDS = {
    'stress_eta': 200,
    'stress_both': 200,
}


def _ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def _savefig(fig, name):
    fig.savefig(os.path.join(OUTPUT_DIR, name))
    plt.close(fig)


def _make_model(name, seed=123, **overrides):
    params = dict(PARAM_SETS[name], seed=seed)
    params.update(overrides)
    return pm.ImprovedSteinStein(**params)


def _moneyness_label(S0, K):
    if S0 > K:
        return 'ITM'
    elif S0 < K:
        return 'OTM'
    return 'ATM'


def _k_panel_label(K, S0_values):
    S0_mid = np.median(S0_values)
    ml = _moneyness_label(S0_mid, K)
    return f'$K = {K:.0f}$ ({ml})'


def _moneyness_label_fixed(S0, K):
    ml = _moneyness_label(S0, K)
    return f'$S_0={S0:.0f},\\ K={K:.0f}$ ({ml})'




# ═══════════════════════════════════════════════════════════════════════
# Plot 1: MC Convergence
# ═══════════════════════════════════════════════════════════════════════

def plot_mc_convergence(model, label, pset_name,
                        S0_values=(70.0, 95.0, 110.0),
                        K=100.0, tau=TAU, n_steps_mc=52,
                        n_paths_values=(200, 500, 1000, 5000, 10000, 50000, 100000),
                        n_seeds=10, base_seed=100,
                        phi_max=300.0, n_phi=513, n_steps_ode=128):

    pre = model.llh_precompute_tau(tau, phi_max, n_phi, n_steps_ode)
    n_panels = len(S0_values)
    mid = n_panels // 2

    fig, axes = plt.subplots(n_panels, 1, figsize=(7, 9), sharex=True)
    if n_panels == 1:
        axes = [axes]

    for i, (ax, S0) in enumerate(zip(axes, S0_values)):
        llh_price = model.price_call_llh(
            S=S0, K=K, tau=tau, vol=model.sigma0, theta=model.theta0, pre=pre
        ).item()

        for np_val in n_paths_values:
            mc_prices = []
            for s in range(n_seeds):
                m = pm.ImprovedSteinStein(
                    r=model.r, rho=model.rho, kappa=model.kappa, nu=model.nu,
                    sigma0=model.sigma0, theta0=model.theta0,
                    lam=model.lam, eta=model.eta, seed=base_seed + s)
                res = m.simulate_prices(S0=S0, T=tau, n_steps_mc=n_steps_mc, n_paths=np_val)
                mc_prices.append(aop.price_call_mc(res['S'], K=K, T=tau, r=model.r)['price'])

            mc_prices = np.array(mc_prices)
            mean_p = mc_prices.mean()
            # 95% CI for the mean of n_seeds replications
            ci_hw = 1.96 * mc_prices.std(ddof=1) / np.sqrt(n_seeds)

            ax.scatter([np_val] * n_seeds, mc_prices, alpha=0.25, s=15, color='#1f77b4', zorder=2)
            ax.errorbar(np_val, mean_p, yerr=ci_hw, fmt='o', color='#d62728',
                        markersize=6, capsize=4, zorder=3)

        ax.axhline(llh_price, color='#2ca02c', ls='--', lw=1.5, label=f'LLH = {llh_price:.2f}')
        ax.set_xscale('log')
        ax.set_xlim(n_paths_values[0] * 0.7, n_paths_values[-1] * 1.5)
        ml = _moneyness_label(S0, K)
        ax.set_title(f'$S_0 = {S0:.0f}$ ({ml})')
        ax.legend(loc='upper right')

        if i == mid:
            ax.set_ylabel('European call price')
        else:
            ax.set_ylabel('')

    axes[-1].set_xlabel('Number of MC paths')
    fig.suptitle(
        f'MC convergence to the LLH formula price\n'
        f'{label} params (Lin, Lin & He, 2024), '
        f'$K = {K:.0f}$, $\\tau = {tau}$',
        fontsize=13, y=1.02)
    fig.tight_layout()
    fname = f'fig1_mc_convergence_{pset_name}.png'
    _savefig(fig, fname)
    print(f"  Saved {fname}")


# ═══════════════════════════════════════════════════════════════════════
# Plots 2a/2b: Price and Bias vs Spot (shared data)
# ═══════════════════════════════════════════════════════════════════════

def _compute_mc_llh_grid(model,
                         S0_values=(90.0, 95.0, 100.0, 105.0, 110.0),
                         K_values=(80.0, 100.0, 120.0),
                         tau=TAU, n_steps_mc=52, n_paths=100_000,
                         phi_max=300.0, n_phi=513, n_steps_ode=128):
    pre = model.llh_precompute_tau(tau, phi_max, n_phi, n_steps_ode)
    results = {}
    for S0 in S0_values:
        res = model.simulate_prices(S0=S0, T=tau, n_steps_mc=n_steps_mc, n_paths=n_paths)
        for K in K_values:
            llh_p = model.price_call_llh(
                S=S0, K=K, tau=tau, vol=model.sigma0, theta=model.theta0, pre=pre
            ).item()
            mc_res = aop.price_call_mc(res['S'], K=K, T=tau, r=model.r)
            results[(S0, K)] = {
                'llh': llh_p,
                'mc': mc_res['price'],
                'mc_ci': mc_res['ci_95'],
            }
    return results


def plot_mc_vs_llh_price(model, grid_data, label, pset_name,
                         S0_values=(90.0, 95.0, 100.0, 105.0, 110.0),
                         K_values=(80.0, 100.0, 120.0),
                         tau=TAU):

    n_panels = len(K_values)
    fig, axes = plt.subplots(1, n_panels, figsize=(14, 4.5), sharey=False)

    for i, (ax, K) in enumerate(zip(axes, K_values)):
        llh_prices = [grid_data[(S0, K)]['llh'] for S0 in S0_values]
        mc_prices = [grid_data[(S0, K)]['mc'] for S0 in S0_values]
        mc_lo = [grid_data[(S0, K)]['mc_ci'][0] for S0 in S0_values]
        mc_hi = [grid_data[(S0, K)]['mc_ci'][1] for S0 in S0_values]
        mc_err_lo = np.array(mc_prices) - np.array(mc_lo)
        mc_err_hi = np.array(mc_hi) - np.array(mc_prices)

        ax.plot(S0_values, llh_prices, 'o-', color='#1f77b4', label='LLH formula')
        ax.errorbar(S0_values, mc_prices, yerr=[mc_err_lo, mc_err_hi],
                    fmt='^--', color='#ff7f0e', capsize=4, label='MC')
        ax.set_xlabel('$S_0$')
        ax.set_title(_k_panel_label(K, S0_values))
        ax.legend()

        if i == 0:
            ax.set_ylabel('European call price')
        else:
            ax.set_ylabel('')

    fig.suptitle(
        f'European call price: LLH formula vs MC\n'
        f'{label} params (Lin, Lin & He, 2024), $\\tau = {tau}$',
        fontsize=13, y=1.03)
    fig.tight_layout()
    fname = f'fig2a_price_vs_spot_{pset_name}.png'
    _savefig(fig, fname)
    print(f"  Saved {fname}")


def plot_mc_vs_llh_bias(model, grid_data, label, pset_name,
                        S0_values=(90.0, 95.0, 100.0, 105.0, 110.0),
                        K_values=(80.0, 100.0, 120.0),
                        tau=TAU):

    n_panels = len(K_values)
    fig, axes = plt.subplots(1, n_panels, figsize=(14, 4.5), sharey=True)

    for i, (ax, K) in enumerate(zip(axes, K_values)):
        biases = []
        for S0 in S0_values:
            d = grid_data[(S0, K)]
            bias = (d['mc'] - d['llh']) / d['mc'] * 100 if abs(d['mc']) > 1e-12 else 0.0
            biases.append(bias)

        ax.plot(S0_values, biases, 'o-', color='#9467bd', lw=1.5)
        ax.axhline(0, color='black', ls='--', lw=0.8)
        ax.set_xlabel('$S_0$')
        ax.set_title(_k_panel_label(K, S0_values))

        if i == 0:
            ax.set_ylabel('Relative bias  (MC $-$ LLH) / MC  [%]')
        else:
            ax.set_ylabel('')

    fig.suptitle(
        f'MC relative bias against the LLH formula\n'
        f'{label} params (Lin, Lin & He, 2024), $\\tau = {tau}$',
        fontsize=13, y=1.03)
    fig.tight_layout()
    fname = f'fig2b_bias_vs_spot_{pset_name}.png'
    _savefig(fig, fname)
    print(f"  Saved {fname}")


# ═══════════════════════════════════════════════════════════════════════
# Plot 3: S&Z vs LLH
# ═══════════════════════════════════════════════════════════════════════

def plot_sz_vs_llh(pset_name, label,
                   S0_values=(90.0, 95.0, 100.0, 105.0, 110.0),
                   K=100.0, tau=TAU,
                   phi_max=300.0, n_phi=513, n_steps_ode=128):

    model_llh = _make_model(pset_name)
    model_sz = _make_model(pset_name, lam=0.0, eta=0.0)

    pre_llh = model_llh.llh_precompute_tau(tau, phi_max, n_phi, n_steps_ode)
    pre_sz = model_sz.llh_precompute_tau(tau, phi_max, n_phi, n_steps_ode)

    llh_prices, sz_prices = [], []
    for S0 in S0_values:
        llh_prices.append(model_llh.price_call_llh(
            S=S0, K=K, tau=tau, vol=model_llh.sigma0, theta=model_llh.theta0, pre=pre_llh
        ).item())
        sz_prices.append(model_sz.price_call_llh(
            S=S0, K=K, tau=tau, vol=model_sz.sigma0, theta=model_sz.theta0, pre=pre_sz
        ).item())

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(S0_values, llh_prices, 'o-', color='#1f77b4', label='LLH (full)')
    ax.plot(S0_values, sz_prices, 's--', color='#ff7f0e',
            label='Sch\u00f6bel-Zhu limit ($\\lambda = \\eta = 0$)')
    ax.set_xlabel('$S_0$')
    ax.set_ylabel('European call price')
    ax.set_title(
        f'Effect of LLH extensions on European call price\n'
        f'{label} params (Lin, Lin & He, 2024), '
        f'$K = {K:.0f}$, $\\tau = {tau}$',
        fontsize=13)
    ax.legend()
    fig.tight_layout()
    fname = f'fig3_sz_vs_llh_{pset_name}.png'
    _savefig(fig, fname)
    print(f"  Saved {fname}")


# ═══════════════════════════════════════════════════════════════════════
# Plot 4a: Price vs Lambda (LLH vs S&Z reference)
# ═══════════════════════════════════════════════════════════════════════

def plot_llh_vs_sz_lambda(pset_name, label,
                          lam_values=np.arange(-1.0, 1.2, 0.2),
                          S0=100.0,
                          K_values=(70.0, 100.0, 120.0),
                          tau=TAU,
                          phi_max=300.0, n_phi=513, n_steps_ode=128):

    model_sz = _make_model(pset_name, lam=0.0, eta=0.0)
    pre_sz = model_sz.llh_precompute_tau(tau, phi_max, n_phi, n_steps_ode)

    n_panels = len(K_values)
    fig, axes = plt.subplots(1, n_panels, figsize=(14, 4.5), sharey=False)

    for i, (ax, K) in enumerate(zip(axes, K_values)):
        sz_price = model_sz.price_call_llh(
            S=S0, K=K, tau=tau, vol=model_sz.sigma0, theta=model_sz.theta0, pre=pre_sz
        ).item()

        prices = []
        for lam in lam_values:
            m = _make_model(pset_name, lam=float(lam))
            p = m.price_call_llh(
                S=S0, K=K, tau=tau, vol=m.sigma0, theta=m.theta0,
                phi_max=phi_max, n_phi=n_phi, n_steps_ode=n_steps_ode
            ).item()
            prices.append(p)

        ax.plot(lam_values, prices, 'o-', color='#1f77b4', label='LLH')
        ax.axhline(sz_price, color='#ff7f0e', ls='--', lw=1.5, label=f'S&Z at {sz_price:.4f}')
        ax.set_xlabel('$\\lambda$')
        ax.set_title(_moneyness_label_fixed(S0, K))
        ax.legend()

        if i == 0:
            ax.set_ylabel('European call price')
        else:
            ax.set_ylabel('')

    fig.suptitle(
        f'Sensitivity of the LLH price to $\\lambda$\n'
        f'{label} params (Lin, Lin & He, 2024), $\\tau = {tau}$',
        fontsize=13, y=1.03)
    fig.tight_layout()
    fname = f'fig4a_price_vs_lambda_{pset_name}.png'
    _savefig(fig, fname)
    print(f"  Saved {fname}")


# ═══════════════════════════════════════════════════════════════════════
# Plot 4b: Price vs Lambda with Eta layers
# ═══════════════════════════════════════════════════════════════════════

def plot_llh_lambda_eta_layers(pset_name, label,
                               lam_values=np.arange(-1.0, 1.2, 0.2),
                               eta_values=(0.1, 0.15, 0.2),
                               S0=100.0,
                               K_values=(70.0, 100.0, 120.0),
                               tau=TAU,
                               phi_max=300.0, n_phi=513, n_steps_ode=128):

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    n_panels = len(K_values)
    fig, axes = plt.subplots(1, n_panels, figsize=(14, 4.5), sharey=False)

    for i, (ax, K) in enumerate(zip(axes, K_values)):
        for eta, color in zip(eta_values, colors):
            prices = []
            for lam in lam_values:
                m = _make_model(pset_name, lam=float(lam), eta=float(eta))
                p = m.price_call_llh(
                    S=S0, K=K, tau=tau, vol=m.sigma0, theta=m.theta0,
                    phi_max=phi_max, n_phi=n_phi, n_steps_ode=n_steps_ode
                ).item()
                prices.append(p)
            ax.plot(lam_values, prices, 'o-', color=color, label=f'$\\eta = {eta}$')

        ax.set_xlabel('$\\lambda$')
        ax.set_title(_moneyness_label_fixed(S0, K))
        ax.legend()

        if i == 0:
            ax.set_ylabel('European call price')
        else:
            ax.set_ylabel('')

    fig.suptitle(
        f'Joint sensitivity of the LLH price to $\\lambda$ and $\\eta$\n'
        f'{label} params (Lin, Lin & He, 2024), $\\tau = {tau}$',
        fontsize=13, y=1.03)
    fig.tight_layout()
    fname = f'fig4b_price_vs_lambda_eta_{pset_name}.png'
    _savefig(fig, fname)
    print(f"  Saved {fname}")


# ═══════════════════════════════════════════════════════════════════════
# Plot 5: American Put Prices (MC Put vs LSM vs CV-LLH)
# ═══════════════════════════════════════════════════════════════════════

HORIZON_CONFIGS = {
    '1m': {'T': 1/12, 'n_steps_mc': 22,  'label': '1-month'},
    '1y': {'T': 1.0,  'n_steps_mc': 52,  'label': '1-year'},
}


def plot_american_put_panels(pset_name, label, horizon_key,
                             S0_values=(85.0, 90.0, 100.0, 110.0, 115.0),
                             K_values=(80.0, 120.0),
                             n_paths=10_000, seed=42,
                             phi_max=300.0, n_phi=513, n_steps_rk4=128):
    """
    Two-panel figure: American put prices across S0 for K=80 and K=120.
    Each panel has 3 lines: MC Put (dashed), Plain LSM, CV-LLH with SE error bars.
    One figure per (param_set, maturity).
    """
    hcfg = HORIZON_CONFIGS[horizon_key]
    T, n_steps_mc = hcfg['T'], hcfg['n_steps_mc']

    # Collect data for each (S0, K)
    data = {}  # (S0, K) -> {mc_put, lsm_price, lsm_se, llh_price, llh_se}
    for S0 in S0_values:
        model = _make_model(pset_name, seed=seed)
        sim = model.simulate_prices(S0=S0, T=T, n_steps_mc=n_steps_mc, n_paths=n_paths)
        for K in K_values:
            mc_put = aop.price_put_mc(sim['S'], K=K, T=T, r=model.r)['price']

            res_plain = aop.price_american_put_lsm_llh(
                model, sim, K, use_cv=False, ridge=1e-5)

            res_llh = aop.price_american_put_lsm_llh(
                model, sim, K, use_cv=True, euro_method='llh', ridge=1e-5,
                phi_max=phi_max, n_phi=n_phi, n_steps_rk4=n_steps_rk4)

            data[(S0, K)] = {
                'mc_put': mc_put,
                'lsm_price': res_plain['price'],
                'lsm_se': res_plain['std_err'],
                'llh_price': res_llh.get('price_imp', res_llh['price']),
                'llh_se': res_llh.get('std_err_imp', res_llh['std_err']),
            }

    n_panels = len(K_values)
    fig, axes = plt.subplots(1, n_panels, figsize=(12, 5), sharey=False)
    if n_panels == 1:
        axes = [axes]

    for i, (ax, K) in enumerate(zip(axes, K_values)):
        mc = [data[(S0, K)]['mc_put'] for S0 in S0_values]
        lsm = [data[(S0, K)]['lsm_price'] for S0 in S0_values]
        lsm_se = [1.96 * data[(S0, K)]['lsm_se'] for S0 in S0_values]
        llh = [data[(S0, K)]['llh_price'] for S0 in S0_values]
        llh_se = [1.96 * data[(S0, K)]['llh_se'] for S0 in S0_values]

        ax.plot(S0_values, mc, 's--', color='#2ca02c', label='Euro Put (MC)')
        ax.errorbar(S0_values, lsm, yerr=lsm_se,
                    fmt='o-', color='#1f77b4', capsize=4, label='Plain LSM')
        ax.errorbar(S0_values, llh, yerr=llh_se,
                    fmt='^-', color='#d62728', capsize=4, label='LSM + CV-LLH')

        ax.set_xlabel('$S_0$')
        ax.set_title(f'$K = {K:.0f}$')
        ax.legend(fontsize=9)

        if i == 0:
            ax.set_ylabel('Put price')

    fig.suptitle(
        f'American put prices: MC Put vs LSM vs CV-LLH\n'
        f'{label} params, {hcfg["label"]} horizon',
        fontsize=13, y=1.03)
    fig.tight_layout()
    fname = f'fig5_american_put_{pset_name}_{horizon_key}.png'
    _savefig(fig, fname)
    print(f"  Saved {fname}")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def _run_param_set(pset_name):
    label = PARAM_LABELS.get(pset_name, pset_name)
    model = _make_model(pset_name)

    print(f"\n{'='*60}")
    print(f"  Generating figures for: {label} ({pset_name})")
    print(f"{'='*60}")

    print("  Plot 1: MC convergence...")
    bs = MC_BASE_SEEDS.get(pset_name, 100)
    plot_mc_convergence(model, label, pset_name, base_seed=bs)

    print("  Plots 2a/2b: Price and bias vs spot...")
    grid_data = _compute_mc_llh_grid(model)
    plot_mc_vs_llh_price(model, grid_data, label, pset_name)
    plot_mc_vs_llh_bias(model, grid_data, label, pset_name)

    print("  Plot 3: S&Z vs LLH...")
    plot_sz_vs_llh(pset_name, label)

    print("  Plot 4a: LLH vs S&Z (lambda sweep)...")
    plot_llh_vs_sz_lambda(pset_name, label)

    print("  Plot 4b: LLH lambda-eta layers...")
    plot_llh_lambda_eta_layers(pset_name, label)

    for hkey in HORIZON_CONFIGS:
        hlbl = HORIZON_CONFIGS[hkey]['label']
        print(f"  Plot 5: American put panels ({hlbl})...")
        plot_american_put_panels(pset_name, label, hkey)


def main():
    plt.rcParams.update(STYLE)
    _ensure_output_dir()

    # CLI: specific sets or all
    requested = sys.argv[1:] if len(sys.argv) > 1 else list(PARAM_SETS.keys())
    for name in requested:
        if name not in PARAM_SETS:
            print(f"Unknown param set '{name}'. Available: {list(PARAM_SETS.keys())}")
            sys.exit(1)

    for name in requested:
        _run_param_set(name)

    print(f"\nAll plots saved to {os.path.abspath(OUTPUT_DIR)}")


if __name__ == '__main__':
    main()
