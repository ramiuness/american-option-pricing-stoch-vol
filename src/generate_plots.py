"""
Publication-quality PNG plots for European pricing under the LLH model.

Usage (from project root):
    cd src && python generate_plots.py
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


def _ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def _savefig(fig, name):
    fig.savefig(os.path.join(OUTPUT_DIR, name))
    plt.close(fig)


# ── Table 1 model factory ──

def _make_model_t1(seed=123, **overrides):
    params = dict(r=0.01, rho=-0.2, kappa=5, nu=0.2,
                  sigma0=0.15, theta0=0.18, lam=0.9, eta=0.01, seed=seed)
    params.update(overrides)
    return pm.ImprovedSteinStein(**params)


def _moneyness_label(S0, K):
    """Return ITM/ATM/OTM label for a European call."""
    if S0 > K:
        return 'ITM'
    elif S0 < K:
        return 'OTM'
    return 'ATM'


# ═══════════════════════════════════════════════════════════════════════
# Plot 1: MC Convergence
# ═══════════════════════════════════════════════════════════════════════

def plot_mc_convergence(model,
                        S0_values=(70.0, 95.0, 110.0),
                        K=100.0, tau=TAU, n_steps_mc=52,
                        n_paths_values=(200, 500, 1000, 5000, 10000),
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
                mc_p = aop.price_call_mc(res['S'], K=K, T=tau, r=model.r)['price']
                mc_prices.append(mc_p)

            mc_prices = np.array(mc_prices)
            mean_p = mc_prices.mean()
            std_p = mc_prices.std()

            ax.scatter([np_val] * n_seeds, mc_prices, alpha=0.25, s=15, color='#1f77b4', zorder=2)
            ax.errorbar(np_val, mean_p, yerr=std_p, fmt='o', color='#d62728',
                        markersize=6, capsize=4, zorder=3)

        ax.axhline(llh_price, color='#2ca02c', ls='--', lw=1.5, label=f'LLH = {llh_price:.4f}')
        ax.set_xscale('log')
        ml = _moneyness_label(S0, K)
        ax.set_title(f'$S_0 = {S0:.0f}$ ({ml})')
        ax.legend(loc='upper right')

        if i == mid:
            ax.set_ylabel('European call price')
        else:
            ax.set_ylabel('')

    axes[-1].set_xlabel('Number of MC paths')
    fig.suptitle(
        'Monte Carlo convergence to the LLH formula price\n'
        f'$K = {K:.0f}$, $\\tau = {tau}$',
        fontsize=13, y=1.02)
    fig.tight_layout()
    _savefig(fig, 'fig1_mc_convergence.png')
    print("  Saved fig1_mc_convergence.png")


# ═══════════════════════════════════════════════════════════════════════
# Plots 2a/2b: Price and Bias vs Spot (shared data)
# ═══════════════════════════════════════════════════════════════════════

def _compute_mc_llh_grid(model,
                         S0_values=(90.0, 95.0, 100.0, 105.0, 110.0),
                         K_values=(80.0, 100.0, 120.0),
                         tau=TAU, n_steps_mc=52, n_paths=100_000,
                         phi_max=300.0, n_phi=513, n_steps_ode=128):
    """Compute LLH and MC call prices for a grid of (S0, K)."""
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


def _k_panel_label(K, S0_values):
    """Label a panel by its dominant moneyness across the S0 range."""
    S0_mid = np.median(S0_values)
    ml = _moneyness_label(S0_mid, K)
    return f'$K = {K:.0f}$ ({ml})'


def plot_mc_vs_llh_price(model, grid_data,
                         S0_values=(90.0, 95.0, 100.0, 105.0, 110.0),
                         K_values=(80.0, 100.0, 120.0),
                         tau=TAU):

    n_panels = len(K_values)
    mid = n_panels // 2
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
        'European call price: LLH formula vs Monte Carlo\n'
        f'$\\tau = {tau}$',
        fontsize=13, y=1.03)
    fig.tight_layout()
    _savefig(fig, 'fig2a_price_vs_spot.png')
    print("  Saved fig2a_price_vs_spot.png")


def plot_mc_vs_llh_bias(model, grid_data,
                        S0_values=(90.0, 95.0, 100.0, 105.0, 110.0),
                        K_values=(80.0, 100.0, 120.0),
                        tau=TAU):

    n_panels = len(K_values)
    mid = n_panels // 2
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
        'Monte Carlo relative bias against the LLH formula\n'
        f'$\\tau = {tau}$',
        fontsize=13, y=1.03)
    fig.tight_layout()
    _savefig(fig, 'fig2b_bias_vs_spot.png')
    print("  Saved fig2b_bias_vs_spot.png")


# ═══════════════════════════════════════════════════════════════════════
# Plot 3: S&Z vs LLH
# ═══════════════════════════════════════════════════════════════════════

def plot_sz_vs_llh(S0_values=(90.0, 95.0, 100.0, 105.0, 110.0),
                   K=100.0, tau=TAU,
                   phi_max=300.0, n_phi=513, n_steps_ode=128):

    model_llh = _make_model_t1()
    model_sz = _make_model_t1(lam=0.0, eta=0.0)

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
        r'Effect of LLH extensions on European call price'
        f'\n$K = {K:.0f}$, $\\tau = {tau}$',
        fontsize=13)
    ax.legend()
    fig.tight_layout()
    _savefig(fig, 'fig3_sz_vs_llh.png')
    print("  Saved fig3_sz_vs_llh.png")


# ═══════════════════════════════════════════════════════════════════════
# Plot 4a: Price vs Lambda (LLH vs S&Z reference)
# ═══════════════════════════════════════════════════════════════════════

def _moneyness_label_fixed(S0, K):
    """Panel label for fixed S0, varying K."""
    ml = _moneyness_label(S0, K)
    return f'$S_0={S0:.0f},\\ K={K:.0f}$ ({ml})'


def plot_llh_vs_sz_lambda(lam_values=np.arange(-1.0, 1.2, 0.2),
                          S0=100.0,
                          K_values=(70.0, 100.0, 120.0),
                          tau=TAU,
                          phi_max=300.0, n_phi=513, n_steps_ode=128):

    model_sz = _make_model_t1(lam=0.0, eta=0.0)
    pre_sz = model_sz.llh_precompute_tau(tau, phi_max, n_phi, n_steps_ode)

    n_panels = len(K_values)
    mid = n_panels // 2
    fig, axes = plt.subplots(1, n_panels, figsize=(14, 4.5), sharey=False)

    for i, (ax, K) in enumerate(zip(axes, K_values)):
        sz_price = model_sz.price_call_llh(
            S=S0, K=K, tau=tau, vol=model_sz.sigma0, theta=model_sz.theta0, pre=pre_sz
        ).item()

        prices = []
        for lam in lam_values:
            m = _make_model_t1(lam=float(lam))
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
        r'Sensitivity of the LLH price to the drift parameter $\lambda$'
        f'\n$\\tau = {tau}$',
        fontsize=13, y=1.03)
    fig.tight_layout()
    _savefig(fig, 'fig4a_price_vs_lambda.png')
    print("  Saved fig4a_price_vs_lambda.png")


# ═══════════════════════════════════════════════════════════════════════
# Plot 4b: Price vs Lambda with Eta layers
# ═══════════════════════════════════════════════════════════════════════

def plot_llh_lambda_eta_layers(lam_values=np.arange(-1.0, 1.2, 0.2),
                               eta_values=(0.1, 0.15, 0.2),
                               S0=100.0,
                               K_values=(70.0, 100.0, 120.0),
                               tau=TAU,
                               phi_max=300.0, n_phi=513, n_steps_ode=128):

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    n_panels = len(K_values)
    mid = n_panels // 2
    fig, axes = plt.subplots(1, n_panels, figsize=(14, 4.5), sharey=False)

    for i, (ax, K) in enumerate(zip(axes, K_values)):
        for eta, color in zip(eta_values, colors):
            prices = []
            for lam in lam_values:
                m = _make_model_t1(lam=float(lam), eta=float(eta))
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
        r'Joint sensitivity of the LLH price to $\lambda$ and $\eta$'
        f'\n$\\tau = {tau}$',
        fontsize=13, y=1.03)
    fig.tight_layout()
    _savefig(fig, 'fig4b_price_vs_lambda_eta.png')
    print("  Saved fig4b_price_vs_lambda_eta.png")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    plt.rcParams.update(STYLE)
    _ensure_output_dir()

    model_t1 = _make_model_t1()

    print("Plot 1: MC convergence...")
    plot_mc_convergence(model_t1)

    print("Plots 2a/2b: Price and bias vs spot...")
    grid_data = _compute_mc_llh_grid(model_t1)
    plot_mc_vs_llh_price(model_t1, grid_data)
    plot_mc_vs_llh_bias(model_t1, grid_data)

    print("Plot 3: S&Z vs LLH...")
    plot_sz_vs_llh()

    print("Plot 4a: LLH vs S&Z (lambda sweep)...")
    plot_llh_vs_sz_lambda()

    print("Plot 4b: LLH lambda-eta layers...")
    plot_llh_lambda_eta_layers()

    print(f"\nAll plots saved to {os.path.abspath(OUTPUT_DIR)}")


if __name__ == '__main__':
    main()
