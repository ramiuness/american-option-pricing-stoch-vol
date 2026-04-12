"""
Publication-quality PNG plots for European and American pricing under the LLH model.

Usage (from project root):
    cd src && python generate_plots.py           # all param sets
    cd src && python generate_plots.py T1        # just Table 1
    cd src && python generate_plots.py T1 T2     # specific sets
"""

import sys
import os
import gc
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import priceModels as pm
import amerPrice as ap

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
    """Create the output directory for figures if it does not exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def _savefig(fig, name):
    """Save figure to OUTPUT_DIR and close it to free memory."""
    fig.savefig(os.path.join(OUTPUT_DIR, name))
    plt.close(fig)


def _make_model(name, seed=123, **overrides):
    """Instantiate an ImprovedSteinStein model from a named parameter set."""
    params = dict(PARAM_SETS[name], seed=seed)
    params.update(overrides)
    return pm.ImprovedSteinStein(**params)


def _moneyness_label(S0, K, option_type='call'):
    """Return 'ITM', 'ATM', or 'OTM' given spot, strike, and option type."""
    if option_type == 'put':
        if S0 < K:
            return 'ITM'
        elif S0 > K:
            return 'OTM'
        return 'ATM'
    else:
        if S0 > K:
            return 'ITM'
        elif S0 < K:
            return 'OTM'
        return 'ATM'


def _k_panel_label(K, S0_values, option_type='call'):
    """Build a panel title like '$K = 100$ (ATM)' using the median S0."""
    S0_mid = np.median(S0_values)
    ml = _moneyness_label(S0_mid, K, option_type)
    return f'$K = {K:.0f}$ ({ml})'


def _moneyness_label_fixed(S0, K, option_type='call'):
    """Build a label like '$S_0=100, K=100$ (ATM)' for a fixed (S0, K) pair."""
    ml = _moneyness_label(S0, K, option_type)
    return f'$S_0={S0:.0f},\\ K={K:.0f}$ ({ml})'




# ═══════════════════════════════════════════════════════════════════════
# Plot 1: MC Convergence
# ═══════════════════════════════════════════════════════════════════════

def plot_mc_convergence(model, label, pset_name,
                        S0_values=(70.0, 95.0, 110.0),
                        K=100.0, tau=TAU, n_steps_mc=52,
                        n_paths_values=(200, 500, 1000, 5000, 10000),
                        n_seeds=10, base_seed=100,
                        phi_max=300.0, n_phi=513, n_steps_ode=128):
    """Plot MC call price convergence to the LLH formula across path counts."""
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
                mc_prices.append(pm.price_call_mc(res['S'], K=K, T=tau, r=model.r)['price'])

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
                         tau=TAU, n_steps_mc=52, n_paths=10_000,
                         phi_max=300.0, n_phi=513, n_steps_ode=128,
                         scheme='euler'):
    """Compute LLH formula and MC call prices over an (S0, K) grid.

    A single Monte Carlo simulation anchored at S0=1 is broadcast across all
    (S0, K) cells via the multiplicative structure of the LLH price SDE
    (S_t = S0 * prod(1 + r*dt + sigma_hat*dW1)). All cells share the same
    random draws — the bias panel therefore isolates the LLH-vs-MC
    discretisation gap from cross-spot sampling noise, and high-MC-count
    regenerations cost one path block per tau instead of len(S0_values).

    The ``scheme`` argument forwards to ``model.simulate_prices`` to select
    the asset-step discretization (``'euler'`` default, ``'log-euler'`` opt-in).
    """
    pre = model.llh_precompute_tau(tau, phi_max, n_phi, n_steps_ode)

    sim = model.simulate_prices(S0=1.0, T=tau,
                                n_steps_mc=n_steps_mc, n_paths=n_paths,
                                scheme=scheme)
    M_T = sim['S'][:, -1]                                    # (n_paths,)

    S0_arr = np.asarray(S0_values, dtype=float)
    K_arr = np.asarray(K_values, dtype=float)
    ST = S0_arr[:, None, None] * M_T[None, None, :]
    Y = np.exp(-model.r * tau) * np.maximum(ST - K_arr[None, :, None], 0.0)
    mc_price = Y.mean(axis=-1)                               # (n_S, n_K)
    mc_se = Y.std(axis=-1, ddof=1) / np.sqrt(n_paths)        # (n_S, n_K)

    results = {}
    for i, S0 in enumerate(S0_values):
        for j, K in enumerate(K_values):
            llh_p = model.price_call_llh(
                S=S0, K=K, tau=tau, vol=model.sigma0,
                theta=model.theta0, pre=pre
            ).item()
            p, se = float(mc_price[i, j]), float(mc_se[i, j])
            results[(S0, K)] = {
                'llh': llh_p,
                'mc': p,
                'mc_ci': (p - 1.96 * se, p + 1.96 * se),
            }
    return results


def plot_mc_vs_llh_price(model, grid_data, label, pset_name,
                         S0_values=(90.0, 95.0, 100.0, 105.0, 110.0),
                         K_values=(80.0, 100.0, 120.0),
                         tau=TAU):
    """Multi-panel figure: European call LLH formula vs MC price across S0."""
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
    """Multi-panel figure: MC relative bias against the LLH formula across S0."""
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
    """Plot European call price: full LLH model vs Schobel-Zhu limit."""
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
    """Multi-panel figure: LLH call price sensitivity to lambda vs S&Z reference."""
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
    """Multi-panel figure: joint sensitivity of the LLH price to lambda and eta."""
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

# ── American pricing defaults ──
AM_K = 100.0
AM_S0_GRID = (85.0, 90.0, 100.0, 110.0, 115.0)
AM_MONEYNESS = ('Deep ITM', 'ITM', 'ATM', 'OTM', 'Deep OTM')
AM_N_PATHS = 10_000
AM_LLH_PARAMS = dict(phi_max=300.0, n_phi=513, n_steps_rk4=256)

def _llh_ode_kw():
    """Map amerPrice key names to priceModels key names for ODE params."""
    return dict(phi_max=AM_LLH_PARAMS['phi_max'],
                n_phi=AM_LLH_PARAMS['n_phi'],
                n_steps_ode=AM_LLH_PARAMS['n_steps_rk4'])

# Basis configs: (basis_type, basis_order, ridge, label)
BASIS_LAGUERRE = ('laguerre', None, 1e-5, 'Laguerre')
BASIS_GAUSSIAN = ('gaussian', 10, 1e-4, 'Gaussian')


def _compute_american_grid(pset_name, basis_type='laguerre', basis_order=None,
                           ridge=1e-5, seed=42):
    """
    Compute American put prices across moneyness and horizons for one param set.

    Returns dict {horizon_key: list of row dicts}.
    """
    results = {}
    for hkey, hcfg in HORIZON_CONFIGS.items():
        T, n_steps_mc = hcfg['T'], hcfg['n_steps_mc']
        rows = []
        for S0, mlabel in zip(AM_S0_GRID, AM_MONEYNESS):
            model = _make_model(pset_name, seed=seed)
            sim = model.simulate_prices(S0=S0, T=T, n_steps_mc=n_steps_mc,
                                        n_paths=AM_N_PATHS)

            mc_put = pm.price_put_mc(sim['S'], K=AM_K, T=T, r=model.r)
            pre = ap.precompute_european(model, sim, AM_K, **AM_LLH_PARAMS)
            res_plain = ap.price_american_put_lsm_llh(
                model, sim, AM_K, use_cv=False, ridge=ridge,
                basis_type=basis_type, basis_order=basis_order,
                precomputed=pre)
            res_llh = ap.price_american_put_lsm_llh(
                model, sim, AM_K, use_cv=True, euro_method='llh',
                ridge=ridge, basis_type=basis_type, basis_order=basis_order,
                precomputed=pre)

            rows.append({
                'Moneyness': mlabel, 'S0': S0,
                'MC_put_price': mc_put['price'], 'MC_put_se': mc_put['std_err'],
                'Plain_price': res_plain['price'],
                'Plain_se': res_plain['std_err'],
                'LLH_price': res_llh.get('price_imp', res_llh['price']),
                'LLH_se': res_llh.get('std_err_imp', res_llh['std_err']),
                'LLH_VR': res_llh.get('vr', np.nan),
            })
        results[hkey] = rows
    return results


def plot_american_put_panels(pset_name, label, horizon_key,
                             S0_values=(85.0, 90.0, 100.0, 110.0, 115.0),
                             K_values=(80.0, 120.0),
                             n_paths=10_000, seed=42):
    """
    Two-panel figure: American put prices across S0 for K=80 and K=120.
    Each panel has 3 lines: MC Put (dashed), Plain LSM, CV-LLH with SE error bars.
    One figure per (param_set, maturity).
    """
    hcfg = HORIZON_CONFIGS[horizon_key]
    T, n_steps_mc = hcfg['T'], hcfg['n_steps_mc']

    data = {}
    for S0 in S0_values:
        model = _make_model(pset_name, seed=seed)
        sim = model.simulate_prices(S0=S0, T=T, n_steps_mc=n_steps_mc, n_paths=n_paths)
        for K in K_values:
            mc_put = pm.price_put_mc(sim['S'], K=K, T=T, r=model.r)['price']

            pre = ap.precompute_european(model, sim, K, **AM_LLH_PARAMS)
            res_plain = ap.price_american_put_lsm_llh(
                model, sim, K, use_cv=False, precomputed=pre)

            res_llh = ap.price_american_put_lsm_llh(
                model, sim, K, use_cv=True, euro_method='llh',
                precomputed=pre)

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
        ax.set_title(_k_panel_label(K, S0_values, option_type='put'))
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


def plot_american_put_panels_floors(pset_name, label, horizon_key,
                                    S0_values=(85.0, 90.0, 100.0, 110.0, 115.0),
                                    K_values=(80.0, 120.0),
                                    n_paths=10_000, seed=42):
    """
    Two-panel figure comparing exercise floor methods across moneyness.
    6 curves: Euro Put (MC), LSM (no floor), LSM (MC floor), LSM (BS floor),
    LSM (LLH floor), LSM + CV-LLH.
    """
    hcfg = HORIZON_CONFIGS[horizon_key]
    T, n_steps_mc = hcfg['T'], hcfg['n_steps_mc']

    data = {}
    for S0 in S0_values:
        model = _make_model(pset_name, seed=seed)
        sim = model.simulate_prices(S0=S0, T=T, n_steps_mc=n_steps_mc, n_paths=n_paths)
        for K in K_values:
            mc_put = pm.price_put_mc(sim['S'], K=K, T=T, r=model.r)['price']

            pre = ap.precompute_european(model, sim, K, **AM_LLH_PARAMS)
            res_no_floor = ap.price_american_put_lsm_llh(
                model, sim, K, use_cv=False, floor_method='none')
            res_mc1_floor = ap.price_american_put_lsm_llh(
                model, sim, K, use_cv=False, floor_method='mc1')
            res_bs_floor = ap.price_american_put_lsm_llh(
                model, sim, K, use_cv=False, floor_method='bs')
            res_llh_floor = ap.price_american_put_lsm_llh(
                model, sim, K, use_cv=False, precomputed=pre)
            res_cv_llh = ap.price_american_put_lsm_llh(
                model, sim, K, use_cv=True, euro_method='llh',
                precomputed=pre)

            data[(S0, K)] = {
                'mc_put': mc_put,
                'no_floor_price': res_no_floor['price'],
                'no_floor_se': res_no_floor['std_err'],
                'mc1_floor_price': res_mc1_floor['price'],
                'mc1_floor_se': res_mc1_floor['std_err'],
                'bs_floor_price': res_bs_floor['price'],
                'bs_floor_se': res_bs_floor['std_err'],
                'llh_floor_price': res_llh_floor['price'],
                'llh_floor_se': res_llh_floor['std_err'],
                'cv_llh_price': res_cv_llh.get('price_imp', res_cv_llh['price']),
                'cv_llh_se': res_cv_llh.get('std_err_imp', res_cv_llh['std_err']),
            }

    n_panels = len(K_values)
    fig, axes = plt.subplots(1, n_panels, figsize=(14, 5), sharey=False)
    if n_panels == 1:
        axes = [axes]

    curves = [
        ('mc_put',        None,           's--', '#2ca02c', 'Euro Put (MC)'),
        ('no_floor_price','no_floor_se',  'x-',  '#8c564b', 'LSM (no floor)'),
        ('mc1_floor_price','mc1_floor_se','p-',  '#9467bd', 'LSM (MC floor)'),
        ('bs_floor_price','bs_floor_se',  'D-',  '#ff7f0e', 'LSM (BS floor)'),
        ('llh_floor_price','llh_floor_se','o-',  '#1f77b4', 'LSM (LLH floor)'),
        ('cv_llh_price',  'cv_llh_se',    '^-',  '#d62728', 'LSM + CV-LLH'),
    ]

    for i, (ax, K) in enumerate(zip(axes, K_values)):
        s0 = list(S0_values)
        for pkey, sekey, fmt, color, clabel in curves:
            vals = [data[(S0, K)][pkey] for S0 in s0]
            if sekey is None:
                ax.plot(s0, vals, fmt, color=color, label=clabel)
            else:
                se = [1.96 * data[(S0, K)][sekey] for S0 in s0]
                ax.errorbar(s0, vals, yerr=se, fmt=fmt, color=color,
                            capsize=4, label=clabel)

        ax.set_xlabel('$S_0$')
        ax.set_title(_k_panel_label(K, S0_values, option_type='put'))
        ax.legend(fontsize=7)

        if i == 0:
            ax.set_ylabel('Put price')

    fig.suptitle(
        f'American put prices: exercise floor comparison\n'
        f'{label} params, {hcfg["label"]} horizon',
        fontsize=13, y=1.03)
    fig.tight_layout()
    fname = f'fig5_american_put_{pset_name}_{horizon_key}_floors.png'
    _savefig(fig, fname)
    print(f"  Saved {fname}")


def plot_estimator_scatter(pset_name, label, S0=100.0, K=100.0,
                           T=1.0, n_steps_mc=52, n_paths=10_000, seed=42):
    """
    Cost-precision scatter for all 6 estimators at a single (S0, K, T).
    Each point: (wall time, SE) on log-log axes.
    """
    import time as _time

    model = _make_model(pset_name, seed=seed)
    sim = model.simulate_prices(S0=S0, T=T, n_steps_mc=n_steps_mc, n_paths=n_paths)

    estimators = [
        ('LSM (no floor)',  'x',  '#8c564b'),
        ('LSM (MC floor)',  'p',  '#9467bd'),
        ('LSM (BS floor)',  'D',  '#ff7f0e'),
        ('LSM (LLH floor)', 'o', '#1f77b4'),
        ('LSM + CV-LLH',   '^',  '#d62728'),
    ]

    results = {}

    # Precompute once (shared by LLH floor and CV-LLH)
    t0 = _time.perf_counter()
    pre = ap.precompute_european(model, sim, K, **AM_LLH_PARAMS)
    t_precomp = _time.perf_counter() - t0

    # No floor
    t0 = _time.perf_counter()
    r = ap.price_american_put_lsm_llh(model, sim, K, use_cv=False, floor_method='none')
    results['LSM (no floor)'] = (_time.perf_counter() - t0, r['std_err'])

    # MC floor
    t0 = _time.perf_counter()
    r = ap.price_american_put_lsm_llh(model, sim, K, use_cv=False, floor_method='mc1')
    results['LSM (MC floor)'] = (_time.perf_counter() - t0, r['std_err'])

    # BS floor
    t0 = _time.perf_counter()
    r = ap.price_american_put_lsm_llh(model, sim, K, use_cv=False, floor_method='bs')
    results['LSM (BS floor)'] = (_time.perf_counter() - t0, r['std_err'])

    # LLH floor (precompute cost included)
    t0 = _time.perf_counter()
    r = ap.price_american_put_lsm_llh(model, sim, K, use_cv=False, precomputed=pre)
    results['LSM (LLH floor)'] = (t_precomp + _time.perf_counter() - t0, r['std_err'])

    # CV-LLH (precompute cost included)
    t0 = _time.perf_counter()
    r = ap.price_american_put_lsm_llh(model, sim, K, use_cv=True, euro_method='llh',
                                       precomputed=pre)
    results['LSM + CV-LLH'] = (t_precomp + _time.perf_counter() - t0,
                                r.get('std_err_imp', r['std_err']))

    fig, ax = plt.subplots(figsize=(8, 6))

    annotate_labels = {'LSM (no floor)', 'LSM + CV-LLH'}
    for elabel, marker, color in estimators:
        t, se = results[elabel]
        ax.scatter(t, se, s=60, marker=marker, color=color, zorder=5, label=elabel)
        if elabel in annotate_labels:
            ax.annotate(f"SE={se:.4f}\n{t:.2f}s",
                        xy=(t, se), xytext=(15, 8), textcoords='offset points',
                        fontsize=8, color=color,
                        arrowprops=dict(arrowstyle='->', color=color, lw=0.8))
        print(f"  {elabel}: {t:.3f}s, SE={se:.6f}")

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Wall time (s)')
    ax.set_ylabel('Standard error')
    ax.legend(fontsize=8, loc='lower left', markerscale=0.8)

    fig.suptitle(
        f'Cost--precision tradeoff ($N={n_paths:,}$, $S_0={S0:.0f}$, '
        f'$K={K:.0f}$, $T={T}$)\n{label} params',
        fontsize=13, y=1.03)
    fig.tight_layout()
    fname = f'fig_estimator_scatter_{pset_name}.png'
    _savefig(fig, fname)
    plt.close(fig)
    print(f"  Saved {fname}")


def plot_mc1_floor_convergence(pset_name, label,
                               S0_cases=((95.0, 100.0, 'ITM'), (100.0, 95.0, 'OTM')),
                               T=1.0, n_steps_mc=52,
                               N_values=(5_000, 10_000, 20_000, 50_000),
                               base_seed=100):
    """
    Price convergence with N: Plain LSM (LLH floor) vs Plain LSM (MC1 floor).
    Two panels (ITM, OTM) with Euro Put (LLH) baseline.
    """
    # European LLH put baseline per S0_case
    euro_put = {}
    for S0, K_case, s0_label in S0_cases:
        model = _make_model(pset_name, seed=base_seed)
        pre = model.llh_precompute_tau(T, **_llh_ode_kw())
        euro_put[s0_label] = model.price_put_llh(
            S=S0, K=K_case, tau=T, vol=model.sigma0, theta=model.theta0,
            pre=pre, **_llh_ode_kw()).item()

    data = {s0_label: [] for _, _, s0_label in S0_cases}
    for S0, K_case, s0_label in S0_cases:
        for n in N_values:
            model = _make_model(pset_name, seed=base_seed)
            sim = model.simulate_prices(S0=S0, T=T, n_steps_mc=n_steps_mc, n_paths=n)
            pre = ap.precompute_european(model, sim, K_case, **AM_LLH_PARAMS)
            res_llh_floor = ap.price_american_put_lsm_llh(
                model, sim, K_case, use_cv=False, precomputed=pre)
            res_mc1_floor = ap.price_american_put_lsm_llh(
                model, sim, K_case, use_cv=False, floor_method='mc1')
            data[s0_label].append({
                'N': n,
                'llh_floor_price': res_llh_floor['price'],
                'llh_floor_se': res_llh_floor['std_err'],
                'mc1_floor_price': res_mc1_floor['price'],
                'mc1_floor_se': res_mc1_floor['std_err'],
            })
            print(f"    {s0_label} N={n:>7}: LLH={res_llh_floor['price']:.4f} "
                  f"MC1={res_mc1_floor['price']:.4f}")
            del sim, pre, res_llh_floor, res_mc1_floor
        gc.collect()

    tick_labels = [f'{n // 1000}k' for n in N_values]

    fig, axes = plt.subplots(1, len(S0_cases), figsize=(14, 5), sharey=False)
    if len(S0_cases) == 1:
        axes = [axes]

    for ax, (S0, K_case, s0_label) in zip(axes, S0_cases):
        rows = data[s0_label]
        ns = [r['N'] for r in rows]
        ep = euro_put[s0_label]
        ax.axhline(ep, color='#2ca02c', ls='--', lw=1.5,
                   label=f'Euro Put (LLH) = {ep:.2f}')
        ax.errorbar(ns, [r['llh_floor_price'] for r in rows],
                    yerr=[1.96 * r['llh_floor_se'] for r in rows],
                    fmt='o-', color='#1f77b4', capsize=4, label='LSM (LLH floor)')
        ax.errorbar(ns, [r['mc1_floor_price'] for r in rows],
                    yerr=[1.96 * r['mc1_floor_se'] for r in rows],
                    fmt='p-', color='#9467bd', capsize=4, label='LSM (MC floor)')
        ax.set_xscale('log')
        ax.set_xticks(N_values)
        ax.set_xticklabels(tick_labels)
        ax.set_xlabel('$N$ (paths)')
        ax.set_title(f'$S_0={S0:.0f}$, $K={K_case:.0f}$ ({s0_label})')
        ax.legend(fontsize=9)

    axes[0].set_ylabel('American put price')
    fig.suptitle(
        f'Price convergence: LLH floor vs MC floor\n'
        f'{label} params, 1-year horizon',
        fontsize=13, y=1.03)
    fig.tight_layout()
    fname = f'fig10_price_convergence_{pset_name}_mc1floor.png'
    _savefig(fig, fname)
    plt.close(fig)
    print(f"  Saved {fname}")


def plot_vr_mc1_comparison(pset_name, label, seed=42):
    """
    1×2 figure (1-month, 1-year): theoretical VR of CV-LLH vs CV-MC1 across S0.
    VR = 1/(1 - rho^2) from pricer output.
    """
    results = {}
    for hkey, hcfg in HORIZON_CONFIGS.items():
        T, n_steps_mc = hcfg['T'], hcfg['n_steps_mc']
        rows = []
        for S0 in AM_S0_GRID:
            model = _make_model(pset_name, seed=seed)
            sim = model.simulate_prices(S0=S0, T=T, n_steps_mc=n_steps_mc,
                                        n_paths=AM_N_PATHS)
            pre = ap.precompute_european(model, sim, AM_K, **AM_LLH_PARAMS)
            res_cv_llh = ap.price_american_put_lsm_llh(
                model, sim, AM_K, use_cv=True, euro_method='llh',
                precomputed=pre)
            res_cv_mc1 = ap.price_american_put_lsm_llh(
                model, sim, AM_K, use_cv=True, euro_method='mc1')
            rows.append({
                'S0': S0,
                'vr_llh': res_cv_llh.get('vr', np.nan),
                'vr_mc1': res_cv_mc1.get('vr', np.nan),
            })
            del sim, pre, res_cv_llh, res_cv_mc1
        gc.collect()
        results[hkey] = rows

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, (hkey, rows) in zip(axes, results.items()):
        hlbl = HORIZON_CONFIGS[hkey]['label']
        s0 = [r['S0'] for r in rows]
        ax.plot(s0, [r['vr_llh'] for r in rows],
                '^-', color='#d62728', label='CV-LLH')
        ax.plot(s0, [r['vr_mc1'] for r in rows],
                'p-', color='#9467bd', label='CV-MC1')
        ax.axhline(1, color='gray', ls='--', lw=0.8, alpha=0.6)
        ax.set_xlabel('$S_0$')
        ax.set_title(f'{hlbl} horizon')
        ax.legend(fontsize=9)

    axes[0].set_ylabel('VR = $1/(1 - \\rho^2)$')
    fig.suptitle(
        f'Variance reduction: CV-LLH vs CV-MC1 ($K = {AM_K:.0f}$)\n{label} params',
        fontsize=13, y=1.03)
    fig.tight_layout()
    fname = f'fig7_vr_mc1_{pset_name}.png'
    _savefig(fig, fname)
    plt.close(fig)
    print(f"  Saved {fname}")


# ═══════════════════════════════════════════════════════════════════════
# Plot 6: American Put Prices vs Spot (K=100, both horizons)
# ═══════════════════════════════════════════════════════════════════════

def plot_american_prices_vs_spot(pset_name, label, am_grid, basis_label=''):
    """
    1×2 figure (1-month, 1-year): MC Put, Plain LSM, CV-LLH vs S0 at K=100.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    for ax, (hkey, rows) in zip(axes, am_grid.items()):
        hlbl = HORIZON_CONFIGS[hkey]['label']
        s0 = [r['S0'] for r in rows]
        ax.plot(s0, [r['MC_put_price'] for r in rows],
                's--', color='#2ca02c', label='Euro Put (MC)')
        ax.errorbar(s0, [r['Plain_price'] for r in rows],
                    yerr=[1.96 * r['Plain_se'] for r in rows],
                    fmt='o-', color='#1f77b4', capsize=4, label='Plain LSM')
        ax.errorbar(s0, [r['LLH_price'] for r in rows],
                    yerr=[1.96 * r['LLH_se'] for r in rows],
                    fmt='^-', color='#d62728', capsize=4, label='LSM + CV-LLH')
        ax.set_xlabel('$S_0$')
        ax.set_title(hlbl)
        ax.legend(fontsize=9)

    axes[0].set_ylabel('Put price')
    basis_suffix = f', {basis_label} basis' if basis_label else ''
    fig.suptitle(
        f'American put prices vs spot ($K = {AM_K:.0f}$)\n{label} params{basis_suffix}',
        fontsize=13, y=1.03)
    fig.tight_layout()
    btag = f'_{basis_label.lower()}' if basis_label else ''
    fname = f'fig6_american_prices_{pset_name}{btag}.png'
    _savefig(fig, fname)
    print(f"  Saved {fname}")


# ═══════════════════════════════════════════════════════════════════════
# Plot 7: Variance Reduction Ratios
# ═══════════════════════════════════════════════════════════════════════

def plot_vr_ratios(pset_name, label, am_grid):
    """
    1×2 figure by horizon: CV-LLH variance reduction ratio vs S0.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, (hkey, rows) in zip(axes, am_grid.items()):
        hlbl = HORIZON_CONFIGS[hkey]['label']
        s0 = [r['S0'] for r in rows]
        vr = [r['LLH_VR'] for r in rows]
        ax.plot(s0, vr, 'o-', color='#d62728', label='CV-LLH')
        ax.axhline(1, color='gray', ls='--', lw=0.8, alpha=0.6)
        ax.set_xlabel('$S_0$')
        ax.set_title(f'{hlbl} horizon')
        ax.legend(fontsize=9)

    axes[0].set_ylabel('VR Ratio')
    fig.suptitle(
        f'Variance reduction ratio ($K = {AM_K:.0f}$)\n{label} params',
        fontsize=13, y=1.03)
    fig.tight_layout()
    fname = f'fig7_vr_ratios_{pset_name}.png'
    _savefig(fig, fname)
    print(f"  Saved {fname}")


# ═══════════════════════════════════════════════════════════════════════
# Plot 7b: Price Shift (bias in SE units) — companion to VR plot
# ═══════════════════════════════════════════════════════════════════════

def plot_price_shift(pset_name, label, am_grid):
    """
    1×2 figure by horizon: |CV-LLH price - Plain price| / Plain price (%) vs S0.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, (hkey, rows) in zip(axes, am_grid.items()):
        hlbl = HORIZON_CONFIGS[hkey]['label']
        s0 = [r['S0'] for r in rows]
        vals = [100 * abs(r['LLH_price'] - r['Plain_price']) / r['Plain_price']
                if r['Plain_price'] > 0 else np.nan for r in rows]
        ax.plot(s0, vals, '^-', color='#d62728', label='CV-LLH')
        ax.set_xlabel('$S_0$')
        ax.set_title(f'{hlbl} horizon')
        ax.legend(fontsize=9)

    axes[0].set_ylabel('$|P_{\\mathrm{CV}} - P_{\\mathrm{Plain}}|\\;/\\;P_{\\mathrm{Plain}}$ (%)')
    fig.suptitle(
        f'Relative price difference vs plain LSM ($K = {AM_K:.0f}$)\n{label} params',
        fontsize=13, y=1.03)
    fig.tight_layout()
    fname = f'fig7b_price_shift_{pset_name}.png'
    _savefig(fig, fname)
    print(f"  Saved {fname}")


# ═══════════════════════════════════════════════════════════════════════
# Plot 8: Early Exercise Premium
# ═══════════════════════════════════════════════════════════════════════

def plot_eep(pset_name, label, am_grid):
    """
    1×2 figure (1-month, 1-year): EEP (%) vs S0 for Plain LSM and CV-LLH.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, (hkey, rows) in zip(axes, am_grid.items()):
        hlbl = HORIZON_CONFIGS[hkey]['label']
        s0 = [r['S0'] for r in rows]
        eep_plain = [(r['Plain_price'] - r['MC_put_price']) / r['MC_put_price'] * 100
                     if r['MC_put_price'] > 0.01 else np.nan for r in rows]
        eep_llh = [(r['LLH_price'] - r['MC_put_price']) / r['MC_put_price'] * 100
                   if r['MC_put_price'] > 0.01 else np.nan for r in rows]
        ax.plot(s0, eep_plain, 'o-', color='#1f77b4', label='Plain LSM')
        ax.plot(s0, eep_llh, '^-', color='#d62728', label='CV-LLH')
        ax.axhline(0, color='gray', ls='--', lw=0.8, alpha=0.6)
        ax.set_xlabel('$S_0$')
        ax.set_title(f'{hlbl} horizon')
        ax.legend(fontsize=9)

    axes[0].set_ylabel('EEP (%)')
    fig.suptitle(
        f'Early exercise premium ($K = {AM_K:.0f}$)\n{label} params',
        fontsize=13, y=1.03)
    fig.tight_layout()
    fname = f'fig8_eep_{pset_name}.png'
    _savefig(fig, fname)
    print(f"  Saved {fname}")


# ═══════════════════════════════════════════════════════════════════════
# Plot 9: Black-Scholes Limit (CV-BS as near-perfect CV)
# ═══════════════════════════════════════════════════════════════════════

def plot_american_bs_limit(n_paths=10_000, seed=42):
    """
    Single figure: MC Put vs Plain LSM vs CV-BS under BS limit (1-year horizon).
    Demonstrates CV-BS as a near-perfect control variate when the model is GBM.
    """
    model = pm.ImprovedSteinStein(
        r=0.05, rho=0.0, kappa=0.0, nu=0.0,
        sigma0=0.2, theta0=0.0, lam=0.0, eta=0.0, seed=seed)

    T, n_steps_mc = 1.0, 52
    s0_list, mc_list, lsm_list, lsm_se_list = [], [], [], []
    bs_list, bs_se_list = [], []

    for S0 in AM_S0_GRID:
        sim = model.simulate_prices(S0=S0, T=T, n_steps_mc=n_steps_mc, n_paths=n_paths)
        mc_put = pm.price_put_mc(sim['S'], K=AM_K, T=T, r=model.r)
        res_plain = ap.price_american_put_lsm_llh(model, sim, AM_K,
                                                    use_cv=False, floor_method='bs')
        res_bs = ap.price_american_put_lsm_llh(model, sim, AM_K,
                                                  use_cv=True, euro_method='bs')
        s0_list.append(S0)
        mc_list.append(mc_put['price'])
        lsm_list.append(res_plain['price'])
        lsm_se_list.append(res_plain['std_err'])
        bs_list.append(res_bs.get('price_imp', res_bs['price']))
        bs_se_list.append(res_bs.get('std_err_imp', res_bs['std_err']))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left panel: price plot
    ax1.plot(s0_list, mc_list, 's--', color='#2ca02c', label='Euro Put (MC)')
    ax1.errorbar(s0_list, lsm_list, yerr=[1.96 * se for se in lsm_se_list],
                 fmt='o-', color='#1f77b4', capsize=4, label='Plain LSM')
    ax1.errorbar(s0_list, bs_list, yerr=[1.96 * se for se in bs_se_list],
                 fmt='^-', color='#d62728', capsize=4, label='LSM + CV-BS')
    ax1.set_xlabel('$S_0$')
    ax1.set_ylabel('Put price')
    ax1.set_title('American put prices')
    ax1.legend(fontsize=9)

    # Right panel: VR line chart
    vr = [(lse / bse)**2 if bse > 0 else np.nan
          for lse, bse in zip(lsm_se_list, bs_se_list)]
    ax2.plot(s0_list, vr, 'o-', color='#d62728')
    ax2.set_xlabel('$S_0$')
    ax2.set_ylabel('VR Ratio')
    ax2.set_title('Variance reduction (CV-BS)')
    ax2.axhline(1, color='gray', ls='--', lw=0.8, alpha=0.6)

    fig.suptitle(
        r'Black--Scholes limit ($r=0.05,\;\sigma_0=0.2,\;'
        r'\kappa=\nu=\lambda=\eta=\rho=0$), $T=1$ yr',
        fontsize=13, y=1.03)
    fig.tight_layout()
    _savefig(fig, 'fig9_american_put_bs_limit.png')
    print("  Saved fig9_american_put_bs_limit.png")


# ═══════════════════════════════════════════════════════════════════════
# Plot 10: MC Path Convergence (Price and EEP vs N)
# ═══════════════════════════════════════════════════════════════════════

def plot_mc_path_convergence(pset_name, label,
                             S0_cases=((95.0, 100.0, 'ITM'), (100.0, 95.0, 'OTM')),
                             T=1.0, n_steps_mc=52,
                             N_values=(5_000, 10_000, 20_000, 50_000),
                             base_seed=100):
    """
    Two figures (price convergence + EEP convergence) showing the effect of
    increasing MC paths on Plain LSM and CV-LLH estimators.

    S0_cases: tuple of (S0, K, label) triples — each panel uses its own strike.
    """
    # European LLH put baseline per S0_case
    euro_put = {}
    for S0, K_case, s0_label in S0_cases:
        model = _make_model(pset_name, seed=base_seed)
        pre = model.llh_precompute_tau(T, **_llh_ode_kw())
        euro_put[s0_label] = model.price_put_llh(
            S=S0, K=K_case, tau=T, vol=model.sigma0, theta=model.theta0,
            pre=pre, **_llh_ode_kw()).item()

    data = {s0_label: [] for _, _, s0_label in S0_cases}
    for S0, K_case, s0_label in S0_cases:
        for n in N_values:
            model = _make_model(pset_name, seed=base_seed)
            sim = model.simulate_prices(S0=S0, T=T, n_steps_mc=n_steps_mc, n_paths=n)
            mc_put = pm.price_put_mc(sim['S'], K=K_case, T=T, r=model.r)['price']
            pre = ap.precompute_european(model, sim, K_case, **AM_LLH_PARAMS)
            res_plain = ap.price_american_put_lsm_llh(
                model, sim, K_case, use_cv=False, precomputed=pre)
            res_llh = ap.price_american_put_lsm_llh(
                model, sim, K_case, use_cv=True, euro_method='llh',
                precomputed=pre)
            data[s0_label].append({
                'N': n,
                'mc_put': mc_put,
                'plain_price': res_plain['price'],
                'plain_se': res_plain['std_err'],
                'llh_price': res_llh.get('price_imp', res_llh['price']),
                'llh_se': res_llh.get('std_err_imp', res_llh['std_err']),
            })
            print(f"    {s0_label} N={n:>7}: Plain={res_plain['price']:.4f} "
                  f"CV-LLH={res_llh.get('price_imp', res_llh['price']):.4f}")
            del sim, pre, res_plain, res_llh
        gc.collect()

    tick_labels = [f'{n // 1000}k' for n in N_values]

    # --- Fig 10a: Price convergence ---
    fig, axes = plt.subplots(1, len(S0_cases), figsize=(14, 5), sharey=False)
    if len(S0_cases) == 1:
        axes = [axes]

    for ax, (S0, K_case, s0_label) in zip(axes, S0_cases):
        rows = data[s0_label]
        ns = [r['N'] for r in rows]
        ep = euro_put[s0_label]
        ax.axhline(ep, color='#2ca02c', ls='--', lw=1.5,
                   label=f'Euro Put (LLH) = {ep:.2f}')
        ax.errorbar(ns, [r['plain_price'] for r in rows],
                    yerr=[1.96 * r['plain_se'] for r in rows],
                    fmt='o-', color='#1f77b4', capsize=4, label='Plain LSM')
        ax.errorbar(ns, [r['llh_price'] for r in rows],
                    yerr=[1.96 * r['llh_se'] for r in rows],
                    fmt='^-', color='#d62728', capsize=4, label='CV-LLH')
        ax.set_xscale('log')
        ax.set_xticks(N_values)
        ax.set_xticklabels(tick_labels)
        ax.set_xlabel('$N$ (paths)')
        ax.set_title(f'$S_0={S0:.0f}$, $K={K_case:.0f}$ ({s0_label})')
        ax.legend(fontsize=9)

    axes[0].set_ylabel('American put price')
    fig.suptitle(
        f'American put price convergence with $N$\n'
        f'{label} params, 1-year horizon',
        fontsize=13, y=1.03)
    fig.tight_layout()
    fname = f'fig10_price_convergence_{pset_name}.png'
    _savefig(fig, fname)
    plt.close(fig)
    print(f"  Saved {fname}")

    # --- Fig 10b: EEP convergence ---
    fig, axes = plt.subplots(1, len(S0_cases), figsize=(14, 5), sharey=False)
    if len(S0_cases) == 1:
        axes = [axes]

    for ax, (S0, K_case, s0_label) in zip(axes, S0_cases):
        rows = data[s0_label]
        ns = [r['N'] for r in rows]
        eep_plain = [100 * (r['plain_price'] - r['mc_put']) / r['mc_put']
                     if r['mc_put'] > 0.01 else np.nan for r in rows]
        eep_llh = [100 * (r['llh_price'] - r['mc_put']) / r['mc_put']
                   if r['mc_put'] > 0.01 else np.nan for r in rows]

        ax.plot(ns, eep_plain, 'o-', color='#1f77b4', label='Plain LSM')
        ax.plot(ns, eep_llh, '^-', color='#d62728', label='CV-LLH')
        ax.set_xscale('log')
        ax.set_xticks(N_values)
        ax.set_xticklabels(tick_labels)
        ax.set_xlabel('$N$ (paths)')
        ax.set_title(f'$S_0={S0:.0f}$, $K={K_case:.0f}$ ({s0_label})')
        ax.axhline(0, color='black', lw=0.5, ls='--')
        ax.legend(fontsize=9)

    axes[0].set_ylabel('EEP (%)')
    fig.suptitle(
        f'Early exercise premium convergence with $N$\n'
        f'{label} params, 1-year horizon',
        fontsize=13, y=1.03)
    fig.tight_layout()
    fname = f'fig10_eep_convergence_{pset_name}.png'
    _savefig(fig, fname)
    plt.close(fig)
    print(f"  Saved {fname}")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def _run_param_set(pset_name):
    """Generate all figures for a single LLH parameter set."""
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

    # Dual basis: Laguerre and Gaussian fig6
    am_grid_laguerre = None
    for btype, border, ridge, blabel in [BASIS_LAGUERRE, BASIS_GAUSSIAN]:
        print(f"  Computing American grid ({blabel})...")
        am_grid = _compute_american_grid(pset_name, basis_type=btype,
                                          basis_order=border, ridge=ridge)
        if btype == 'laguerre':
            am_grid_laguerre = am_grid
        gc.collect()

        print(f"  Plot 6: American prices vs spot ({blabel})...")
        plot_american_prices_vs_spot(pset_name, label, am_grid, basis_label=blabel)

    # Reuse Laguerre grid for VR/EEP/price-shift plots
    print("  Plot 7: Variance reduction ratios...")
    plot_vr_ratios(pset_name, label, am_grid_laguerre)

    print("  Plot 7b: Price shift...")
    plot_price_shift(pset_name, label, am_grid_laguerre)

    print("  Plot 8: Early exercise premium...")
    plot_eep(pset_name, label, am_grid_laguerre)

    del am_grid_laguerre
    gc.collect()

    print("  Plot 10: MC path convergence (price + EEP)...")
    plot_mc_path_convergence(pset_name, label)
    gc.collect()


def main():
    """Entry point: generate all publication-quality plots for selected parameter sets."""
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
        gc.collect()

    print("\n  Plot 9: BS-limit American put...")
    plot_american_bs_limit()

    print(f"\nAll plots saved to {os.path.abspath(OUTPUT_DIR)}")


if __name__ == '__main__':
    main()
