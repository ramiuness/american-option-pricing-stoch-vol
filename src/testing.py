"""
Testing and comparison utilities for regression basis and CI experiments.

Used by notebooks/regression_basis_comparison.ipynb and notebooks/ci_comparison.ipynb.
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import priceModels as pm
import amOptPricer as aop


# ═══════════════════════════════════════════════════════════════════════
# Regression basis comparison
# ═══════════════════════════════════════════════════════════════════════

def basis_comparison_grid(params, K, S0_grid, moneyness_labels,
                          T, n_steps_mc, n_paths, seed,
                          llh_params, configs):
    """
    Price American puts across moneyness with multiple basis types.

    Parameters
    ----------
    params      : dict of model parameters (r, kappa, nu, ...)
    configs     : list of (name, basis_type, basis_order, ridge) tuples

    Returns
    -------
    pd.DataFrame with columns per basis type for plain/CV prices, SEs, VR, timing.
    """
    rows = []
    for S0, mlabel in zip(S0_grid, moneyness_labels):
        model = pm.ImprovedSteinStein(**params, seed=seed)
        sim = model.simulate_prices(S0=S0, T=T, n_steps_mc=n_steps_mc, n_paths=n_paths)

        row = {'S0': S0, 'Moneyness': mlabel}
        for name, btype, border, ridge in configs:
            t0 = time.perf_counter()
            res_p = aop.price_american_put_lsm_llh(
                model, sim, K, basis_order=border, basis_type=btype,
                use_cv=False, ridge=ridge)
            row[f'{name}_plain_time'] = time.perf_counter() - t0
            row[f'{name}_plain_price'] = res_p['price']
            row[f'{name}_plain_se'] = res_p['std_err']

            t0 = time.perf_counter()
            res_cv = aop.price_american_put_lsm_llh(
                model, sim, K, basis_order=border, basis_type=btype,
                use_cv=True, euro_method='llh', ridge=ridge, **llh_params)
            row[f'{name}_cv_time'] = time.perf_counter() - t0
            row[f'{name}_cv_price'] = res_cv.get('price_imp', res_cv['price'])
            row[f'{name}_cv_se'] = res_cv.get('std_err_imp', res_cv['std_err'])
            row[f'{name}_cv_vr'] = (res_p['std_err'] / row[f'{name}_cv_se'])**2 \
                if row[f'{name}_cv_se'] > 0 else np.nan

        rows.append(row)
        print(f"  {mlabel}: done")

    return pd.DataFrame(rows)


def basis_sensitivity(params, S0, K, T, n_steps_mc, n_paths, seed,
                      llh_params, orders, ridge=1e-4):
    """
    Sweep Gaussian basis_order at fixed S0. Also computes Laguerre baseline.

    Returns
    -------
    (sens_df, lag_plain_res, lag_cv_res)
    """
    model = pm.ImprovedSteinStein(**params, seed=seed)
    sim = model.simulate_prices(S0=S0, T=T, n_steps_mc=n_steps_mc, n_paths=n_paths)

    lag_plain = aop.price_american_put_lsm_llh(model, sim, K, basis_type='laguerre',
        basis_order=3, use_cv=False, ridge=1e-5)
    lag_cv = aop.price_american_put_lsm_llh(model, sim, K, basis_type='laguerre',
        basis_order=3, use_cv=True, euro_method='llh', ridge=1e-5, **llh_params)

    sens_rows = []
    for m in orders:
        res_p = aop.price_american_put_lsm_llh(model, sim, K, basis_type='gaussian',
            basis_order=m, use_cv=False, ridge=ridge)
        res_cv = aop.price_american_put_lsm_llh(model, sim, K, basis_type='gaussian',
            basis_order=m, use_cv=True, euro_method='llh', ridge=ridge, **llh_params)
        sens_rows.append({
            'M': m,
            'Plain price': res_p['price'],
            'Plain SE': res_p['std_err'],
            'CV-LLH price': res_cv.get('price_imp', res_cv['price']),
            'CV-LLH SE': res_cv.get('std_err_imp', res_cv['std_err']),
            'VR': (res_p['std_err'] / res_cv.get('std_err_imp', res_cv['std_err']))**2,
        })
        print(f"  M={m}: Plain={res_p['price']:.4f}, "
              f"CV-LLH={res_cv.get('price_imp', res_cv['price']):.4f}")

    return pd.DataFrame(sens_rows), lag_plain, lag_cv


def format_basis_table(df):
    """Format raw basis comparison DataFrame for display."""
    cols = {
        'Moneyness': df['Moneyness'],
        'S0': df['S0'].astype(int),
        'Lag Plain': df['Laguerre_plain_price'].round(4),
        'Lag SE': df['Laguerre_plain_se'].round(4),
        'Gau Plain': df['Gaussian_plain_price'].round(4),
        'Gau SE': df['Gaussian_plain_se'].round(4),
        'Lag CV-LLH': df['Laguerre_cv_price'].round(4),
        'Lag CV SE': df['Laguerre_cv_se'].round(4),
        'Gau CV-LLH': df['Gaussian_cv_price'].round(4),
        'Gau CV SE': df['Gaussian_cv_se'].round(4),
        'Lag VR': df['Laguerre_cv_vr'].round(1),
        'Gau VR': df['Gaussian_cv_vr'].round(1),
    }
    return pd.DataFrame(cols).set_index('Moneyness')


def plot_basis_comparison(df, moneyness_labels, title_suffix=''):
    """3-panel figure: Plain prices, CV-LLH prices (vs S0), VR bars."""
    s0 = df['S0'].values
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: Plain LSM prices
    ax = axes[0]
    ax.errorbar(s0, df['Laguerre_plain_price'], yerr=1.96*df['Laguerre_plain_se'],
                fmt='o-', color='#1f77b4', capsize=4, label='Laguerre')
    ax.errorbar(s0, df['Gaussian_plain_price'], yerr=1.96*df['Gaussian_plain_se'],
                fmt='D--', color='#d62728', capsize=4, label='Gaussian')
    ax.set_xlabel('$S_0$')
    ax.set_ylabel('American put price')
    ax.set_title('Plain LSM')
    ax.legend(fontsize=9)

    # Panel 2: CV-LLH prices
    ax = axes[1]
    ax.errorbar(s0, df['Laguerre_cv_price'], yerr=1.96*df['Laguerre_cv_se'],
                fmt='o-', color='#1f77b4', capsize=4, label='Laguerre')
    ax.errorbar(s0, df['Gaussian_cv_price'], yerr=1.96*df['Gaussian_cv_se'],
                fmt='D--', color='#d62728', capsize=4, label='Gaussian')
    ax.set_xlabel('$S_0$')
    ax.set_ylabel('American put price')
    ax.set_title('CV-LLH')
    ax.legend(fontsize=9)

    # Panel 3: VR ratios
    ax = axes[2]
    x = np.arange(len(moneyness_labels))
    bar_w = 0.35
    ax.bar(x - bar_w/2, df['Laguerre_cv_vr'], bar_w,
           color='#1f77b4', alpha=0.85, label='Laguerre')
    ax.bar(x + bar_w/2, df['Gaussian_cv_vr'], bar_w,
           color='#d62728', alpha=0.85, label='Gaussian')
    ax.set_xticks(x)
    ax.set_xticklabels(moneyness_labels, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('VR Ratio')
    ax.set_title('Variance Reduction')
    ax.axhline(1, color='gray', ls='--', lw=0.8, alpha=0.6)
    ax.legend(fontsize=9)

    fig.suptitle(f'Laguerre vs Gaussian RBF{title_suffix}',
                 fontsize=13, y=1.03)
    fig.tight_layout()
    plt.show()


def plot_basis_sensitivity(sens_df, lag_plain_res, lag_cv_res, S0, K):
    """2-panel figure: price vs M, SE vs M with Laguerre baselines."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(sens_df['M'], sens_df['Plain price'], 'o-', label='Gaussian Plain')
    ax1.plot(sens_df['M'], sens_df['CV-LLH price'], 'D-', label='Gaussian CV-LLH')
    ax1.axhline(lag_plain_res['price'], color='#1f77b4', ls='--', alpha=0.6,
                label='Laguerre Plain')
    ax1.axhline(lag_cv_res.get('price_imp', lag_cv_res['price']),
                color='#ff7f0e', ls='--', alpha=0.6, label='Laguerre CV-LLH')
    ax1.set_xlabel('Number of RBF centers ($M$)')
    ax1.set_ylabel('American put price')
    ax1.set_title('Price vs basis order')
    ax1.legend(fontsize=9)

    ax2.plot(sens_df['M'], sens_df['Plain SE'], 'o-', label='Gaussian Plain')
    ax2.plot(sens_df['M'], sens_df['CV-LLH SE'], 'D-', label='Gaussian CV-LLH')
    ax2.axhline(lag_plain_res['std_err'], color='#1f77b4', ls='--', alpha=0.6,
                label='Laguerre Plain')
    ax2.axhline(lag_cv_res.get('std_err_imp', lag_cv_res['std_err']),
                color='#ff7f0e', ls='--', alpha=0.6, label='Laguerre CV-LLH')
    ax2.set_xlabel('Number of RBF centers ($M$)')
    ax2.set_ylabel('Standard error')
    ax2.set_title('SE vs basis order')
    ax2.legend(fontsize=9)

    fig.suptitle(
        f'Gaussian RBF sensitivity to number of centers '
        f'($S_0={S0:.0f}$, $K={K:.0f}$)',
        fontsize=13, y=1.03)
    fig.tight_layout()
    plt.show()


def plot_basis_by_moneyness(df, title_suffix=''):
    """
    3-panel figure: prices vs S0 for All, ITM-only, OTM-only moneyness ranges.
    Each panel overlays Laguerre and Gaussian CV-LLH prices with error bars.
    """
    filters = [
        ('All', df),
        ('ITM only', df[df['S0'] <= 100.0]),
        ('OTM only', df[df['S0'] >= 100.0]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, (flabel, sub) in zip(axes, filters):
        s0 = sub['S0'].values
        ax.errorbar(s0, sub['Laguerre_cv_price'], yerr=1.96*sub['Laguerre_cv_se'],
                    fmt='o-', color='#1f77b4', capsize=4, label='Laguerre')
        ax.errorbar(s0, sub['Gaussian_cv_price'], yerr=1.96*sub['Gaussian_cv_se'],
                    fmt='D--', color='#d62728', capsize=4, label='Gaussian')
        ax.plot(s0, sub['Laguerre_plain_price'], 's:', color='#1f77b4',
                alpha=0.4, label='Lag Plain')
        ax.plot(s0, sub['Gaussian_plain_price'], '^:', color='#d62728',
                alpha=0.4, label='Gau Plain')
        ax.set_xlabel('$S_0$')
        ax.set_title(flabel)
        ax.legend(fontsize=8)

    axes[0].set_ylabel('American put price')
    fig.suptitle(f'CV-LLH prices by moneyness range{title_suffix}',
                 fontsize=13, y=1.03)
    fig.tight_layout()
    plt.show()


def plot_basis_table(dfs, labels):
    """
    Line-plot comparison of Laguerre vs Gaussian prices by moneyness,
    one panel per parameter set.

    Parameters
    ----------
    dfs    : list of pd.DataFrame (one per parameter set)
    labels : list of str (panel titles, e.g. ['Table 1', 'Table 2'])
    """
    fig, axes = plt.subplots(1, len(dfs), figsize=(7 * len(dfs), 5), sharey=False)
    if len(dfs) == 1:
        axes = [axes]

    for ax, df, lbl in zip(axes, dfs, labels):
        s0 = df['S0'].values
        # Plain (dotted)
        ax.plot(s0, df['Laguerre_plain_price'], 's:', color='#1f77b4',
                alpha=0.5, label='Laguerre Plain')
        ax.plot(s0, df['Gaussian_plain_price'], 'D:', color='#d62728',
                alpha=0.5, label='Gaussian Plain')
        # CV-LLH (solid)
        ax.plot(s0, df['Laguerre_cv_price'], 'o-', color='#1f77b4',
                label='Laguerre CV-LLH')
        ax.plot(s0, df['Gaussian_cv_price'], '^-', color='#d62728',
                label='Gaussian CV-LLH')
        ax.set_xlabel('$S_0$')
        ax.set_title(lbl)
        ax.legend(fontsize=8)

    axes[0].set_ylabel('American put price')
    fig.suptitle('Price comparison by parameter set', fontsize=13, y=1.03)
    fig.tight_layout()
    plt.show()


def run_multi_basis(params, S0, K, T, n_steps_mc, N_paths, R,
                    basis_type, basis_order, ridge, use_cv,
                    llh_params=None, base_seed=1000):
    """
    Run R independent replications with a specific basis configuration.

    Always uses CV-LLH when use_cv=True (for comparing bases with/without CV).

    Returns
    -------
    (prices, ses) : arrays of shape (R,)
    """
    prices, ses = [], []
    kw = llh_params or {}
    for r in range(R):
        model = pm.ImprovedSteinStein(**params, seed=base_seed + r)
        sim = model.simulate_prices(S0=S0, T=T, n_steps_mc=n_steps_mc, n_paths=N_paths)
        res = aop.price_american_put_lsm_llh(
            model, sim, K,
            basis_type=basis_type, basis_order=basis_order, ridge=ridge,
            use_cv=use_cv, euro_method='llh' if use_cv else 'bs',
            **(kw if use_cv else {}))
        if use_cv:
            p = res.get('price_imp', res['price'])
            se = res.get('std_err_imp', res['std_err'])
        else:
            p, se = res['price'], res['std_err']
        prices.append(p)
        ses.append(se)
    return np.array(prices), np.array(ses)


def bias_convergence(params, S0, K, T, n_steps_mc, seed,
                     N_values, R, llh_params, configs):
    """
    Estimate bias by comparing estimator means across R replications at each N
    against a high-N reference price for each basis configuration.

    The reference for each config is the mean at the largest N.

    Parameters
    ----------
    configs : list of (display_name, basis_type, basis_order, ridge, use_cv) tuples.
              E.g.:
                ('Laguerre Plain',   'laguerre', 3,  1e-5, False)
                ('Laguerre CV-LLH',  'laguerre', 3,  1e-5, True)
                ('Gaussian Plain',   'gaussian', 15, 1e-4, False)
                ('Gaussian CV-LLH',  'gaussian', 15, 1e-4, True)

    Returns
    -------
    (bias_df, ref_prices) where ref_prices is {config_name: reference_price}
    """
    all_data = {}
    for name, btype, border, ridge, use_cv in configs:
        all_data[name] = {}
        for N in N_values:
            prices, ses = run_multi_basis(
                params, S0, K, T, n_steps_mc, N, R,
                basis_type=btype, basis_order=border, ridge=ridge,
                use_cv=use_cv, llh_params=llh_params)
            all_data[name][N] = {
                'mean': prices.mean(),
                'se_multi': prices.std(ddof=1) / np.sqrt(R),
                'se_single_mean': ses.mean(),
            }
            print(f"  {name:22s} N={N:>7}: mean={prices.mean():.4f}")

    N_ref = max(N_values)
    ref_prices = {name: all_data[name][N_ref]['mean'] for name in all_data}

    rows = []
    for name in all_data:
        for N in N_values:
            d = all_data[name][N]
            rows.append({
                'Config': name, 'N': N,
                'Price': d['mean'],
                'SE_multi': d['se_multi'],
                'Bias (vs ref)': d['mean'] - ref_prices[name],
                'Bias / SE': (d['mean'] - ref_prices[name]) / d['se_multi']
                             if d['se_multi'] > 0 else np.nan,
            })

    return pd.DataFrame(rows), ref_prices


def plot_bias_convergence(bias_df, ref_prices, S0, K):
    """
    2-panel figure: (1) price vs N with reference line, (2) bias/SE vs N.
    Each basis configuration is plotted as a separate series.
    """
    configs = bias_df['Config'].unique()
    # Colors by basis, linestyle by CV status
    style_map = {
        'Laguerre Plain':  ('#1f77b4', 'o', '--'),
        'Laguerre CV-LLH': ('#1f77b4', 'o', '-'),
        'Gaussian Plain':  ('#d62728', '^', '--'),
        'Gaussian CV-LLH': ('#d62728', '^', '-'),
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for name in configs:
        sub = bias_df[bias_df['Config'] == name]
        c, m, ls = style_map.get(name, ('gray', 'o', '-'))
        ax1.errorbar(sub['N'], sub['Price'], yerr=1.96*sub['SE_multi'],
                     fmt=f'{m}{ls}', color=c, capsize=4, label=name)
        ax2.plot(sub['N'], sub['Bias / SE'], f'{m}{ls}', color=c, label=name)

    for name in configs:
        c, _, _ = style_map.get(name, ('gray', 'o', '-'))
        ax1.axhline(ref_prices[name], color=c, ls=':', alpha=0.3)

    ax1.set_xscale('log')
    ax1.set_xlabel('$N$ (paths)')
    ax1.set_ylabel('American put price')
    ax1.set_title('Price convergence')
    ax1.legend(fontsize=8)

    ax2.set_xscale('log')
    ax2.axhline(0, color='gray', ls='--', lw=0.8, alpha=0.6)
    ax2.set_xlabel('$N$ (paths)')
    ax2.set_ylabel('Bias / SE')
    ax2.set_title('Bias in SE units')
    ax2.legend(fontsize=8)

    fig.suptitle(f'Bias convergence ($S_0={S0:.0f}$, $K={K:.0f}$)',
                 fontsize=13, y=1.03)
    fig.tight_layout()
    plt.show()


# ═══════════════════════════════════════════════════════════════════════
# CI comparison (single-run CLT vs multi-run)
# ═══════════════════════════════════════════════════════════════════════

def ci_comparison_grid(params, K, T, n_steps_mc, S0_cases,
                       N_values, R, configs, llh_params=None):
    """
    Run multi-run CI experiment across S0 cases, N values, and basis configs.

    Compares two estimates of the per-run standard error sigma:
      - SD_empirical : sample std of R independent price estimates
      - SE_single (mean) : mean of the R per-run CLT SE estimates

    Both estimate the same quantity (the per-run SE). Their ratio should be
    approximately 1 if the single-run CLT approximation is adequate.

    The CI of the *mean* of R runs is computed from SE_mean = SD_empirical/sqrt(R).
    """
    from scipy.stats import t as t_dist

    rows = []
    for S0, mlabel in S0_cases:
        for N in N_values:
            for name, btype, border, ridge, use_cv in configs:
                print(f"  {mlabel} N={N:>6} {name}...", end=' ')
                prices, ses = run_multi_basis(
                    params, S0, K, T, n_steps_mc, N, R,
                    basis_type=btype, basis_order=border, ridge=ridge,
                    use_cv=use_cv, llh_params=llh_params)

                sd_empirical = prices.std(ddof=1)         # empirical per-run SD
                se_mean = sd_empirical / np.sqrt(R)       # SE of the mean
                se_single_mean = ses.mean()               # mean of per-run CLT SEs
                ratio = sd_empirical / se_single_mean if se_single_mean > 0 else np.nan

                t_crit = t_dist.ppf(0.975, df=R-1)
                rows.append({
                    'S0': S0, 'Moneyness': mlabel, 'N': N, 'Config': name,
                    'Price (multi)': prices.mean(),
                    'SD_empirical': sd_empirical,
                    'SE_single (mean)': se_single_mean,
                    'Ratio': ratio,
                    'SE_of_mean': se_mean,
                    'CI_mean_lo': prices.mean() - t_crit * se_mean,
                    'CI_mean_hi': prices.mean() + t_crit * se_mean,
                })
                print(f"ratio={ratio:.3f}")

    return pd.DataFrame(rows)


def format_ci_table(ci_df):
    """Format CI comparison DataFrame for display."""
    out = ci_df[['Moneyness', 'N', 'Config', 'Price (multi)',
                 'SD_empirical', 'SE_single (mean)', 'Ratio']].copy()
    out['Price (multi)'] = out['Price (multi)'].round(4)
    out['SD_empirical'] = out['SD_empirical'].round(4)
    out['SE_single (mean)'] = out['SE_single (mean)'].round(4)
    out['Ratio'] = out['Ratio'].round(3)
    return out.set_index(['Moneyness', 'N', 'Config'])


def plot_ci_levels(ci_df, S0_cases, R):
    """
    Grid figure: rows = S0_cases, cols = configs.

    Each subplot plots the empirical per-run SD across R replications (solid red)
    and the mean of per-run CLT SE estimates (dashed blue) vs N on log-log axes.
    Both estimate the same quantity (the per-run SE sigma); their ratio should
    be ~1 if the single-run CLT approximation is adequate.
    """
    configs_present = list(ci_df['Config'].unique())
    n_rows = len(S0_cases)
    n_cols = len(configs_present)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5.5 * n_cols, 4.5 * n_rows),
                             squeeze=False, sharex=True)

    for i, (S0, mlabel) in enumerate(S0_cases):
        for j, cname in enumerate(configs_present):
            ax = axes[i, j]
            sub = ci_df[(ci_df['S0'] == S0) & (ci_df['Config'] == cname)]
            sub = sub.sort_values('N')
            ns = sub['N'].values
            ax.plot(ns, sub['SD_empirical'], 'o-', color='#d62728',
                    label='SD empirical ($R$ runs)')
            ax.plot(ns, sub['SE_single (mean)'], 's--', color='#1f77b4',
                    label='$\\overline{\\mathrm{SE}}_{\\mathrm{single}}$ (CLT)')
            ax.set_xscale('log')
            ax.set_yscale('log')

            # Annotate ratio at largest N
            if len(sub) > 0:
                last = sub.iloc[-1]
                ax.annotate(f'ratio @ N={int(last["N"]):,}: {last["Ratio"]:.2f}',
                            xy=(0.02, 0.05), xycoords='axes fraction',
                            fontsize=9, alpha=0.8)

            ax.set_title(f'{mlabel} ($S_0={S0:.0f}$) — {cname}', fontsize=10)
            if i == n_rows - 1:
                ax.set_xlabel('$N$ (paths)')
            if j == 0:
                ax.set_ylabel('Per-run SE estimate')
            if i == 0 and j == n_cols - 1:
                ax.legend(fontsize=9, loc='upper right')

    fig.suptitle(
        f'Per-run SE: empirical SD across $R={R}$ runs vs mean single-run CLT SE',
        fontsize=13, y=1.00)
    fig.tight_layout()
    plt.show()
