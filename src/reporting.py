import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import priceModels as pm
import amOptPricer as aop


def llh_vs_mc(model, S_vals, K_vals, tau, n_steps_mc, n_paths,
              phi_max=300.0, n_phi=513, n_steps_ode=128):
    """
    Build call and put DataFrames comparing LLH analytical prices with
    Monte Carlo estimates across a grid of (S₀, K) pairs.

    Parameters
    ----------
    model       : pm.ImprovedSteinStein
    S_vals      : list of initial spot prices
    K_vals      : list of strikes (same length as S_vals)
    tau         : time to maturity
    n_steps_mc  : MC time discretization steps
    n_paths     : number of MC paths
    phi_max     : maximum φ for quadrature
    n_phi       : number of φ nodes
    n_steps_ode : RK4 steps for ODE integration

    Returns
    -------
    call_df, put_df : pd.DataFrame indexed by S₀
    """
    rows = []
    for s, k in zip(S_vals, K_vals):
        res = pm.european_prices(model, S0=s, K=k, tau=tau,
                                 n_steps_mc=n_steps_mc, n_paths=n_paths,
                                 phi_max=phi_max, n_phi=n_phi,
                                 n_steps_ode=n_steps_ode)
        rows.append({
            'S₀': s, 'K': k,
            'llh_call': res['llh_call'], 'mc_call': res['mc_call'],
            'mc_call_ci': res['mc_call_ci'],
            'llh_put': res['llh_put'], 'mc_put': res['mc_put'],
            'mc_put_ci': res['mc_put_ci'],
        })

    df = pd.DataFrame(rows).set_index('S₀')

    call_df = df[['K']].copy()
    call_df['LLH'] = df['llh_call']
    call_df['MC'] = df['mc_call']
    call_df['Bias (%)'] = ((df['mc_call'] - df['llh_call'])
                           / df['mc_call'] * 100).round(2)
    call_df['MC 95% CI'] = df['mc_call_ci'].apply(
        lambda ci: f"[{ci[0]:.4f}, {ci[1]:.4f}]")

    put_df = df[['K']].copy()
    put_df['LLH'] = df['llh_put']
    put_df['MC'] = df['mc_put']
    put_df['Bias (%)'] = ((df['mc_put'] - df['llh_put'])
                          / df['mc_put'] * 100).round(2)
    put_df['MC 95% CI'] = df['mc_put_ci'].apply(
        lambda ci: f"[{ci[0]:.4f}, {ci[1]:.4f}]")

    return call_df, put_df


def llh_vs_mc_timing(model, S0, K, scenarios, n_paths,
                     phi_max=300.0, n_phi=513, n_steps_ode=128,
                     n_runs=5):
    """
    Compare LLH analytical vs MC call prices with timing across maturities.

    Parameters
    ----------
    model       : pm.ImprovedSteinStein
    S0          : initial spot price
    K           : strike
    scenarios   : list of (label, tau, n_steps_mc) tuples
    n_paths     : number of MC paths
    phi_max, n_phi, n_steps_ode : LLH quadrature params
    n_runs      : number of timing repetitions (reports median)

    Returns
    -------
    pd.DataFrame with columns: Label, tau, LLH, MC, Time_LLH, Time_MC, Speedup

    Timing
    ------
    One warmup call is discarded, then ``n_runs`` repetitions are timed with
    ``time.perf_counter()``. Reported times are the median across runs.
    """
    rows = []
    for label, tau, n_steps_mc in scenarios:
        # ── Warmup (discarded) ──
        model.price_call_llh(S=S0, K=K, tau=tau,
                             vol=model.sigma0, theta=model.theta0,
                             phi_max=phi_max, n_phi=n_phi, n_steps_ode=n_steps_ode)

        # ── Numerical LLH (median of n_runs) ──
        times_llh = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            p_llh = model.price_call_llh(
                S=S0, K=K, tau=tau,
                vol=model.sigma0, theta=model.theta0,
                phi_max=phi_max, n_phi=n_phi, n_steps_ode=n_steps_ode).item()
            times_llh.append(time.perf_counter() - t0)
        t_llh = np.median(times_llh)

        # ── Monte Carlo (median of n_runs) ──
        times_mc = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            res_mc = model.simulate_prices(S0=S0, T=tau,
                                           n_steps_mc=n_steps_mc, n_paths=n_paths)
            p_mc = aop.price_call_mc(res_mc['S'], K=K, T=tau, r=model.r)['price']
            times_mc.append(time.perf_counter() - t0)
        t_mc = np.median(times_mc)

        rows.append({
            'Label': label, 'tau': tau,
            'LLH': p_llh, 'MC': p_mc,
            'Time_LLH': t_llh, 'Time_MC': t_mc,
            'Speedup': t_mc / t_llh,
        })

        # print(f"\n── {label} ──")
        # print(f"{'Method':<20} {'Price':>12} {'Time (s)':>10}")
        # print(f"{'-'*44}")
        # print(f"{'Numerical LLH':<20} {p_llh:>12.6f} {t_llh:>10.4f}")
        # print(f"{'Monte Carlo':<20} {p_mc:>12.6f} {t_mc:>10.4f}")
        # print(f"{'Speedup':<20} {'':>12} {t_mc/t_llh:>10.1f}x")

    return pd.DataFrame(rows)


def sz_table2(rho_cases, theta_values, K_vals, bs_prices,
              r, kappa, nu, sigma0, S0, tau,
              n_steps_mc=52, n_paths=100_000,
              phi_max=300.0, n_phi=513, n_steps_ode=128, seed=123):
    """
    Reproduce Schöbel & Zhu (1999) Table 2: European call prices across
    (ρ, θ₀, K) under the S&Z limit (λ=η=0) of the LLH model.

    Parameters
    ----------
    rho_cases    : dict  {label: rho_value}
    theta_values : list of θ₀ values
    K_vals       : list of strikes
    bs_prices    : dict  {K: BS_price} — Black-Scholes reference row
    r, kappa, nu, sigma0, S0, tau : model / market scalars
    n_steps_mc, n_paths : MC parameters
    phi_max, n_phi, n_steps_ode : LLH quadrature params
    seed         : RNG seed

    Returns
    -------
    dict  {case_name: pd.DataFrame}  — one table per ρ case
    """
    tables = {}
    for case_name, rho_val in rho_cases.items():
        rows = {'BS': bs_prices}
        for theta0_val in theta_values:
            model_sz = pm.ImprovedSteinStein(
                r=r, rho=rho_val, kappa=kappa, nu=nu,
                sigma0=sigma0, theta0=theta0_val,
                lam=0.0, eta=0.0, seed=seed)
            llh_row, mc_row = {}, {}
            for K in K_vals:
                res = pm.european_prices(
                    model_sz, S0=S0, K=K, tau=tau,
                    n_steps_mc=n_steps_mc, n_paths=n_paths,
                    phi_max=phi_max, n_phi=n_phi, n_steps_ode=n_steps_ode)
                llh_row[K] = round(res['llh_call'], 4)
                mc_row[K]  = round(res['mc_call'],  4)
            rows[f'LLH θ₀={theta0_val}'] = llh_row
            rows[f'MC  θ₀={theta0_val}'] = mc_row
        df = pd.DataFrame(rows, index=K_vals).T
        df.index.name = f'{case_name} / K'
        tables[case_name] = df
    return tables


# ═══════════════════════════════════════════════════════════════════════════
# American pricing comparison helpers
# ═══════════════════════════════════════════════════════════════════════════

def _model_attrs(model):
    """Extract constructor kwargs from an ImprovedSteinStein instance."""
    return {k: getattr(model, k)
            for k in ['r', 'rho', 'kappa', 'nu', 'sigma0', 'theta0', 'lam', 'eta']}


def american_put_comparison(model, K, S0_grid, moneyness_labels, T, n_steps_mc,
                            n_paths, n_paths_llh=None, llh_params=None,
                            include_llh=True, seed=42):
    """
    Price American puts across moneyness levels.

    When include_llh=True (default), three methods are compared: Plain LSM,
    CV-BS, and CV-LLH.  When include_llh=False, only Plain LSM and CV-BS are
    computed (no LLH ODE solver is invoked).

    Returns a DataFrame with columns for each method's price, SE, CI width,
    variance reduction ratio (VR), and MC European put price.
    """
    rows = []
    for s0, label in zip(S0_grid, moneyness_labels):
        row = {'Moneyness': label, 'S0': s0}

        # --- Shared simulation for plain LSM + CV-BS (n_paths) ---
        m = pm.ImprovedSteinStein(**_model_attrs(model), seed=seed)
        sim = m.simulate_prices(S0=s0, T=T, n_steps_mc=n_steps_mc, n_paths=n_paths)

        # MC European put
        res_mc_put = m.price_put_mc(sim, K=K)
        row['MC_put_price'] = res_mc_put['price']
        row['MC_put_se'] = res_mc_put['std_err']

        # Plain LSM
        res_plain = m.price_american_put(sim, K=K, use_cv=False, ridge=1e-5)
        row['Plain_price'] = res_plain['price']
        row['Plain_se'] = res_plain['std_err']
        row['Plain_ci_w'] = res_plain['ci_95'][1] - res_plain['ci_95'][0]

        # CV-BS
        res_bs = m.price_american_put(sim, K=K, use_cv=True, euro_method='bs', ridge=1e-5)
        row['BS_price'] = res_bs.get('price_imp', res_bs['price'])
        row['BS_se'] = res_bs.get('std_err_imp', res_bs['std_err'])
        ci_bs = res_bs.get('ci_95_imp', res_bs['ci_95'])
        row['BS_ci_w'] = ci_bs[1] - ci_bs[0]
        row['BS_VR'] = (res_plain['std_err'] / row['BS_se'])**2 if row['BS_se'] > 0 else np.nan

        if include_llh:
            # --- Separate simulation for CV-LLH (n_paths_llh) ---
            m2 = pm.ImprovedSteinStein(**_model_attrs(model), seed=seed)
            sim_llh = m2.simulate_prices(S0=s0, T=T, n_steps_mc=n_steps_mc, n_paths=n_paths_llh)

            # Plain LSM on the same small sim (for apples-to-apples VR)
            res_plain_small = m2.price_american_put(sim_llh, K=K, use_cv=False, ridge=1e-5)

            # CV-LLH
            res_llh = m2.price_american_put(sim_llh, K=K, use_cv=True, euro_method='llh',
                                            ridge=1e-5, **llh_params)
            row['LLH_price'] = res_llh.get('price_imp', res_llh['price'])
            row['LLH_se'] = res_llh.get('std_err_imp', res_llh['std_err'])
            ci_llh = res_llh.get('ci_95_imp', res_llh['ci_95'])
            row['LLH_ci_w'] = ci_llh[1] - ci_llh[0]
            row['LLH_VR'] = (res_plain_small['std_err'] / row['LLH_se'])**2 if row['LLH_se'] > 0 else np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def format_results_table(df):
    """Format the raw comparison DataFrame for display.

    Automatically includes CV-LLH columns only when present in df.
    """
    cols = {
        'S0':            df['S0'].astype(int),
        'MC Put':        df['MC_put_price'].round(4),
        'MC Put SE':     df['MC_put_se'].round(4),
        'LSM Price':     df['Plain_price'].round(4),
        'LSM SE':        df['Plain_se'].round(4),
        'CV-BS Price':   df['BS_price'].round(4),
        'CV-BS SE':      df['BS_se'].round(4),
        'CV-BS VR':      df['BS_VR'].round(1),
    }
    if 'LLH_price' in df.columns:
        cols['CV-LLH Price'] = df['LLH_price'].round(4)
        cols['CV-LLH SE']    = df['LLH_se'].round(4)
        cols['CV-LLH VR']    = df['LLH_VR'].round(1)
    return pd.DataFrame(cols).set_index(df['Moneyness'])


def build_vr_summary(results_dict):
    """
    Consolidate variance reduction ratios from multiple comparison DataFrames.

    Parameters
    ----------
    results_dict : dict  {label: DataFrame} e.g. {'T1, 1m': df_t1_1m, ...}

    Returns
    -------
    vr_df : DataFrame with columns Setting, Moneyness, CV-BS VR, CV-LLH VR
    """
    vr_rows = []
    for label, df in results_dict.items():
        for _, row in df.iterrows():
            vr_rows.append({
                'Setting': label,
                'Moneyness': row['Moneyness'],
                'CV-BS VR': row['BS_VR'],
                'CV-LLH VR': row['LLH_VR'],
            })
    return pd.DataFrame(vr_rows)


def plot_vr_bars(vr_df, moneyness_labels):
    """Plot variance reduction ratio bar charts for CV-BS and CV-LLH."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, method, col in zip(axes, ['CV-BS', 'CV-LLH'], ['CV-BS VR', 'CV-LLH VR']):
        pivot = vr_df.pivot(index='Moneyness', columns='Setting', values=col)
        pivot.loc[moneyness_labels].plot.bar(ax=ax, rot=0)
        ax.set_title(f'Variance Reduction Ratio — {method}')
        ax.set_ylabel('VR Ratio')
        ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='VR = 1')
        ax.legend(title='Setting', fontsize=8)
    plt.tight_layout()
    plt.show()


def plot_american_put_prices(df, title='', cv_method='llh'):
    """
    Line chart: MC Put, Plain LSM, and a CV American put price vs S0.

    Parameters
    ----------
    df         : raw DataFrame from american_put_comparison()
    title      : figure title string
    cv_method  : 'llh' (default) or 'bs' — selects which CV series to plot
    """
    s0 = df['S0'].values

    if cv_method == 'bs':
        cv_price, cv_se, cv_label = df['BS_price'], df['BS_se'], 'LSM + CV-BS'
    else:
        cv_price, cv_se, cv_label = df['LLH_price'], df['LLH_se'], 'LSM + CV-LLH'

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(s0, df['MC_put_price'], 's--', color='#2ca02c', label='Euro Put (MC)')
    ax.errorbar(s0, df['Plain_price'], yerr=1.96 * df['Plain_se'],
                fmt='o-', color='#1f77b4', capsize=4, label='Plain LSM')
    ax.errorbar(s0, cv_price, yerr=1.96 * cv_se,
                fmt='^-', color='#d62728', capsize=4, label=cv_label)

    ax.set_xlabel('$S_0$')
    ax.set_ylabel('Put price')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def build_eep_table(results_dict, models_dict, K,
                    phi_max=300.0, n_phi=513, n_steps_ode=128):
    """
    Compute early exercise premium (EEP) table.

    EEP is computed relative to the MC European put price (already in the
    raw DataFrames from american_put_comparison).  The LLH analytical
    European put is included as a reference column.

    Includes EEP for Plain LSM, CV-BS, and (when present) CV-LLH.

    Parameters
    ----------
    results_dict : dict  {(params_label, horizon_label): DataFrame}
        e.g. {('Table 1', '1-month'): df_t1_1m, ...}
    models_dict : dict  {params_label: model}
        e.g. {'Table 1': model_t1, 'Table 2': model_t2}
    K : float  strike

    Returns
    -------
    eep_df : DataFrame indexed by (Params, Horizon, Moneyness)
    """
    horizons = {'1-month': 1/12, '1-year': 1.0}
    eep_rows = []
    for (params_label, horizon_label), df in results_dict.items():
        model = models_dict[params_label]
        T = horizons[horizon_label]
        has_llh = 'LLH_price' in df.columns
        for _, row in df.iterrows():
            s0 = row['S0']
            mc_put = row['MC_put_price']
            llh_put = model.price_put_llh(
                S=s0, K=K, tau=T, vol=model.sigma0, theta=model.theta0,
                phi_max=phi_max, n_phi=n_phi, n_steps_ode=n_steps_ode
            ).item()

            am_lsm = row['Plain_price']
            eep_lsm = am_lsm - mc_put

            am_bs = row['BS_price']
            eep_bs = am_bs - mc_put

            entry = {
                'Params': params_label,
                'Horizon': horizon_label,
                'Moneyness': row['Moneyness'],
                'S0': int(s0),
                'Euro Put (MC)': round(mc_put, 4),
                'Euro Put (LLH)': round(llh_put, 4),
                'Amer Put (LSM)': round(am_lsm, 4),
                'EEP (LSM)': round(eep_lsm, 4),
                'EEP % (LSM)': round(eep_lsm / mc_put * 100, 2) if mc_put > 0.01 else np.nan,
                'Amer Put (CV-BS)': round(am_bs, 4),
                'EEP (CV-BS)': round(eep_bs, 4),
                'EEP % (CV-BS)': round(eep_bs / mc_put * 100, 2) if mc_put > 0.01 else np.nan,
            }

            if has_llh:
                am_llh = row['LLH_price']
                eep_llh = am_llh - mc_put
                entry['Amer Put (CV-LLH)'] = round(am_llh, 4)
                entry['EEP (CV-LLH)'] = round(eep_llh, 4)
                entry['EEP % (CV-LLH)'] = round(eep_llh / mc_put * 100, 2) if mc_put > 0.01 else np.nan

            eep_rows.append(entry)

    eep_df = pd.DataFrame(eep_rows)
    return eep_df.set_index(['Params', 'Horizon', 'Moneyness'])


def plot_eep_table(eep_df):
    """
    Multi-panel grouped bar chart of the early exercise premium by method.

    Layout: one row per parameter set, one column per horizon.
    Each panel shows grouped bars (LSM, CV-BS, and optionally CV-LLH)
    across moneyness levels.

    Parameters
    ----------
    eep_df : DataFrame from build_eep_table(), indexed by
             (Params, Horizon, Moneyness)
    """
    params_list = eep_df.index.get_level_values('Params').unique().tolist()
    horizon_list = eep_df.index.get_level_values('Horizon').unique().tolist()
    n_rows = len(params_list)
    n_cols = len(horizon_list)

    has_llh = 'EEP % (CV-LLH)' in eep_df.columns

    methods = ['EEP % (LSM)', 'EEP % (CV-BS)']
    labels = ['Plain LSM', 'CV-BS']
    colors = ['#1f77b4', '#d62728']
    if has_llh:
        methods.append('EEP % (CV-LLH)')
        labels.append('CV-LLH')
        colors.append('#ff7f0e')

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(6 * n_cols, 4.5 * n_rows),
                             squeeze=False, sharey='row')

    n_methods = len(methods)
    bar_width = 0.8 / n_methods

    for i, params in enumerate(params_list):
        for j, horizon in enumerate(horizon_list):
            ax = axes[i, j]
            try:
                sub = eep_df.loc[(params, horizon)]
            except KeyError:
                ax.set_visible(False)
                continue

            moneyness = sub.index.tolist()
            x = np.arange(len(moneyness))

            for k, (method, label, color) in enumerate(zip(methods, labels, colors)):
                vals = sub[method].values
                offset = (k - (n_methods - 1) / 2) * bar_width
                ax.bar(x + offset, vals, bar_width, label=label, color=color,
                       alpha=0.85)

            ax.set_xticks(x)
            ax.set_xticklabels(moneyness, rotation=30, ha='right', fontsize=9)
            ax.set_title(f'{params}, {horizon}', fontsize=11)
            ax.axhline(0, color='black', lw=0.5, ls='--')

            if j == 0:
                ax.set_ylabel('EEP (%)')
            if i == 0 and j == n_cols - 1:
                ax.legend(fontsize=9)

    fig.suptitle('Early Exercise Premium by Method and Moneyness',
                 fontsize=13, y=1.02)
    fig.tight_layout()
    plt.show()


def build_timing_table(model, K, S0, horizons, n_paths, n_paths_llh,
                       llh_params, seed=42):
    """
    Time the three pricing methods across horizons.

    Parameters
    ----------
    model : ImprovedSteinStein
    K, S0 : float
    horizons : dict  {label: {'T': float, 'n_steps_mc': int}}
    n_paths, n_paths_llh : int
    llh_params : dict  passed to price_american_put for LLH mode

    Returns
    -------
    pd.DataFrame indexed by Horizon
    """
    timing_rows = []
    for horizon_label, hparams in horizons.items():
        T, n_steps = hparams['T'], hparams['n_steps_mc']
        m = pm.ImprovedSteinStein(**_model_attrs(model), seed=seed)

        # Shared simulation for plain + CV-BS
        sim = m.simulate_prices(S0=S0, T=T, n_steps_mc=n_steps, n_paths=n_paths)

        # Plain LSM
        t0 = time.perf_counter()
        res_p = m.price_american_put(sim, K=K, use_cv=False, ridge=1e-5)
        t_plain = time.perf_counter() - t0

        # CV-BS
        t0 = time.perf_counter()
        res_b = m.price_american_put(sim, K=K, use_cv=True, euro_method='bs', ridge=1e-5)
        t_bs = time.perf_counter() - t0

        # CV-LLH
        sim_llh = m.simulate_prices(S0=S0, T=T, n_steps_mc=n_steps, n_paths=n_paths_llh)
        t0 = time.perf_counter()
        res_l = m.price_american_put(sim_llh, K=K, use_cv=True, euro_method='llh',
                                     ridge=1e-5, **llh_params)
        t_llh = time.perf_counter() - t0

        timing_rows.append({
            'Horizon': horizon_label,
            'Plain LSM (s)':  round(t_plain, 3),
            'Plain LSM SE':   round(res_p['std_err'], 4),
            'CV-BS (s)':      round(t_bs, 3),
            'CV-BS SE':       round(res_b.get('std_err_imp', res_b['std_err']), 4),
            'CV-LLH (s)':     round(t_llh, 3),
            'CV-LLH SE':      round(res_l.get('std_err_imp', res_l['std_err']), 4),
        })

    return pd.DataFrame(timing_rows).set_index('Horizon')
