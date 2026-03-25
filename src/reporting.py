import time
import numpy as np
import pandas as pd
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
