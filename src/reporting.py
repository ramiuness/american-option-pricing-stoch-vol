import numpy as np
import pandas as pd
import priceModels as pm


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
