"""
Calibrate LLH stochastic volatility model to S&P 500 European options via MC simulation.

Usage:
    import calibrate
    spot, r, raw = calibrate.fetch_spx_options()
    df = calibrate.filter_options(raw, spot)
    result = calibrate.calibrate(spot, r, df)
    eval_df = calibrate.report_calibration(result, spot, r, df)
    calibrate.plot_calibration(eval_df, spot)
"""

import numpy as np
import pandas as pd
from math import ceil
from scipy.optimize import differential_evolution, minimize
from scipy.stats import norm
import warnings

import priceModels as pm

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

PARAM_NAMES = ["kappa", "nu", "lam", "eta", "rho", "sigma0", "theta0"]

PARAM_BOUNDS = [
    (0.1, 20.0),    # κ
    (0.01, 2.0),    # ν
    (-1.0, 1.0),    # λ
    (0.01, 2.0),    # η
    (-0.99, 0.99),  # ρ
    (0.05, 1.0),    # σ₀
    (0.01, 1.0),    # θ₀
]


# ──────────────────────────────────────────────────────────────────────────────
# Step 1: Data fetching
# ──────────────────────────────────────────────────────────────────────────────

def fetch_spx_options(ticker="^SPX"):
    """
    Fetch spot price, risk-free rate, and all available option chains.

    Returns
    -------
    spot : float
    r : float  (annualized, continuously compounded)
    options_df : pd.DataFrame
        Columns: strike, tau, mid_price, volume, openInterest, option_type,
                 impliedVolatility, bid, ask
    """
    import yfinance as yf
    from datetime import datetime

    # --- Spot price ---
    tk = yf.Ticker(ticker)
    hist = tk.history(period="1d")
    if hist.empty:
        raise RuntimeError(f"Could not fetch spot price for {ticker}")
    spot = float(hist["Close"].iloc[-1])

    # --- Risk-free rate (13-week T-bill via ^IRX) ---
    try:
        irx = yf.Ticker("^IRX")
        irx_hist = irx.history(period="5d")
        if not irx_hist.empty:
            r_annual = float(irx_hist["Close"].iloc[-1]) / 100.0
        else:
            raise ValueError("empty")
    except Exception:
        r_annual = 0.05  # fallback
        warnings.warn(f"Could not fetch risk-free rate; using fallback r={r_annual}")

    # Convert discount rate to continuously compounded
    r = np.log(1 + r_annual)

    # --- Option chains ---
    today = datetime.now()
    expirations = tk.options  # list of date strings
    rows = []
    for exp_str in expirations:
        chain = tk.option_chain(exp_str)
        exp_date = datetime.strptime(exp_str, "%Y-%m-%d")
        tau = (exp_date - today).days / 365.0
        if tau <= 0:
            continue
        for otype, frame in [("call", chain.calls), ("put", chain.puts)]:
            sub = frame[["strike", "bid", "ask", "volume", "openInterest",
                         "impliedVolatility"]].copy()
            sub["tau"] = tau
            sub["option_type"] = otype
            sub["mid_price"] = (sub["bid"] + sub["ask"]) / 2.0
            rows.append(sub)

    if not rows:
        raise RuntimeError("No option chains returned")
    options_df = pd.concat(rows, ignore_index=True)
    return spot, r, options_df


# ──────────────────────────────────────────────────────────────────────────────
# Step 2: Filtering
# ──────────────────────────────────────────────────────────────────────────────

def filter_options(df, spot,
                   moneyness_range=(0.85, 1.15),
                   tau_range=(14/365, 1.0),
                   min_volume=10,
                   min_open_interest=100,
                   min_bid=0.10,
                   otm_only=True):
    """Filter raw option chain data to liquid, near-the-money contracts.

    Applies sequential filters on bid quality, mid-price positivity,
    volume, open interest, moneyness (K/S), and maturity range.
    When ``otm_only=True`` (default), retains only OTM calls (K > S)
    and OTM puts (K <= S) to avoid redundant ATM/ITM contracts.

    Parameters
    ----------
    df                : DataFrame from fetch_spx_options() with columns
                        strike, bid, mid_price, volume, openInterest,
                        option_type, tau
    spot              : float — current underlying price
    moneyness_range   : tuple (lo, hi) — keep K/S in [lo, hi]
    tau_range         : tuple (lo, hi) — keep maturities in [lo, hi] years
    min_volume        : int — minimum daily volume
    min_open_interest : int — minimum open interest
    min_bid           : float — minimum bid price
    otm_only          : bool — if True, retain only out-of-the-money contracts

    Returns
    -------
    pd.DataFrame — filtered copy of the input, reset-indexed
    """
    df = df.copy()

    # Basic quality filters
    df = df[df["bid"] > min_bid]
    df = df[df["mid_price"] > 0]
    df = df[df["volume"] >= min_volume]
    df = df[df["openInterest"] >= min_open_interest]

    # Moneyness
    m = df["strike"] / spot
    df = df[(m >= moneyness_range[0]) & (m <= moneyness_range[1])]

    # Maturity
    df = df[(df["tau"] >= tau_range[0]) & (df["tau"] <= tau_range[1])]

    # OTM only: calls where K > S, puts where K <= S
    if otm_only:
        is_otm_call = (df["option_type"] == "call") & (df["strike"] > spot)
        is_otm_put = (df["option_type"] == "put") & (df["strike"] <= spot)
        df = df[is_otm_call | is_otm_put]

    df = df.reset_index(drop=True)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Step 3: Pre-generate random draws (CRN)
# ──────────────────────────────────────────────────────────────────────────────

def generate_random_draws(n_paths, n_steps_mc, seed=42):
    """
    Pre-generate the three independent normal arrays for CRN.

    Z1, Z2_indep drive (W1, W2) with Z2 = ρ·Z1 + √(1−ρ²)·Z2_indep reconstructed
    per evaluation. Z_B drives the geometric Brownian B.

    Returns dict with keys 'Z1', 'Z2_indep', 'Z_B', each shape (n_paths, n_steps_mc).
    """
    rng = np.random.default_rng(seed)
    return {
        "Z1": rng.standard_normal((n_paths, n_steps_mc)),
        "Z2_indep": rng.standard_normal((n_paths, n_steps_mc)),
        "Z_B": rng.standard_normal((n_paths, n_steps_mc)),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Step 4: Simulation from pre-generated draws + objective
# ──────────────────────────────────────────────────────────────────────────────

def _simulate_from_draws(params, r, S0, T_max, draws, steps_per_year=252):
    """
    Run one LLH simulation using pre-generated draws.

    Parameters
    ----------
    params : array-like, length 7
        (kappa, nu, lam, eta, rho, sigma0, theta0)
    r : float
    S0 : float
    T_max : float
    draws : dict from generate_random_draws
    steps_per_year : int

    Returns
    -------
    S : ndarray, shape (n_paths, n_steps_mc + 1)
    """
    kappa, nu, lam, eta, rho, sigma0, theta0 = params
    n_steps_mc = max(1, ceil(T_max * steps_per_year))
    dt = T_max / n_steps_mc

    Z1 = draws["Z1"][:, :n_steps_mc]
    Z2_indep = draws["Z2_indep"][:, :n_steps_mc]
    Z_B = draws["Z_B"][:, :n_steps_mc]

    # Correlated Z2 from current ρ
    Z2 = rho * Z1 + np.sqrt(max(0.0, 1.0 - rho**2)) * Z2_indep

    # Brownian motions
    dW1, _ = pm.brownian_from_normals(Z1, dt)
    _, W2 = pm.brownian_from_normals(Z2, dt)
    _, W_B = pm.brownian_from_normals(Z_B, dt)
    B = pm.gbm_from_formula(W_B, dt)

    # Volatility
    sigma_hat = pm.sigma_hat_from_components(
        n_steps_mc, dt, kappa, sigma0, theta0, lam, nu, eta, W2, B
    )

    # Prices
    S_sim = pm.multiplicative_euler_prices(S0, r, sigma_hat, dW1, dt)
    S0_col = np.full((S_sim.shape[0], 1), S0)
    S = np.concatenate([S0_col, S_sim], axis=1)
    return S


def objective(params, r, spot, options_df, draws, steps_per_year=252):
    """
    Relative MSE between MC model prices and market mid-prices.

    Minimizes mean((V_model - V_market)² / V_market²), giving each option
    equal weight in percentage terms. Without this normalization, expensive
    options (ATM, long-dated) dominate the loss and bias the calibration
    away from the vol smile wings.

    Parameters
    ----------
    params : array-like, length 7
    r, spot : float
    options_df : DataFrame with columns strike, tau, mid_price, option_type
    draws : dict from generate_random_draws
    steps_per_year : int

    Returns
    -------
    rel_mse : float  (or 1e10 on failure)
    """
    try:
        T_max = options_df["tau"].max()
        n_steps_mc = max(1, ceil(T_max * steps_per_year))
        dt = T_max / n_steps_mc

        S = _simulate_from_draws(params, r, spot, T_max, draws, steps_per_year)

        rel_errors_sq = []
        for tau_val, grp in options_df.groupby("tau"):
            j = max(1, min(n_steps_mc, round(tau_val / dt)))
            S_T = S[:, j]
            disc = np.exp(-r * tau_val)
            for _, row in grp.iterrows():
                K = row["strike"]
                if row["option_type"] == "call":
                    payoff = np.maximum(S_T - K, 0.0)
                else:
                    payoff = np.maximum(K - S_T, 0.0)
                model_price = disc * payoff.mean()
                rel_errors_sq.append(
                    ((model_price - row["mid_price"]) / row["mid_price"]) ** 2
                )

        rel_mse = np.mean(rel_errors_sq)
        if not np.isfinite(rel_mse):
            return 1e10
        return float(rel_mse)

    except Exception:
        return 1e10


def _objective_vec(params, r, spot, options_df, draws, steps_per_year=252):
    """
    Relative MSE objective with pre-extracted numpy arrays and cached S_T slices.

    Faster than objective() (avoids pandas iterrows/groupby overhead) but still
    loops over options in Python. Each iteration is vectorized across paths.
    """
    try:
        T_max = options_df["tau"].max()
        n_steps_mc = max(1, ceil(T_max * steps_per_year))
        dt = T_max / n_steps_mc

        S = _simulate_from_draws(params, r, spot, T_max, draws, steps_per_year)

        strikes = options_df["strike"].values
        taus = options_df["tau"].values
        mids = options_df["mid_price"].values
        is_call = (options_df["option_type"] == "call").values

        # Time indices
        j_indices = np.clip(np.round(taus / dt).astype(int), 1, n_steps_mc)

        # Unique time steps to avoid redundant slicing
        unique_j = np.unique(j_indices)
        S_T_map = {j: S[:, j] for j in unique_j}

        model_prices = np.empty(len(options_df))
        for i in range(len(options_df)):
            S_T = S_T_map[j_indices[i]]
            K = strikes[i]
            if is_call[i]:
                payoff = np.maximum(S_T - K, 0.0)
            else:
                payoff = np.maximum(K - S_T, 0.0)
            disc = np.exp(-r * taus[i])
            model_prices[i] = disc * payoff.mean()

        rel_mse = np.mean(((model_prices - mids) / mids) ** 2)
        if not np.isfinite(rel_mse):
            return 1e10
        return float(rel_mse)

    except Exception:
        return 1e10


# ──────────────────────────────────────────────────────────────────────────────
# Step 5: Calibration (DE + L-BFGS-B)
# ──────────────────────────────────────────────────────────────────────────────

def calibrate(spot, r, options_df,
              n_paths=10_000,
              steps_per_year=252,
              seed=42,
              de_maxiter=50,
              de_popsize=15,
              de_tol=1e-6,
              lbfgsb_maxiter=200,
              verbose=True):
    """
    Two-stage calibration: Differential Evolution → L-BFGS-B.

    Parameters
    ----------
    spot, r : float
    options_df : DataFrame (filtered)
    n_paths : int
    steps_per_year : int
    seed : int
    de_maxiter, de_popsize, de_tol : DE parameters
    lbfgsb_maxiter : int
    verbose : bool

    Returns
    -------
    dict with keys: params (dict), params_array, mse_de, mse_final,
                    de_result, lbfgsb_result, n_options, draws
    """
    T_max = options_df["tau"].max()
    n_steps_mc = max(1, ceil(T_max * steps_per_year))

    if verbose:
        print(f"Calibrating: {len(options_df)} options, T_max={T_max:.4f}yr, "
              f"n_steps_mc={n_steps_mc}, n_paths={n_paths}")

    draws = generate_random_draws(n_paths, n_steps_mc, seed=seed)

    obj_fn = lambda p: _objective_vec(p, r, spot, options_df, draws, steps_per_year)

    # --- Stage A: Differential Evolution ---
    gen_count = [0]
    def callback(xk, convergence):
        gen_count[0] += 1
        if verbose and gen_count[0] % 10 == 0:
            rel_mse = obj_fn(xk)
            print(f"  DE gen {gen_count[0]:3d}: RelMSE = {rel_mse:.6f}")

    if verbose:
        print("Stage A: Differential Evolution...")
    de_result = differential_evolution(
        obj_fn, bounds=PARAM_BOUNDS,
        maxiter=de_maxiter, popsize=de_popsize, tol=de_tol,
        seed=seed, callback=callback, polish=False
    )
    rel_mse_de = de_result.fun
    if verbose:
        print(f"  DE done: RelMSE = {rel_mse_de:.6f}")

    # --- Stage B: L-BFGS-B refinement ---
    if verbose:
        print("Stage B: L-BFGS-B refinement...")
    lb_result = minimize(
        obj_fn, x0=de_result.x, method="L-BFGS-B",
        bounds=PARAM_BOUNDS,
        options={"maxiter": lbfgsb_maxiter, "ftol": 1e-10}
    )
    rel_mse_final = lb_result.fun
    if verbose:
        print(f"  L-BFGS-B done: RelMSE = {rel_mse_final:.6f}")

    best = lb_result.x
    params_dict = dict(zip(PARAM_NAMES, best))
    if verbose:
        print("\nCalibrated parameters:")
        for name, val in params_dict.items():
            print(f"  {name:8s} = {val:.6f}")
        print(f"\nFinal RelMSE = {rel_mse_final:.6f}  "
              f"(RRMSE = {np.sqrt(rel_mse_final)*100:.2f}%)")

    return {
        "params": params_dict,
        "params_array": best,
        "rel_mse_de": rel_mse_de,
        "rel_mse_final": rel_mse_final,
        "de_result": de_result,
        "lbfgsb_result": lb_result,
        "n_options": len(options_df),
        "draws": draws,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Step 6: Reporting & Visualization
# ──────────────────────────────────────────────────────────────────────────────

def implied_vol_bisect(price, S, K, tau, r, option_type, tol=1e-6, max_iter=100):
    """
    Invert Black-Scholes to find implied vol via bisection.

    Returns implied vol or NaN if not found.
    """
    if tau <= 0 or price <= 0:
        return np.nan
    lo, hi = 1e-4, 5.0
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        d1 = (np.log(S / K) + (r + 0.5 * mid**2) * tau) / (mid * np.sqrt(tau))
        d2 = d1 - mid * np.sqrt(tau)
        if option_type == "call":
            bs = S * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)
        else:
            bs = K * np.exp(-r * tau) * norm.cdf(-d2) - S * norm.cdf(-d1)
        if bs > price:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            return (lo + hi) / 2.0
    return np.nan


def report_calibration(result, spot, r, options_df, n_paths=50_000,
                       steps_per_year=252, seed=123):
    """
    Re-evaluate calibrated model with more paths and build evaluation DataFrame.

    Returns DataFrame with: strike, tau, option_type, market_price, model_price,
                            error, pct_error, market_iv, model_iv
    """
    params = result["params_array"]
    T_max = options_df["tau"].max()
    n_steps_mc = max(1, ceil(T_max * steps_per_year))
    dt = T_max / n_steps_mc

    draws = generate_random_draws(n_paths, n_steps_mc, seed=seed)
    S = _simulate_from_draws(params, r, spot, T_max, draws, steps_per_year)

    records = []
    for _, row in options_df.iterrows():
        K = row["strike"]
        tau_val = row["tau"]
        otype = row["option_type"]
        mkt = row["mid_price"]

        j = max(1, min(n_steps_mc, round(tau_val / dt)))
        S_T = S[:, j]
        disc = np.exp(-r * tau_val)
        if otype == "call":
            model_price = disc * np.maximum(S_T - K, 0.0).mean()
        else:
            model_price = disc * np.maximum(K - S_T, 0.0).mean()

        err = model_price - mkt
        pct_err = err / mkt * 100 if mkt > 0 else np.nan

        mkt_iv = implied_vol_bisect(mkt, spot, K, tau_val, r, otype)
        mod_iv = implied_vol_bisect(model_price, spot, K, tau_val, r, otype)

        records.append({
            "strike": K,
            "tau": tau_val,
            "option_type": otype,
            "moneyness": K / spot,
            "market_price": mkt,
            "model_price": model_price,
            "error": err,
            "pct_error": pct_err,
            "market_iv": mkt_iv,
            "model_iv": mod_iv,
        })

    eval_df = pd.DataFrame(records)
    rmse = np.sqrt((eval_df["error"] ** 2).mean())
    mae = eval_df["error"].abs().mean()
    mape = eval_df["pct_error"].abs().mean()
    print(f"Evaluation ({n_paths:,} paths): RMSE=${rmse:.2f}, MAE=${mae:.2f}, MAPE={mape:.1f}%")
    return eval_df


def plot_calibration(eval_df, spot):
    """
    Four-panel calibration diagnostic plots.

    Panels distinguish calls (blue) vs puts (red) to reveal systematic
    mispricing by option type. Error axes show relative error (%).
    """
    import matplotlib.pyplot as plt

    calls = eval_df[eval_df["option_type"] == "call"]
    puts = eval_df[eval_df["option_type"] == "put"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Model vs Market scatter — colored by option type
    ax = axes[0, 0]
    ax.scatter(calls["market_price"], calls["model_price"],
               alpha=0.5, s=15, c="tab:blue", label="Calls")
    ax.scatter(puts["market_price"], puts["model_price"],
               alpha=0.5, s=15, c="tab:red", label="Puts")
    lims = [0, max(eval_df["market_price"].max(), eval_df["model_price"].max()) * 1.05]
    ax.plot(lims, lims, "k--", lw=0.8)
    ax.set_xlabel("Market price ($)")
    ax.set_ylabel("Model price ($)")
    ax.set_title("Model vs Market")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.legend(fontsize=9)

    # 2. Relative error (%) vs Moneyness — colored by option type
    ax = axes[0, 1]
    ax.scatter(calls["moneyness"], calls["pct_error"], alpha=0.5, s=15,
               c="tab:blue", label="Calls")
    ax.scatter(puts["moneyness"], puts["pct_error"], alpha=0.5, s=15,
               c="tab:red", label="Puts")
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("Moneyness (K/S)")
    ax.set_ylabel("Relative error (%)")
    ax.set_title("Relative pricing error vs Moneyness")
    ax.legend(fontsize=9)

    # 3. Relative error (%) vs Maturity — colored by option type
    ax = axes[1, 0]
    ax.scatter(calls["tau"], calls["pct_error"], alpha=0.5, s=15,
               c="tab:blue", label="Calls")
    ax.scatter(puts["tau"], puts["pct_error"], alpha=0.5, s=15,
               c="tab:red", label="Puts")
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("Time to maturity (years)")
    ax.set_ylabel("Relative error (%)")
    ax.set_title("Relative pricing error vs Maturity")
    ax.legend(fontsize=9)

    # 4. Implied vol smile (market vs model) — select representative maturities
    ax = axes[1, 1]
    valid = eval_df.dropna(subset=["market_iv", "model_iv"])
    all_taus = sorted(valid["tau"].unique())
    # Pick up to 6 evenly spaced maturities to keep the plot readable
    if len(all_taus) > 6:
        idx = np.linspace(0, len(all_taus) - 1, 6, dtype=int)
        sel_taus = [all_taus[i] for i in idx]
    else:
        sel_taus = all_taus
    colors = plt.cm.viridis(np.linspace(0, 1, max(1, len(sel_taus))))
    for i, tau_val in enumerate(sel_taus):
        sub = valid[valid["tau"] == tau_val].sort_values("moneyness")
        label = f"τ={tau_val:.2f}"
        ax.plot(sub["moneyness"], sub["market_iv"], "o-",
                color=colors[i], ms=3, lw=0.8, alpha=0.7, label=f"{label} mkt")
        ax.plot(sub["moneyness"], sub["model_iv"], "x--",
                color=colors[i], ms=4, lw=0.8, alpha=0.7, label=f"{label} mdl")
    ax.set_xlabel("Moneyness (K/S)")
    ax.set_ylabel("Implied volatility")
    ax.set_title("IV smile: market (o) vs model (x)")
    ax.legend(fontsize=7, ncol=2, loc="best")

    fig.suptitle(f"LLH Calibration — S={spot:.0f}", fontsize=14)
    fig.tight_layout()
    plt.show()
