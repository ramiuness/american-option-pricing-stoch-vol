from dataclasses import dataclass 
import numpy as np
from scipy.stats import norm
from numpy.polynomial.laguerre import lagvander


#############################################################################
########################### European Pricer #################################
#############################################################################
def price_call_bs(S, K=90, tau=1.0, r=0.05, vol=0.2):
    """
    Black-Scholes European call price using the Black-Scholes formula.`
    """
    if tau <= 0:
        return max(0.0, S - K)
    v = vol * np.sqrt(tau)
    d1 = (np.log(S/K) + (r + 0.5*vol*vol)*tau) / v
    d2 = d1 - v
    return float(S * norm.cdf(d1) - K * np.exp(-r*tau) * norm.cdf(d2))

def price_call_mc(paths, K, T, r=None, seed=None):
    """
    Monte Carlo European call pricing 
    K: strike
    r: discount rate (or taken from simulator if None)
    Returns dict with price, std_err, ci_95, n_paths.
    """
    # Discover steps M, initial spot S0, dt
    n_paths = paths.shape[0]
    disc = np.exp(-r * T)

    # Payoff at maturity
    Y = disc * np.maximum(paths[:, -1] - K, 0.0)

    # Statistics
    price = Y.mean()
    std_err = Y.std(ddof=1) / np.sqrt(n_paths) #verify the ci
    ci95 = (price - 1.96*std_err, price + 1.96*std_err)

    return {
        'price': price,
        'std_err': std_err,
        'ci_95': ci95,
        'n_paths': n_paths
    }


def price_put_mc(paths, K, T, r=None):
    """
    Monte Carlo European put pricing.
    K: strike
    r: discount rate
    Returns dict with price, std_err, ci_95, n_paths.
    """
    n_paths = paths.shape[0]
    disc = np.exp(-r * T)
    Y = disc * np.maximum(K - paths[:, -1], 0.0)
    price = Y.mean()
    std_err = Y.std(ddof=1) / np.sqrt(n_paths)
    ci95 = (price - 1.96*std_err, price + 1.96*std_err)
    return {
        'price': price,
        'std_err': std_err,
        'ci_95': ci95,
        'n_paths': n_paths
    }


#############################################################################
####################### American Put Pricer (LSM + CV) ######################
#############################################################################

def _laguerre_basis(x, order=3):
    """
    Laguerre basis L_0..L_order evaluated at x (vector).
    Uses numpy.polynomial.laguerre.lagvander. Returns Phi (n, order+1).
    """
    x = np.asarray(x, dtype=float)
    return lagvander(x, order)


def _ols_fit_predict(Phi, y, Phi_all, ridge=0.0, ridge_eps=1e-12):
    """
    Solve min ||Phi * beta - y||_2 (optionally ridge) and predict on Phi_all.
    Pure NumPy backend. If ridge>0, solves (Phi^T Phi + (ridge+eps) I) beta = Phi^T y.
    Safeguards: if Phi has zero rows, return zeros; if lstsq fails, fall back to tiny ridge.
    """
    Phi = np.asarray(Phi, dtype=float, order="C")
    y = np.asarray(y, dtype=float).ravel()
    Phi_all = np.asarray(Phi_all, dtype=float, order="C")

    if Phi.shape[0] == 0:
        return np.zeros(Phi_all.shape[0], dtype=float)

    n_features = Phi.shape[1]
    if ridge and ridge > 0.0:
        A = Phi.T @ Phi
        b = Phi.T @ y
        reg = float(ridge) + float(ridge_eps)
        beta = np.linalg.solve(A + reg * np.eye(n_features), b)
    else:
        try:
            beta, *_ = np.linalg.lstsq(Phi, y, rcond=None)
        except np.linalg.LinAlgError:
            A = Phi.T @ Phi
            b = Phi.T @ y
            beta = np.linalg.solve(A + ridge_eps * np.eye(n_features), b)
    return Phi_all @ beta


def _theta_lr_grid(B, theta0, lam, eta, dt, n_steps, n_paths):
    """
    Build long-run mean process θ_t on the simulation grid.
    Returns (n_paths, n_steps+1); col 0 = θ0; for j>=1:
      θ[:, j] = θ0 + lam*(j*dt) + eta*(B[:, j-1] - 1).

    Note: B has shape (n_paths, n_steps), so broadcasting B to theta[:, 1:]
    correctly implements theta[:, j] = ... + eta*(B[:, j-1] - 1) for j=1..n_steps.
    """
    theta = np.empty((n_paths, n_steps + 1), dtype=float)
    theta[:, 0] = float(theta0)
    t_index = np.arange(1, n_steps + 1, dtype=float) * dt
    # This broadcast correctly aligns: theta[:, j] uses B[:, j-1]
    theta[:, 1:] = theta0 + lam * t_index[None, :] + eta * (B - 1.0)
    return theta


# ========================= European PUT providers ==========================

def _bs_put_vec(S, K, tau, r, vol):
    """
    Vectorized Black–Scholes PUT price with continuous compounding (no dividends).
    Inputs broadcast along the first dimension; returns 1-D array.
    """
    S = np.asarray(S, dtype=float).ravel()
    vol = np.asarray(vol, dtype=float).ravel()
    tau = float(tau)
    r = float(r)
    if tau <= 0.0:
        return np.maximum(K - S, 0.0)

    sqrt_tau = np.sqrt(tau)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(S / K) + (r + 0.5 * vol**2) * tau) / (vol * sqrt_tau)
        d2 = d1 - vol * sqrt_tau
    N = norm.cdf
    discK = K * np.exp(-r * tau)
    put = discK * N(-d2) - S * N(-d1)
    # handle vol=0 edge
    zero_vol = (vol <= 0.0)
    if np.any(zero_vol):
        intrinsic = np.maximum(discK - S, 0.0)
        put = np.where(zero_vol, intrinsic, put)
    return put


def _euro_put_llh_slice(model, S_col, vol_col, theta_col, tau, K, pre=None):
    """
    European PUT under Lin–Lin–He via parity at a fixed time slice (vectorized):
      Put = Call - S + K*exp(-r*tau), using model.price_call_llh_vec.
    """
    S_col = np.asarray(S_col, dtype=float).reshape(-1)
    vol_col = np.asarray(vol_col, dtype=float).reshape(-1)
    theta_col = np.asarray(theta_col, dtype=float).reshape(-1)
    calls = None
    if hasattr(model, 'price_call_llh_vec'):
        calls = model.price_call_llh_vec(S_col, K, tau, vol_col, theta_col, pre=pre)
    else:
        calls = np.empty_like(S_col, dtype=float)
        for i in range(S_col.shape[0]):
            calls[i] = model.price_call_llh(S_col[i], K, tau, vol_col[i], theta_col[i])
    return calls - S_col + K * np.exp(-model.r * tau)


def _euro_put_bs_slice(model, S_col, vol_col, tau, K):
    """
    European PUT via Black–Scholes proxy at slice (vectorized):
      P_BS(S_j, K, tau_j, r, sigma_hat_j).
    """
    return _bs_put_vec(S_col, K, tau, model.r, vol_col)


def _euro_put_mc1_slice(sim_out, j, r, K):
    """
    European PUT via one-sample MC using terminal payoff on the same path:
      E_j^i ≈ e^{-r(T - t_j)} (K - S_T^i)^+.
    """
    S_T = np.asarray(sim_out['S'][:, -1], dtype=float)
    dt = float(sim_out['dt'])
    n_steps = sim_out['S'].shape[1] - 1
    disc_jT = np.exp(-r * (n_steps - j) * dt)
    return disc_jT * np.maximum(K - S_T, 0.0)


def _euro_put_slice(method, model, sim_out, j, S_col, vol_col, theta_col, tau, K, llh_pre=None):
    """
    Dispatcher for E_j at a slice:
      method ∈ {'llh','bs','mc1'}.
    """
    if method == 'llh':
        return _euro_put_llh_slice(model, S_col, vol_col, theta_col, tau, K, pre=llh_pre)
    elif method == 'bs':
        return _euro_put_bs_slice(model, S_col, vol_col, tau, K)
    elif method == 'mc1':
        return _euro_put_mc1_slice(sim_out, j, model.r, K)
    else:
        raise ValueError("euro_method must be one of {'llh','bs','mc1'}.")


# ===================== Main LSM pricer with CV (PUT) =======================

def price_american_put_lsm_llh(model, sim_out, K, basis_order=3,
                               use_cv=True, improved=True, ridge=0.0,
                               euro_method='llh',
                               phi_max=300.0, n_phi=513, n_steps_rk4=128, eps0=1e-6):
    """
    LSM pricer for an American PUT with optional Rasmussen-style control variates.
    Dynamics: Lin–Lin–He (Improved Stein–Stein); paths come from `sim_out`.

    Parameters
    ----------
    model : ImprovedSteinStein
        The LLH model instance with parameters.
    sim_out : dict
        Simulated paths from model.simulate_prices(), contains 'S', 'sigma_hat', 'B', 'dt'.
    K : float
        Strike price.
    basis_order : int, default=3
        Order of Laguerre polynomial basis for regression.
    use_cv : bool, default=True
        Whether to use control variates (Rasmussen approach).
    improved : bool, default=True
        Whether to compute improved estimator with global control parameter.
    ridge : float, default=0.0
        Ridge regularization parameter for OLS regressions.
    euro_method : {'llh', 'bs', 'mc1'}, default='llh'
        Method for European option pricing:
        - 'llh': Lin-Lin-He formula (accurate but slow)
        - 'bs': Black-Scholes proxy (fast approximation)
        - 'mc1': Same-path Monte Carlo estimate

    **LLH Integration Parameters** (only used if euro_method='llh'):
    phi_max : float, default=300.0
        Maximum frequency for Simpson quadrature. Higher captures tail better.
        Recommended: 200 (FAST), 300 (STANDARD), 400 (HIGH_ACCURACY).
    n_phi : int, default=513
        Number of quadrature points (must be odd). Error ~ O(phi_max/n_phi)^4.
        Recommended: 257 (FAST), 513 (STANDARD), 1025 (HIGH_ACCURACY).
    n_steps_rk4 : int, default=128
        Number of RK4 steps for ODE integration. Error ~ O(tau/n_steps)^5.
        Recommended: 32 (FAST), 128 (STANDARD), 256 (HIGH_ACCURACY).
    eps0 : float, default=1e-6
        Minimum frequency to avoid division by zero in quadrature.

    Returns
    -------
    dict with keys:
        'price' : float
            American put price (LSM estimate).
        'std_err' : float
            Standard error of the estimate.
        'ci_95' : tuple
            95% confidence interval (lower, upper).
        'n_paths' : int
            Number of simulated paths used.
        'price_imp' : float (if improved=True)
            Improved estimate with global control parameter.
        'std_err_imp' : float (if improved=True)
            Standard error of improved estimate.
        'ci_95_imp' : tuple (if improved=True)
            95% CI for improved estimate.

    Notes
    -----
    Defaults (phi_max=300, n_phi=513, n_steps_rk4=128) correspond to STANDARD preset,
    which balances accuracy and speed. For faster testing, use FAST preset
    (phi_max=200, n_phi=257, n_steps_rk4=32). For publication, use HIGH_ACCURACY
    (phi_max=400, n_phi=1025, n_steps_rk4=256).
    """
    S = sim_out['S']
    sigma_hat = sim_out['sigma_hat']
    B = sim_out['B']
    dt = float(sim_out['dt'])
    r = float(model.r)

    n_paths, n_cols = S.shape
    n_steps = n_cols - 1
    disc = np.exp(-r * dt)

    # θ_t aligned with S
    theta_lr = _theta_lr_grid(B, model.theta0, model.lam, model.eta, dt, n_steps, n_paths)

    # Initialize at maturity t_N
    CF = np.maximum(K - S[:, -1], 0.0)
    CV = CF.copy()
    stop_idx = np.full(n_paths, n_steps, dtype=int)

    CF_1 = CV_1 = E_1 = None

    # Precompute all unique τ values (only if using LLH)
    tau_cache = {}
    if euro_method == 'llh':
        for j in range(1, n_steps):
            tauj = (n_steps - j) * dt
            if tauj not in tau_cache:
                tau_cache[tauj] = model.llh_precompute_tau(
                    tauj, phi_max=phi_max, n_phi=n_phi,
                    n_steps_ode=n_steps_rk4, eps0=eps0
                )

    # Backward loop: j = N-1,...,1
    for j in range(n_steps - 1, 0, -1):
        # (i) discount one step
        CF = disc * CF
        CV = disc * CV

        # Node data at t_j
        Sj = S[:, j]
        # FIX: Use sigma_hat[:, j] for forward pricing from t_j
        # sigma_hat[:, j] is volatility for interval [t_j, t_{j+1}]
        volj = sigma_hat[:, j]  # Was: sigma_hat[:, j-1] (incorrect - off by one)
        thetaj = theta_lr[:, j]
        tauj = (n_steps - j) * dt

        # (A) Use cached τ-precompute if using LLH; otherwise None
        pre_j = tau_cache.get(tauj) if euro_method == 'llh' else None

        # (B) E_j: European PUT via selected method (vectorized)
        Ej = _euro_put_slice(euro_method, model, sim_out, j, Sj, volj, thetaj, tauj, K, llh_pre=pre_j)

        # Immediate exercise for PUT
        Ij = np.maximum(K - Sj, 0.0)

        # ITM set and basis on x = S/K
        x = Sj / K
        Phi_all = _laguerre_basis(x, order=basis_order)
        itm = (Ij > 0.0)

        # (ii) estimators (regressions on ITM)
        if itm.any():
            Phi = Phi_all[itm, :]
            tildeV  = _ols_fit_predict(Phi, CF[itm], Phi_all, ridge=ridge)
            tildeVE = _ols_fit_predict(Phi, CV[itm], Phi_all, ridge=ridge)
            aux1    = _ols_fit_predict(Phi, (Ej[itm] * CV[itm]), Phi_all, ridge=ridge)
            aux2    = _ols_fit_predict(Phi, (CV[itm]**2),      Phi_all, ridge=ridge)
        else:
            tildeV = np.zeros_like(Sj)
            tildeVE = np.zeros_like(Sj)
            aux1 = np.zeros_like(Sj)
            aux2 = np.ones_like(Sj)

        # (Rasmussen CV test value)
        if use_cv:
            denom = aux2 - (tildeVE**2)
            theta_cv = np.zeros_like(Sj)
            ok = np.abs(denom) > 1e-12
            theta_cv[ok] = - (aux1[ok] - tildeV[ok]*tildeVE[ok]) / denom[ok]
            TV = tildeV + theta_cv * (tildeVE - Ej)
        else:
            TV = tildeV

        # (iii) exercise decision
        ex_mask = Ij > np.maximum(Ej, TV)
        if np.any(ex_mask):
            CF[ex_mask] = Ij[ex_mask]
            CV[ex_mask] = Ej[ex_mask]
            stop_idx[ex_mask] = j

        # save j==1 slice for final estimator
        if j == 1:
            CF_1 = CF.copy()
            CV_1 = CV.copy()
            E_1  = Ej.copy()

    # Time-0 estimators
    disc1 = np.exp(-r * dt)
    base_samples = disc1 * CF_1
    I0 = max(K - S[0, 0], 0.0)
    price = max(I0, float(base_samples.mean()))
    std_err = base_samples.std(ddof=1) / np.sqrt(n_paths)
    ci95 = (price - 1.96*std_err, price + 1.96*std_err)

    out = {
        'price': price,
        'std_err': std_err,
        'ci_95': ci95,
        'n_paths': int(n_paths)
    }

    if improved:
        # Global θ̂ from OLS slope of CF_1 on CV_1
        X = CV_1 - CV_1.mean()
        Y = CF_1 - CF_1.mean()
        varX = X.var(ddof=1)
        theta_global = 0.0 if varX <= 1e-16 else float((X*Y).sum() / ((n_paths - 1) * varX))

        imp_samples = disc1 * (CF_1 + theta_global * (CV_1 - E_1))
        price_imp = max(I0, float(imp_samples.mean()))
        std_err_imp = imp_samples.std(ddof=1) / np.sqrt(n_paths)
        ci95_imp = (price_imp - 1.96*std_err_imp, price_imp + 1.96*std_err_imp)

        out.update({
            'price_imp': price_imp,
            'std_err_imp': std_err_imp,
            'ci_95_imp': ci95_imp
        })

    return out
