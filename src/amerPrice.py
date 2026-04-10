"""
American put pricing via LSM with optional Rasmussen control variates.

Implements Longstaff-Schwartz (2001) with the functional control-variate
extension of Rasmussen (2005) under the Lin-Lin-He (2024) improved
Stein-Stein stochastic volatility model. See american_pricing_report.pdf
section 3.2.2 for the mathematical formulation.
"""

from dataclasses import dataclass
import numpy as np
from scipy.special import ndtr
from numpy.polynomial.laguerre import lagvander

# Re-exports for backward compatibility (canonical home: priceModels)
from priceModels import price_call_mc, price_put_mc, price_call_bs  # noqa: F401


#############################################################################
####################### American Put Pricer (LSM + CV) ######################
#############################################################################

def _laguerre_basis(x, order=3):
    """Laguerre polynomial basis L_0..L_order. Returns (n, order+1)."""
    x = np.asarray(x, dtype=float)
    return lagvander(x, order)


def _gaussian_rbf_basis(x, centers, bandwidth):
    """Gaussian RBF basis. Returns (n, n_centers): Phi[i,k] = exp(-|x_i - c_k|^2 / (2h^2))."""
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    c = np.asarray(centers, dtype=float).reshape(1, -1)
    return np.exp(-((x - c) ** 2) / (2 * bandwidth ** 2))


def _compute_rbf_params(x_itm, n_centers):
    """Quantile-spaced RBF centers and median-spacing bandwidth from ITM spots."""
    n = len(x_itm)
    if n < 2:
        return np.array([x_itm.mean() if n > 0 else 1.0]), 1.0
    centers = np.unique(np.quantile(x_itm, np.linspace(0, 1, n_centers)))
    spacings = np.diff(centers)
    pos = spacings[spacings > 0]
    bandwidth = np.median(pos) if len(pos) > 0 else np.ptp(x_itm)
    return centers, max(bandwidth, 0.01 * np.ptp(x_itm))


_RIDGE_EPS = 1e-12


def _ols_fit_predict(Phi, y, Phi_all, ridge=0.0):
    """Ridge regression fit on (Phi, y) with prediction on Phi_all. Returns (n_all,)."""
    Phi = np.asarray(Phi, dtype=float, order="C")
    y = np.asarray(y, dtype=float).ravel()
    Phi_all = np.asarray(Phi_all, dtype=float, order="C")

    if Phi.shape[0] == 0:
        return np.zeros(Phi_all.shape[0], dtype=float)

    n_features = Phi.shape[1]
    A = Phi.T @ Phi
    b = Phi.T @ y
    reg = max(float(ridge), _RIDGE_EPS)
    try:
        beta = np.linalg.solve(A + reg * np.eye(n_features), b)
    except np.linalg.LinAlgError:
        beta, *_ = np.linalg.lstsq(A + reg * np.eye(n_features), b, rcond=None)
    return Phi_all @ beta


def _ols_fit_predict_multi(Phi, Y_mat, Phi_all, ridge=0.0):
    """Multi-target ridge regression with shared Gram factorization.

    Parameters
    ----------
    Phi     : (n_itm, p) design matrix on ITM paths.
    Y_mat   : (n_itm, k) response columns.
    Phi_all : (n_paths, p) design matrix for prediction.
    ridge   : L2 regularization strength.

    Returns
    -------
    ndarray, (n_paths, k) — predicted values for each target.
    """
    if Phi.shape[0] == 0:
        return np.zeros((Phi_all.shape[0], Y_mat.shape[1]), dtype=float)

    n_features = Phi.shape[1]
    A = Phi.T @ Phi
    B_rhs = Phi.T @ Y_mat

    reg = max(float(ridge), _RIDGE_EPS)
    try:
        Beta = np.linalg.solve(A + reg * np.eye(n_features), B_rhs)
    except np.linalg.LinAlgError:
        Beta, *_ = np.linalg.lstsq(A + reg * np.eye(n_features), B_rhs, rcond=None)
    return Phi_all @ Beta


def _theta_lr_grid(B, theta0, lam, eta, dt, n_steps, n_paths):
    """Long-run mean theta_t on the simulation grid. Returns (n_paths, n_steps+1)."""
    theta = np.empty((n_paths, n_steps + 1), dtype=float)
    theta[:, 0] = float(theta0)
    t_index = np.arange(1, n_steps + 1, dtype=float) * dt
    theta[:, 1:] = theta0 + lam * t_index[None, :] + eta * (B - 1.0)
    return theta


# ========================= European PUT providers ==========================

def _bs_put_vec(S, K, tau, r, vol):
    """Vectorized Black-Scholes put price (continuous compounding, no dividends). Returns (n,)."""
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
    N = ndtr
    discK = K * np.exp(-r * tau)
    put = discK * N(-d2) - S * N(-d1)
    zero_vol = (vol <= 0.0)
    if np.any(zero_vol):
        intrinsic = np.maximum(discK - S, 0.0)
        put = np.where(zero_vol, intrinsic, put)
    return put


def _euro_put_llh_slice(model, S_col, vol_col, theta_col, tau, K, pre=None):
    """European put under LLH via put-call parity at a single time slice. Returns (n,)."""
    S_col = np.asarray(S_col, dtype=float).reshape(-1)
    vol_col = np.asarray(vol_col, dtype=float).reshape(-1)
    theta_col = np.asarray(theta_col, dtype=float).reshape(-1)
    calls = model.price_call_llh(S_col, K, tau, vol_col, theta_col, pre=pre)
    return calls - S_col + K * np.exp(-model.r * tau)


def _euro_put_bs_slice(model, S_col, vol_col, tau, K):
    """European put via Black-Scholes proxy using instantaneous vol at a time slice."""
    return _bs_put_vec(S_col, K, tau, model.r, vol_col)


def _euro_put_mc1_slice(sim_out, j, r, K):
    """European put via single-path MC: E_j^i = e^{-r(T-t_j)} (K - S_T^i)^+."""
    S_T = np.asarray(sim_out['S'][:, -1], dtype=float)
    dt = float(sim_out['dt'])
    n_steps = sim_out['S'].shape[1] - 1
    disc_jT = np.exp(-r * (n_steps - j) * dt)
    return disc_jT * np.maximum(K - S_T, 0.0)


def _euro_put_slice(method, model, sim_out, j, S_col, vol_col, theta_col, tau, K, llh_pre=None):
    """Dispatch European put valuation at time slice j. method in {'llh', 'bs', 'mc1'}."""
    if method == 'llh':
        return _euro_put_llh_slice(model, S_col, vol_col, theta_col, tau, K, pre=llh_pre)
    elif method == 'bs':
        return _euro_put_bs_slice(model, S_col, vol_col, tau, K)
    elif method == 'mc1':
        return _euro_put_mc1_slice(sim_out, j, model.r, K)
    else:
        raise ValueError("euro_method must be one of {'llh','bs','mc1'}.")


# ===================== Shared European precomputation =======================

@dataclass
class PrecomputedEuropean:
    """Shared European pricing data for both plain LSM and CV-LLH.

    Attributes
    ----------
    tau_cache : dict[int, object]
        ODE coefficients keyed by steps_remaining (1..M-1).
    theta_lr : ndarray, (n_paths, n_steps+1)
        Long-run mean process grid.
    euro_grid : dict[int, ndarray]
        European put prices at each step j, shape (n_paths,).
        j > 1: computed for ITM paths only, zeros for OTM.
        j == 1: computed for ALL paths (needed by CV-LLH global correction).
    """
    tau_cache: dict
    theta_lr: np.ndarray
    euro_grid: dict


def precompute_european(model, sim_out, K,
                        phi_max=300.0, n_phi=513, n_steps_rk4=256, eps0=1e-6):
    """Precompute European put prices shared by plain LSM and CV-LLH.

    Builds ODE coefficients, theta_lr grid, and European put prices at every
    exercise date. The result can be passed to price_american_put_lsm_llh via
    the precomputed parameter to avoid redundant computation.

    Parameters
    ----------
    model    : ImprovedSteinStein instance
    sim_out  : dict from model.simulate_prices()
    K        : strike
    phi_max, n_phi, n_steps_rk4, eps0 : LLH quadrature params

    Returns
    -------
    PrecomputedEuropean
    """
    S = sim_out['S']
    sigma_hat = sim_out['sigma_hat']
    B = sim_out['B']
    dt = float(sim_out['dt'])

    n_paths, n_cols = S.shape
    n_steps = n_cols - 1

    # 1. Long-run mean grid
    theta_lr = _theta_lr_grid(B, model.theta0, model.lam, model.eta,
                              dt, n_steps, n_paths)

    # 2. ODE coefficients for each unique remaining maturity
    tau_cache = {}
    for j in range(1, n_steps):
        steps_remaining = n_steps - j
        if steps_remaining not in tau_cache:
            tau_cache[steps_remaining] = model.llh_precompute_tau(
                steps_remaining * dt, phi_max=phi_max, n_phi=n_phi,
                n_steps_ode=n_steps_rk4, eps0=eps0
            )

    # 3. European put prices at each exercise date
    euro_grid = {}
    for j in range(n_steps - 1, 0, -1):
        Sj = S[:, j]
        volj = sigma_hat[:, j]
        thetaj = theta_lr[:, j]
        tauj = (n_steps - j) * dt
        steps_remaining = n_steps - j
        pre_j = tau_cache.get(steps_remaining)

        if j == 1:
            # j=1: all paths (CV-LLH global correction needs this)
            euro_grid[j] = _euro_put_llh_slice(model, Sj, volj, thetaj,
                                               tauj, K, pre=pre_j)
        else:
            # j>1: ITM paths only
            Ij = np.maximum(K - Sj, 0.0)
            itm = (Ij > 0.0)
            Ej = np.zeros(n_paths, dtype=float)
            if itm.any():
                thetaj_itm = thetaj[itm]
                Ej[itm] = _euro_put_llh_slice(model, Sj[itm], volj[itm],
                                              thetaj_itm, tauj, K, pre=pre_j)
            euro_grid[j] = Ej

    return PrecomputedEuropean(tau_cache=tau_cache, theta_lr=theta_lr,
                               euro_grid=euro_grid)


# ===================== LSM pricer internals ================================

def _setup(model, sim_out, K, use_cv, euro_method, floor_method,
           phi_max, n_phi, n_steps_rk4, eps0, precomputed=None):
    """Initialize LSM state: arrays, theta grid, maturity payoffs, and ODE cache.

    When precomputed is provided, tau_cache and theta_lr are taken from it,
    skipping ODE and theta grid construction. Otherwise computes internally.

    Returns
    -------
    dict
        Shared state consumed by _backward_loop, _base_estimator,
        and _improved_estimator.
    """
    S = sim_out['S']
    sigma_hat = sim_out['sigma_hat']
    B = sim_out['B']
    dt = float(sim_out['dt'])
    r = float(model.r)

    n_paths, n_cols = S.shape
    n_steps = n_cols - 1
    disc = np.exp(-r * dt)

    if precomputed is not None:
        theta_lr = precomputed.theta_lr
        tau_cache = precomputed.tau_cache
    else:
        # Step 1: theta_t grid (needed when LLH is used for CV or floor)
        need_theta = (use_cv and euro_method == 'llh') or (not use_cv and floor_method == 'llh')
        theta_lr = _theta_lr_grid(B, model.theta0, model.lam, model.eta,
                                  dt, n_steps, n_paths) if need_theta else None

        # Precompute ODE coefficients for all unique remaining maturities
        tau_cache = {}
        if (use_cv and euro_method == 'llh') or (not use_cv and floor_method == 'llh'):
            for j in range(1, n_steps):
                steps_remaining = n_steps - j
                if steps_remaining not in tau_cache:
                    tau_cache[steps_remaining] = model.llh_precompute_tau(
                        steps_remaining * dt, phi_max=phi_max, n_phi=n_phi,
                        n_steps_ode=n_steps_rk4, eps0=eps0
                    )

    # Initialize at maturity t_N
    CF = np.maximum(K - S[:, -1], 0.0)
    CV = CF.copy() if use_cv else None

    return {
        'S': S, 'sigma_hat': sigma_hat, 'theta_lr': theta_lr,
        'CF': CF, 'CV': CV, 'tau_cache': tau_cache,
        'dt': dt, 'r': r, 'disc': disc,
        'n_paths': n_paths, 'n_steps': n_steps,
        '_Ij': np.empty(n_paths, dtype=float),
        '_x': np.empty(n_paths, dtype=float),
    }


def _backward_loop(state, model, sim_out, K, use_cv,
                   basis_type, basis_order, ridge, euro_method, floor_method,
                   precomputed=None):
    """Backward induction from j = N-1 to j = 1.

    At each exercise date, fits continuation-value regressions, computes
    the optimal exercise decision (with or without control variates), and
    updates the cash-flow and control-variate vectors in place.

    When precomputed is provided, European prices are read from
    precomputed.euro_grid[j] instead of being computed on the fly.

    Returns
    -------
    CF_1 : ndarray, (n_paths,)
        Discounted cash-flow snapshot at j=1.
    CV_1 : ndarray or None
        European control-variate snapshot at j=1 (None when use_cv=False).
    """
    S         = state['S']
    sigma_hat = state['sigma_hat']
    theta_lr  = state['theta_lr']
    CF        = state['CF']
    CV        = state['CV']
    tau_cache = state['tau_cache']
    disc      = state['disc']
    n_paths   = state['n_paths']
    n_steps   = state['n_steps']
    dt        = state['dt']
    _Ij       = state['_Ij']
    _x        = state['_x']

    euro_grid = precomputed.euro_grid if precomputed is not None else None

    CF_1 = CV_1 = None

    for j in range(n_steps - 1, 0, -1):
        # Step 3(i): discount one step
        CF *= disc
        if use_cv:
            CV *= disc

        # Node data at t_j
        Sj = S[:, j]
        volj = sigma_hat[:, j]
        thetaj = theta_lr[:, j] if theta_lr is not None else None
        tauj = (n_steps - j) * dt

        # Intrinsic value I_j = max(K - S_j, 0) for PUT
        np.subtract(K, Sj, out=_Ij)
        np.maximum(_Ij, 0.0, out=_Ij)
        Ij = _Ij
        itm = (Ij > 0.0)

        # Basis on x = S/K
        np.divide(Sj, K, out=_x)
        if basis_type == 'laguerre':
            Phi_all = _laguerre_basis(_x, order=basis_order)
        elif basis_type == 'gaussian':
            x_itm_vals = _x[itm] if itm.any() else _x
            centers, bw = _compute_rbf_params(x_itm_vals, basis_order)
            Phi_all = _gaussian_rbf_basis(_x, centers, bw)
        else:
            raise ValueError(f"Unknown basis_type: '{basis_type}'")

        if use_cv:
            # Step 3(ii): European PUT E_j
            if euro_grid is not None:
                Ej = euro_grid[j]
            else:
                steps_remaining = n_steps - j
                pre_j = tau_cache.get(steps_remaining) if euro_method == 'llh' else None
                if j == 1 or euro_method == 'mc1':
                    Ej = _euro_put_slice(euro_method, model, sim_out, j,
                                         Sj, volj, thetaj, tauj, K, llh_pre=pre_j)
                else:
                    Ej = np.zeros(n_paths, dtype=float)
                    if itm.any():
                        thetaj_itm = thetaj[itm] if thetaj is not None else None
                        Ej[itm] = _euro_put_slice(
                            euro_method, model, sim_out, j,
                            Sj[itm], volj[itm], thetaj_itm, tauj, K, llh_pre=pre_j
                        )

            # Step 3(ii-iii): 4 regressions with shared Gram matrix
            #   col 0: CF      → tildeV   (continuation value)
            #   col 1: CV      → tildeV_E (European continuation)
            #   col 2: CF·CV   → aux_CCV  (CV^{aux1})
            #   col 3: CV²     → aux_CV2  (CV^{aux2})
            if itm.any():
                Phi = Phi_all[itm, :]
                CF_itm = CF[itm]
                CV_itm = CV[itm]
                Y_mat = np.column_stack([CF_itm, CV_itm, CF_itm * CV_itm, CV_itm**2])
                preds = _ols_fit_predict_multi(Phi, Y_mat, Phi_all, ridge=ridge)
                tildeV, tildeV_E, aux_CCV, aux_CV2 = (
                    preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3])
            else:
                tildeV   = np.zeros_like(Sj)
                tildeV_E = np.zeros_like(Sj)
                aux_CCV  = np.zeros_like(Sj)
                aux_CV2  = np.ones_like(Sj)

            # Step 3(iii): per-path CV coefficient
            #   θ̂^i = -(aux_CCV - tildeV·tildeV_E) / (aux_CV2 - tildeV_E²)
            denom = aux_CV2 - tildeV_E**2
            theta_cv = np.zeros_like(Sj)
            ok = np.abs(denom) > 1e-12
            theta_cv[ok] = -(aux_CCV[ok] - tildeV[ok] * tildeV_E[ok]) / denom[ok]

            # Test value and exercise decision
            TV = tildeV + theta_cv * (tildeV_E - Ej)
            ex_mask = Ij > np.maximum(Ej, TV)
            if np.any(ex_mask):
                CF[ex_mask] = Ij[ex_mask]
                CV[ex_mask] = Ej[ex_mask]

            if j == 1:
                CF_1 = CF.copy()
                CV_1 = CV.copy()

        else:
            # Plain LSM: 1 regression + European floor
            if euro_grid is not None:
                Ej = euro_grid[j]
            elif floor_method == 'none':
                Ej = np.zeros(n_paths, dtype=float)
            elif floor_method in ('bs', 'mc1'):
                Ej = _euro_put_slice(floor_method, model, sim_out, j,
                                     Sj, volj, thetaj, tauj, K)
            else:
                steps_remaining = n_steps - j
                pre_j = tau_cache.get(steps_remaining)
                Ej = np.zeros(n_paths, dtype=float)
                if itm.any():
                    thetaj_itm = thetaj[itm] if thetaj is not None else None
                    Ej[itm] = _euro_put_slice(
                        floor_method, model, sim_out, j,
                        Sj[itm], volj[itm], thetaj_itm, tauj, K, llh_pre=pre_j)

            if itm.any():
                Phi = Phi_all[itm, :]
                tildeV = _ols_fit_predict(Phi, CF[itm], Phi_all, ridge=ridge)
            else:
                tildeV = np.zeros_like(Sj)

            ex_mask = Ij > np.maximum(Ej, tildeV)
            if np.any(ex_mask):
                CF[ex_mask] = Ij[ex_mask]

            if j == 1:
                CF_1 = CF.copy()

    return CF_1, CV_1


def _base_estimator(state, CF_1, K):
    """Compute the base (plain LSM) price estimate: max(I_0, mean(disc * CF_1)).

    Returns
    -------
    dict
        price, std_err, ci_95, n_paths.
    """
    disc    = state['disc']
    n_paths = state['n_paths']
    S0      = float(state['S'][0, 0])

    base_samples = disc * CF_1
    I0 = max(K - S0, 0.0)
    price = max(I0, float(base_samples.mean()))
    std_err = base_samples.std(ddof=1) / np.sqrt(n_paths)
    ci95 = (price - 1.96 * std_err, price + 1.96 * std_err)

    return {
        'price': price,
        'std_err': std_err,
        'ci_95': ci95,
        'n_paths': int(n_paths),
    }


def _improved_estimator(state, CF_1, CV_1, model, K,
                        euro_method, phi_max, n_phi, n_steps_rk4, eps0, sim_out):
    """Compute the global control-variate (improved) price estimate.

    Evaluates the European put E_0 = V^E(S_0, K, T), estimates the global
    CV coefficient theta via sample covariance, and forms the variance-reduced
    samples: disc * (CF_1 - theta * (CV_1 - E_0)).

    Returns
    -------
    dict
        price_imp, std_err_imp, ci_95_imp.
    """
    disc    = state['disc']
    n_paths = state['n_paths']
    n_steps = state['n_steps']
    dt      = state['dt']
    S0      = float(state['S'][0, 0])
    I0      = max(K - S0, 0.0)

    # Compute E_0 = V^E(S_0, K, T)
    T_total = n_steps * dt
    if euro_method == 'llh':
        pre_0 = model.llh_precompute_tau(
            T_total, phi_max=phi_max, n_phi=n_phi,
            n_steps_ode=n_steps_rk4, eps0=eps0)
        E_0 = _euro_put_llh_slice(
            model, np.array([S0]), np.array([model.sigma0]),
            np.array([model.theta0]), T_total, K, pre=pre_0).item()
    elif euro_method == 'bs':
        E_0 = _bs_put_vec(
            np.array([S0]), K, T_total, model.r,
            np.array([model.sigma0])).item()
    elif euro_method == 'mc1':
        E_0 = float(_euro_put_mc1_slice(sim_out, 0, model.r, K).mean())

    # Global CV coefficient: theta = Cov(CF_1, CV_1) / Var(CV_1)
    X = CV_1 - CV_1.mean()
    Y = CF_1 - CF_1.mean()
    varX = X.var(ddof=1)
    theta_global = 0.0 if varX <= 1e-16 else float(
        (X * Y).sum() / ((n_paths - 1) * varX))

    # Theoretical VR: 1/(1 - rho^2) where rho^2 = Cov^2 / (Var_CF * Var_CV)
    varY = Y.var(ddof=1)
    covXY = (X * Y).sum() / (n_paths - 1)
    rho_sq = 0.0 if (varX <= 1e-16 or varY <= 1e-16) else (covXY**2) / (varX * varY)
    vr = 1.0 / (1.0 - rho_sq) if rho_sq < 1.0 else np.inf

    imp_samples = disc * (CF_1 - theta_global * (CV_1 - E_0))
    price_imp = max(I0, float(imp_samples.mean()))
    std_err_imp = imp_samples.std(ddof=1) / np.sqrt(n_paths)
    ci95_imp = (price_imp - 1.96 * std_err_imp,
                price_imp + 1.96 * std_err_imp)

    return {
        'price_imp': price_imp,
        'std_err_imp': std_err_imp,
        'ci_95_imp': ci95_imp,
        'rho_squared': rho_sq,
        'vr': vr,
    }


# ===================== Public API ==========================================

def price_american_put_lsm_llh(model, sim_out, K, basis_order=None,
                               basis_type='laguerre',
                               use_cv=True, improved=True, ridge=1e-5,
                               euro_method='llh', floor_method='llh',
                               phi_max=300.0, n_phi=513, n_steps_rk4=256, eps0=1e-6,
                               precomputed=None):
    """
    LSM American put pricer with optional Rasmussen control variates.

    Parameters
    ----------
    model        : ImprovedSteinStein instance
    sim_out      : dict from model.simulate_prices() — keys 'S', 'sigma_hat', 'B', 'dt'
    K            : strike
    basis_order  : None (auto) or int. Default: Laguerre=2 (3 functions), Gaussian=5.
    use_cv       : if False, runs plain LSM with European exercise floor
    improved     : if True and use_cv=True, adds global CV estimator
    euro_method  : 'llh' | 'bs' | 'mc1' — European pricing for CV (only when use_cv=True)
    floor_method : 'bs' | 'llh' — European floor for exercise safety when use_cv=False.
                   Default 'bs' is fast; 'llh' is accurate but ~100x slower.
    phi_max, n_phi, n_steps_rk4, eps0 : LLH quadrature params (only when euro_method='llh')
    precomputed  : PrecomputedEuropean or None — shared European grid from
                   precompute_european(). Skips ODE and European pricing when provided.

    Returns
    -------
    dict:
      price, std_err, ci_95, n_paths
      + price_imp, std_err_imp, ci_95_imp  (when use_cv=True and improved=True)

    Notation
    --------
    CF      = discounted eventual exercise (cash flow)
    CV      = European eventual exercise (control variate)
    Ej      = European option value at node (S_j, tau_j)
    tildeV  = Ṽ_j   — regression of CF on basis (continuation value)
    tildeV_E= Ṽ^E_j — regression of CV on basis
    aux_CCV = CV^{aux1} — regression of CF·CV on basis
    aux_CV2 = CV^{aux2} — regression of CV² on basis
    theta_cv= θ̂^i_j — per-path functional CV coefficient
    TV      = test value: tildeV + theta_cv·(tildeV_E - E_j)

    Notes
    -----
    When use_cv=True, the exercise rule is Ij > max(Ej, TV) using euro_method.
    When use_cv=False, the exercise rule is Ij > max(Ej, tildeV) using floor_method.
    """
    # Default basis_order: Laguerre=2 (3 functions), Gaussian=5 (5 centers)
    if basis_order is None:
        basis_order = 5 if basis_type == 'gaussian' else 2

    state = _setup(model, sim_out, K, use_cv, euro_method, floor_method,
                   phi_max, n_phi, n_steps_rk4, eps0, precomputed=precomputed)

    CF_1, CV_1 = _backward_loop(state, model, sim_out, K, use_cv,
                                 basis_type, basis_order, ridge,
                                 euro_method, floor_method,
                                 precomputed=precomputed)

    out = _base_estimator(state, CF_1, K)

    if use_cv and improved:
        out.update(_improved_estimator(state, CF_1, CV_1, model, K,
                                       euro_method, phi_max, n_phi,
                                       n_steps_rk4, eps0, sim_out))
    return out
