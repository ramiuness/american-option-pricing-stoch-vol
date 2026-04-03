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


def _ols_fit_predict_multi(Phi, Y_mat, Phi_all, ridge=0.0, ridge_eps=1e-12):
    """
    Solve multiple regressions sharing the same design matrix Phi.
    Factorizes (Phi^T Phi + ridge*I) once, solves for all k RHS columns.

    Phi     : (n_itm, p)
    Y_mat   : (n_itm, k)
    Phi_all : (n_paths, p)
    Returns : (n_paths, k)
    """
    if Phi.shape[0] == 0:
        return np.zeros((Phi_all.shape[0], Y_mat.shape[1]), dtype=float)

    n_features = Phi.shape[1]
    A = Phi.T @ Phi
    B_rhs = Phi.T @ Y_mat

    reg = max(float(ridge), 0.0) + float(ridge_eps)
    try:
        Beta = np.linalg.solve(A + reg * np.eye(n_features), B_rhs)
    except np.linalg.LinAlgError:
        Beta, *_ = np.linalg.lstsq(A + reg * np.eye(n_features), B_rhs, rcond=None)
    return Phi_all @ Beta


def _theta_lr_grid(B, theta0, lam, eta, dt, n_steps, n_paths):
    """
    Build long-run mean process theta_t on the simulation grid.
    Returns (n_paths, n_steps+1); col 0 = theta0; for j>=1:
      theta[:, j] = theta0 + lam*(j*dt) + eta*(B[:, j-1] - 1).
    """
    theta = np.empty((n_paths, n_steps + 1), dtype=float)
    theta[:, 0] = float(theta0)
    t_index = np.arange(1, n_steps + 1, dtype=float) * dt
    theta[:, 1:] = theta0 + lam * t_index[None, :] + eta * (B - 1.0)
    return theta


# ========================= European PUT providers ==========================

def _bs_put_vec(S, K, tau, r, vol):
    """
    Vectorized Black-Scholes PUT price with continuous compounding (no dividends).
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
    N = ndtr
    discK = K * np.exp(-r * tau)
    put = discK * N(-d2) - S * N(-d1)
    zero_vol = (vol <= 0.0)
    if np.any(zero_vol):
        intrinsic = np.maximum(discK - S, 0.0)
        put = np.where(zero_vol, intrinsic, put)
    return put


def _euro_put_llh_slice(model, S_col, vol_col, theta_col, tau, K, pre=None):
    """
    European PUT under Lin-Lin-He via parity at a fixed time slice (vectorized):
      Put = Call - S + K*exp(-r*tau), using model.price_call_llh.
    """
    S_col = np.asarray(S_col, dtype=float).reshape(-1)
    vol_col = np.asarray(vol_col, dtype=float).reshape(-1)
    theta_col = np.asarray(theta_col, dtype=float).reshape(-1)
    calls = model.price_call_llh(S_col, K, tau, vol_col, theta_col, pre=pre)
    return calls - S_col + K * np.exp(-model.r * tau)


def _euro_put_bs_slice(model, S_col, vol_col, tau, K):
    """
    European PUT via Black-Scholes proxy at slice (vectorized):
      P_BS(S_j, K, tau_j, r, sigma_hat_j).
    """
    return _bs_put_vec(S_col, K, tau, model.r, vol_col)


def _euro_put_mc1_slice(sim_out, j, r, K):
    """
    European PUT via one-sample MC using terminal payoff on the same path:
      E_j^i = e^{-r(T - t_j)} (K - S_T^i)^+.
    """
    S_T = np.asarray(sim_out['S'][:, -1], dtype=float)
    dt = float(sim_out['dt'])
    n_steps = sim_out['S'].shape[1] - 1
    disc_jT = np.exp(-r * (n_steps - j) * dt)
    return disc_jT * np.maximum(K - S_T, 0.0)


def _euro_put_slice(method, model, sim_out, j, S_col, vol_col, theta_col, tau, K, llh_pre=None):
    """
    Dispatcher for E_j at a slice:
      method in {'llh','bs','mc1'}.
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
                               euro_method='llh', floor_method='bs',
                               phi_max=300.0, n_phi=513, n_steps_rk4=128, eps0=1e-6):
    """
    LSM American put pricer with optional Rasmussen control variates.

    Implements the LSM-CV algorithm from report §3.3.1, based on West (2012) §3.1.2
    and Rasmussen (2005). When use_cv=True, four regressions sharing the same Laguerre
    basis yield a per-path optimal CV coefficient (functional form theta).

    Parameters
    ----------
    model        : ImprovedSteinStein instance
    sim_out      : dict from model.simulate_prices() — keys 'S', 'sigma_hat', 'B', 'dt'
    K            : strike
    use_cv       : if False, runs plain LSM with European exercise floor
    improved     : if True and use_cv=True, adds global CV estimator (West 2012 §3.1.2)
    euro_method  : 'llh' | 'bs' | 'mc1' — European pricing for CV (only when use_cv=True)
    floor_method : 'bs' | 'llh' — European floor for exercise safety when use_cv=False.
                   Default 'bs' is fast; 'llh' is accurate but ~100x slower.
    phi_max, n_phi, n_steps_rk4, eps0 : LLH quadrature params (only when euro_method='llh')

    Returns
    -------
    dict:
      price, std_err, ci_95, n_paths
      + price_imp, std_err_imp, ci_95_imp  (when use_cv=True and improved=True)

    Notation (report §3.3.1 / West 2012)
    -------------------------------------
    CF      = EE  in West  — discounted eventual exercise (cash flow)
    CV      = EEE in West  — European eventual exercise (control variate)
    Ej      = E   in West  — European option value at node (S_j, tau_j)
    tildeV  = Ṽ_j          — regression of CF on Laguerre basis (continuation value)
    tildeV_E= Ṽ^E_j        — regression of CV on basis
    aux_ECV = CV^{aux1}     — regression of E_j·CV on basis
    aux_CV2 = CV^{aux2}     — regression of CV² on basis
    theta_cv= θ̂^i_j        — per-path functional CV coefficient
    TV      = TV            — test value: tildeV + theta_cv·(tildeV_E - E_j)

    Regressions are fitted on ITM paths only (I_j > 0) per Longstaff-Schwartz,
    but predicted for all paths.

    Notes
    -----
    When use_cv=True, the exercise rule is Ij > max(Ej, TV) using euro_method.
    When use_cv=False, the exercise rule is Ij > max(Ej, tildeV) using floor_method.

    Defaults (phi_max=300, n_phi=513, n_steps_rk4=128) correspond to STANDARD preset.
    """
    S = sim_out['S']
    sigma_hat = sim_out['sigma_hat']
    B = sim_out['B']
    dt = float(sim_out['dt'])
    r = float(model.r)

    n_paths, n_cols = S.shape
    n_steps = n_cols - 1
    disc = np.exp(-r * dt)

    # Step 1: Compute theta_t, needed if LLH is used for CV or floor (as it enters the PDE coefficients).
    need_theta = (use_cv and euro_method == 'llh') or (not use_cv and floor_method == 'llh')
    theta_lr = _theta_lr_grid(B, model.theta0, model.lam, model.eta, dt, n_steps, n_paths) if need_theta else None

    # Step 2: Initialize at maturity t_N 
    CF = np.maximum(K - S[:, -1], 0.0)   # EE in West — eventual exercise
    CV = CF.copy() if use_cv else None    # EEE in West — European eventual exercise

    CF_1 = CV_1 = E_1 = None  # saved at j=1 for the final estimators

    # Precompute all unique tau values (when LLH European pricing is used in any mode)
    tau_cache = {}
    if (use_cv and euro_method == 'llh') or (not use_cv and floor_method == 'llh'):
        for j in range(1, n_steps):
            steps_remaining = n_steps - j
            if steps_remaining not in tau_cache:
                tau_cache[steps_remaining] = model.llh_precompute_tau(
                    steps_remaining * dt, phi_max=phi_max, n_phi=n_phi,
                    n_steps_ode=n_steps_rk4, eps0=eps0
                )

    _Ij = np.empty(n_paths, dtype=float)
    _x  = np.empty(n_paths, dtype=float)

    # Step 3: Backward loop j = N-1,...,1 
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

        # ITM set and basis on x = S/K
        np.divide(Sj, K, out=_x)
        Phi_all = _laguerre_basis(_x, order=basis_order)
        itm = (Ij > 0.0)

        if use_cv:
            # Step 3(ii): European PUT E_j via selected method
            steps_remaining = n_steps - j
            pre_j = tau_cache.get(steps_remaining) if euro_method == 'llh' else None

            # Optimization: for j > 1, only compute E_j for ITM paths (OTM never
            # exercise since I_j=0). At j==1, compute for ALL paths (needed for
            # E_1 in the improved estimator).
            if j == 1 or euro_method == 'mc1':
                Ej = _euro_put_slice(euro_method, model, sim_out, j, Sj, volj, thetaj, tauj, K, llh_pre=pre_j)
            else:
                Ej = np.zeros(n_paths, dtype=float)
                if itm.any():
                    thetaj_itm = thetaj[itm] if thetaj is not None else None
                    Ej[itm] = _euro_put_slice(
                        euro_method, model, sim_out, j,
                        Sj[itm], volj[itm], thetaj_itm, tauj, K, llh_pre=pre_j
                    )

            # Step 3(ii-iii): 4 regressions with shared Gram matrix factorization.
            #   col 0: CF      → tildeV   (Ṽ_j  — continuation value)
            #   col 1: CV      → tildeV_E (Ṽ^E_j — European continuation)
            #   col 2: E_j·CV  → aux_ECV  (CV^{aux1} in report)
            #   col 3: CV²     → aux_CV2  (CV^{aux2} in report)
            if itm.any():
                Phi = Phi_all[itm, :]
                CF_itm = CF[itm]
                CV_itm = CV[itm]
                Ej_itm = Ej[itm]
                Y_mat = np.column_stack([CF_itm, CV_itm, Ej_itm * CV_itm, CV_itm**2])
                preds = _ols_fit_predict_multi(Phi, Y_mat, Phi_all, ridge=ridge)
                tildeV, tildeV_E, aux_ECV, aux_CV2 = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
            else:
                tildeV   = np.zeros_like(Sj)
                tildeV_E = np.zeros_like(Sj)
                aux_ECV  = np.zeros_like(Sj)
                aux_CV2  = np.ones_like(Sj)

            # Step 3(iii): per-path CV coefficient θ̂^i
            #   θ̂^i = -(aux_ECV - tildeV·tildeV_E) / (aux_CV2 - tildeV_E²)
            denom = aux_CV2 - tildeV_E**2
            theta_cv = np.zeros_like(Sj)
            ok = np.abs(denom) > 1e-12
            theta_cv[ok] = -(aux_ECV[ok] - tildeV[ok] * tildeV_E[ok]) / denom[ok]

            # Test value: TV = Ṽ_j + θ̂^i·(Ṽ^E_j - E_j)
            TV = tildeV + theta_cv * (tildeV_E - Ej)

            # Step 3(iii): exercise if I_j > max(E_j, TV); update CF=I, CV=E
            ex_mask = Ij > np.maximum(Ej, TV)
            if np.any(ex_mask):
                CF[ex_mask] = Ij[ex_mask]
                CV[ex_mask] = Ej[ex_mask]

            # Save j==1 vectors for final estimators (Steps 4-5)
            if j == 1:
                CF_1 = CF.copy()
                CV_1 = CV.copy()
                E_1  = Ej.copy()

        else:
            # Plain LSM: 1 regression + European floor for exercise safety.
            # Default floor_method='bs' is fast (~free); 'llh' is available but expensive
            # as it triggers ODE solves at every exercise date.
            if floor_method == 'bs':
                Ej = _euro_put_bs_slice(model, Sj, volj, tauj, K)
            else:
                steps_remaining = n_steps - j
                pre_j = tau_cache.get(steps_remaining)
                Ej = _euro_put_slice(floor_method, model, sim_out, j, Sj, volj, thetaj, tauj, K, llh_pre=pre_j)

            if itm.any():
                Phi = Phi_all[itm, :]
                tildeV = _ols_fit_predict(Phi, CF[itm], Phi_all, ridge=ridge)
            else:
                tildeV = np.zeros_like(Sj)

            # Exercise decision with European floor: Ij > max(Ej, tildeV)
            ex_mask = Ij > np.maximum(Ej, tildeV)
            if np.any(ex_mask):
                CF[ex_mask] = Ij[ex_mask]

            if j == 1:
                CF_1 = CF.copy()

    # Step 4: Base estimator
    # max(I_0, (1/N) Σ e^{-r t_1} CF_1^i)
    base_samples = disc * CF_1
    I0 = max(K - S[0, 0], 0.0)
    price = max(I0, float(base_samples.mean()))
    std_err = base_samples.std(ddof=1) / np.sqrt(n_paths)
    ci95 = (price - 1.96 * std_err, price + 1.96 * std_err)

    out = {
        'price': price,
        'std_err': std_err,
        'ci_95': ci95,
        'n_paths': int(n_paths)
    }

    # Step 5: Improved estimator (Rasmussen 2002 / West 2012) 
    if use_cv and improved:
        X = CV_1 - CV_1.mean()
        Y = CF_1 - CF_1.mean()
        varX = X.var(ddof=1)
        theta_global = 0.0 if varX <= 1e-16 else float((X * Y).sum() / ((n_paths - 1) * varX))

        imp_samples = disc * (CF_1 - theta_global * (CV_1 - E_1))
        price_imp = max(I0, float(imp_samples.mean()))
        std_err_imp = imp_samples.std(ddof=1) / np.sqrt(n_paths)
        ci95_imp = (price_imp - 1.96 * std_err_imp, price_imp + 1.96 * std_err_imp)

        out.update({
            'price_imp': price_imp,
            'std_err_imp': std_err_imp,
            'ci_95_imp': ci95_imp
        })

    return out
