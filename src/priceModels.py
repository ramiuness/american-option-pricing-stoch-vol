"""
LLH (Lin-Lin-He 2024) improved Stein-Stein stochastic volatility model.

Provides Monte Carlo simulation, closed-form European pricing via characteristic
function / ODE integration, and Black-Scholes baselines.
See reports/llh-formula_report.pdf for the theoretical derivation.
"""

from dataclasses import dataclass
import numpy as np
from scipy.stats import lognorm, kstest


# ─── Simulation primitives (private) ─────────────────────────────────────────

def _draw_correlated_normals(rng, n_paths: int, n_steps_mc: int, rho: float):
    """Correlated standard normals Z1, Z2 with Corr(Z1,Z2)=rho, each (n_paths, n_steps_mc)."""
    Z1 = rng.standard_normal((n_paths, n_steps_mc))
    Z2i = rng.standard_normal((n_paths, n_steps_mc))
    Z2 = rho * Z1 + np.sqrt(max(0.0, 1.0 - rho**2)) * Z2i
    return Z1, Z2

def _brownian_from_normals(Z: np.ndarray, dt: float):
    """Increments dW and cumulative Brownian W from standard normals Z, both (n_paths, n_steps_mc)."""
    dW = np.sqrt(dt) * Z
    W = np.cumsum(dW, axis=1)
    return dW, W

def _gbm_from_formula(W: np.ndarray, dt: float):
    """Unit-drift geometric Brownian motion B = exp(W - t/2), shape (n_paths, n_steps_mc)."""
    n = W.shape[1]
    t_grid = dt * np.arange(1, n + 1)[None, :]
    B = np.exp(W - 0.5 * t_grid)
    return B

def _causal_exp_conv(X: np.ndarray, a1: float):
    """Causal exponential-kernel convolution s_j = sum_{i=1}^{j} a1^i X_{j-i}.

    Returns array with same shape as X (n_paths, n_steps_mc).
    """
    n = X.shape[1]
    j = np.arange(n)
    Y = X * (a1 ** (-j))[None, :]
    P = np.cumsum(Y, axis=1)
    return P * (a1 ** (j + 1))[None, :]


def _causal_exp_conv_pair(X1, X2, a1):
    """Paired causal exponential convolution reusing power arrays for efficiency.

    Returns (conv(X1, a1), conv(X2, a1)), each (n_paths, n_steps_mc).
    """
    n = X1.shape[1]
    j = np.arange(n)
    pow_neg = (a1 ** (-j))[None, :]
    pow_pos = (a1 ** (j + 1))[None, :]

    Y1 = X1 * pow_neg
    P1 = np.cumsum(Y1, axis=1)
    r1 = P1 * pow_pos

    Y2 = X2 * pow_neg
    P2 = np.cumsum(Y2, axis=1)
    r2 = P2 * pow_pos

    return r1, r2


def _sigma_hat_from_components(
    n_steps_mc: int, dt: float, kappa: float, sigma0: float, theta0: float, lam: float,
    nu: float, eta: float, W2: np.ndarray, B: np.ndarray
):
    """LLH volatility process sigma_hat[:,j] = sigma(t_j), adapted to F_{t_j}.

    Ensures correct pairing with dW1[:,j] (the increment on [t_j, t_{j+1}]).

    Returns
    -------
    sigma_hat : ndarray, shape (n_paths, n_steps_mc)
    """
    n_paths = W2.shape[0]
    idx = np.arange(0, n_steps_mc)[None, :]
    exp_kdt_idx = np.exp(-kappa * idx * dt)
    a1 = np.exp(-kappa * dt)

    # In-place shifted arrays (avoid concatenate copies)
    W2_shifted = np.empty_like(W2)
    W2_shifted[:, 0] = 0.0
    W2_shifted[:, 1:] = W2[:, :-1]

    Bm1_shifted = np.empty_like(B)
    Bm1_shifted[:, 0] = 0.0
    np.subtract(B[:, :-1], 1.0, out=Bm1_shifted[:, 1:])

    # Shared power arrays for both convolutions
    term_W2, term_Bm1 = _causal_exp_conv_pair(W2_shifted, Bm1_shifted, a1)

    if kappa != 0:
        lam_drift = lam * idx * dt + lam * np.expm1(-kappa * idx * dt) / kappa
    else:
        lam_drift = np.zeros_like(idx, dtype=float)
    sigma_hat = (
        exp_kdt_idx * (sigma0 - theta0)
        + theta0
        + lam_drift
        + nu * W2_shifted
        - (nu * kappa * dt) * term_W2
        + (eta * kappa * dt) * term_Bm1
    )
    return sigma_hat

def _multiplicative_euler_prices(S0: float, r: float, sigma_hat: np.ndarray, dW1: np.ndarray, dt: float):
    """Multiplicative Euler scheme for the asset price, shape (n_paths, n_steps_mc)."""
    increments = 1.0 + r * dt + sigma_hat * dW1
    return S0 * np.cumprod(increments, axis=1)


# ─── ODE system & quadrature (private) ───────────────────────────────────────

def _trap_weights(n, simplified=False):
    """Composite trapezoid weights for n uniform-grid nodes.

    When simplified=True, halves only the first weight (Boyarchenko-Levendorskii
    Eq. 2.31); otherwise standard half-weights at both endpoints.

    Returns
    -------
    w : ndarray, shape (n,)
        Weights to be scaled by the step size h.
    """
    w = np.ones(n, dtype=np.float64)
    w[0] = 0.5
    if not simplified:
        w[-1] = 0.5
    return w

def _phi_grid(phi_max, n_phi, eps0=1e-6):
    """Uniform phi-grid on [0, phi_max] with phi[0]=eps0 to avoid 1/phi singularity."""
    phi = np.linspace(0.0, phi_max, n_phi, dtype=np.float64)
    phi[0] = eps0
    return phi


def _rhs_factory(phi, r, kappa, nu, lam, eta, rho):
    """Build the ODE right-hand side for the LLH characteristic-exponent system (Eq. 4).

    Batched over phi and both j=1,2 branches simultaneously.

    Parameters
    ----------
    phi : ndarray, shape (n_phi,)
    r, kappa, nu, lam, eta, rho : float
        LLH model parameters.

    Returns
    -------
    rhs : callable
        rhs(Y, out=None) -> dY, where Y and dY have shape (n_phi, 2, 6)
        with components ordered (C, D, E, F, G, H).
    """
    # Pre-compute constants (avoids recomputation in 512 rhs calls)
    nu2 = nu ** 2
    eta2 = eta ** 2
    two_nu2 = 2.0 * nu2
    two_eta2 = 2.0 * eta2
    half_nu2 = 0.5 * nu2
    half_eta2 = 0.5 * eta2
    two_kappa = 2.0 * kappa

    u = np.array([+0.5, -0.5])[None, :]
    b = np.array([1.0, 0.0])[None, :]
    PHI = phi[:, None]
    iPHI = 1j * PHI
    half_PHI2 = 0.5 * PHI ** 2
    A = kappa - nu * rho * (b + iPHI)
    two_A = 2.0 * A
    r_iPHI = r * iPHI

    # Pre-compute the constant part of dD (independent of Y)
    dD_const = u * iPHI - half_PHI2

    def rhs(Y, out=None):
        C, D, E, F, G, H = [Y[..., k] for k in range(6)]
        if out is None:
            out = np.empty_like(Y)

        # dD (index 1 in CDEFGH ordering)
        out[..., 1] = dD_const + two_nu2 * D*D - two_A*D + half_eta2 * E*E
        # dE
        out[..., 2] = two_kappa*D - A*E + two_nu2 * D*E + two_eta2 * E*F
        # dF
        out[..., 3] = kappa*E + half_nu2 * E*E + two_eta2 * F*F
        # dG
        out[..., 4] = kappa*H + 2*lam*F + nu2 * E*H + two_eta2 * F*G
        # dH
        out[..., 5] = -A*H + lam*E + two_nu2 * D*H + eta2 * E*G
        # dC
        out[..., 0] = r_iPHI + lam*G + half_nu2 * H*H + nu2 * D + half_eta2 * G*G + eta2 * F
        return out

    return rhs


def _rk4_integrate(rhs, tau, n_steps_ode, n_phi):
    """Integrate the characteristic-exponent ODE from tau=0 to tau via RK4.

    Returns
    -------
    coeffs : dict
        Keys 'C','D','E','F','G','H', each ndarray of shape (n_phi, 2).
    """
    Z = np.zeros((n_phi, 2), dtype=np.complex128)
    Y = np.stack([Z,Z,Z,Z,Z,Z], axis=-1)
    if tau == 0 or n_steps_ode == 0:
        return {k: Y[..., j] for j, k in enumerate("CDEFGH")}

    dt = tau / n_steps_ode

    # Pre-allocate RK4 workspace
    k1 = np.empty_like(Y)
    k2 = np.empty_like(Y)
    k3 = np.empty_like(Y)
    k4 = np.empty_like(Y)
    Y_tmp = np.empty_like(Y)

    half_dt = 0.5 * dt
    sixth_dt = dt / 6.0

    for _ in range(n_steps_ode):
        rhs(Y, out=k1)
        np.add(Y, half_dt * k1, out=Y_tmp)
        rhs(Y_tmp, out=k2)
        np.add(Y, half_dt * k2, out=Y_tmp)
        rhs(Y_tmp, out=k3)
        np.add(Y, dt * k3, out=Y_tmp)
        rhs(Y_tmp, out=k4)
        # Weighted sum: Y += (dt/6)(k1 + 2k2 + 2k3 + k4)
        Y += sixth_dt * (k1 + 2*k2 + 2*k3 + k4)

    return {k: Y[..., j] for j, k in enumerate("CDEFGH")}


# ─── Characteristic function transform (private) ─────────────────────────────

def _build_transform_vec(coeffs, S_vec, v_vec, theta_vec, phi):
    """Characteristic-function transform f(phi) = exp(C + Dv^2 + Ev*theta + ...) over N paths.

    Parameters
    ----------
    coeffs : dict or ndarray
        Dict with keys 'C'..'H' each (n_phi, 2), or pre-stacked (n_phi, 2, 6).
    S_vec, v_vec, theta_vec : array-like, shape (N,)
        Spot prices, instantaneous volatilities, and long-run volatilities.
    phi : ndarray, shape (n_phi,)

    Returns
    -------
    f : ndarray, shape (n_phi, 2, N)
    """
    S_vec = np.asarray(S_vec, dtype=float).reshape(-1)
    v_vec = np.asarray(v_vec, dtype=float).reshape(-1)
    theta_vec = np.asarray(theta_vec, dtype=float).reshape(-1)

    x  = np.log(S_vec)                 # (N,)

    # Support both dict and pre-stacked array
    if isinstance(coeffs, dict):
        C = coeffs['C'][..., None]
        D = coeffs['D'][..., None]
        E = coeffs['E'][..., None]
        F = coeffs['F'][..., None]
        G = coeffs['G'][..., None]
        H = coeffs['H'][..., None]
    else:
        # coeffs is (n_phi, 2, 6) pre-stacked
        C = coeffs[..., 0:1]   # (n_phi, 2, 1)
        D = coeffs[..., 1:2]
        E = coeffs[..., 2:3]
        F = coeffs[..., 3:4]
        G = coeffs[..., 4:5]
        H = coeffs[..., 5:6]

    v2 = (v_vec**2)[None, None, :]     # (1,1,N)
    v  = v_vec[None, None, :]
    th = theta_vec[None, None, :]
    xv = x[None, None, :]
    PHI = phi[:, None, None]           # (n_phi,1,1)

    quad = (C + D*v2 + E*v*th + F*th*th + G*th + H*v + 1j*PHI*xv)
    return np.exp(quad)


def _compute_P_vec(f, K, phi, w):
    """Risk-neutral probabilities P1, P2 via trapezoid quadrature of the transform.

    Parameters
    ----------
    f : ndarray, shape (n_phi, 2, N)
        Characteristic-function values from ``_build_transform_vec``.
    K : float or ndarray, shape (N,)
    phi : ndarray, shape (n_phi,)
    w : ndarray, shape (n_phi,)
        Trapezoid weights (including step size).

    Returns
    -------
    P1, P2 : ndarray, each shape (N,)
    """
    lnK = np.log(K)
    kernel = np.exp(-1j * phi * lnK) / (1j * phi)         # (n_phi,)
    integrand = np.real(kernel[:, None, None] * f)        # (n_phi, 2, N)
    P = 0.5 + (1.0/np.pi) * np.tensordot(w, integrand, axes=(0, 0))  # (2, N)
    return P[0, :], P[1, :]


# ─── Black-Scholes closed forms (public) ─────────────────────────────────────

def price_call_bs(S, K=90, tau=1.0, r=0.05, vol=0.2):
    """Black-Scholes European call price.

    Parameters
    ----------
    S : float
        Spot price.
    K : float
        Strike price.
    tau : float
        Time to maturity in years.
    r : float
        Risk-free rate.
    vol : float
        Constant volatility.

    Returns
    -------
    float
        Call price (intrinsic value if tau <= 0).
    """
    if tau <= 0:
        return max(0.0, S - K)
    v = vol * np.sqrt(tau)
    d1 = (np.log(S/K) + (r + 0.5*vol*vol)*tau) / v
    d2 = d1 - v
    from scipy.stats import norm as _norm
    return float(S * _norm.cdf(d1) - K * np.exp(-r*tau) * _norm.cdf(d2))


def price_put_bs(S, K=90, tau=1.0, r=0.05, vol=0.2):
    """Black-Scholes European put price via put-call parity.

    Parameters
    ----------
    S, K, tau, r, vol : float
        Same as ``price_call_bs``.

    Returns
    -------
    float
    """
    return price_call_bs(S, K, tau, r, vol) - S + K * np.exp(-r * tau)


# ─── MC pricing (public) ─────────────────────────────────────────────────────

def price_call_mc(paths, K, T, r=None):
    """Monte Carlo European call price from simulated terminal values.

    Parameters
    ----------
    paths : ndarray, shape (n_paths, n_steps+1)
        Simulated price paths; only the terminal column is used.
    K : float
        Strike price.
    T : float
        Time to maturity.
    r : float
        Risk-free rate.

    Returns
    -------
    dict
        Keys: 'price', 'std_err', 'ci_95' (tuple), 'n_paths'.
    """
    n_paths = paths.shape[0]
    disc = np.exp(-r * T)
    Y = disc * np.maximum(paths[:, -1] - K, 0.0)
    price = Y.mean()
    std_err = Y.std(ddof=1) / np.sqrt(n_paths)
    ci95 = (price - 1.96*std_err, price + 1.96*std_err)
    return {
        'price': price,
        'std_err': std_err,
        'ci_95': ci95,
        'n_paths': n_paths
    }


def price_put_mc(paths, K, T, r=None):
    """Monte Carlo European put price from simulated terminal values.

    Parameters
    ----------
    paths : ndarray, shape (n_paths, n_steps+1)
        Simulated price paths; only the terminal column is used.
    K : float
        Strike price.
    T : float
        Time to maturity.
    r : float
        Risk-free rate.

    Returns
    -------
    dict
        Keys: 'price', 'std_err', 'ci_95' (tuple), 'n_paths'.
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


# ─── Convenience / notebook utilities (public) ───────────────────────────────

def plot(paths, title="Simulated Price Paths"):
    """Quick line plot of simulated price paths.

    Parameters
    ----------
    paths : ndarray
        Price paths (columns = time steps).
    title : str
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(paths, color='#1f77b4', alpha=0.6, lw=0.8)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel('Time steps')
    ax.set_ylabel('Price')
    ax.grid(True, ls=':', lw=0.6, alpha=0.7)
    ax.margins(x=0)
    fig.tight_layout()
    plt.show()


def test_lognormality(data):
    """Kolmogorov-Smirnov test for log-normality of positive data.

    Parameters
    ----------
    data : array-like
        Sample values (non-positive entries are dropped).
    """
    data = data[data > 0]
    shape, loc, scale = lognorm.fit(data, floc=0)
    D, p_value = kstest(data, 'lognorm', args=(shape, loc, scale))

    print(f"KS statistic: {D:.4f}, p-value: {p_value:.4g}")
    if p_value > 0.05:
        print("Fail to reject null: data may be lognormal.")
    else:
        print("Reject null: data is not lognormal.")


def european_prices(model, S0=100.0, K=90.0, tau=1.0,
                    n_steps_mc=252, n_paths=50000,
                    phi_max=300.0, n_phi=513, n_steps_ode=128):
    """European call and put prices under both LLH closed-form and Monte Carlo.

    Parameters
    ----------
    model : ImprovedSteinStein
    S0 : float
        Initial spot price.
    K : float
        Strike price.
    tau : float
        Time to maturity.
    n_steps_mc : int
        MC time discretization steps.
    n_paths : int
        Number of MC paths.
    phi_max, n_phi, n_steps_ode : float, int, int
        Quadrature / ODE integration settings.

    Returns
    -------
    dict
        Keys: 'llh_call', 'llh_put', 'mc_call', 'mc_call_ci',
        'mc_put', 'mc_put_ci'.
    """
    res = model.simulate_prices(S0=S0, T=tau, n_steps_mc=n_steps_mc, n_paths=n_paths)

    llh_call = model.price_call_llh(S=S0, K=K, tau=tau,
                                    vol=model.sigma0, theta=model.theta0,
                                    phi_max=phi_max, n_phi=n_phi, n_steps_ode=n_steps_ode).item()
    llh_put  = model.price_put_llh(S=S0, K=K, tau=tau,
                                   vol=model.sigma0, theta=model.theta0,
                                   phi_max=phi_max, n_phi=n_phi, n_steps_ode=n_steps_ode).item()

    res_mc_call = price_call_mc(res['S'], K=K, T=tau, r=model.r)
    res_mc_put  = price_put_mc(res['S'],  K=K, T=tau, r=model.r)

    return {
        'llh_call':   llh_call,
        'llh_put':    llh_put,
        'mc_call':    res_mc_call['price'],
        'mc_call_ci': res_mc_call['ci_95'],
        'mc_put':     res_mc_put['price'],
        'mc_put_ci':  res_mc_put['ci_95'],
    }


def compare_european_prices(model, S0=100.0, K=90.0, tau=1.0,
                            n_steps_mc=252, n_paths=50000,
                            phi_max=300.0, n_phi=513, n_steps_ode=128):
    """Print LLH and MC European prices side-by-side.

    Parameters
    ----------
    model : ImprovedSteinStein
    S0, K, tau : float
        Spot, strike, maturity.
    n_steps_mc, n_paths : int
        MC settings.
    phi_max, n_phi, n_steps_ode : float, int, int
        Quadrature / ODE settings.
    """
    r = european_prices(model, S0=S0, K=K, tau=tau,
                        n_steps_mc=n_steps_mc, n_paths=n_paths,
                        phi_max=phi_max, n_phi=n_phi, n_steps_ode=n_steps_ode)
    print(
        f"European call LLH price:  {r['llh_call']},\n"
        f"European call MC price:   {r['mc_call']}, MC 95% CI: {r['mc_call_ci']},\n"
        f"European put LLH price:   {r['llh_put']},\n"
        f"European put MC price:    {r['mc_put']}, MC 95% CI: {r['mc_put_ci']}"
    )


# ─── Main class ──────────────────────────────────────────────────────────────

@dataclass
class LLHPrecompute:
    """Cached quadrature and ODE data for a fixed maturity tau.

    Attributes
    ----------
    tau : float
        Time to maturity for which coefficients were computed.
    phi : ndarray, shape (n_phi,)
        Quadrature grid in the transform variable.
    w : ndarray, shape (n_phi,)
        Trapezoid weights (including step size).
    coeffs : dict
        Keys 'C','D','E','F','G','H', each ndarray of shape (n_phi, 2).
    coeffs_stacked : ndarray, shape (n_phi, 2, 6)
        Same coefficients pre-stacked for ``_build_transform_vec``.
    n_phi : int
        Number of quadrature nodes.
    """
    tau: float
    phi: np.ndarray
    w: np.ndarray
    coeffs: dict
    coeffs_stacked: np.ndarray
    n_phi: int


@dataclass
class ImprovedSteinStein:
    """Lin-Lin-He (2024) improved Stein-Stein stochastic volatility model.

    Provides Monte Carlo path simulation and closed-form European option
    pricing via characteristic-function inversion.

    Attributes
    ----------
    r : float
        Risk-free interest rate.
    sigma0 : float
        Initial instantaneous volatility.
    theta0 : float
        Initial long-run volatility level.
    rho : float
        Correlation between asset and volatility Brownians.
    kappa : float
        Mean-reversion speed of the volatility process.
    lam : float
        Drift coefficient of the long-run volatility theta(t).
    nu : float
        Diffusion coefficient (vol-of-vol) driven by W2.
    eta : float
        Diffusion coefficient of theta(t) driven by geometric BM.
    seed : int or None
        Random seed for reproducible simulation.
    """
    r: float
    sigma0: float
    theta0: float
    rho: float
    kappa: float
    lam: float
    nu: float
    eta: float
    seed: int | None = None

    def simulate_prices(
        self,
        S0: float,
        T: float,
        n_steps_mc: int,
        n_paths: int
    ) -> dict:
        """Simulate LLH price and volatility paths via Euler discretization.

        Parameters
        ----------
        S0 : float
            Initial spot price.
        T : float
            Time horizon in years.
        n_steps_mc : int
            Number of Euler time steps.
        n_paths : int
            Number of Monte Carlo paths.

        Returns
        -------
        dict
            'dt'        : float -- step size T/n_steps_mc
            'W'         : ndarray (n_paths, n_steps_mc) -- auxiliary BM
            'B'         : ndarray (n_paths, n_steps_mc) -- geometric BM
            'W2'        : ndarray (n_paths, n_steps_mc) -- volatility-driver BM
            'sigma_hat' : ndarray (n_paths, n_steps_mc) -- instantaneous vol
            'S'         : ndarray (n_paths, n_steps_mc+1) -- prices incl. S0
        """
        dt = T / n_steps_mc
        sqrt_dt = np.sqrt(dt)
        rng = np.random.default_rng(self.seed)

        # (i) Brownian for B
        Zb = rng.standard_normal((n_paths, n_steps_mc))
        dW, W = _brownian_from_normals(Zb, dt)
        B = _gbm_from_formula(W, dt)

        # (ii) Correlated Brownian motions for price and sigma drivers
        Z1, Z2 = _draw_correlated_normals(rng, n_paths, n_steps_mc, self.rho)
        # Inline dW1 to avoid discarded cumsum allocation
        dW1 = sqrt_dt * Z1
        dW2, W2 = _brownian_from_normals(Z2, dt)

        # (iii) sigma_hat
        sigma_hat = _sigma_hat_from_components(
            n_steps_mc, dt, self.kappa, self.sigma0, self.theta0, self.lam, self.nu, self.eta, W2, B
        )

        # (iv) prices — avoid S0 column concatenation
        S = _multiplicative_euler_prices(S0, self.r, sigma_hat, dW1, dt)
        S_full = np.empty((S.shape[0], S.shape[1] + 1), dtype=float)
        S_full[:, 0] = S0
        S_full[:, 1:] = S

        return {"dt": dt, "W": W, "B": B, "W2": W2, "sigma_hat": sigma_hat,
                "S": S_full}


    def llh_precompute_tau(self, tau, phi_max, n_phi, n_steps_ode, eps0=1e-6) -> LLHPrecompute:
        """Precompute quadrature grid and ODE coefficients for a fixed maturity.

        Parameters
        ----------
        tau : float
            Time to maturity.
        phi_max : float
            Upper bound of the phi integration grid.
        n_phi : int
            Number of quadrature nodes.
        n_steps_ode : int
            RK4 integration steps for the characteristic-exponent ODE.
        eps0 : float
            Small offset replacing phi=0 to avoid 1/phi singularity.

        Returns
        -------
        LLHPrecompute
        """
        phi = _phi_grid(phi_max, n_phi, eps0)
        dphi = phi_max / (n_phi - 1)
        w = dphi * _trap_weights(n_phi, simplified=True)
        rhs = _rhs_factory(phi, self.r, self.kappa, self.nu, self.lam, self.eta, self.rho)
        coeffs = _rk4_integrate(rhs, tau, n_steps_ode, n_phi)

        # Pre-stack coefficients for _build_transform_vec
        coeffs_stacked = np.stack(
            [coeffs[k] for k in "CDEFGH"], axis=-1
        )  # (n_phi, 2, 6)

        return LLHPrecompute(tau=float(tau), phi=phi, w=w, coeffs=coeffs,
                             coeffs_stacked=coeffs_stacked, n_phi=n_phi)


    def price_call_llh(self, S, K, tau, vol, theta, phi_max=300.0, n_phi=513, n_steps_ode=128, pre=None, eps0=1e-6):
        """Vectorized European call prices under the LLH characteristic-function formula.

        Parameters
        ----------
        S, vol, theta : float or array-like, shape (N,)
            Spot prices, instantaneous vols, and long-run vols.
        K : float
            Strike price.
        tau : float
            Time to maturity.
        phi_max, n_phi, n_steps_ode : float, int, int
            Quadrature / ODE settings (ignored if *pre* is given).
        pre : LLHPrecompute or None
            Reusable precomputed coefficients for this tau.
        eps0 : float
            Phi-grid singularity offset.

        Returns
        -------
        ndarray, shape (N,)
        """
        S = np.asarray(S, dtype=float).reshape(-1)
        vol = np.asarray(vol, dtype=float).reshape(-1)
        theta = np.asarray(theta, dtype=float).reshape(-1)
        if S.shape != vol.shape or S.shape != theta.shape:
            raise ValueError("S, vol, theta must have the same shape")

        if pre is None:
            pre = self.llh_precompute_tau(tau, phi_max=phi_max, n_phi=n_phi, n_steps_ode=n_steps_ode, eps0=eps0)

        # Use pre-stacked coefficients for efficiency
        f = _build_transform_vec(pre.coeffs_stacked, S, vol, theta, pre.phi)
        P1, P2 = _compute_P_vec(f, K, pre.phi, pre.w)
        disc = np.exp(-self.r * tau)
        return np.real(S * P1 - K * disc * P2)


    def price_put_llh(self, S, K, tau, vol, theta, phi_max, n_phi, n_steps_ode, pre=None, eps0=1e-6):
        """Vectorized European put prices under LLH via put-call parity.

        Parameters and returns follow ``price_call_llh``.
        """
        call = self.price_call_llh(S, K, tau, vol, theta, phi_max, n_phi, n_steps_ode, pre=pre, eps0=eps0)
        return call - S + K * np.exp(-self.r * tau)


    # ---------- MC European pricing ----------

    def price_call_mc(self, sim_out, K):
        """European call price via MC on pre-simulated paths.

        Parameters
        ----------
        sim_out : dict
            Output of ``simulate_prices``.
        K : float
            Strike price.

        Returns
        -------
        dict
            Same as module-level ``price_call_mc``.
        """
        T = sim_out['dt'] * (sim_out['S'].shape[1] - 1)
        return price_call_mc(sim_out['S'], K=K, T=T, r=self.r)

    def price_put_mc(self, sim_out, K):
        """European put price via MC on pre-simulated paths.

        Parameters
        ----------
        sim_out : dict
            Output of ``simulate_prices``.
        K : float
            Strike price.

        Returns
        -------
        dict
            Same as module-level ``price_put_mc``.
        """
        T = sim_out['dt'] * (sim_out['S'].shape[1] - 1)
        return price_put_mc(sim_out['S'], K=K, T=T, r=self.r)


    # ---------- American pricing (delegates to amerPrice) ----------

    def price_american_put(self, sim_out, K, basis_order=3,
                           basis_type='laguerre',
                           use_cv=True, improved=True, ridge=0.0,
                           euro_method='llh', floor_method='bs',
                           phi_max=300.0, n_phi=513, n_steps_rk4=128, eps0=1e-6):
        """American put price via LSM with optional Rasmussen control variates.

        Delegates to ``amOptPricer.price_american_put_lsm_llh``; see that
        function for full parameter and return documentation.
        """
        import amerPrice as ap
        return ap.price_american_put_lsm_llh(
            self, sim_out, K, basis_order=basis_order,
            basis_type=basis_type,
            use_cv=use_cv, improved=improved, ridge=ridge,
            euro_method=euro_method, floor_method=floor_method,
            phi_max=phi_max, n_phi=n_phi, n_steps_rk4=n_steps_rk4, eps0=eps0
        )
