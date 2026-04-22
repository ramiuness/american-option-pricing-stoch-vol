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
    Z2 = rho * Z1 + np.sqrt(max(0.0, 1.0 - rho ** 2)) * Z2i
    return Z1, Z2


def _increments_from_normals(rho: float, dt: float, n_paths: int, n_steps_mc: int, rng):
    """Draw BM increments (dW1, dW2, dW), each (n_paths, n_steps_mc).

    ``dW1, dW2`` are correlated with ``Corr = rho``; ``dW`` is independent
    of both. All three scale as sqrt(dt).
    """
    Z1, Z2 = _draw_correlated_normals(rng, n_paths, n_steps_mc, rho)
    Zi = rng.standard_normal((n_paths, n_steps_mc))
    sqdt = np.sqrt(dt)
    return sqdt * Z1, sqdt * Z2, sqdt * Zi


def _euler_step_inplace(S, sigma, theta, dW1_n, dW2_n, dW_n, n, dt,
                        r, kappa, nu, lam, eta, theta0,
                        theta_driver='bm'):
    """Advance 1D state (S, sigma, theta) by one Euler step in place.

    Ordering: S is updated first (uses old sigma), then sigma (uses old theta),
    then theta (updates itself). All three arrays are shape ``(batch,)``.

    ``theta_driver`` selects the SDE for theta:
      * ``'bm'`` : dtheta = lam dt + eta dW  (standard BM; matches the LLH
        formula's PDE, which assumes constant diffusion ``eta``).
      * ``'gbm'``: dtheta = lam dt + (theta - theta0 + eta - lam*t) dW
        (Ito-integrated form of the report's original dB = B dW driver).
    """
    S *= np.exp((r - 0.5 * sigma * sigma) * dt + sigma * dW1_n)
    sigma += kappa * (theta - sigma) * dt + nu * dW2_n
    if theta_driver == 'bm':
        theta += lam * dt + eta * dW_n
    else:  # 'gbm'
        theta += lam * dt + (theta - theta0 + eta - lam * n * dt) * dW_n


def _milstein_step_inplace(S, sigma, theta, dW1_n, dW2_n, dW_n, xi_n, n, dt,
                           r, kappa, nu, lam, eta, rho, theta0):
    """Advance 1D state (S, sigma, theta) by one Milstein step in place.

    ``xi_n`` carries a fair Bernoulli ±1 per path for the mixed-noise term.
    Correlated-driver scheme: see ``eur_price_llh/milstein.tex``.
    Ordering matches ``_euler_step_inplace``.
    """
    sig = sigma
    th = theta
    Gamma = th - theta0 + eta - lam * n * dt
    dt2 = dt * dt
    sqrt_1mr2 = np.sqrt(max(0.0, 1.0 - rho * rho))

    log_inc = (
        (r - 0.5 * sig * sig) * dt
        + sig * dW1_n
        - (kappa * sig * (th - sig) + 0.5 * nu * nu) * dt2
        + 0.5 * kappa * (th - sig) * dW1_n * dt
        - 0.5 * nu * sig * dW2_n * dt
        + 0.5 * nu * (dW1_n * dW2_n - rho * dt - sqrt_1mr2 * xi_n * dt)
    )
    S *= np.exp(log_inc)

    sigma += (
        kappa * (th - sig) * dt
        + nu * dW2_n
        + kappa * (lam - kappa * (th - sig)) * dt2
        - 0.5 * kappa * nu * dW2_n * dt
        + 0.5 * kappa * Gamma * dW_n * dt
    )

    theta += (
        lam * dt
        + Gamma * dW_n
        + 0.5 * Gamma * (dW_n * dW_n - dt)
    )


# ─── ODE system & quadrature (private) ───────────────────────────────────────

def _trap_weights(n, simplified=False):
    """Composite trapezoid weights for n uniform-grid nodes."""
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
    """Build the ODE right-hand side for the LLH characteristic-exponent system (Eq. 4)."""
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

    dD_const = u * iPHI - half_PHI2

    def rhs(Y, out=None):
        C, D, E, F, G, H = [Y[..., k] for k in range(6)]
        if out is None:
            out = np.empty_like(Y)
        out[..., 1] = dD_const + two_nu2 * D*D - two_A*D + half_eta2 * E*E
        out[..., 2] = two_kappa*D - A*E + two_nu2 * D*E + two_eta2 * E*F
        out[..., 3] = kappa*E + half_nu2 * E*E + two_eta2 * F*F
        out[..., 4] = kappa*H + 2*lam*F + nu2 * E*H + two_eta2 * F*G
        out[..., 5] = -A*H + lam*E + two_nu2 * D*H + eta2 * E*G
        out[..., 0] = r_iPHI + lam*G + half_nu2 * H*H + nu2 * D + half_eta2 * G*G + eta2 * F
        return out

    return rhs


def _rk4_integrate(rhs, tau, n_steps_ode, n_phi):
    """Integrate the characteristic-exponent ODE from tau=0 to tau via RK4."""
    Z = np.zeros((n_phi, 2), dtype=np.complex128)
    Y = np.stack([Z, Z, Z, Z, Z, Z], axis=-1)
    if tau == 0 or n_steps_ode == 0:
        return {k: Y[..., j] for j, k in enumerate("CDEFGH")}

    dt = tau / n_steps_ode
    k1 = np.empty_like(Y); k2 = np.empty_like(Y)
    k3 = np.empty_like(Y); k4 = np.empty_like(Y)
    Y_tmp = np.empty_like(Y)
    half_dt = 0.5 * dt
    sixth_dt = dt / 6.0

    for _ in range(n_steps_ode):
        rhs(Y, out=k1)
        np.add(Y, half_dt * k1, out=Y_tmp); rhs(Y_tmp, out=k2)
        np.add(Y, half_dt * k2, out=Y_tmp); rhs(Y_tmp, out=k3)
        np.add(Y, dt * k3, out=Y_tmp);      rhs(Y_tmp, out=k4)
        Y += sixth_dt * (k1 + 2*k2 + 2*k3 + k4)

    return {k: Y[..., j] for j, k in enumerate("CDEFGH")}


# ─── Characteristic function transform (private) ─────────────────────────────

def _build_transform_vec(coeffs, S_vec, v_vec, theta_vec, phi):
    """Characteristic-function transform f(phi) over N paths, shape (n_phi, 2, N)."""
    S_vec = np.asarray(S_vec, dtype=float).reshape(-1)
    v_vec = np.asarray(v_vec, dtype=float).reshape(-1)
    theta_vec = np.asarray(theta_vec, dtype=float).reshape(-1)
    x = np.log(S_vec)

    if isinstance(coeffs, dict):
        C = coeffs['C'][..., None]; D = coeffs['D'][..., None]
        E = coeffs['E'][..., None]; F = coeffs['F'][..., None]
        G = coeffs['G'][..., None]; H = coeffs['H'][..., None]
    else:
        C = coeffs[..., 0:1]; D = coeffs[..., 1:2]
        E = coeffs[..., 2:3]; F = coeffs[..., 3:4]
        G = coeffs[..., 4:5]; H = coeffs[..., 5:6]

    v2 = (v_vec ** 2)[None, None, :]
    v = v_vec[None, None, :]
    th = theta_vec[None, None, :]
    xv = x[None, None, :]
    PHI = phi[:, None, None]

    quad = (C + D*v2 + E*v*th + F*th*th + G*th + H*v + 1j*PHI*xv)
    return np.exp(quad)


def _compute_P_vec(f, K, phi, w):
    """Risk-neutral probabilities P1, P2 via trapezoid quadrature of the transform."""
    lnK = np.log(K)
    kernel = np.exp(-1j * phi * lnK) / (1j * phi)
    integrand = np.real(kernel[:, None, None] * f)
    P = 0.5 + (1.0 / np.pi) * np.tensordot(w, integrand, axes=(0, 0))
    return P[0, :], P[1, :]


# ─── Black-Scholes closed forms (public) ─────────────────────────────────────

def price_call_bs(S, K=90, tau=1.0, r=0.05, vol=0.2):
    """Black-Scholes European call price."""
    if tau <= 0:
        return max(0.0, S - K)
    v = vol * np.sqrt(tau)
    d1 = (np.log(S / K) + (r + 0.5 * vol * vol) * tau) / v
    d2 = d1 - v
    from scipy.stats import norm as _norm
    return float(S * _norm.cdf(d1) - K * np.exp(-r * tau) * _norm.cdf(d2))


def price_put_bs(S, K=90, tau=1.0, r=0.05, vol=0.2):
    """Black-Scholes European put price via put-call parity."""
    return price_call_bs(S, K, tau, r, vol) - S + K * np.exp(-r * tau)


# ─── MC pricing (public) ─────────────────────────────────────────────────────

def price_call_mc(paths, K, T, r=None):
    """Monte Carlo European call price from simulated terminal values."""
    n_paths = paths.shape[0]
    disc = np.exp(-r * T)
    Y = disc * np.maximum(paths[:, -1] - K, 0.0)
    price = Y.mean()
    std_err = Y.std(ddof=1) / np.sqrt(n_paths)
    ci95 = (price - 1.96 * std_err, price + 1.96 * std_err)
    return {'price': price, 'std_err': std_err, 'ci_95': ci95, 'n_paths': n_paths}


def price_put_mc(paths, K, T, r=None):
    """Monte Carlo European put price from simulated terminal values."""
    n_paths = paths.shape[0]
    disc = np.exp(-r * T)
    Y = disc * np.maximum(K - paths[:, -1], 0.0)
    price = Y.mean()
    std_err = Y.std(ddof=1) / np.sqrt(n_paths)
    ci95 = (price - 1.96 * std_err, price + 1.96 * std_err)
    return {'price': price, 'std_err': std_err, 'ci_95': ci95, 'n_paths': n_paths}


# ─── Convenience / notebook utilities (public) ───────────────────────────────

def plot(paths, title="Simulated Price Paths"):
    """Quick line plot of simulated price paths."""
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
    """Kolmogorov-Smirnov test for log-normality of positive data."""
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
    """European call and put prices under both LLH closed-form and Monte Carlo."""
    res = model.simulate_prices(S0=S0, T=tau, n_steps_mc=n_steps_mc, n_paths=n_paths)

    llh_call = model.price_call_llh(S=S0, K=K, tau=tau,
                                    vol=model.sigma0, theta=model.theta0,
                                    phi_max=phi_max, n_phi=n_phi, n_steps_ode=n_steps_ode).item()
    llh_put = model.price_put_llh(S=S0, K=K, tau=tau,
                                  vol=model.sigma0, theta=model.theta0,
                                  phi_max=phi_max, n_phi=n_phi, n_steps_ode=n_steps_ode).item()

    res_mc_call = price_call_mc(res['S'], K=K, T=tau, r=model.r)
    res_mc_put = price_put_mc(res['S'], K=K, T=tau, r=model.r)

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
    """Print LLH and MC European prices side-by-side."""
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
    """Cached quadrature and ODE data for a fixed maturity tau."""
    tau: float
    phi: np.ndarray
    w: np.ndarray
    coeffs: dict
    coeffs_stacked: np.ndarray
    n_phi: int


@dataclass
class ImprovedSteinStein:
    """LLH model parameters and methods for simulation and pricing.
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
        n_paths: int,
        batch_size: int = 20_000,
        terminal_only: bool = False,
        scheme: str = 'euler',
        theta_driver: str = 'bm',
        **kwargs,
    ) -> dict:
        """Batched simulation of the LLH SDE system.

        Parameters
        ----------
        S0 : float
            Initial spot price.
        T : float
            Time horizon in years.
        n_steps_mc : int
            Number of time steps.
        n_paths : int
            Number of Monte Carlo paths.
        batch_size : int, default 20_000
            Paths processed per batch. Caps the peak memory contribution of
            the three BM-increment arrays to ``3 * batch_size * n_steps_mc * 8 B``
            (plus one Bernoulli array of the same size when ``scheme='milstein'``).
        terminal_only : bool, default False
            When True, discard intermediate states and return only ``S_T`` of
            shape ``(n_paths,)``. Use for European MC when the full grid is
            not needed. Not consumed by any downstream module.
        scheme : {'euler', 'milstein'}, default 'euler'
            Discretization of the LLH SDE system. See
            ``eur_price_llh/generated_report/european_pricing_report.tex``
            §Simulation/Discretization Schemes for the reference formulas.
        theta_driver : {'bm', 'gbm'}, default 'bm'
            Diffusion form of the theta SDE. ``'bm'`` uses the standard
            Brownian driver ``dtheta = lam dt + eta dW`` -- this matches the
            assumption baked into the LLH formula's ODE/PDE. ``'gbm'`` uses
            the Ito-integrated form ``dtheta = lam dt +
            (theta - theta0 + eta - lam*t) dW`` of the report's original
            geometric-BM driver ``dB = B dW``; useful only for back-comparing
            the pre-alignment bias. Euler-only: combining ``theta_driver='bm'``
            with ``scheme='milstein'`` raises ``ValueError``.

        Returns
        -------
        dict
            If ``terminal_only=False`` (default):
                ``'S'``         : (n_paths, n_steps_mc+1) -- prices incl. S0
                ``'sigma_hat'`` : (n_paths, n_steps_mc)   -- sigma[:, :-1]
                ``'theta'``     : (n_paths, n_steps_mc+1) -- theta incl. theta0
                ``'dt'``        : float
            If ``terminal_only=True``:
                ``'S_T'`` : (n_paths,) -- terminal prices only
                ``'dt'``  : float

        Raises
        ------
        ValueError
            If ``scheme`` is not in ``{'euler', 'milstein'}``, if
            ``theta_driver`` is not in ``{'bm', 'gbm'}``, or if
            ``theta_driver='bm'`` is passed with ``scheme='milstein'``.
        TypeError
            If any unknown kwarg is passed.
        """
        if kwargs:
            raise TypeError(
                f"simulate_prices() got unexpected kwargs: {sorted(kwargs)}"
            )
        if scheme not in ('euler', 'milstein'):
            raise ValueError(
                f"unknown scheme {scheme!r}; expected 'euler' or 'milstein'"
            )
        if theta_driver not in ('bm', 'gbm'):
            raise ValueError(
                f"unknown theta_driver {theta_driver!r}; expected 'bm' or 'gbm'"
            )
        if scheme == 'milstein' and theta_driver == 'bm':
            raise ValueError(
                "theta_driver='bm' is Euler-only; milstein retains the 'gbm' "
                "form. Use scheme='euler' or pass theta_driver='gbm'."
            )

        dt = T / n_steps_mc
        n_batches = (n_paths + batch_size - 1) // batch_size
        sub_seeds = np.random.SeedSequence(self.seed).spawn(n_batches)
        use_milstein = (scheme == 'milstein')

        if terminal_only:
            chunks = []
            remaining = n_paths
            for b in range(n_batches):
                bs = min(batch_size, remaining)
                remaining -= bs
                rng = np.random.default_rng(sub_seeds[b])
                dW1, dW2, dW = _increments_from_normals(self.rho, dt, bs, n_steps_mc, rng)
                if use_milstein:
                    xi = (2 * rng.integers(0, 2, size=(bs, n_steps_mc)) - 1).astype(np.float64)

                S = np.full(bs, S0, dtype=np.float64)
                sigma = np.full(bs, self.sigma0, dtype=np.float64)
                theta = np.full(bs, self.theta0, dtype=np.float64)
                for n in range(n_steps_mc):
                    if use_milstein:
                        _milstein_step_inplace(
                            S, sigma, theta,
                            dW1[:, n], dW2[:, n], dW[:, n], xi[:, n],
                            n, dt, self.r, self.kappa, self.nu, self.lam,
                            self.eta, self.rho, self.theta0,
                        )
                    else:
                        _euler_step_inplace(
                            S, sigma, theta,
                            dW1[:, n], dW2[:, n], dW[:, n],
                            n, dt, self.r, self.kappa, self.nu, self.lam,
                            self.eta, self.theta0,
                            theta_driver=theta_driver,
                        )
                chunks.append(S.copy())
                del dW1, dW2, dW, S, sigma, theta
                if use_milstein:
                    del xi

            return {'S_T': np.concatenate(chunks), 'dt': dt}

        # Full-grid mode
        S = np.empty((n_paths, n_steps_mc + 1), dtype=np.float64)
        sigma = np.empty_like(S)
        theta = np.empty_like(S)
        S[:, 0] = S0
        sigma[:, 0] = self.sigma0
        theta[:, 0] = self.theta0

        r = self.r
        kappa = self.kappa
        nu = self.nu
        lam = self.lam
        eta = self.eta
        rho = self.rho
        theta0 = self.theta0
        dt2 = dt * dt
        sqrt_1mr2 = np.sqrt(max(0.0, 1.0 - rho * rho))

        remaining = n_paths
        start = 0
        for b in range(n_batches):
            bs = min(batch_size, remaining)
            remaining -= bs
            end = start + bs
            rng = np.random.default_rng(sub_seeds[b])
            dW1, dW2, dW = _increments_from_normals(rho, dt, bs, n_steps_mc, rng)
            if use_milstein:
                xi = (2 * rng.integers(0, 2, size=(bs, n_steps_mc)) - 1).astype(np.float64)

            Sv = S[start:end]
            sv = sigma[start:end]
            tv = theta[start:end]
            for n in range(n_steps_mc):
                sig_n = sv[:, n]
                th_n = tv[:, n]
                dW1_n = dW1[:, n]
                dW2_n = dW2[:, n]
                dW_n = dW[:, n]

                if use_milstein:
                    Gamma = th_n - theta0 + eta - lam * n * dt
                    xi_n = xi[:, n]
                    log_inc = (
                        (r - 0.5 * sig_n * sig_n) * dt
                        + sig_n * dW1_n
                        - (kappa * sig_n * (th_n - sig_n) + 0.5 * nu * nu) * dt2
                        + 0.5 * kappa * (th_n - sig_n) * dW1_n * dt
                        - 0.5 * nu * sig_n * dW2_n * dt
                        + 0.5 * nu * (dW1_n * dW2_n - rho * dt - sqrt_1mr2 * xi_n * dt)
                    )
                    Sv[:, n + 1] = Sv[:, n] * np.exp(log_inc)
                    sv[:, n + 1] = (
                        sig_n
                        + kappa * (th_n - sig_n) * dt
                        + nu * dW2_n
                        + kappa * (lam - kappa * (th_n - sig_n)) * dt2
                        - 0.5 * kappa * nu * dW2_n * dt
                        + 0.5 * kappa * Gamma * dW_n * dt
                    )
                    tv[:, n + 1] = (
                        th_n
                        + lam * dt
                        + Gamma * dW_n
                        + 0.5 * Gamma * (dW_n * dW_n - dt)
                    )
                else:
                    # Euler (original v1 path).
                    Sv[:, n + 1] = Sv[:, n] * np.exp(
                        (r - 0.5 * sig_n * sig_n) * dt + sig_n * dW1_n
                    )
                    sv[:, n + 1] = sig_n + kappa * (th_n - sig_n) * dt + nu * dW2_n
                    if theta_driver == 'bm':
                        tv[:, n + 1] = th_n + lam * dt + eta * dW_n
                    else:  # 'gbm'
                        tv[:, n + 1] = (
                            th_n + lam * dt
                            + (th_n - theta0 + eta - lam * n * dt) * dW_n
                        )

            del dW1, dW2, dW
            if use_milstein:
                del xi
            start = end

        return {'S': S, 'sigma_hat': sigma[:, :-1], 'theta': theta, 'dt': dt}

    def llh_precompute_tau(self, tau, phi_max, n_phi, n_steps_ode, eps0=1e-6) -> LLHPrecompute:
        """Precompute quadrature grid and ODE coefficients for a fixed maturity."""
        phi = _phi_grid(phi_max, n_phi, eps0)
        dphi = phi_max / (n_phi - 1)
        w = dphi * _trap_weights(n_phi, simplified=True)
        rhs = _rhs_factory(phi, self.r, self.kappa, self.nu, self.lam, self.eta, self.rho)
        coeffs = _rk4_integrate(rhs, tau, n_steps_ode, n_phi)
        coeffs_stacked = np.stack([coeffs[k] for k in "CDEFGH"], axis=-1)
        return LLHPrecompute(tau=float(tau), phi=phi, w=w, coeffs=coeffs,
                             coeffs_stacked=coeffs_stacked, n_phi=n_phi)

    def price_call_llh(self, S, K, tau, vol, theta, phi_max=300.0, n_phi=513,
                       n_steps_ode=128, pre=None, eps0=1e-6):
        """Vectorized European call prices under the LLH characteristic-function formula."""
        S = np.asarray(S, dtype=float).reshape(-1)
        vol = np.asarray(vol, dtype=float).reshape(-1)
        theta = np.asarray(theta, dtype=float).reshape(-1)
        if S.shape != vol.shape or S.shape != theta.shape:
            raise ValueError("S, vol, theta must have the same shape")
        if pre is None:
            pre = self.llh_precompute_tau(tau, phi_max=phi_max, n_phi=n_phi,
                                          n_steps_ode=n_steps_ode, eps0=eps0)
        f = _build_transform_vec(pre.coeffs_stacked, S, vol, theta, pre.phi)
        P1, P2 = _compute_P_vec(f, K, pre.phi, pre.w)
        disc = np.exp(-self.r * tau)
        return np.real(S * P1 - K * disc * P2)

    def price_put_llh(self, S, K, tau, vol, theta, phi_max, n_phi, n_steps_ode,
                      pre=None, eps0=1e-6):
        """Vectorized European put prices under LLH via put-call parity."""
        call = self.price_call_llh(S, K, tau, vol, theta, phi_max, n_phi, n_steps_ode,
                                   pre=pre, eps0=eps0)
        return call - S + K * np.exp(-self.r * tau)

    # ---------- MC European pricing ----------

    def price_call_mc(self, sim_out, K):
        """European call price via MC on pre-simulated paths."""
        T = sim_out['dt'] * (sim_out['S'].shape[1] - 1)
        return price_call_mc(sim_out['S'], K=K, T=T, r=self.r)

    def price_put_mc(self, sim_out, K):
        """European put price via MC on pre-simulated paths."""
        T = sim_out['dt'] * (sim_out['S'].shape[1] - 1)
        return price_put_mc(sim_out['S'], K=K, T=T, r=self.r)

    # ---------- American pricing (delegates to amerPrice) ----------

    def price_american_put(self, sim_out, K, basis_order=3,
                           basis_type='laguerre',
                           use_cv=True, improved=True, ridge=0.0,
                           euro_method='llh', floor_method='bs',
                           phi_max=300.0, n_phi=513, n_steps_rk4=128, eps0=1e-6):
        """American put price via LSM with optional Rasmussen control variates."""
        import amerPrice as ap
        return ap.price_american_put_lsm_llh(
            self, sim_out, K, basis_order=basis_order,
            basis_type=basis_type,
            use_cv=use_cv, improved=improved, ridge=ridge,
            euro_method=euro_method, floor_method=floor_method,
            phi_max=phi_max, n_phi=n_phi, n_steps_rk4=n_steps_rk4, eps0=eps0,
        )
