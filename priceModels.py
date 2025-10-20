from dataclasses import dataclass 
import amOptPricer as aop 
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import lognorm, kstest

#############################################################################
######################## LLH Model Simulation ###############################
#############################################################################

# General utilities for the LLH model simulation

def draw_correlated_normals(rng, n_paths: int, n_steps_mc: int, rho: float):
    """Return Z1, Z2 ~ N(0,1) with Corr(Z1, Z2)=rho. Shapes: (n_paths, n_steps_mc)."""
    Z1 = rng.standard_normal((n_paths, n_steps_mc))
    Z2i = rng.standard_normal((n_paths, n_steps_mc))
    Z2 = rho * Z1 + np.sqrt(max(0.0, 1.0 - rho**2)) * Z2i
    return Z1, Z2

def brownian_from_normals(Z: np.ndarray, dt: float):
    """dW = sqrt(dt)*Z, W = cumsum(dW, axis=1)."""
    dW = np.sqrt(dt) * Z
    W = np.cumsum(dW, axis=1)
    return dW, W

def gbm_from_formula(W: np.ndarray, dt: float):
    """B_t = exp(W_t - t/2), ΔB = B - B_prev with B_0=1."""
    n = W.shape[1]
    t_grid = dt * np.arange(1, n + 1)[None, :] # broadcast to shape (1, n); same effect as [np.newaxis, :]
    B = np.exp(W - 0.5 * t_grid)
    return B 

def causal_exp_conv(X: np.ndarray, a1: float):
    """
    Causal exponential-kernel convolution:
      s_j = sum_{i=1}^j (a1**i) * X_{j-i},  j=1..n
    Implemented via reweight + cumsum (fully vectorized).
    """
    n = X.shape[1]
    j = np.arange(n)
    Y = X * (a1 ** (-j))[None, :]
    P = np.cumsum(Y, axis=1)
    return P * (a1 ** (j + 1))[None, :]

def sigma_hat_from_components(
    n_steps_mc: int, dt: float, kappa: float, sigma0: float, theta0: float, lam: float,
    nu: float, eta: float, W2: np.ndarray, B: np.ndarray
):
    """Compute sigma_hat's using the reported formula (15) and given inputs. Returns (n_paths, n_steps_mc)."""
    idx = np.arange(1, n_steps_mc + 1)[None, :]
    exp_kdt_idx = np.exp(-kappa * idx * dt)
    a1 = np.exp(-kappa * dt)
    term_W2  = causal_exp_conv(W2, a1)
    term_Bm1 = causal_exp_conv(B - 1.0, a1)
    sigma_hat = (
        exp_kdt_idx * (sigma0 + lam - theta0)
        + theta0
        + lam * (idx * dt - 1.0)
        + nu * W2
        - (nu * kappa * dt) * term_W2
        + (eta * kappa * dt) * term_Bm1
    )
    return sigma_hat

def multiplicative_euler_prices(S0: float, r: float, sigma_hat: np.ndarray, dW1: np.ndarray, dt: float):
    """S = S0 * cumprod(1 + r*dt + σ̂ * dW1) along time. Returns shape (n_paths, n_steps_mc)."""
    increments = 1.0 + r * dt + sigma_hat * dW1
    return S0 * np.cumprod(increments, axis=1)

def plot(paths, title="Simulated Price Paths"):
    """Plot the simulated prices."""
    plt.plot(paths)
    plt.title(title)
    plt.xlabel("Time Steps")
    plt.ylabel("Price")
    plt.grid()
    plt.tight_layout()
    plt.show()


def test_lognormality(data):
    """
    Perform the Kolmogorov-Smirnov test to check if the data follows a lognormal distribution.
    
    Parameters:
    data (array-like): The data to be tested.
    
    Returns:
    D (float): KS statistic.
    p_value (float): p-value of the test.
    """
    data = data[data > 0]  # lognormal is only defined for positive values

    # Estimate lognormal parameters from data (MLE)
    shape, loc, scale = lognorm.fit(data, floc=0)  # force location=0

    # Perform the Kolmogorov-Smirnov test
    D, p_value = kstest(data, 'lognorm', args=(shape, loc, scale))

    print(f"KS statistic: {D:.4f}, p-value: {p_value:.4g}")
    if p_value > 0.05:
        print("Fail to reject null: data may be lognormal.")
    else:
        print("Reject null: data is not lognormal.")


#############################################################################
########## European price formula under the LLH Model  ######################
#############################################################################

# ---------- Quadrature utilities ----------
# Simpson quadrature to compute P_j's, fixed-step RK4 tau, complex128, batched over phi, 1<= j<= 2

def simpson_weights(n):
    """
    Composite Simpson weights for n nodes (n must be odd).
    Returns array of shape (n,), to be scaled by step/3.
    """
    if n % 2 == 0:
        raise ValueError("Simpson requires odd number of nodes.")
    w = np.ones(n, dtype=np.float64)
    w[1:-1:2] = 4.0
    w[2:-2:2] = 2.0
    return w

def phi_grid(phi_max, n_phi, eps0=1e-6):
    """Uniform φ-grid on [0, phi_max], with φ[0]=eps0 to avoid division by zero."""
    phi = np.linspace(0.0, phi_max, n_phi, dtype=np.float64)
    phi[0] = eps0
    return phi

# ---------- ODE system & integrator ----------
# Coefficients C,D,E,F,G,H via RK4 integration of ODE system Eq. (4) of Lin-Lin-He paper

def rhs_factory(phi, r, kappa, nu, lam, eta, rho): 
    """
    Returns RHS (callable) for ODE system Eq. (4) of Lin-Lin-He paper for both j=1,2, batched over φ.
    Shapes:
      - Y: (n_phi, 2, 6) complex, with components (C,D,E,F,G,H)
      - dY: same shape
    """
    u = np.array([+0.5, -0.5])[None, :]  # j=1: u=0.5; j=2: u=-0.5
    b = np.array([1.0, 0.0])[None, :]    # j=1: b=1;   j=2: b=0
    PHI = phi[:, None]
    A = kappa - nu * rho * (b + 1j * PHI)  # (n_phi, 2)

    def rhs(Y):
        C, D, E, F, G, H = [Y[..., k] for k in range(6)] # Y[..., k]: index the last axis by k, keep all others.
    
        dD = (u * 1j*PHI - 0.5 * PHI ** 2) + 2*nu**2 * D*D - 2*A*D + 0.5*eta**2 * E*E
        dE = 2*kappa*D - A*E + 2*nu**2 * D*E + 2*eta**2 * E*F
        dF = kappa*E + 0.5*nu**2 * E*E + 2*eta**2 * F*F
        dG = kappa*H + 2*lam*F + nu**2 * E*H + 2*eta**2 * F*G
        dH = -A*H + lam*E + 2*nu**2 * D*H + eta**2 * E*G
        dC = r*1j * PHI + lam*G + 0.5*nu**2 * H*H + nu**2 * D + 0.5*eta**2 * G*G + eta**2 * F
        return np.stack([dC,dD,dE,dF,dG,dH], axis=-1) # build a "channel" dimension, where channel 0 is dC, channel 1 is dD, …, channel 5 is dH
    
    return rhs

def rk4_integrate(rhs, tau, n_steps_ode, n_phi):
    """Integrate from τ=0 to τ with RK4. Returns dict of coeff arrays (n_phi,2) with keys C,D,E,F,G,H.
    rhs: callable(Y) -> dY
    tau: final time
    n_steps_ode: number of RK4 steps for ODE integration
    n_phi: number of φ nodes

    """
    Z = np.zeros((n_phi, 2), dtype=np.complex128)
    Y = np.stack([Z,Z,Z,Z,Z,Z], axis=-1) # shape (n_phi, 2, 6); initial values C=D=E=F=G=H=0
    if tau == 0 or n_steps_ode == 0:
        return {k:Y[...,j] for j,k in enumerate("CDEFGH")}
    dt = tau / n_steps_ode
    for _ in range(n_steps_ode):
        k1 = rhs(Y)
        k2 = rhs(Y + 0.5*dt*k1)
        k3 = rhs(Y + 0.5*dt*k2)
        k4 = rhs(Y + dt*k3)
        Y = Y + (dt/6.0)*(k1+2*k2+2*k3+k4)
    return {k:Y[...,j] for j,k in enumerate("CDEFGH")}

# ---------- Transform & pricing (scalar) ----------

# def build_transform(coeffs, S, v, theta, phi):
#     """
#     f_j(τ; φ) = exp(C + D v^2 + E v θ + F θ^2 + G θ + H v + i φ ln S).
#     Returns (n_phi,2) complex.
#     """
#     x = np.log(S)
#     quad = (coeffs['C']

#             + coeffs['D']*v**2
#             + coeffs['E']*v*theta
#             + coeffs['F']*theta**2
#             + coeffs['G']*theta
#             + coeffs['H']*v
#             + 1j*phi[:,None]*x)
#     return np.exp(quad)

# def compute_P(f, K, phi, w):
#     """Compute P1,P2 via Simpson quadrature (scalar path)."""
#     lnK = np.log(K)
#     kernel = np.exp(-1j*phi*lnK)/(1j*phi)        # (n_phi,)
#     integrand = np.real(kernel[:,None]*f)        # (n_phi,2)
#     P = 0.5 + (1/np.pi)*(w[:,None]*integrand).sum(axis=0)
#     return float(P[0]), float(P[1])

# ---------- Vectorized transform & pricing (paths) ----------

def build_transform_vec(coeffs, S_vec, v_vec, theta_vec, phi):
    """
    Vectorized transform over paths:
      f(φ) = exp(C + D v^2 + E v θ + F θ^2 + G θ + H v + i φ ln S).
    Returns array with shape (n_phi, 2, N) where N = len(S_vec).

    coeffs: dict with keys 'C','D','E','F','G','H' each of shape (n_phi,2)
    S_vec: array of shape (N,)
    v_vec: array of shape (N,)
    theta_vec: array of shape (N,)
    phi: array of shape (n_phi,)
    """

    S_vec = np.asarray(S_vec, dtype=float).reshape(-1)
    v_vec = np.asarray(v_vec, dtype=float).reshape(-1)
    theta_vec = np.asarray(theta_vec, dtype=float).reshape(-1)

    x  = np.log(S_vec)                 # (N,)
    C  = coeffs['C'][..., None]        # (n_phi,2,1)
    D  = coeffs['D'][..., None]
    E  = coeffs['E'][..., None]
    F  = coeffs['F'][..., None]
    G  = coeffs['G'][..., None]
    H  = coeffs['H'][..., None]
    v2 = (v_vec**2)[None, None, :]     # (1,1,N)
    v  = v_vec[None, None, :]
    th = theta_vec[None, None, :]
    xv = x[None, None, :]
    PHI = phi[:, None, None]           # (n_phi,1,1)

    quad = (C + D*v2 + E*v*th + F*th*th + G*th + H*v + 1j*PHI*xv)  # (n_phi,2,N)
    return np.exp(quad)

def compute_P_vec(f, K, phi, w):
    """
    Vectorized Simpson integration for P1,P2.
    f: (n_phi, 2, N)  -> returns P1,P2 with shape (N,)
    """
    lnK = np.log(K)
    kernel = np.exp(-1j*phi*lnK) / (1j*phi)      # (n_phi,)
    integrand = np.real(kernel[:, None, None] * f)  # (n_phi,2,N)
    P = 0.5 + (1/np.pi) * np.tensordot(w, integrand, axes=(0,0))  # (2,N)
    return np.real(P[0, :]), np.real(P[1, :])

# ---------- Precompute holder ----------

@dataclass
class LLHPrecompute:
    """Precomputed phi-grid, Simpson weights, and RK4 coefficients for a given tau."""
    tau: float
    phi: np.ndarray
    w: np.ndarray
    coeffs: dict   # keys: 'C','D','E','F','G','H'
    n_phi: int

@dataclass
class ImprovedSteinStein:
    """Simulator for the four-step price algorithm (vectorized over paths)."""
    r: float
    sigma0: float
    theta0: float
    rho: float
    kappa: float
    lam: float
    nu: float
    eta: float
    seed: int | None = None

    # Complexity: Time O(M*n), Space O(M*n)
    def simulate_prices(
        self,
        S0: float,
        T: float,
        n_steps_mc: int,
        n_paths: int

    ) -> dict:
        """
        Simulate n_paths price paths.

        Parameters:
          S0         : initial price (scalar)
          T          : time horizon (scalar)
          n_steps_mc : number of Monte Carlo time steps (scalar)
          n_paths    : number of paths to simulate (scalar)

        Returns: dict with dt, W, B, W2, sigma_hat, S (all arrays shape (n_paths, n_steps_mc + 1)).
        """
        dt = T / n_steps_mc
        rng = np.random.default_rng(self.seed)

        # (i) Brownian for B (optionally injected)
        Zb = rng.standard_normal((n_paths, n_steps_mc))
        dW, W = brownian_from_normals(Zb, dt)
        B = gbm_from_formula(W, dt)

        # (ii) Correlated Brownian motions for price and sigma drivers
        Z1, Z2 = draw_correlated_normals(rng, n_paths, n_steps_mc, self.rho)
        dW1, _ = brownian_from_normals(Z1, dt)
        dW2, W2 = brownian_from_normals(Z2, dt)

        # (iii) sigma_hat
        sigma_hat = sigma_hat_from_components(
            n_steps_mc, dt, self.kappa, self.sigma0, self.theta0, self.lam, self.nu, self.eta, W2, B
        )

        # (iv) prices
        S = multiplicative_euler_prices(S0, self.r, sigma_hat, dW1, dt)
        S0_ = S0*np.ones((S.shape[0], 1))

        return {"dt": dt, "W": W, "B": B, "W2": W2, "sigma_hat": sigma_hat,
                "S": np.concatenate([S0_, S], axis=1)}

    # ---------- European call pricing (scalar, with optional precompute) ----------
    def llh_precompute_tau(self, tau, phi_max, n_phi, n_steps_ode, eps0=1e-6) -> LLHPrecompute:
        """Precompute phi-grid, Simpson weights and RK4 coeffs at fixed tau.

        Parameters:
          tau         : time to maturity
          phi_max     : maximum φ value for integration
          n_phi       : number of φ nodes (must be odd)
          n_steps_ode : number of RK4 steps for ODE integration
          eps0        : small value to avoid division by zero at φ=0
        """
        if n_phi % 2 == 0:
            raise ValueError("n_phi must be odd. Simpson requires odd number of nodes.")
        phi = phi_grid(phi_max, n_phi, eps0)
        w = simpson_weights(n_phi) * ((phi_max)/(n_phi-1)/3.0)
        rhs = rhs_factory(phi, self.r, self.kappa, self.nu, self.lam, self.eta, self.rho)
        coeffs = rk4_integrate(rhs, tau, n_steps_ode, n_phi)
        return LLHPrecompute(tau=float(tau), phi=phi, w=w, coeffs=coeffs, n_phi=n_phi)

    # def price_call_llh_scalar(self, S, K, tau, vol, theta,
    #                    phi_max=200.0, n_phi=1025, n_steps=252, eps0=1e-6, pre=None): 
    #     """European call under Lin–Lin–He (scalar). If `pre` is provided, reuse its τ-setup."""
    #     if pre is None:
    #         if n_phi%2==0: 
    #             raise ValueError("n_phi must be odd.\n Simpson requires odd number of nodes.")
    #         phi = phi_grid(phi_max,n_phi,eps0)
    #         w = simpson_weights(n_phi)*((phi_max)/(n_phi-1)/3.0)
    #         rhs = rhs_factory(phi,self.r,self.kappa,self.nu,self.lam,self.eta,self.rho)
    #         coeffs = rk4_integrate(rhs,tau,n_steps,n_phi)
    #     else:
    #         phi = pre.phi
    #         w = pre.w
    #         coeffs = pre.coeffs

    #     f = build_transform(coeffs,S,vol,theta,phi)
    #     P1,P2 = compute_P(f,K,phi,w)
    #     return float(np.real(S*P1 - K*np.exp(-self.r*tau)*P2))

    # ---------- European call pricing (vectorized over paths for fixed τ) ----------

    def price_call_llh(self, S, K, tau, vol, theta, phi_max, n_phi, n_steps_ode, pre=None, eps0=1e-6):
        """
        Vectorized European call prices under Lin–Lin–He for a fixed tau.

        Parameters:
          S           : spot prices (scalar or array)
          K           : strike price
          tau         : time to maturity
          vol         : current volatility σ_t (scalar or array, matching S)
          theta       : current long-term mean θ_t (scalar or array, matching S)
          phi_max     : maximum φ value for Simpson integration
          n_phi       : number of φ nodes (must be odd)
          n_steps_ode : number of RK4 steps for ODE integration
          pre         : optional precomputed LLHPrecompute object
          eps0        : small value to avoid division by zero at φ=0

        Returns: np.ndarray of shape (N,), matching S.
        """
        S = np.asarray(S, dtype=float).reshape(-1)
        vol = np.asarray(vol, dtype=float).reshape(-1)
        theta = np.asarray(theta, dtype=float).reshape(-1)
        if S.shape != vol.shape or S.shape != theta.shape:
            raise ValueError("S , vol, theta must have the same shape")

        if pre is None:
            pre = self.llh_precompute_tau(tau, phi_max=phi_max, n_phi=n_phi, n_steps_ode=n_steps_ode, eps0=eps0)

        f = build_transform_vec(pre.coeffs, S, vol, theta, pre.phi)  # (n_phi,2,N)
        P1, P2 = compute_P_vec(f, K, pre.phi, pre.w)                     # (N,)
        disc = np.exp(-self.r * tau)
        return np.real(S * P1 - K * disc * P2)

        # ---------- European put pricing (vectorized over paths for fixed τ) ----------
    def price_put_llh(self, S, K, tau, vol, theta, phi_max, n_phi, n_steps_ode, pre=None, eps0=1e-6):
        """
        Vectorized European put prices under Lin–Lin–He for a fixed tau (via put-call parity).

        Parameters:
          S           : spot prices (scalar or array)
          K           : strike price
          tau         : time to maturity
          vol         : current volatility σ_t (scalar or array, matching S)
          theta       : current long-term mean θ_t (scalar or array, matching S)
          phi_max     : maximum φ value for Simpson integration
          n_phi       : number of φ nodes (must be odd)
          n_steps_ode : number of RK4 steps for ODE integration
          pre         : optional precomputed LLHPrecompute object
          eps0        : small value to avoid division by zero at φ=0

        Returns: np.ndarray of shape (N,), matching S.
        """
        call = self.price_call_llh(S, K, tau, vol, theta, phi_max, n_phi, n_steps_ode, pre=pre, eps0=eps0)
        return call - S + K * np.exp(-self.r * tau)
    
    

def compare_european_prices(model, S0=100.0, K=90.0, tau=1.0,
                           n_steps_mc=252, n_paths=50000,
                           phi_max=300.0, n_phi=513, n_steps_ode=128):
    """
    Compute the prices of European call and put using the LLH formula and MC simulations.

    Parameters:
      model       : ImprovedSteinStein model instance
      S0          : initial spot price
      K           : strike price
      tau         : time to maturity
      n_steps_mc  : number of MC time discretization steps
      n_paths     : number of MC paths
      phi_max     : maximum φ value for Simpson integration
      n_phi       : number of φ nodes (must be odd)
      n_steps_ode : number of RK4 steps for ODE integration
    """

    # Simulate paths for MC pricing
    res = model.simulate_prices(S0=S0, T=tau, n_steps_mc=n_steps_mc, n_paths=n_paths)

    # European call price using LLH analytical formula
    price_call = model.price_call_llh(S=S0, K=K, tau=tau,
                                      vol=model.sigma0, theta=model.theta0,
                                      phi_max=phi_max, n_phi=n_phi, n_steps_ode=n_steps_ode).item()

    # European put price using LLH analytical formula
    price_put = model.price_put_llh(S=S0, K=K, tau=tau,
                                     vol=model.sigma0, theta=model.theta0,
                                     phi_max=phi_max, n_phi=n_phi, n_steps_ode=n_steps_ode).item()

    # Monte Carlo pricing using the simulated paths
    res_mc = aop.price_call_mc(res['S'], K=K, T=tau, r=model.r)
    price_put_mc = res_mc.get('price') - S0 + K * np.exp(-model.r * tau)

    print(f"European call LLH price:  {price_call},\nEuropean call MC price:  {res_mc.get('price')}, MC 95% CI: {res_mc.get('ci_95')},\nEuropean put LLH price:  {price_put},\nEuropean put MC price:  {price_put_mc}")
    
