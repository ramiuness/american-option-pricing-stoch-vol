# American Option Pricing under the LLH Stochastic Volatility Model

Pricing American put options via Longstaff-Schwartz Monte Carlo (LSM) with Rasmussen control variates, under the Lin-Lin-He (2024) improved Stein-Stein stochastic volatility model.

**Author:** Rami Younes
**Supervisor:** Prof. Fabian Bastin, Université de Montréal

---

## Overview

This project extends LSM American option pricing from Black-Scholes to the LLH stochastic volatility model. European option prices — computed semi-analytically via characteristic functions — serve as Rasmussen-style control variates to reduce Monte Carlo variance.

The central research question is whether control variates remain effective under stochastic volatility, as they are under GBM.

---

## Model

The LLH model (Lin, Lin & He 2024) is an improved Stein-Stein process:

```
dS / S  =  r dt + σ_t dW¹
dσ      =  κ(θ_t − σ) dt + ν dW²
dθ      =  λ dt + η dB,    B_t = exp(W_t − t/2)
Cov(W¹, W²) = ρ t
```

The θ process drives a stochastic long-run mean for σ, making this a two-factor stochastic volatility model. The model nests:
- **Black-Scholes** when κ = ν = λ = η = ρ = 0
- **Stein-Stein (1991)** when λ = η = ρ = 0

Reference parameters from Lin-Lin-He Table 2:

| r | κ | ν | λ | η | ρ | σ₀ | θ₀ |
|---|---|---|---|---|---|----|----|
| 0.3943 | 4.9394 | 0.4 | 0.3115 | 0.4112 | 0.1691 | 0.2924 | 0.1319 |

---

## Repository Structure

```
src/
  priceModels.py        # LLH simulation + European pricing (ODE/quadrature)
  amOptPricer.py        # LSM + Rasmussen control variates (production)

notebooks/
  demo.ipynb                  # End-to-end pricing demo
  european_pricing.ipynb      # European price validation + S&Z Table 2 comparison
  char_func_symbolic.ipynb    # SymPy proof: ansatz non-closure (degree-4 argument)

reports/
  eur_price_llh.pdf           # Theoretical derivation of the European price formula
```

---

## Pricing Methods

### European Price (Semi-Analytic)

The European call price follows the Fourier inversion formula:

```
C = S · P₁ − K · e^{−rτ} · P₂
```

where P₁, P₂ are computed via the characteristic functions f₁, f₂ obtained by solving a system of autonomous Riccati ODEs (RK4 integration).

**Precision presets:**

| Preset | φ_max | n_φ | n_steps_ode | Bias (τ=1) |
|--------|-------|-----|-------------|------------|
| STANDARD | 300 | 513 | 128 | < 5% |
| HIGH_ACCURACY | 400 | 1025 | 256 | < 2% |

### American Put Price (LSM + CV)

The Longstaff-Schwartz algorithm uses Laguerre polynomial regression to estimate continuation values at each exercise date. European LLH prices serve as Rasmussen control variates at each backward step, reducing estimator variance.

---

## Theoretical Notes

### Ansatz Non-Closure

The covariance matrix of (σ, θ) is **quadratic** in the state, not affine. This means the exponential-quadratic ansatz used to derive the characteristic function ODEs is mathematically unjustified. `notebooks/char_func_symbolic.ipynb` proves this symbolically: the residual P = L[y]/y has total degree **4** in (σ, θ) for the full LLH PDE, collapsing to degree 2 only in the Stein-Stein limit. The ODE system from LLH is therefore an approximation.

### Exact Alternative (Planned)

Conditioning on the GBM driver filtration ℱ^W renders θ(t) deterministic, reducing the characteristic function PDE to a 1D problem with an exact quadratic ansatz. The outer expectation over ℱ^W is Gaussian and has a closed form. This approach is expected to eliminate the residual 3–5% bias.

---

## Quick Start

```python
import sys
sys.path.insert(0, 'src')

import priceModels as pm
import amOptPricer as aop

# Simulate LLH paths
rng = pm.np.random.default_rng(42)
params = dict(r=0.3943, kappa=4.9394, nu=0.4, lam=0.3115,
              eta=0.4112, rho=0.1691, sigma0=0.2924, theta0=0.1319)

S, sigma_hat = pm.simulate_llh(rng, S0=100, T=1.0, n_paths=10_000,
                                n_steps_mc=50, **params)

# Price American put via LSM + control variates
result = aop.price_american_put(S, sigma_hat, K=100, T=1.0, **params)
print(result)
```

---

## Dependencies

- Python 3.9+
- NumPy, SciPy, Matplotlib
- SymPy (for `char_func_symbolic.ipynb`)

---

## References

- Lin, Lin & He (2024). *Improved Stein-Stein stochastic volatility model.*
- Longstaff & Schwartz (2001). *Valuing American options by simulation.* RFS.
- Rasmussen (2005). *Smoothing and interpolation with model selection.* PhD thesis.
- Schöbel & Zhu (1999). *Stochastic volatility with an Ornstein-Uhlenbeck process.* EF.
- Stein & Stein (1991). *Stock price distributions with stochastic volatility.* RFS.
