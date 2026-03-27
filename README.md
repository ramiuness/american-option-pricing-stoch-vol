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
| 0.01 | 4.9394 | 0.3943 | 0.3115 | 0.4112 | 0.1691 | 0.2924 | 0.1319 |

---

## Repository Structure

```
src/
  priceModels.py        # LLH simulation + European pricing (ODE/quadrature)
  amOptPricer.py        # LSM + Rasmussen control variates (production)
  calibrate.py          # LLH calibration to market options (DE + L-BFGS-B, CRN)

notebooks/
  european_pricing.ipynb      # European price validation + S&Z Table 2 comparison
  char_func_symbolic.ipynb    # SymPy proof: ansatz non-closure (degree-4 argument)
  calibration.ipynb           # Calibrate LLH to S&P 500 options + diagnostic plots
  stylized_facts.ipynb        # Parameter impact on return distributions (5 experiments)

reports/
  llh-formula-report.pdf           # Report on the validatio of the implementatio of llh formula
  llh-formula.pdf           # Theoretical derivation of the European price formula
  pricing-project.pdf        # Theoretical framework for pricing under LLH model
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

## Quick Start

```python
import sys
sys.path.insert(0, 'src')

import priceModels as pm
from amOptPricer import *

# Simulate LLH paths
rng = pm.np.random.default_rng(42)
params = dict(r=0.01, kappa=4.9394, nu=0.3943, lam=0.3115,
              eta=0.4112, rho=0.1691, sigma0=0.2924, theta0=0.1319)

S, sigma_hat = pm.simulate_llh(rng, S0=100, T=1.0, n_paths=10_000,
                                n_steps_mc=50, **params)

# Price American put via LSM + control variates
result = price_american_put(S, sigma_hat, K=100, T=1.0, **params)
print(result)
```

---

## Dependencies

- Python 3.9+
- NumPy, SciPy, Matplotlib
- SymPy (for `char_func_symbolic.ipynb`)
- yfinance (for `calibrate.py` market data fetching)
- statsmodels (for ACF computation in `stylized_facts.ipynb`)

---

## References

- Lin, Lin & He (2024). *Improved Stein-Stein stochastic volatility model.*
- Longstaff & Schwartz (2001). *Valuing American options by simulation.* RFS.
- Rasmussen (2005). *Smoothing and interpolation with model selection.* PhD thesis.
- Schöbel & Zhu (1999). *Stochastic volatility with an Ornstein-Uhlenbeck process.* EF.
- Stein & Stein (1991). *Stock price distributions with stochastic volatility.* RFS.
