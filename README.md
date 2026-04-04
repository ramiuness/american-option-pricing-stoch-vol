# American Option Pricing under the LLH Stochastic Volatility Model

Pricing American put options via Longstaff-Schwartz Monte Carlo (LSM) with Rasmussen control variates, under the Lin-Lin-He (2024) improved Stein-Stein stochastic volatility model.

**Author:** Rami Younes
**Supervisor:** Prof. Fabian Bastin, Universite de Montreal

---

## Overview

This project extends LSM American option pricing from Black-Scholes to the LLH stochastic volatility model. European option prices — computed semi-analytically via characteristic functions — serve as Rasmussen-style control variates to reduce Monte Carlo variance.

The central research question is whether control variates remain effective under stochastic volatility, as they are under GBM.

---

## Model

The LLH model (Lin, Lin & He 2024) is an improved Stein-Stein process:

```
dS / S  =  r dt + sigma_t dW1
dsigma  =  kappa(theta_t - sigma) dt + nu dW2
dtheta  =  lambda dt + eta dB,    B_t = exp(W_t - t/2)
Cov(W1, W2) = rho t
```

The theta process drives a stochastic long-run mean for sigma, making this a two-factor stochastic volatility model. The model nests:
- **Black-Scholes** when kappa = nu = lambda = eta = rho = 0
- **Stein-Stein (1991)** when lambda = eta = rho = 0

Reference parameters from Lin-Lin-He:

| Set | r | kappa | nu | lambda | eta | rho | sigma0 | theta0 |
|-----|---|-------|----|--------|-----|-----|--------|--------|
| Table 1 | 0.01 | 5 | 0.2 | 0.9 | 0.01 | -0.2 | 0.15 | 0.18 |
| Table 2 | 0.01 | 4.9394 | 0.3943 | 0.3115 | 0.4112 | 0.1691 | 0.2924 | 0.1319 |

---

## Repository Structure

```
src/
  priceModels.py        # LLH simulation + European pricing (ODE/quadrature + BS/MC)
  amOptPricer.py        # LSM + Rasmussen control variates; Laguerre and Gaussian RBF bases
  calibrate.py          # LLH calibration to market options (DE + L-BFGS-B)
  reporting.py          # Notebook presentation helpers
  generate_plots.py     # Plot generation for European and American pricing reports
  timing_analysis.py    # Empirical timing analysis of LSM+CV-LLH (scaling, stage breakdown)
  testing.py            # Regression-basis and CI comparison utilities
  *_v0.py               # Archived originals (gitignored)

notebooks/
  european_pricing.ipynb              # European price validation + S&Z Table 2 comparison
  american_pricing.ipynb              # American put pricing: LSM vs CV-BS vs CV-LLH + BS-limit validation
  regression_basis_comparison.ipynb   # Laguerre vs Gaussian RBF: prices, sensitivity, bias
  ci_comparison.ipynb                 # Single-run CLT vs multi-run confidence intervals
  calibration.ipynb                   # Calibrate LLH to S&P 500 options
  char_func_symbolic.ipynb            # SymPy proof: ansatz non-closure (degree-4 argument)
  stylized_facts.ipynb                # Parameter impact on return distributions

reports/
  llh-formula.pdf             # Theoretical derivation of the European price formula
  llh-formula-report.pdf      # Extended report with European pricing results
  pricing-project.pdf         # Project report
```

---

## Pricing Methods

### European Price (Semi-Analytic)

The European call price follows the Fourier inversion formula:

```
C = S * P1 - K * exp(-r*tau) * P2
```

where P1, P2 are computed via the characteristic functions f1, f2 obtained by solving a system of autonomous Riccati ODEs (RK4 integration).

### American Put Price (LSM + CV)

The Longstaff-Schwartz algorithm estimates continuation values at each exercise date by regression on a basis of the current spot. Two basis types are supported via `basis_type`:
- **Laguerre** (`basis_type='laguerre'`): Laguerre polynomials of order `basis_order` (default 3, giving 4 basis functions)
- **Gaussian** (`basis_type='gaussian'`): Gaussian RBFs at `basis_order` quantile-grid centers (default 15), Silverman bandwidth, adaptive per time step

Three European put estimators are available as Rasmussen control variates:
- **LLH** (`euro_method='llh'`): exact LLH European put via ODE/quadrature (slow, best VR)
- **BS** (`euro_method='bs'`): Black-Scholes proxy (fast, moderate VR)
- **MC1** (`euro_method='mc1'`): single-path terminal payoff estimate

---

## Quick Start

```python
import sys
sys.path.insert(0, 'src')

import priceModels as pm
import amOptPricer as aop

# Create model with Table 2 parameters
model = pm.ImprovedSteinStein(
    r=0.01, kappa=4.9394, nu=0.3943, lam=0.3115,
    eta=0.4112, rho=0.1691, sigma0=0.2924, theta0=0.1319, seed=42
)

# Simulate paths
sim = model.simulate_prices(S0=100.0, T=1.0, n_steps_mc=52, n_paths=10_000)

# Price American put via LSM + control variates
result = aop.price_american_put_lsm_llh(model, sim, K=100.0, use_cv=True, euro_method='bs')
print(f"Price: {result['price_imp']:.4f}, SE: {result['std_err_imp']:.4f}")
```

---

## Theoretical Notes

### Ansatz Non-Closure

The covariance matrix of (sigma, theta) is **quadratic** in the state, not affine. This means the exponential-quadratic ansatz used to derive the characteristic function ODEs in the LLH paper is mathematically unjustified. `notebooks/char_func_symbolic.ipynb` proves this symbolically: the residual P = L[y]/y has total degree **4** in (sigma, theta) for the full LLH PDE, collapsing to degree 2 only in the Stein-Stein limit. The ODE system from LLH is therefore an approximation.

---

## Dependencies

- Python 3.9+
- NumPy, SciPy, Matplotlib, Pandas
- SymPy (for `char_func_symbolic.ipynb`)
- yfinance (for `calibrate.py` market data fetching)
- statsmodels (for ACF computation in `stylized_facts.ipynb`)

---

## References

- Lin, Lin & He (2024). *Improved Stein-Stein stochastic volatility model.*
- Longstaff & Schwartz (2001). *Valuing American options by simulation.* RFS.
- Rasmussen (2005). *Control variates for Monte Carlo valuation of American options.* JCAM.
- Schobel & Zhu (1999). *Stochastic volatility with an Ornstein-Uhlenbeck process.* EF.
- Stein & Stein (1991). *Stock price distributions with stochastic volatility.* RFS.
