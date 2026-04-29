# American Option Pricing under the LLH Stochastic Volatility Model

Pricing American put options via Longstaff-Schwartz Monte Carlo (LSM) with Rasmussen control variates, under the Lin, Lin & He (2024) three-factor Stein-Stein stochastic volatility model.

**Author:** Rami Younes, Universite de Montreal
 
---

## Overview

American options are widely traded yet their valuation remains the subject of extensive academic research. Among the available numerical methods, least-squares Monte Carlo (LSM, Longstaff & Schwartz 2001) continues to stand as one of the most competitive — particularly when combined with variance reduction.

Under Black-Scholes, Rasmussen-style control variates built from European option prices yield significant accuracy gains for LSM, even outperforming neural-network approaches (Chavez Aquino et al. 2022). This pattern presupposes a closed-form European price — available for Black-Scholes, Stein-Stein (1991), and Schöbel-Zhu (1999) but not for richer stochastic-volatility models.

Lin, Lin & He (2024) proposed a stochastic-volatility model that nests all three and derived a *semi-analytical* European price via the Heston-PDE / Fourier-inversion / ODE-numerics technique — reportedly faster than Monte Carlo. This makes the LLH formula a natural drop-in for Rasmussen's control-variate framework.

This repository replicates the European-pricing experiments of Lin, Lin & He, and then estimates American put prices under the same dynamics by combining LSM with Rasmussen control variates, following Rasmussen (2005) and West (2013). The pipeline is computationally feasible and highly effective for variance reduction despite the semi-analytical European pricing; accuracy and stability are reached at lower path counts, in agreement with Rasmussen (2005).

---

## Model

The LLH model specifies the following risk-neutral dynamics:

$$
\begin{aligned}
\frac{dS_t}{S_t} &= r dt + \sigma_t dW^1_t, \\
d\sigma_t       &= \kappa(\theta_t - \sigma_t) dt + \nu dW^2_t, \\
d\theta_t       &= \lambda dt + \eta dW_t,
\end{aligned}
$$

with $\langle W^1, W^2\rangle_t = \rho t$ and $W_t$ independent of $(W^1_t, W^2_t)$. The $\theta_t$ process drives a stochastic long-run mean for $\sigma_t$. The model nests:

- **Black-Scholes** when $\kappa = \nu = \lambda = \eta = \rho = 0$
- **Stein-Stein (1991)** when $\lambda = \eta = \rho = 0$
- **Schöbel-Zhu (1999)** when $\lambda = \eta = 0$

Reference parameter sets from Lin, Lin & He (2024):

| Set | $r$ | $\kappa$ | $\nu$ | $\lambda$ | $\eta$ | $\rho$ | $\sigma_0$ | $\theta_0$ |
|-----|---|-------|----|--------|-----|-----|--------|--------|
| Table 1 | 0.01 | 5 | 0.2 | 0.9 | 0.01 | -0.2 | 0.15 | 0.18 |
| Table 2 | 0.01 | 4.9394 | 0.3943 | 0.3115 | 0.4112 | 0.1691 | 0.2924 | 0.1319 |

Full mathematical derivation, discretization, and proofs are in `reports/american_pricing_report.pdf`.

---

## Repository Structure

```
src/
  priceModels.py        # LLH simulation + European pricing (semi-analytic via ODE/quadrature, plus BS/MC helpers)
  amerPrice.py          # LSM + Rasmussen control variates; Laguerre/Gaussian RBF bases; multivariate basis_vars
  generate_plots.py     # Figure generation for European and American pricing reports
  timing_analysis.py    # Empirical scaling/timing breakdown of LSM+CV-LLH
  testing.py            # Regression-basis and CI comparison utilities
  reporting.py          # Notebook presentation helpers

notebooks/
  european_pricing.ipynb     # European price validation: LLH formula vs MC; edge-case BS/SS/SZ recovery
  american_pricing.ipynb     # American put: plain LSM vs CV-BS vs CV-LLH; BS-limit validation
  char_func_symbolic.ipynb   # SymPy proof: LLH ansatz non-closure (degree-4 argument)

reports/
  american_pricing_report.pdf  # Project report (LSM + CV results, VR, EEP)
  llh-formula-report.pdf       # Extended European-pricing report
  pricing-project.pdf          # Project overview

scripts/
  regen_report_figs.py   # One-shot: re-run generate_plots.py and timing_analysis.py and copy report PNGs
  report_figs.txt        # Whitelist of report-relevant PNG filenames

tests/                   # Diagnostic scripts (indexing, bias, T2 discretization, regression). See tests/README.md
figs/                    # 96 PNGs produced by generate_plots.py and timing_analysis.py
```

---

## Pricing Methods

### European Price (Semi-Analytic)

The European call price follows the Fourier-inversion formula:

$$ C = S_t P_1 - K e^{-r\tau} P_2 $$

where $P_1, P_2$ are recovered from the characteristic functions $f_1, f_2$ obtained by solving an autonomous Riccati ODE system (RK4 integration) and inverting via trapezoid quadrature. European put prices follow by put-call parity.

### American Put Price (LSM + CV)

The Longstaff-Schwartz algorithm estimates continuation values at each exercise date by ridge regression on a basis of the current state. 

**1. Regression basis** (`basis_type`, `basis_vars`)
- `basis_type='laguerre'` — Laguerre polynomials of order `basis_order` (default 2 → 3 functions).
- `basis_type='gaussian'` — Gaussian RBFs at `basis_order` quantile-grid centres (default 5), median-spacing bandwidth, recomputed per time step.
- `basis_vars` — single-variable `('S',)` (default; bitwise-identical to legacy spot-only) or multivariate `('S','sigma','theta')` (per-variable Laguerre/RBF blocks concatenated; duplicate $L_0$ columns dropped after the first block for Laguerre).

**2. Control variate** (`use_cv=True`)
Rasmussen control variate using the closed-form LLH European put. 

**3. Exercise floor** (`floor_method` / `euro_method`)
Three options:
- `'llh'` — exact LLH European put via ODE/quadrature (slow; the canonical choice when CV is on).
- `'bs'` — Black-Scholes evaluated at instantaneous simulated vol (fast proxy).
- `'mc1'` — single-path discounted terminal payoff (cheapest; surprisingly competitive).

When `use_cv=True`, the floor uses `euro_method`. When `use_cv=False`, it uses `floor_method`.

**Discretization.** Two SDE schemes are implemented: Euler (weak order 1, default) and Milstein (weak order 2). For Monte Carlo European pricing, `terminal_only=True` skips the path grid and returns only $S_T$.

---

## Key Findings

Headline results from `reports/american_pricing_report.pdf`:

- **Plain LSM is unstable under full LLH dynamics** — remains noisy at 50,000 paths. **LSM+CV-LLH converges at ~5,000 paths**, in agreement with Rasmussen (2005).
- **CV-LLH dominates plain LSM** and harvests **positive early-exercise premium** across the moneyness ladder under both reference parameter regimes.
- **Variance reduction factor ~10⁵-10⁶×** under both regimes and both bases.
- **LLH European pricing dominates runtime** — ~264× longer than plain LSM with an MC1 floor at default settings. When tight confidence intervals are not required, plain LSM with an MC1 exercise floor is a competitive cheap baseline.

---

## Quick Start

```python
import sys
sys.path.insert(0, 'src')

import priceModels as pm
import amerPrice as ap

# Create model with LLH Table 1 parameters
model = pm.ImprovedSteinStein(
    r=0.01, kappa=5.0, nu=0.2, lam=0.9,
    eta=0.01, rho=-0.2, sigma0=0.15, theta0=0.18, seed=42,
)

# Simulate paths (Euler scheme, BM-driver theta — both are defaults)
sim = model.simulate_prices(S0=100.0, T=1.0, n_steps_mc=52, n_paths=10_000)

# Price the American put via LSM + Rasmussen CV using the LLH European put as both
# control variate and exercise floor; multivariate basis over (S, sigma, theta)
result = ap.price_american_put_lsm_llh(
    model, sim, K=100.0,
    use_cv=True, euro_method='llh',
    basis_type='laguerre', basis_vars=('S', 'sigma', 'theta'),
    ridge=1e-3,
)
print(f"Price: {result['price_imp']:.4f}, SE: {result['std_err_imp']:.4f}")
```

For paired plain-LSM and CV-LLH calls on the same simulation, share an ODE/quadrature grid:

```python
grid = ap.precompute_european(model, sim, K=100.0)
plain = ap.price_american_put_lsm_llh(model, sim, K=100.0, use_cv=False, precomputed=grid)
cv    = ap.price_american_put_lsm_llh(model, sim, K=100.0, use_cv=True,  precomputed=grid)
```

---

## Regenerating Report Figures

The figures in `figs/` already exist on disk and ship with the repository. From the project root:

```bash
python scripts/regen_report_figs.py --out-dir /path/to/your/report/figs
```

The script invokes `src/generate_plots.py` (T1 and T2 parameter sets) and `src/timing_analysis.py` in-process, then copies the 20 report-relevant PNGs (listed in `scripts/report_figs.txt`) into `--out-dir`. Pipeline parameters (spots, horizons, ridge, basis_vars, seeds) live as module-level constants in the two source files.

Available flags:

- `--out-dir PATH` — destination for the 20 copied PNGs (required unless `--list-only`).
- `--list-only` — print the figure filenames and exit (no execution, no copying).
- `--param-sets T1` or `--param-sets T1,T2` — restrict which LLH parameter set(s) to run.
- `--skip-plots` — reuse existing `figs/` contents and run only `timing_analysis.py`.
- `--skip-timing` — run only `generate_plots.py`.

Runtime: ~30 min for `generate_plots.py`, ~5-8 min for `timing_analysis.py`.

---

## Dependencies

- Python 3.10+
- NumPy, SciPy, Matplotlib, Pandas
- SymPy (for `char_func_symbolic.ipynb`)

---

## References

- Black & Scholes (1973). *The pricing of options and corporate liabilities.* JPE.
- Chavez Aquino, Bastin, Benazzouz & Kharrat (2022). *Monte Carlo methods for pricing American options.* Springer.
- Clément, Lamberton & Protter (2002). *An analysis of a least-squares regression method for American option valuation.* Finance and Stochastics.
- Glasserman (2004). *Monte Carlo methods in financial engineering.* Springer.
- Heston (1993). *A closed-form solution for options with stochastic volatility.* RFS.
- Lin, Lin & He (2024). *Analytically pricing European options with a two-factor Stein-Stein model.* JCAM.
- Longstaff & Schwartz (2001). *Valuing American options by simulation.* RFS.
- Lord & Kahl (2006). *Optimal Fourier inversion in semi-analytical option pricing.* Tinbergen Institute.
- Rasmussen (2005). *Control variates for Monte Carlo valuation of American options.* JCAM.
- Schöbel & Zhu (1999). *Stochastic volatility with an Ornstein-Uhlenbeck process.* Review of Finance.
- Stein & Stein (1991). *Stock price distributions with stochastic volatility.* RFS.
- West (2013). *American Monte Carlo option pricing under pure jump Lévy models.* PhD thesis, Stellenbosch.
