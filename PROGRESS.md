# Project Progress: American Option Pricing under LLH Model

## Overview

Pricing American put options via LSM + Rasmussen control variates under the LLH (Lin-Lin-He 2024)
stochastic volatility model. European option prices are computed semi-analytically via characteristic
functions and used as control variates to reduce variance in the LSM estimator.

---

## Completed Work

### Phase 1: Code Restructuring ✅

Reorganized monolithic scripts (`archive/`) into a clean modular `src/` package.

**Archive preserved (read-only reference):**
- `archive/priceModels.py` — original LLH model + European pricing (autonomous ODE, correct)
- `archive/priceModels_tau_fix.py` — τ-multiplier variant (wrong ODE — kept for reference)
- `archive/amOptPricer.py` — original LSM pricer (has Bug #1: volatility index off-by-one)
- `archive/amOptPricer_corrected.py` — Bug #1 fixed LSM pricer

**New `src/` package:**

| Module | File | Purpose |
|--------|------|---------|
| Simulation | `src/simulation/stoch_processes.py` | Path generation primitives |
| Model | `src/models/llh_model.py` | `ImprovedSteinStein` class + `LLHPrecompute` |
| European BS | `src/pricing/european/bs.py` | Black-Scholes European pricing |
| European MC | `src/pricing/european/mc.py` | Monte Carlo European pricing |
| European LLH | `src/pricing/european/llh.py` | ODE/quadrature infrastructure |
| American LSM | `src/pricing/american/lsm.py` | LSM + Rasmussen CV |
| Diagnostics | `src/utils/diagnostics.py` | Validation utilities |

**Runtime bug fixed during migration:**
`archive/amOptPricer_corrected.py` ~line 287 passes `n_steps=n_steps_rk4` but
`llh_precompute_tau` expects `n_steps_ode`. Fixed in `src/pricing/american/lsm.py`.

---

### Phase 2: Formula Corrections (from eur_price_llh report + verification audit) ✅

Two substantive errors identified and corrected:

#### Fix 1 — σ(t) Deterministic Coefficients (`src/simulation/stoch_processes.py`)

The integrated solution for σ(t) had wrong constants in the discretization formula.

| | Prefactor on e^{-κt} | Time term |
|---|---|---|
| **Before (wrong)** | `σ₀ + λ - θ₀` | `λ(t - 1)` |
| **After (correct)** | `σ₀ - θ₀ + λ/κ` | `λ(t - 1/κ)` |

Verified by substituting both into the original ODE `dσ/dt = κ(θ-σ)` with ν=η=0;
only the correct formula satisfies it identically. Error did not affect the
characteristic function derivation, Feynman-Kac PDE, or ODE system.

#### Fix 2 — ODE System is Autonomous (`src/pricing/european/llh.py`)

The LLH paper has a typo: a spurious τ multiplier in the dD/dτ equation.
The symbolic audit (SymPy) confirms the ODE is fully autonomous.

| | dD equation |
|---|---|
| **Before (wrong)** | `(u·iφ - φ²/2) * τ + 2ν²D² - 2AD + η²E²/2` |
| **After (correct)** | `(u·iφ - φ²/2) + 2ν²D² - 2AD + η²E²/2` |

The `rhs` callable is now `rhs(Y)` (no `current_tau` argument), and `rk4_integrate`
no longer tracks or passes τ through the RK4 stages.

---

## Known Bugs (Historical)

| Bug | Status | Location |
|-----|--------|----------|
| #1 Volatility index off-by-one in LSM loop | ✅ Fixed | `src/pricing/american/lsm.py` uses `sigma_hat[:, j]` (not `j-1`) |
| #2 σ(t) deterministic coefficients wrong | ✅ Fixed | `src/simulation/stoch_processes.py` (Phase 2 above) |
| #3 ODE has spurious τ multiplier | ✅ Fixed | `src/pricing/european/llh.py` (Phase 2 above) |
| #4 `n_steps` kwarg mismatch in LSM | ✅ Fixed | `src/pricing/american/lsm.py` uses `n_steps_ode` |

---

## Precision Presets

```python
# STANDARD (recommended for most use cases)
phi_max=300.0, n_phi=513, n_steps_ode=128
# Expected bias: < 5% for τ = 1.0

# HIGH_ACCURACY (critical calculations)
phi_max=400.0, n_phi=1025, n_steps_ode=256
# Expected bias: < 2% for τ = 1.0
```

Observed bias before fixes (old defaults `phi_max=200, n_phi=257, n_steps_ode=64`):

| Maturity | LLH bias vs MC |
|----------|---------------|
| 1 month  | ~3.6%         |
| 1 year   | ~13.5%        |

---

## Phase 3: Theoretical Analysis — Ansatz Non-Closure + Conditioning on B (in progress)

### Completed deliverables (2026-03-18)

#### `notebooks/char_func_symbolic.ipynb` (new) ✅

Symbolic proof (SymPy) — §1 only: non-closure of the quadratic-exponential ansatz for the full LLH PDE.

- Constructs `P = L[y]/y` symbolically using the exact PDE (21) with θ-diffusion
  coefficient `½(η−θ₀−λt+θ)²` (quadratic in θ)
- P displayed in decreasing graded lex order (θ⁴ term first); `sp.Poly(P, σ, θ).total_degree()` = **4**
- Coefficient of `θ⁴` = `½Γ₂₂²` → forces `Γ₂₂=0` → forces `Γ₁₂=0` → `σθ` cost `-s₂` unbalanced
- Same decreasing-order display applied to P after each substitution (Γ₂₂=0, then Γ₁₂=0)
- Formal impossibility table; confirms ODE system in LLH paper is an **approximation**

The exact alternative (conditioning on ℱ^W) is planned in `notebooks/conditioning_on_B.ipynb` (not yet created).

#### `notebooks/conditioning_on_B.ipynb` — **planned, not yet created**

Will contain symbolic derivation (SymPy) — exact 1D reduction of the LLH characteristic function by conditioning on ℱ^W.

**§1 — 1D closure verification**
- Conditions on the GBM driver filtration: `θ(u)=θ₀+λu+η(B_u−1)` becomes deterministic
- 1D Feynman-Kac PDE for σ (time-inhomogeneous O-U): verifies degree = 2 regardless of `θ(t)`
- P1D displayed in decreasing σ order (σ² first)
- Extracts three ODEs: `Ḋ` (Riccati, θ-free — same as S&Z), `Ḃ_coef` (linear, θ(t) enters),
  `Ċ` (quadrature, θ(t) enters)

**§2 — Gaussian outer expectation**
- Decomposes C: deterministic + `η∫f(u)B_u du`; shows the latter is Gaussian → closed-form
  outer expectation via `E[exp(M)] = exp(½Var(M))`
- Derives `Var(M) = η² ∫∫ f(u)f(v) Cov(B_u,B_v) du dv` with `Cov(B_u,B_v)=e^{min(u,v)−(u+v)/2}−1`
- Summary table: LLH approximation vs S&Z exact (SS limit) vs conditioning-on-B exact

#### `reports/conditioning_derivation.tex` — **planned, not yet created**

Will be a standalone LaTeX document (amsmath/amssymb/amsthm/geometry). Six sections:
1. Introduction — two approaches, why ansatz fails
2. Model Setup — LLH SDEs, θ solution, characteristic-function decomposition
3. Non-Affine Observation — Proposition + proof (citing symbolic notebook)
4. S&Z Limit — exact ODEs (3 equations) and cosh/sinh structure
5. Conditioning on B — full derivation: formal conditioning, 1D PDE, quadratic ansatz
   → 3 ODEs, C decomposition, Gaussian integral → final formula (Eq. 15)
6. Conclusion — what remains for numerical implementation

#### `notebooks/european_pricing.ipynb` — §5 appended ✅

New §5 "Comparison with Schöbel-Zhu (1999) Table 2 — Impact of θ₀":
- SS-limit parameters: `r=0.0953, κ=4, ν=0.1, σ₀=0.15, τ=0.5, S=100`
- 3 panels (D: ρ=0.5, E: ρ=0.0, F: ρ=−0.5) × 4 θ₀ values × 7 strikes
- DataFrame output: BS / LLH / MC rows per panel
- Remarks on stationary case, bias, correlation skew, long-run mean effect

---

## Open Tasks

### Validation

- [ ] Run `notebooks/char_func_symbolic.ipynb` to confirm symbolic degree = 4 and θ⁴ term visible at top of displayed P
- [ ] Run `notebooks/conditioning_on_B.ipynb` to confirm `Poly_P1D.degree()` = 2 and Ḋ prints without θ(t) in free symbols
- [ ] Run `notebooks/european_pricing.ipynb` §5 panels — Panel E (ρ=0, θ₀=0.2) LLH should match S&Z Table 2
- [ ] **P_A ≥ P_E** no-arbitrage check: BS limit, SS limit, full LLH model
- [ ] **Variance reduction**: CV run shows lower std_err than no-CV; improved estimator reduces further

### Research

- [ ] Implement conditioning-on-B pricing numerically in `src/priceModels.py` (replaces approximate ODE)
  - Step 1: RK4 for `D, B_coef, C_det` with θ(t) = θ₀+λt−η (deterministic part)
  - Step 2: 2D quadrature for `∫∫ f(u)f(v) Cov(B_u,B_v) du dv`
  - Expected: eliminates the ~3–5% bias at τ=0.5
- [ ] If European bias persists with current ODE: try `scipy.integrate.solve_ivp` (adaptive step)
- [ ] Finalize STANDARD and HIGH_ACCURACY preset documentation

---

## Reference Values

| Test | Expected | Source |
|------|----------|--------|
| BS limit European call | Black-Scholes formula | Standard |
| Stein-Stein European call | ≈ 10.77 | Stein-Stein (1991) Table 1 |
| LLH ATM call (S=K=100, T=1) | ≈ 7–8 | LLH paper Figure 2(a) (visual) |

## LLH Reference Parameters (Lin-Lin-He Table 2)

`r=0.3943, κ=4.9394, ν=0.4, σ₀=0.2924, θ₀=0.1319, λ=0.3115, η=0.4112, ρ=0.1691`

---

## Theoretical Notes

- **Non-affine diffusion**: The covariance matrix of (σ, θ) is quadratic in the state (not affine),
  so the exponential-quadratic ansatz for the characteristic function is mathematically inconsistent.
  The ODE system from LLH is used in practice anyway as an approximation. (Audit: ✓ argument correct.)
- **Variance of σ when η=0**: `ν²(1 - e^{-2κt}) / (2κ)` — the report had a draft formula here,
  flagged and not used in code.
