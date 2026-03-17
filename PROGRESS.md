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

## Open Tasks

### Immediate: Formula Fix (archive/)

- [ ] Apply σ(t) fix to `archive/priceModels.py` (`sigma_hat_from_components`):
  - `exp_kdt_idx * (sigma0 + lam - theta0)` → `exp_kdt_idx * (sigma0 - theta0 + lam / kappa)`
  - `lam * (idx * dt - 1.0)` → `lam * (idx * dt - 1.0 / kappa)`
- [x] Autonomous ODE fix — already present in `archive/priceModels.py` (no `* current_tau`)

### Validation (run after σ(t) fix)

Work through these in order using the numbered notebooks in `notebooks/`.

#### Step 1: Verify Imports

```bash
python -c "from src.models.llh_model import ImprovedSteinStein; print('OK')"
python -c "from src.pricing.american.lsm import price_american_put_lsm_llh; print('OK')"
python -c "from src.utils.diagnostics import compare_european_prices; print('OK')"
```

#### Step 2: Notebook 01 — LLH Simulation

- [ ] **MC drift**: `E[S_T] / S₀ ≈ exp(r·T)` for T = 0.25, 1.0, 2.0 — key test for σ fix
- [ ] **BS limit**: KS lognormality test passes (p-value > 0.05)
- [ ] `sigma_hat` values non-negative
- [ ] Path plots qualitatively reasonable

#### Step 3: Notebook 02 — European Pricing

- [ ] **BS limit**: LLH formula matches Black-Scholes (error < 1e-10)
- [ ] **Stein-Stein limit**: LLH formula ≈ 10.77 (Stein-Stein 1991 Table 1, error < 3e-3)
- [ ] **LLH ATM** (S=K=100, T=1): ≈ 7–8 per LLH paper Figure 2(a)
- [ ] **European bias table**: LLH vs MC for T = 1/12, 1/4, 1/2, 1.0; K = 90, 100, 110
- [ ] Convergence study: `n_steps_ode` ∈ {64, 128, 256, 512}; `phi_max` ∈ {200, 300, 400, 500}
- [ ] Confirm STANDARD (`phi_max=300, n_phi=513, n_steps_ode=128`) and HIGH presets

#### Step 4: Notebook 03 — American Pricing

- [ ] **P_A ≥ P_E** no-arbitrage: BS limit, SS limit, full LLH model
- [ ] **Variance reduction**: CV run shows lower std_err than no-CV run
- [ ] **Improved estimator** (`improved=True`) reduces std_err further
- [ ] Test with n_paths = 1000, 5000, 10000

### Research (after validation)

- [ ] If European bias persists: try `scipy.integrate.solve_ivp` (adaptive ODE)
- [ ] If European bias persists: try `scipy.integrate.quad` (adaptive quadrature)
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
