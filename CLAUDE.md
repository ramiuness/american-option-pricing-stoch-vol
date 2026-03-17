# CLAUDE.md — American Options Pricing (LLH Model)

American put pricing via LSM + Rasmussen control variates under the Lin-Lin-He (2024)
improved Stein-Stein stochastic volatility model.

---

## Working Code

**Primary working directory:** `archive/`
**Modular refactor (side experiment, not primary):** `src/`

### File Map

| File | Purpose | Status |
|------|---------|--------|
| `archive/priceModels.py` | LLH simulation + European pricing (ODE/quadrature) | ✅ USE THIS |
| `archive/amOptPricer_corrected.py` | LSM + Rasmussen CV (Bug #1 fixed) | ✅ USE THIS |
| `archive/amOptPricer.py` | Original LSM pricer | ⚠️ Bug #1: wrong volatility index |
| `archive/priceModels_tau_fix.py` | Wrong ODE variant (τ multiplier) | ❌ DO NOT USE |
| `archive/tests/` | Diagnostic and validation scripts | Reference |

---

## CRITICAL: Array Indexing Convention

```
S.shape         = (n_paths, n_steps_mc + 1)   # includes S₀ at column 0
sigma_hat.shape = (n_paths, n_steps_mc)        # no initial value
```

**Rule:** `sigma_hat[:, j]` drives the transition from `S[:, j]` to `S[:, j+1]`.

In the LSM backward loop at time `t_j`, use **`sigma_hat[:, j]`** (not `j-1`) for
forward-pricing the European control variate. Using `j-1` is Bug #1.

---

## CRITICAL: Parameter Naming

| Name | Meaning |
|------|---------|
| `n_steps_mc` | Monte Carlo time discretization steps |
| `n_steps_ode` | RK4 steps for ODE integration |
| `n_steps_rk4` | Same as `n_steps_ode` (used in `amOptPricer_corrected.py`) |

**Never confuse these.** They are completely independent parameters.

---

## Formula Corrections Applied

### Fix 1 — σ(t) discretization (`sigma_hat_from_components`)

| | Prefactor on e^{-κt} | Time term |
|---|---|---|
| Wrong | `σ₀ + λ − θ₀` | `λ(t − 1)` |
| **Correct** | `σ₀ − θ₀ + λ/κ` | `λ(t − 1/κ)` |

Status in `archive/priceModels.py`: **needs Fix 1 applied** (see PROGRESS.md).
Status in `src/simulation/stoch_processes.py`: ✅ already corrected.

### Fix 2 — ODE is autonomous (no τ multiplier in dD)

The `rhs` callable is `rhs(Y)`, not `rhs(Y, tau)`. No `* current_tau` in `dD`.
Status in `archive/priceModels.py`: ✅ already autonomous.

---

## Precision Presets

```python
# STANDARD (default)
phi_max=300.0, n_phi=513, n_steps_ode=128

# HIGH_ACCURACY
phi_max=400.0, n_phi=1025, n_steps_ode=256
```

---

## LLH Reference Parameters (Lin-Lin-He Table 2)

```python
r=0.3943, kappa=4.9394, nu=0.4, lam=0.3115,
eta=0.4112, rho=0.1691, sigma0=0.2924, theta0=0.1319
```

Special cases: BS limit (`kappa=nu=lam=eta=rho=0`), SS limit (`lam=eta=rho=0`).

---

## Quick Commands

```bash
# Run from project/
python -c "import archive.priceModels as pm; print('OK')"

# Validate imports (src/)
python -c "from src.models.llh_model import ImprovedSteinStein; print('OK')"
python -c "from src.pricing.american.lsm import price_american_put_lsm_llh; print('OK')"
```

---

## Do Not

- Edit `archive/priceModels_tau_fix.py` — it is a historical reference of a wrong variant
- Use `sigma_hat[:, j-1]` in the LSM loop — always `sigma_hat[:, j]`
- Pass `n_steps_mc` to ODE functions or vice versa
- Remove the safeguard `ex_mask = Ij > max(Ej, TV)` — it is correct per LSM theory
