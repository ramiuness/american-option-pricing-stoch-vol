# Tests

Diagnostic and validation scripts for the LLH pricing implementation.

## Scripts

| File | Purpose |
|------|---------|
| `test_v2_regression.py` | 42-test pytest suite: v1/v2 agreement, EEP bounds, OLS multi-regression, Laguerre basis, European pricing parity, RK4 convergence |
| `test_final.py` | Compares original vs corrected LSM: verifies P_Am >= P_Eu after vol-index fix |
| `test_notebook_conditions.py` | Validates notebook exercise conditions and regression logic |
| `diagnostic_fast.py` | Quick diagnostic for European pricing accuracy and arbitrage checks |
| `diagnose_bias.py` | Investigates European pricing bias across maturities |
| `analyze_indexing.py` | Demonstrates that `sigma_hat[:, j]` drives S[:,j] -> S[:,j+1] |
| `sweep_t2_discretization_bias.py` | Discretization-bias convergence diagnostic: multi-seed sweep over `n_steps_mc` for T2 params, fitting `bias(dt) ≈ a + b·dt` |

## Usage

Run from `project/`:

```bash
# Full pytest suite (v2 regression + unit tests)
python -m pytest tests/test_v2_regression.py -v

# Legacy diagnostic scripts
cd tests
python test_final.py
python diagnostic_fast.py
```

All scripts use `seed=42` for reproducibility.

## v1 coverage gap

Every script here imports the **v0** modules only (`priceModels`,
`priceModels_v0`, `amerPrice`). None exercises `priceModels_v1`,
`amerPrice_v1`, `generate_plots_v1`, or `timing_analysis_v1`. As a
consequence:

- The existing `test_v2_regression.py` failures (per the most recent
  audit) are pre-existing and unrelated to the v1 promotion work.
- No regression suite protects the v1 schema (`sim_out['theta']` vs
  v0's `sim_out['B']`), the `theta_driver='bm'|'gbm'` toggle, the
  `terminal_only` optimization, or the driver-agnostic
  `amerPrice_v1._setup`/`precompute_european`.

If v1 correctness needs to be locked in, add dedicated tests that
import `priceModels_v1` / `amerPrice_v1` and verify:

1. `sim = model.simulate_prices(..., theta_driver='bm')` vs
   `theta_driver='gbm'` — BS-limit paths are bit-identical
   (both collapse to the same θ when `η=λ=0`).
2. `amerPrice_v1.precompute_european(model, sim, K)` runs against
   v1 sim output without `KeyError: 'B'`.
3. MC European price under `'bm'` converges to the LLH formula
   within 95% CI at 10⁵ paths (driver-aligned convergence).
