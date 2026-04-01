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
