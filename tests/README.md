# Testing & Diagnostic Files

This directory contains files generated during the debugging investigation on 2025-10-15.

## Diagnostic Scripts (Run to verify bugs)

### `diagnostic.py`
**Comprehensive diagnostic analysis**
- Verifies array shape mismatches
- Tests European pricing accuracy (BS limit)
- Tests American pricing for arbitrage violations
- Tests both BS limit and full LLH model

**Run**: `python diagnostic.py`

### `diagnostic_fast.py`
**Fast version of diagnostic.py**
- Same tests, fewer paths for quick verification
- Use for rapid iteration during debugging

**Run**: `python diagnostic_fast.py`

### `analyze_indexing.py`
**Mathematical proof of volatility index bug**
- Shows time grid vs array indexing
- Proves `sigma_hat[:, j]` drives transition S[:, j] → S[:, j+1]
- Demonstrates current code uses wrong index (j-1)

**Run**: `python analyze_indexing.py`

---

## Validation Tests (Run to verify fixes)

### `test_final.py` ⭐ **RUN THIS FIRST**
**Compares original vs corrected implementation**
- Tests BS limit model
- Tests full LLH model (1 month)
- Verifies P_A ≥ P_E after index fix
- Uses BS control variate to isolate Bug #1 from Bug #2

**Run**: `python test_final.py`

**Expected Result**: Corrected version should show P_A ≥ P_E

### `test_varying_vol.py`
**Tests sensitivity to volatility variation**
- Tests Stein-Stein limit (varying volatility)
- Tests full LLH model (maximum variation)
- Measures if index fix makes measurable difference

**Run**: `python test_varying_vol.py`

**Expected Result**: Larger difference when volatility varies significantly

### `test_comparison.py`
**Tests discarded fix (safeguard removal)**
- Compares original with safeguard removed
- Tests with 3 control variate methods: 'bs', 'mc1', none
- **Note**: This fix was REJECTED - safeguard is correct

**Run**: `python test_comparison.py`

**Purpose**: Documents why safeguard removal doesn't work

---

## Code Variants

### `amOptPricer_fixed.py` ❌ **DISCARDED**
**What it changes**: Removes safeguard from exercise decision
```python
ex_mask = Ij > TV  # Instead of: Ij > max(Ej, TV)
```

**Why discarded**: Violates LSM theory, leads to spurious early exercise

**Status**: Kept for historical reference only

---

## Quick Validation Workflow

1. **Verify the bug exists**:
   ```bash
   python diagnostic_fast.py
   ```
   Should show P_A < P_E violations

2. **Verify the fix works**:
   ```bash
   python test_final.py
   ```
   Should show corrected version has P_A ≥ P_E

3. **Measure impact**:
   ```bash
   python test_varying_vol.py
   ```
   Should show index fix makes measurable difference

---

## Notes

- All scripts use `seed=42` for reproducibility
- BS limit tests avoid Bug #2 (European pricing bias)
- Default paths: 1000-5000 (increase for production)
- Use `euro_method='bs'` to isolate indexing issues from European pricing issues
