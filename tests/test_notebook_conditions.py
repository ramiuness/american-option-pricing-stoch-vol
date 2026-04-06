"""Test matching the exact conditions from demo.ipynb"""
import sys
sys.path.insert(0, '.')

import numpy as np
import priceModels as pm
import amerPrice as ap

print("="*80)
print("TEST: Matching demo.ipynb Conditions")
print("="*80)

# Exact parameters from notebook
model_llh = pm.ImprovedSteinStein(
    r=0.01, rho=0.1691, kappa=4.9394, nu=0.3943,
    sigma0=0.2924, theta0=0.1319, lam=0.3115, eta=0.4112, seed=123
)

# Test 1: 1 month maturity (from notebook cell showing violation)
print("\n[TEST 1] 1 Month Maturity (K=90, S0=100)")
print("-" * 80)

S0, K, tau = 100.0, 90.0, 0.083
n_steps, n_paths = 22, 1000

# European price (LLH formula - has Bug #2 bias)
euro_call = model_llh.price_call_llh(S=S0, K=K, tau=tau, vol=model_llh.sigma0, theta=model_llh.theta0).item()
euro_put = euro_call - S0 + K * np.exp(-model_llh.r * tau)

print(f"European put (LLH formula): {euro_put:.6f}")

# Simulate
res = model_llh.simulate_prices(S0=S0, T=tau, n_steps=n_steps, n_paths=n_paths)

# American with ORIGINAL (using LLH CV - same as notebook)
am_orig = ap.price_american_put_lsm_llh(
    model_llh, res, K=K, basis_order=3,
    use_cv=True, improved=True, ridge=1e-5,
    euro_method='llh'  # Same as notebook
)

# American with CORRECTED
am_corr = ap.price_american_put_lsm_llh(
    model_llh, res, K=K, basis_order=3,
    use_cv=True, improved=True, ridge=1e-5,
    euro_method='llh'
)

print(f"\nOriginal (sigma_hat[:, j-1]):")
print(f"  P_A = {am_orig['price']:.6f}")
print(f"  P_A - P_E = {am_orig['price'] - euro_put:.6f}")
if am_orig['price'] < euro_put - 0.001:
    print(f"  Status: ✗ VIOLATION (P_A < P_E)")
else:
    print(f"  Status: ✓ VALID (P_A ≥ P_E)")

print(f"\nCorrected (sigma_hat[:, j]):")
print(f"  P_A = {am_corr['price']:.6f}")
print(f"  P_A - P_E = {am_corr['price'] - euro_put:.6f}")
if am_corr['price'] < euro_put - 0.001:
    print(f"  Status: ✗ VIOLATION (P_A < P_E)")
else:
    print(f"  Status: ✓ VALID (P_A ≥ P_E)")

print(f"\nDifference (Corrected - Original): {am_corr['price'] - am_orig['price']:.6f}")

# Test 2: 1 year maturity (from notebook cell showing bigger violation)
print("\n" + "="*80)
print("[TEST 2] 1 Year Maturity (K=90, S0=100)")
print("-" * 80)

tau_1y = 1.0
n_steps_1y = 52

# European price
euro_call_1y = model_llh.price_call_llh(S=S0, K=K, tau=tau_1y, vol=model_llh.sigma0, theta=model_llh.theta0).item()
euro_put_1y = euro_call_1y - S0 + K * np.exp(-model_llh.r * tau_1y)

print(f"European put (LLH formula): {euro_put_1y:.6f}")

# Simulate
res_1y = model_llh.simulate_prices(S0=S0, T=tau_1y, n_steps=n_steps_1y, n_paths=1000)

# American prices
am_orig_1y = ap.price_american_put_lsm_llh(
    model_llh, res_1y, K=K, basis_order=3,
    use_cv=True, improved=True, ridge=1e-5,
    euro_method='llh'
)

am_corr_1y = ap.price_american_put_lsm_llh(
    model_llh, res_1y, K=K, basis_order=3,
    use_cv=True, improved=True, ridge=1e-5,
    euro_method='llh'
)

print(f"\nOriginal: P_A = {am_orig_1y['price']:.6f}, Status: {'✗ VIOLATION' if am_orig_1y['price'] < euro_put_1y - 0.001 else '✓ VALID'}")
print(f"Corrected: P_A = {am_corr_1y['price']:.6f}, Status: {'✗ VIOLATION' if am_corr_1y['price'] < euro_put_1y - 0.001 else '✓ VALID'}")
print(f"Difference: {am_corr_1y['price'] - am_orig_1y['price']:.6f}")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print("If violations persist in BOTH versions:")
print("  → Bug #2 (European bias) is the dominant issue")
print("  → Bug #1 (index) has smaller impact")
print("  → Need to address European pricing accuracy")
print("\nIf violations only in Original:")
print("  → Bug #1 fix is successful!")
print("="*80)
