"""
Final Test: Original vs Corrected LSM with Index Fixes
Tests the volatility index alignment fix (Error #1).
"""

import numpy as np
import priceModels as pm
import amOptPricer as aop_original
import amOptPricer_corrected as aop_corrected

print("="*80)
print("FINAL TEST: Original vs Corrected (with Index Fix)")
print("="*80)

# BS limit model
model_bs = pm.ImprovedSteinStein(
    r=0.05, rho=0.0, kappa=0.0, nu=0.0,
    sigma0=0.2, theta0=0.0, lam=0.0, eta=0.0, seed=42
)

S0, K, tau = 100.0, 100.0, 0.25

# Analytical European put
bs_call = aop_original.price_call_bs(S=S0, K=K, tau=tau, r=0.05, vol=0.2)
bs_put = bs_call - S0 + K * np.exp(-0.05 * tau)

print(f"\n[1] BS Limit Model Test")
print("-" * 80)
print(f"Parameters: S0={S0}, K={K}, tau={tau}, r=0.05, vol=0.2")
print(f"European put (BS analytical): {bs_put:.6f}")

# Simulate paths
n_paths_test = 5000
res_bs = model_bs.simulate_prices(S0=S0, T=tau, n_steps=22, n_paths=n_paths_test)

# Test with BS control variate
print(f"\n--- With BS Control Variate ---")

am_orig = aop_original.price_american_put_lsm_llh(
    model_bs, res_bs, K=K, basis_order=3,
    use_cv=True, improved=True, ridge=1e-5,
    euro_method='bs'
)

am_corr = aop_corrected.price_american_put_lsm_llh(
    model_bs, res_bs, K=K, basis_order=3,
    use_cv=True, improved=True, ridge=1e-5,
    euro_method='bs'
)

print(f"\nOriginal (sigma_hat[:, j-1]):")
print(f"  P_A = {am_orig['price']:.6f}")
print(f"  P_A - P_E = {am_orig['price'] - bs_put:.6f}")
status_orig = "✓ VALID" if am_orig['price'] >= bs_put - 0.001 else "✗ VIOLATION"
print(f"  Status: {status_orig}")

print(f"\nCorrected (sigma_hat[:, j]):")
print(f"  P_A = {am_corr['price']:.6f}")
print(f"  P_A - P_E = {am_corr['price'] - bs_put:.6f}")
status_corr = "✓ VALID" if am_corr['price'] >= bs_put - 0.001 else "✗ VIOLATION"
print(f"  Status: {status_corr}")

print(f"\nImprovement: {am_corr['price'] - am_orig['price']:.6f}")

# Test LLH model
print(f"\n[2] LLH Model Test (1 month)")
print("-" * 80)
model_llh = pm.ImprovedSteinStein(
    r=0.3943, rho=0.1691, kappa=4.9394, nu=0.4,
    sigma0=0.2924, theta0=0.1319, lam=0.3115, eta=0.4112, seed=42
)

tau_1m = 0.083
K_test = 90.0
S0_test = 100.0

res_llh = model_llh.simulate_prices(S0=S0_test, T=tau_1m, n_steps=22, n_paths=5000)

# Use BS proxy to avoid LLH timing issues
am_orig_llh = aop_original.price_american_put_lsm_llh(
    model_llh, res_llh, K=K_test, basis_order=3,
    use_cv=True, improved=True, ridge=1e-5,
    euro_method='bs'
)

am_corr_llh = aop_corrected.price_american_put_lsm_llh(
    model_llh, res_llh, K=K_test, basis_order=3,
    use_cv=True, improved=True, ridge=1e-5,
    euro_method='bs'
)

print(f"Original: P_A = {am_orig_llh['price']:.6f}")
print(f"Corrected: P_A = {am_corr_llh['price']:.6f}")
print(f"Difference: {am_corr_llh['price'] - am_orig_llh['price']:.6f}")

print(f"\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"European put (benchmark): {bs_put:.6f}")
print(f"\nBS Limit Model:")
print(f"  Original:  {am_orig['price']:.6f} ({status_orig})")
print(f"  Corrected: {am_corr['price']:.6f} ({status_corr})")
print(f"\nLLH Model (1 month):")
print(f"  Original:  {am_orig_llh['price']:.6f}")
print(f"  Corrected: {am_corr_llh['price']:.6f}")

if status_corr == "✓ VALID":
    print(f"\n✓ SUCCESS: Index alignment fix improves results!")
    print(f"   The safeguard max(Ej, TV) + correct indexing = valid prices")
else:
    print(f"\n⚠ PARTIAL: More investigation needed")
print("="*80)
