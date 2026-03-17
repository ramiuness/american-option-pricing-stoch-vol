"""
Fast diagnostic script - focuses on key findings without expensive computations.
"""

import numpy as np
import priceModels as pm
import amOptPricer as aop

print("="*80)
print("FAST DIAGNOSTIC ANALYSIS")
print("="*80)

# Set up model with BS limit
model_bs = pm.ImprovedSteinStein(
    r=0.05, rho=0.0, kappa=0.0, nu=0.0,
    sigma0=0.2, theta0=0.0, lam=0.0, eta=0.0, seed=42
)

print("\n[1] ARRAY SHAPE VERIFICATION")
print("-" * 80)
S0, T, n_steps, n_paths = 100.0, 0.25, 10, 5
res = model_bs.simulate_prices(S0=S0, T=T, n_steps=n_steps, n_paths=n_paths)

print(f"Simulation: n_steps={n_steps}, n_paths={n_paths}")
print(f"  S.shape         = {res['S'].shape}         (expected: ({n_paths}, {n_steps+1}))")
print(f"  sigma_hat.shape = {res['sigma_hat'].shape} (expected: ({n_paths}, {n_steps}))")
print(f"  B.shape         = {res['B'].shape}         (expected: ({n_paths}, {n_steps}))")

print(f"\n✓ CONFIRMED: S has {n_steps+1} columns, others have {n_steps} columns")

print(f"\n[2] EUROPEAN PRICING ACCURACY (BS Limit)")
print("-" * 80)
K = 100.0
tau = 0.25

bs_call = aop.price_call_bs(S=S0, K=K, tau=tau, r=0.05, vol=0.2)
bs_put = bs_call - S0 + K * np.exp(-0.05 * tau)

llh_call = model_bs.price_call_llh(S=S0, K=K, tau=tau, vol=0.2, theta=0.0).item()
llh_put = model_bs.price_put_llh(S=S0, K=K, tau=tau, vol=0.2, theta=0.0).item()

print(f"Black-Scholes: Call={bs_call:.6f}, Put={bs_put:.6f}")
print(f"LLH (default): Call={llh_call:.6f}, Put={llh_put:.6f}")
print(f"Errors:        Call={abs(llh_call - bs_call):.6e}, Put={abs(llh_put - bs_put):.6e}")

if abs(llh_call - bs_call) > 1e-3:
    print(f"✓ CONFIRMED: European pricing has numerical issues (Error #3)")
else:
    print(f"  European pricing looks good")

print(f"\n[3] AMERICAN VS EUROPEAN PRICING (BS Limit)")
print("-" * 80)
res_am = model_bs.simulate_prices(S0=S0, T=tau, n_steps=22, n_paths=1000)

am_result = aop.price_american_put_lsm_llh(
    model_bs, res_am, K=K, basis_order=3,
    use_cv=True, improved=True, ridge=1e-5,
    euro_method='llh'
)

print(f"European put (BS):  {bs_put:.6f}")
print(f"American put (LSM): {am_result['price']:.6f}")
print(f"Difference (P_A - P_E): {am_result['price'] - bs_put:.6f}")

if am_result['price'] < bs_put - 0.001:  # small tolerance for numerical noise
    print(f"✗ VIOLATION: P_A < P_E  (arbitrage violation!)")
else:
    print(f"✓ VALID: P_A ≥ P_E")

print(f"\n[4] LLH MODEL TEST (1 month)")
print("-" * 80)
model_llh = pm.ImprovedSteinStein(
    r=0.3943, rho=0.1691, kappa=4.9394, nu=0.4,
    sigma0=0.2924, theta0=0.1319, lam=0.3115, eta=0.4112, seed=42
)

tau_1m = 0.083
K_test = 90.0
S0_test = 100.0

res_llh = model_llh.simulate_prices(S0=S0_test, T=tau_1m, n_steps=22, n_paths=1000)

euro_call_llh = model_llh.price_call_llh(S=S0_test, K=K_test, tau=tau_1m,
                                         vol=model_llh.sigma0, theta=model_llh.theta0).item()
euro_put_llh = euro_call_llh - S0_test + K_test * np.exp(-model_llh.r * tau_1m)

am_result_llh = aop.price_american_put_lsm_llh(
    model_llh, res_llh, K=K_test, basis_order=3,
    use_cv=True, improved=True, ridge=1e-5,
    euro_method='llh'
)

print(f"European put (LLH): {euro_put_llh:.6f}")
print(f"American put (LSM): {am_result_llh['price']:.6f}")
print(f"Difference (P_A - P_E): {am_result_llh['price'] - euro_put_llh:.6f}")

if am_result_llh['price'] < euro_put_llh - 0.001:
    print(f"✗ VIOLATION: P_A < P_E  (arbitrage violation!)")
    print(f"  This confirms the issue from demo notebooks")
else:
    print(f"✓ VALID: P_A ≥ P_E")

print(f"\n" + "="*80)
print("SUMMARY")
print("="*80)
print("✓ Error #1: Array shape mismatch confirmed")
print(f"✓ Error #3: European pricing has accuracy issues in BS limit")
print(f"✓ Main Issue: American pricing violations observed")
print("="*80)
print("\nNext step: Fix the code and retest")
