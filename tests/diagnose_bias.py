"""
Diagnostic script to understand the LLH pricing bias.

Tests:
1. Black-Scholes limit (all stochastic terms = 0)
2. Parameter sensitivity
3. MC convergence
"""

import numpy as np
import sys
sys.path.insert(0, '/home/ramiuness/Documents/study/umontreal/myCourses/amerOptionsPricing/project/code')
import priceModels as pm
import amOptPricer_corrected as aop
from scipy.stats import norm

print("=" * 80)
print("LLH PRICING BIAS DIAGNOSTIC")
print("=" * 80)

# Black-Scholes reference for validation
def bs_call(S, K, tau, r, sigma):
    """Black-Scholes call price formula."""
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    return S * norm.cdf(d1) - K * np.exp(-r*tau) * norm.cdf(d2)

# Test parameters
S0, K, tau = 100.0, 90.0, 1.0
r = 0.01
sigma_const = 0.15

print("\n" + "=" * 80)
print("TEST 1: BLACK-SCHOLES LIMIT (all stochastic terms = 0)")
print("=" * 80)
print("If kappa=nu=lam=eta=rho=0, LLH should reduce to constant-vol BS")

# Create BS-limit model
model_bs = pm.ImprovedSteinStein(
    r=r, rho=0.0, kappa=0.0, nu=0.0,
    sigma0=sigma_const, theta0=0.0, lam=0.0, eta=0.0, seed=42
)

# Analytical BS price
bs_analytical = bs_call(S0, K, tau, r, sigma_const)
print(f"\nBS analytical:  {bs_analytical:.6f}")

# LLH formula (should match BS)
llh_price = model_bs.price_call_llh(
    S=S0, K=K, tau=tau, vol=sigma_const, theta=0.0,
    phi_max=300.0, n_phi=513, n_steps_ode=128
).item()
print(f"LLH formula:    {llh_price:.6f}")
print(f"Difference:     {llh_price - bs_analytical:.6f} ({(llh_price/bs_analytical-1)*100:.2f}%)")

# MC simulation
res_bs = model_bs.simulate_prices(S0=S0, T=tau, n_steps_mc=252, n_paths=500_000)
mc_result = aop.price_call_mc(res_bs['S'], K=K, T=tau, r=r)
print(f"MC simulation:  {mc_result['price']:.6f} ± {np.diff(mc_result['ci_95'])[0]/2:.6f}")

if abs(llh_price - bs_analytical) / bs_analytical > 0.05:
    print("\n⚠️  WARNING: LLH formula does NOT match BS in the limit!")
    print("    This indicates a FUNDAMENTAL PROBLEM with the ODE system or implementation.")
else:
    print("\n✓ LLH formula correctly reduces to BS in the limit")

print("\n" + "=" * 80)
print("TEST 2: LLH MODEL WITH STOCHASTIC VOLATILITY")
print("=" * 80)
print("Using Table 1 parameters from Lin-Lin-He paper")

# Table 1 parameters
model_llh = pm.ImprovedSteinStein(
    r=0.01, rho=-0.2, kappa=5, nu=0.2,
    sigma0=0.15, theta0=0.18, lam=0.9, eta=0.01, seed=42
)

print(f"\nParameters: r={model_llh.r}, κ={model_llh.kappa}, ν={model_llh.nu}")
print(f"           σ₀={model_llh.sigma0}, θ₀={model_llh.theta0}")
print(f"           λ={model_llh.lam}, η={model_llh.eta}, ρ={model_llh.rho}")

# LLH formula price
llh_stoch = model_llh.price_call_llh(
    S=S0, K=K, tau=tau, vol=model_llh.sigma0, theta=model_llh.theta0,
    phi_max=300.0, n_phi=513, n_steps_ode=128
).item()
print(f"\nLLH formula:    {llh_stoch:.6f}")

# MC simulation with MANY paths
print("\nMC simulation (converging)...")
for n_paths in [50_000, 100_000, 250_000]:
    res_llh = model_llh.simulate_prices(S0=S0, T=tau, n_steps_mc=252, n_paths=n_paths)
    mc_result = aop.price_call_mc(res_llh['S'], K=K, T=tau, r=model_llh.r)
    print(f"  n={n_paths:7d}: {mc_result['price']:.6f} ± {np.diff(mc_result['ci_95'])[0]/2:.4f}")

res_llh = model_llh.simulate_prices(S0=S0, T=tau, n_steps_mc=252, n_paths=250_000)
mc_result = aop.price_call_mc(res_llh['S'], K=K, T=tau, r=model_llh.r)
mc_price = mc_result['price']

bias_pct = (llh_stoch - mc_price) / mc_price * 100
print(f"\nBias: {llh_stoch - mc_price:.6f} ({bias_pct:.1f}%)")

if abs(bias_pct) > 20:
    print("\n⚠️  WARNING: Bias > 20% indicates SERIOUS PROBLEM")
    print("    Possible causes:")
    print("    1. ODE system does not match the SDE dynamics")
    print("    2. Simulation discretization does not match continuous model")
    print("    3. Wrong initial conditions or parameter interpretation")

print("\n" + "=" * 80)
print("TEST 3: PARAMETER SENSITIVITY")
print("=" * 80)
print("Test if bias changes with different parameters")

test_configs = [
    ("Low λ", dict(r=0.01, rho=-0.2, kappa=5, nu=0.2, sigma0=0.15, theta0=0.18, lam=0.1, eta=0.01)),
    ("Zero λ,η", dict(r=0.01, rho=-0.2, kappa=5, nu=0.2, sigma0=0.15, theta0=0.18, lam=0.0, eta=0.0)),
]

for name, params in test_configs:
    model_test = pm.ImprovedSteinStein(**params, seed=42)
    llh_test = model_test.price_call_llh(
        S=S0, K=K, tau=tau, vol=model_test.sigma0, theta=model_test.theta0,
        phi_max=300.0, n_phi=513, n_steps_ode=128
    ).item()
    res_test = model_test.simulate_prices(S0=S0, T=tau, n_steps_mc=252, n_paths=100_000)
    mc_test = aop.price_call_mc(res_test['S'], K=K, T=tau, r=model_test.r)['price']

    print(f"\n{name}:")
    print(f"  LLH: {llh_test:.4f}, MC: {mc_test:.4f}, Bias: {(llh_test-mc_test)/mc_test*100:.1f}%")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
