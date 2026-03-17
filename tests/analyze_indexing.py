"""
Analyze the exact indexing convention to determine correct alignment.
"""

import numpy as np
import priceModels as pm

print("="*80)
print("INDEXING CONVENTION ANALYSIS")
print("="*80)

# Simple BS model
model = pm.ImprovedSteinStein(
    r=0.05, rho=0.0, kappa=0.0, nu=0.0,
    sigma0=0.2, theta0=0.0, lam=0.0, eta=0.0, seed=42
)

# Small simulation for clarity
res = model.simulate_prices(S0=100.0, T=0.1, n_steps=5, n_paths=1)

S = res['S'][0, :]
sigma_hat = res['sigma_hat'][0, :]
B = res['B'][0, :]
dt = res['dt']

print(f"\nn_steps = 5, dt = {dt:.4f}")
print(f"\nTime grid: t_0=0.0, t_1={dt:.4f}, t_2={2*dt:.4f}, ..., t_5={5*dt:.4f}")
print(f"\n{'Index':<8} {'Time':<10} {'S':<12} {'sigma_hat':<12} {'B':<12}")
print("-" * 54)

for j in range(6):
    S_val = f"{S[j]:.6f}" if j < len(S) else "N/A"
    sigma_val = f"{sigma_hat[j]:.6f}" if j < len(sigma_hat) else "N/A"
    B_val = f"{B[j]:.6f}" if j < len(B) else "N/A"
    print(f"{j:<8} {j*dt:<10.4f} {S_val:<12} {sigma_val:<12} {B_val:<12}")

print(f"\nKey observations:")
print(f"- S has indices 0 to 5 (6 values)")
print(f"- sigma_hat has indices 0 to 4 (5 values)")
print(f"- B has indices 0 to 4 (5 values)")

print(f"\nSimulation logic (multiplicative_euler_prices):")
print(f"  S[:, j+1] = S[:, j] * (1 + r*dt + sigma_hat[:, j] * dW1[:, j])")
print(f"  ")
print(f"  So sigma_hat[i] drives transition from S[i] to S[i+1]")
print(f"  Equivalently: sigma_hat[i] is the volatility DURING interval [t_i, t_{i+1}]")

print(f"\nIn LSM at time step j (backward loop):")
print(f"  - We are at time t_j with price S[:, j]")
print(f"  - We need to price a European option from t_j to T")
print(f"  - Time remaining: tau_j = (n_steps - j) * dt")
print(f"  ")
print(f"  Question: What volatility to use?")
print(f"  ")
print(f"  Option 1: sigma_hat[:, j-1]  (volatility that drove us TO t_j)")
print(f"    - This is the PAST volatility")
print(f"    - Wrong for forward pricing!")
print(f"  ")
print(f"  Option 2: sigma_hat[:, j]    (volatility for interval [t_j, t_{{j+1}}])")
print(f"    - This is the CURRENT/NEXT volatility")
print(f"    - Correct for forward pricing!")
print(f"  ")
print(f"  HOWEVER: At j=n_steps (maturity), sigma_hat[:, n_steps] doesn't exist!")
print(f"    - The loop goes j = n_steps-1 down to 1")
print(f"    - At j=n_steps-1: sigma_hat[:, n_steps-1] exists ✓")
print(f"    - At j=1: sigma_hat[:, 1] exists ✓")

print(f"\nCurrent code uses: volj = sigma_hat[:, j-1]")
print(f"  At j=n_steps-1={5-1}: uses sigma_hat[:, {5-2}] = sigma_hat[:, 3]")
print(f"  At j=1: uses sigma_hat[:, 0]")

print(f"\nCorrect should be: volj = sigma_hat[:, min(j, n_steps-1)]")
print(f"  At j=n_steps-1=4: uses sigma_hat[:, 4] ✓")
print(f"  At j=1: uses sigma_hat[:, 1] ✓")
print(f"  ")
print(f"  But wait - we're in BACKWARD loop, so j goes from n_steps-1 down to 1")
print(f"  j ranges from {5-1} to 1")
print(f"  So j is always < n_steps, meaning sigma_hat[:, j] always exists!")

print(f"\nCONCLUSION:")
print(f"  Current: volj = sigma_hat[:, j-1]  (WRONG - off by one)")
print(f"  Correct: volj = sigma_hat[:, j]    (RIGHT - current volatility for forward pricing)")
print("="*80)
