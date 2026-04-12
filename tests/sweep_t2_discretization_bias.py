"""
Multi-seed sweep over n_steps_mc to characterise the LLH-vs-MC bias floor
for LLH Table 2 parameters.

Background
----------
Single-seed sweeps revealed that the T2 OTM bias has two components:
  1. A linear-in-Δt component visible at small n_steps_mc
  2. An asymptotic floor at ~−5% to −6% that doesn't shrink with refinement

Sweep B (LLH precision sweep) confirmed the LLH closed-form is bulletproof
to 8+ decimals at the STANDARD preset, so the floor is entirely on the
MC simulation side. The structural hypothesis is the leverage-coupling bias
of the multiplicative Euler scheme: for stochastic σ correlated with the
asset Brownian (T2 has ρ=0.169), the truncation σ̂[j]·dW1[j] carries a
non-zero-mean error that does not shrink with Δt.

This script averages the bias across 5 RNG seeds per (n_steps_mc, K) cell
to remove single-realization noise and fits bias(Δt) ≈ a + b·Δt to extract
the asymptote `a` (the floor scheme changes need to fix) and the linear
coefficient `b` (the part that log-Euler will close).

Run
---
    python tests/sweep_t2_discretization_bias.py                              # default: euler
    python tests/sweep_t2_discretization_bias.py --scheme log-euler           # log-Euler variant
    python tests/sweep_t2_discretization_bias.py --scheme predictor-corrector # midpoint sigma
    python tests/sweep_t2_discretization_bias.py --scheme milstein            # 2D Milstein leverage term
    cd tests && python sweep_t2_discretization_bias.py [--scheme ...]

Wall clock: ~4 minutes. Peak RSS: ~2.5 GiB at the largest n_steps_mc.
"""

import argparse
import os
import sys
import time

import numpy as np

# Script-relative path so the file works regardless of cwd
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'src'))

import generate_plots as gp  # noqa: E402


# ── Configuration ───────────────────────────────────────────────────────────

N_PATHS = 200_000
SEEDS = (123, 124, 125, 126, 127)
N_STEPS_VALUES = (13, 26, 52, 104, 208, 416)
S0 = 100.0
K_VALUES = (80.0, 100.0, 120.0)
MONEYNESS = {80.0: 'ITM', 100.0: 'ATM', 120.0: 'OTM'}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument(
        '--scheme',
        choices=('euler', 'log-euler', 'predictor-corrector', 'milstein'),
        default='euler',
        help="asset-step discretization (default: 'euler' = multiplicative Euler)",
    )
    args = parser.parse_args()
    scheme = args.scheme

    print(f'  T2 params, S0={S0}, n_paths={N_PATHS:,}, n_seeds={len(SEEDS)}')
    print(f'  scheme: {scheme}')
    print()

    # results[n_steps][K] = list of bias_pct across SEEDS
    results: dict = {n: {k: [] for k in K_VALUES} for n in N_STEPS_VALUES}

    for n_steps in N_STEPS_VALUES:
        dt = 1.0 / n_steps
        t0 = time.perf_counter()
        for seed in SEEDS:
            m = gp._make_model('T2', seed=seed)
            g = gp._compute_mc_llh_grid(
                m,
                n_paths=N_PATHS,
                n_steps_mc=n_steps,
                S0_values=(S0,),
                K_values=K_VALUES,
                scheme=scheme,
            )
            for K in K_VALUES:
                v = g[(S0, K)]
                bias_pct = (v['mc'] - v['llh']) / v['llh'] * 100
                results[n_steps][K].append(bias_pct)
        elapsed = time.perf_counter() - t0

        line = f'  n_steps={n_steps:>3}  dt={dt:.4f}  ({elapsed:5.1f}s) | '
        for K in K_VALUES:
            biases = results[n_steps][K]
            mean = float(np.mean(biases))
            sem = float(np.std(biases, ddof=1) / np.sqrt(len(SEEDS)))
            line += f'K={int(K):3} {MONEYNESS[K]:>3}: {mean:>+7.3f}±{sem:.2f}%  '
        print(line)

    # ── Linear fit on OTM (K=120) bias ──
    print()
    print('  Asymptote fit for K=120 OTM: bias(dt) ≈ a + b·dt')

    otm_means = np.array([np.mean(results[n][120.0]) for n in N_STEPS_VALUES])
    otm_sems = np.array(
        [np.std(results[n][120.0], ddof=1) / np.sqrt(len(SEEDS)) for n in N_STEPS_VALUES]
    )
    dts = np.array([1.0 / n for n in N_STEPS_VALUES])

    A = np.vstack([np.ones_like(dts), dts]).T
    (a_fit, b_fit), *_ = np.linalg.lstsq(A, otm_means, rcond=None)

    print(f'    a (asymptote)         = {a_fit:>+7.3f}%')
    print(f'    b (linear coefficient) = {b_fit:>+7.3f}% per dt-unit')
    print(f'    floor estimate (dt=0)  = {a_fit:>+7.3f}%')
    print()
    print(f'  {"n_steps":>8} {"dt":>8}   {"observed":>13}   {"predicted":>10}   {"residual":>10}')
    for n, dt, obs, sem in zip(N_STEPS_VALUES, dts, otm_means, otm_sems):
        pred = a_fit + b_fit * dt
        print(
            f'  {n:>8} {dt:>8.4f}   {obs:>+8.3f}±{sem:.2f}%   '
            f'{pred:>+9.3f}%   {obs - pred:>+9.3f}%'
        )


if __name__ == '__main__':
    main()
