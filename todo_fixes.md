# TODO: Remaining Fixes

## Completed

- [x] A–G, 1–9: Initial fixes (VR formula, CV-BS removal, bar→line, T2 hardening, dual basis, convergence, report updates)
- [x] 10A: Pricer defaults → `floor_method='llh'`, `n_steps_rk4=256`
- [x] 10B: BS-limit override + dead param cleanup + Laguerre grid dedup
- [x] 10C: `K_RK4 = 256` in timing analysis
- [x] 10D: Report setup table — `n_steps_ODE` 128→256, exercise floor row
- [x] 10E: Plain LSM ITM-only LLH floor — 2.5x speedup
- [x] 10F: Merged timing figures into `fig_timing_combined.png` 1×2 panel
- [x] 10-plots: All affected figures regenerated (fig5/6/7/7b/8 × T1/T2 + timing)

- [x] 11: Precompute shared European grid (`PrecomputedEuropean` + `precompute_european()`)
- [x] 11-callers: Updated 3 call sites in `generate_plots.py` to use `precomputed=`
- [x] 12: Timing breakdown uses shared precompute — 4 honest bars (sim, precomp, plain, CV-LLH)
- [x] 13: Updated §4.3 convergence plots — S0/K=(95,100) ITM + (100,95) OTM, N up to 50K. Memory-safe defaults (del+gc.collect in inner loops, plt.close after saves). All 6 figures regenerated.

- [x] v1 promotion: `_v1` files promoted to working versions, old files backed up as `_v0`. Imports fixed.

## Completed (details)

### 11. Precompute shared European grid

**Problem:** Every call site that compares plain LSM vs CV-LLH on the
same simulation independently computes `tau_cache`, `theta_lr`, and
European prices — doubling the ODE and quadrature cost.

**Affected call sites** (3 pairs, each runs both pricers on same `sim`):

| Function | Pairs | Savings |
|----------|-------|---------|
| `_compute_american_grid` | 5 S0 × 2 horizons = 10 | 10 European passes |
| `plot_american_put_panels` | 5 S0 × 2 K × 2 horizons = 20 | 20 European passes |
| `plot_mc_path_convergence` | 2 cases × multiple N | per (case, N) |

(`plot_american_bs_limit` excluded — uses `floor_method='bs'`, no ODE.)

**Fix — 4 functions (1 new, 3 modified):**

#### New: `precompute_european(model, sim_out, K, phi_max, n_phi, n_steps_rk4, eps0)`

Builds and returns a `PrecomputedEuropean` dataclass:

```python
@dataclass
class PrecomputedEuropean:
    tau_cache: dict[int, object]
    # tau_cache[steps_remaining] → ODE coefficients
    # keys: unique remaining-step counts 1..M-1

    theta_lr: np.ndarray   # (n_paths, n_steps)
    # long-run mean process grid

    euro_grid: dict[int, np.ndarray]
    # euro_grid[j] → European put prices, shape (n_paths,)
    # j > 1: computed for ITM paths only, zeros for OTM
    # j == 1: computed for ALL paths (CV-LLH global correction needs this)
```

#### Modified: `_setup(... precomputed=None)`

When `precomputed` is provided:
- Use `precomputed.tau_cache` and `precomputed.theta_lr`
- Skip internal ODE precomputation and theta grid construction

When `None`: current behavior unchanged.

#### Modified: `_backward_loop(... precomputed=None)`

When `precomputed` is provided:
- Read `Ej = precomputed.euro_grid[j]` instead of calling
  `_euro_put_slice` / `_euro_put_bs_slice`
- Skip the ITM-subsetting and dispatch logic for European prices

When `None`: current behavior unchanged.

#### Modified: `price_american_put_lsm_llh(... precomputed=None)`

New optional kwarg. Threads `precomputed` to `_setup` and `_backward_loop`.

**Caller pattern:**

```python
pre = ap.precompute_european(model, sim, K, **AM_LLH_PARAMS)
res_plain = ap.price_american_put_lsm_llh(model, sim, K, use_cv=False, precomputed=pre)
res_llh   = ap.price_american_put_lsm_llh(model, sim, K, use_cv=True,
                                           euro_method='llh', precomputed=pre)
```

**Verification:**
1. Prices must match current output (precompute is a pure refactor)
2. Timing: paired calls should take ~50% less than current
3. `precomputed=None` path unchanged — no existing caller breaks

### 12. Fix timing breakdown asymmetry

**Problem:** `_time_stages` gives CV-LLH a 3-bar decomposition (sim, ODE,
backward) but lumps plain LSM's ODE into "backward". With `floor_method='llh'`
both methods do ODE work, so both need 3 bars.

**Depends on:** task 11 — with `precompute_european`, the ODE cost is
measured once and attributed to a shared "precompute" bar, making the
breakdown honest by construction.

**Fix:** After task 11, restructure `_time_stages` to time:
1. Simulation
2. `precompute_european` (shared)
3. Plain LSM backward (pricer with `precomputed`)
4. CV-LLH backward (pricer with `precomputed`)

Update `plot_timing_combined` left panel to show 4 bars:
Shared sim, Shared European precomp, Plain backward, CV-LLH backward.


### 13. Update §4.3 convergence plots — new S0/K + N to 50K

**Changes made (code + report):**
- `plot_mc_path_convergence` defaults: S0_cases
((95,100,'ITM'), (100,95,'OTM')), N_values
(5k,10k,20k,50k)
- `plot_mc1_floor_convergence` defaults: same
- Report §4.3 text: S0=95/K=100 ITM, S0=100/K=95
OTM
- Memory-safe: `del` + `gc.collect()` after each
iteration, `plt.close(fig)` after saves,
`gc.collect()` between parameter sets in `main()`

**Figures regenerated (6 total):**
- fig10_price_convergence_T1.png
- fig10_eep_convergence_T1.png
- fig10_price_convergence_T1_mc1floor.png
- fig10_price_convergence_T2.png
- fig10_eep_convergence_T2.png
- fig10_price_convergence_T2_mc1floor.png

## Pending

(none)
