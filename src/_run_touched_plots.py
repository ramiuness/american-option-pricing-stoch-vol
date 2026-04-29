"""Run only the three plot functions touched in the recent edits.

Invoked as:   python _run_touched_plots.py T1
              python _run_touched_plots.py T2

For T1: produces fig6, fig7_vr_ratios, fig5_american_put_T1_1y_floors.
For T2: produces fig6, fig7_vr_ratios (no floor plot per existing orchestration).
"""
import gc
import sys
import time

import matplotlib.pyplot as plt

import generate_plots as gp


def run(pset_name: str) -> None:
    label = gp.PARAM_LABELS.get(pset_name, pset_name)
    print(f"\n=== {label} ({pset_name}) ===")

    t0 = time.time()
    print(f"  [{time.time()-t0:6.1f}s] Computing American grid (Laguerre)...")
    am_lag = gp._compute_american_grid(
        pset_name,
        basis_type=gp.BASIS_LAGUERRE[0],
        basis_order=gp.BASIS_LAGUERRE[1],
        ridge=gp.BASIS_LAGUERRE[2],
    )
    gc.collect()

    print(f"  [{time.time()-t0:6.1f}s] Computing American grid (Gaussian)...")
    am_gauss = gp._compute_american_grid(
        pset_name,
        basis_type=gp.BASIS_GAUSSIAN[0],
        basis_order=gp.BASIS_GAUSSIAN[1],
        ridge=gp.BASIS_GAUSSIAN[2],
    )
    gc.collect()

    print(f"  [{time.time()-t0:6.1f}s] Plot 6: American prices vs spot (Laguerre + RBF)...")
    gp.plot_american_prices_vs_spot(pset_name, label, am_lag, am_gauss)
    del am_gauss
    gc.collect()

    print(f"  [{time.time()-t0:6.1f}s] Plot 7: VR ratios (1-year only)...")
    gp.plot_vr_ratios(pset_name, label, am_lag)
    del am_lag
    gc.collect()

    if pset_name == 'T1':
        print(f"  [{time.time()-t0:6.1f}s] Plot 5 floors: American put K=100 floor comparison...")
        gp.plot_american_put_panels_floors(pset_name, label, '1y')
        gc.collect()

    print(f"  [{time.time()-t0:6.1f}s] done: {pset_name}")


if __name__ == '__main__':
    plt.rcParams.update(gp.STYLE)
    gp._ensure_output_dir()
    for name in sys.argv[1:]:
        run(name)
