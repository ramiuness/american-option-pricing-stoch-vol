#!/usr/bin/env python3
"""Regenerate the code-produced figures referenced in the LaTeX report.

The two underlying pipelines (``src/generate_plots.py``,
``src/timing_analysis.py``) still write their PNGs into the project's own
``figs/`` directory as today; this script invokes them in-process and
then copies the report-relevant subset (listed in
``scripts/report_figs.txt``) into the caller-specified ``--out-dir``.

External figures that the script deliberately does NOT regenerate
(static or authored outside the codebase) are ``udem_logo.png``,
``fig_paths_table1.png``, and ``fig_paths_table2.png``. The script
checks for them in ``--out-dir`` and warns if absent.

Usage
-----
    python scripts/regen_report_figs.py --out-dir /path/to/report/figs
    python scripts/regen_report_figs.py --out-dir ./out --param-sets T1
    python scripts/regen_report_figs.py --out-dir ./out --skip-timing
    python scripts/regen_report_figs.py --list-only
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE.parent
SRC_DIR = PROJECT_DIR / "src"
PROJECT_FIGS = PROJECT_DIR / "figs"
FIG_LIST_PATH = HERE / "report_figs.txt"
EXTERNAL_FIGS = (
    "udem_logo.png",
    "fig_paths_table1.png",
    "fig_paths_table2.png",
)


def _load_fig_list(path: Path) -> list[str]:
    figs: list[str] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            figs.append(line)
    return figs


def _run_generate_plots(param_sets: list[str]) -> None:
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    import generate_plots as gp  # type: ignore

    for name in param_sets:
        if name not in gp.PARAM_SETS:
            raise SystemExit(
                f"Unknown param set '{name}'. "
                f"Available: {list(gp.PARAM_SETS)}"
            )

    import matplotlib.pyplot as plt
    plt.rcParams.update(gp.STYLE)
    gp._ensure_output_dir()
    for name in param_sets:
        gp._run_param_set(name)
    print("\n  Plot 9: BS-limit American put...")
    gp.plot_american_bs_limit()


def _run_timing_analysis() -> None:
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    import timing_analysis as ta  # type: ignore

    if hasattr(ta, "main") and callable(ta.main):
        ta.main()
        return
    # Fallback: exec the module body under __name__ == '__main__' so its
    # bottom-of-file entrypoint fires.
    code = (SRC_DIR / "timing_analysis.py").read_text()
    exec(compile(code, str(SRC_DIR / "timing_analysis.py"), "exec"),
         {"__name__": "__main__", "__file__": str(SRC_DIR / "timing_analysis.py")})


def _copy_figs(out_dir: Path, fig_list: list[str]) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    missing = 0
    for name in fig_list:
        src = PROJECT_FIGS / name
        if src.exists():
            shutil.copy2(src, out_dir / name)
        else:
            print(f"  MISSING {name}", file=sys.stderr)
            missing += 1
    return missing


def _check_external(out_dir: Path) -> None:
    for name in EXTERNAL_FIGS:
        if (out_dir / name).exists():
            print(f"  OK      {name}")
        else:
            print(f"  WARNING {name} not found in {out_dir}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Regenerate the LaTeX-report figures and copy them to "
                    "a user-specified output directory.",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=None,
        help="Destination directory for the copied report figures "
             "(typically your LaTeX project's figs/ directory). "
             "Required unless --list-only is given.",
    )
    parser.add_argument(
        "--param-sets", default="T1,T2",
        help="Comma-separated LLH parameter-set labels to run "
             "(default: T1,T2).",
    )
    parser.add_argument(
        "--skip-plots", action="store_true",
        help="Skip generate_plots.py and reuse the existing figs/ contents.",
    )
    parser.add_argument(
        "--skip-timing", action="store_true",
        help="Skip timing_analysis.py.",
    )
    parser.add_argument(
        "--list-only", action="store_true",
        help="Print the figure list (comments stripped) and exit without "
             "running or copying.",
    )
    args = parser.parse_args()

    fig_list = _load_fig_list(FIG_LIST_PATH)
    if args.list_only:
        for name in fig_list:
            print(name)
        return 0

    if args.out_dir is None:
        parser.error("--out-dir is required unless --list-only is given")
    out_dir = args.out_dir.resolve()
    print(f"Project dir : {PROJECT_DIR}")
    print(f"Copying to  : {out_dir}\n")

    if not args.skip_plots:
        param_sets = [s.strip() for s in args.param_sets.split(",") if s.strip()]
        print(f"=== generate_plots.py ({', '.join(param_sets)}; ~30 min) ===")
        _run_generate_plots(param_sets)
        print()

    if not args.skip_timing:
        print("=== timing_analysis.py (~5-8 min) ===")
        _run_timing_analysis()
        print()

    print(f"=== copying {len(fig_list)} figures to {out_dir} ===")
    missing = _copy_figs(out_dir, fig_list)

    print("\n=== Unmanaged report figures (checked, not regenerated) ===")
    _check_external(out_dir)

    if missing:
        print(f"\nDone with {missing} missing generated figure(s).",
              file=sys.stderr)
        return 1
    print(f"\nDone. All {len(fig_list)} generated figures copied to {out_dir}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
