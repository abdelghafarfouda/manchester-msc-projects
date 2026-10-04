#!/usr/bin/env python3
"""Run the whole workflow and write every table and figure.

    python scripts/run_all.py                 # into results/
    python scripts/run_all.py --out-dir DIR   # somewhere else (CI compares DIR with results/)

The original workflow writes the 12 tables and summary.json to <out>/tables and
the six figures to <out>/figures.  The depth-block test of the October 2026
revision writes to <out>/depth_blocks.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seisgeomech import analysis, depth_blocks, figures  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Regenerate every table and figure.")
    ap.add_argument("--out-dir", default=None,
                    help="output folder (default: the project's results/)")
    args = ap.parse_args(argv)
    out = Path(args.out_dir) if args.out_dir else ROOT / "results"

    t0 = time.time()
    print("SeisGeoMech - running workflow")
    res = analysis.run(results_root=out)
    paths = figures.make_all(res, out / "figures")

    s = res.scalars
    print("\n--- headline results ---")
    print(f"overlap interval          {s['overlap_top_m']:.1f}-{s['overlap_base_m']:.1f} m "
          f"({s['overlap_n']} samples)")
    print(f"Gardner bias              {s['gardner_bias_gcc']:+.4f} g/cm3 "
          f"({s['gardner_bias_pct']:+.2f} %)")
    print(f"Gardner RMSE              {s['gardner_rmse_gcc']:.4f} g/cm3")
    print(f"rho g gradient            {s['rho_g_gradient_measured_MPa_per_km']:.3f} MPa/km measured, "
          f"{s['rho_g_gradient_gardner_MPa_per_km']:.3f} MPa/km Gardner "
          f"({s['rho_g_gradient_difference_pct']:+.2f} %)")
    print(f"RC RMS ratio (pre-conv.)  {s['rc_rms_ratio_gardner_over_measured_pre_convolution']:.3f} "
          f"(correlation {s['rc_correlation']:.3f})")
    print(f"synthetic RMS ratio       {s['synthetic_rms_ratio_min']:.3f} - "
          f"{s['synthetic_rms_ratio_max']:.3f} across the tested frequencies")
    print(f"olivine VRH Vp            {res.arrays['olivine'].vp_m_s:.1f} m/s")
    print(f"Merivale E, nu            {res.arrays['merivale'].E_Pa/1e9:.2f} GPa, "
          f"{res.arrays['merivale'].nu:.3f}")

    blocks = depth_blocks.run(out / "depth_blocks")
    block_fig = figures.fig_depth_blocks(blocks, out / "depth_blocks")
    print("\n--- depth-block test (configs/depth_blocks.json) ---")
    for name, d in blocks["scores"]["directions"].items():
        h, r = d["held_out_block_fit"], d["held_out_supplied_gardner"]
        print(f"{name:6s} fit {d['fitted_coefficient_a']:.4f} Vp^{d['fitted_exponent_b']:.4f}: "
              f"held-out RMSE {h['rmse_gcc']:.4f} g/cm3 (bias {h['bias_gcc']:+.4f}); "
              f"supplied Gardner on the same samples {r['rmse_gcc']:.4f} ({r['bias_gcc']:+.4f})")

    print(f"\n{len(res.tables)} tables -> {out / 'tables'}")
    print(f"{len(paths)} figures -> {out / 'figures'}")
    print(f"depth-block outputs and figure -> {block_fig.parent}")
    print(f"done in {time.time() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
