#!/usr/bin/env python3
"""Run the whole workflow and write every table and figure.

    python scripts/run_all.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seisgeomech import analysis, figures  # noqa: E402


def main() -> int:
    t0 = time.time()
    print("SeisGeoMech - running workflow")
    res = analysis.run()
    paths = figures.make_all(res)

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

    print(f"\n{len(res.tables)} tables -> results/tables")
    print(f"{len(paths)} figures -> results/figures")
    print(f"done in {time.time() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
