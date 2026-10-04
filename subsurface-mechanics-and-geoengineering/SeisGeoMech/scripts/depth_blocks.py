#!/usr/bin/env python3
"""The two-direction depth-block test (configs/depth_blocks.json).

    python scripts/depth_blocks.py               # split, score, write, figure
    python scripts/depth_blocks.py --split-only  # the frozen split, no scoring
    python scripts/depth_blocks.py --out-dir DIR # write somewhere else (CI)

``run_all.py`` runs the full version after the original workflow.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seisgeomech import depth_blocks as db  # noqa: E402


def report(result) -> None:
    split = result["split"]
    lo, hi = split["excluded_interval_m"]
    print(f"midpoint {split['midpoint_m']:.6f} m; gap {split['exclusion_gap_m']:.0f} m, "
          f"excluded if within [{lo:.6f}, {hi:.6f}] m (inclusive)")
    for name in ("A", "B", "excluded"):
        b = split["blocks"][name]
        print(f"  {name:8s} n={b['n']:4d}  {b['depth_m_first']:.6f}-{b['depth_m_last']:.6f} m  "
              f"LAS rows {b['las_row_first']}-{b['las_row_last']}  "
              f"Vp {b['vp_min_m_s']:.0f}-{b['vp_max_m_s']:.0f} m/s")
    scores = result.get("scores")
    if not scores:
        return
    print("\nheld-out density error (g/cm3) on the evaluation block; "
          "rho g: 1-D, along the logged coordinate")
    for name, d in scores["directions"].items():
        h, s = d["held_out_block_fit"], d["held_out_supplied_gardner"]
        g = d["rho_g_gradient_over_evaluation_block"]
        print(f"  {name}: rho = {d['fitted_coefficient_a']:.4f} Vp^{d['fitted_exponent_b']:.4f}"
              f"  -> {d['reading']}")
        print(f"     block fit         bias {h['bias_gcc']:+.4f}  RMSE {h['rmse_gcc']:.4f}  "
              f"MAE {h['mae_gcc']:.4f}  rho g {g['block_fit_minus_measured_MPa_per_km']:+.3f} MPa/km")
        print(f"     supplied Gardner  bias {s['bias_gcc']:+.4f}  RMSE {s['rmse_gcc']:.4f}  "
              f"MAE {s['mae_gcc']:.4f}  rho g {g['supplied_minus_measured_MPa_per_km']:+.3f} MPa/km")
    print(f"  overall: {scores['overall_reading']}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split-only", action="store_true",
                    help="derive the split, check it against the frozen one, and write it without scoring")
    ap.add_argument("--out-dir", default=None, help="output folder (default results/depth_blocks)")
    ap.add_argument("--no-figure", action="store_true", help="skip the figure")
    args = ap.parse_args(argv)

    if args.split_only:
        split, _ = db.write_split(args.out_dir)
        report({"split": split})
        return 0

    result = db.run(args.out_dir)
    report(result)
    if not args.no_figure:
        from seisgeomech import figures
        figures.fig_depth_blocks(result, result["out_dir"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
