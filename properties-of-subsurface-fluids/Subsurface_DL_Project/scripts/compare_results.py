"""Compare recorded numerical results with a fresh run, value by value.

    python scripts/compare_results.py RECORDED NEW [--expect-count N] [--profile metrics|derived]

RECORDED and NEW are two JSON files, two CSV files, or two folders (every
``*.json`` and ``*.csv`` directly inside RECORDED is compared with the file of
the same name in NEW).

* Numbers pass if ``|new - recorded| <= atol + rtol * |recorded|``.  Profile
  ``metrics`` (the default), ``rtol = 1e-6`` and ``atol = 1e-6``, is used for the
  185 original metrics and every other directly computed output; profile
  ``derived`` (``--profile derived``), ``rtol = 1e-4`` and ``atol = 2e-6``, for the
  error-analysis tables, whose numbers are differences and shares of nearly
  equal errors and are partly written with six decimals.
* Keys, list lengths, CSV headers and every non-numerical value (labels,
  statuses, phase names, True/False) must match exactly.  A key present on one
  side only is a failure, so a missing output cannot pass.
* Subtrees under keys that hold run-time measurements or run metadata are not
  compared: ``timing`` (wall-clock timings), ``environment`` (versions,
  platform, timestamps) and ``run`` (dates and commands).  Figures (PNG) are
  never compared.

Why these tolerances.  The networks compute in float32, so each prediction is
known to one unit in the last place (ULP), up to 6e-8 near F_V = 1, and a
different CPU or PyTorch build moves predictions by about that much.
``scripts/ulp_sensitivity.py`` moves every prediction of every saved model by
1 and by 3 ULPs and records how far each compared number moves
(``results/reproducibility/ulp_sensitivity.json``):

* the 185 original metrics move by at most 1.7e-7 (1 ULP) and 4.6e-7 (3 ULPs)
  in absolute terms -- a median of the Rachford-Rice residual |h|, about
  6e-3 -- and by more than 1e-6 relative only where they are that small.
  ``rtol = 1e-6`` with ``atol = 1e-6`` covers even the 3-ULP shift with a
  factor of two.  The absolute part is still smaller than the smallest
  non-zero recorded metric (4.4e-6, the baseline's test R^2, which is pure
  NumPy, does not depend on the network, and is reproduced exactly);
* the error-analysis outputs -- percentage changes and shares formed from
  differences of nearly equal errors -- amplify the same rounding (up to about
  1e-4 relative, 2.5e-5 absolute in percentage points), and those written to
  CSV with six decimals can change in their last printed digit (1e-6).
  Profile ``derived`` covers this and is still far below any digit quoted in
  the documentation.

The sensitivity script also checks that a 3-ULP perturbation of every
prediction stays inside the tolerance of each file's profile.

In the first CI run on GitHub's CPU-only PyTorch build, 182 of the 185
metrics agreed within 1e-6 relative; the other three, all medians of |h|,
differed by up to 5.5e-8 absolute, inside the 1-ULP range measured here.

``--expect-count N`` also requires exactly N numerical values to have been
compared (185 for ``results/metrics/evaluation.json``).
Exits with status 1 on any difference.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys

RTOL = 1.0e-6
ATOL = 1.0e-6
#: (rtol, atol) per kind of output; see the module docstring and
#: results/reproducibility/ulp_sensitivity.json for the measurements behind them.
PROFILES = {"metrics": (RTOL, ATOL), "derived": (1.0e-4, 2.0e-6)}
#: which profile each compared output uses (paths relative to the project folder)
FILE_PROFILES = {
    "results/metrics/evaluation.json": "metrics",
    "results/metrics/verify_flash.json": "metrics",
    "configs/prediction_domain.json": "metrics",
    "results/guarded/demo_cases.json": "metrics",
    "results/capacity/capacity_results.json": "metrics",
    "results/capacity/capacity_table.csv": "metrics",
    "results/analysis": "derived",
}
SKIP_KEYS = {"timing", "environment", "run"}


class Report:
    def __init__(self, rtol=RTOL, atol=ATOL):
        self.rtol, self.atol = rtol, atol
        self.numbers = 0
        self.texts = 0
        self.max_rel = 0.0
        self.failures = []

    def number(self, where, new, old):
        self.numbers += 1
        if math.isnan(old) or math.isnan(new):
            if not (math.isnan(old) and math.isnan(new)):
                self.failures.append(f"{where}: {new!r} vs recorded {old!r}")
            return
        if math.isinf(old) or math.isinf(new):
            if new != old:
                self.failures.append(f"{where}: {new!r} vs recorded {old!r}")
            return
        diff = abs(new - old)
        if old != 0.0:
            self.max_rel = max(self.max_rel, diff / abs(old))
        if diff > self.atol + self.rtol * abs(old):
            self.failures.append(f"{where}: {new!r} vs recorded {old!r} (difference {diff:.3e})")

    def text(self, where, new, old):
        self.texts += 1
        if new != old or type(new) is not type(old):
            self.failures.append(f"{where}: {new!r} vs recorded {old!r}")


def is_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def compare_json(where, new, old, rep):
    if isinstance(old, dict):
        if not isinstance(new, dict):
            rep.failures.append(f"{where}: expected an object")
            return
        keys_old = set(old) - SKIP_KEYS
        keys_new = set(new) - SKIP_KEYS
        if keys_old != keys_new:
            rep.failures.append(f"{where}: keys missing {sorted(keys_old - keys_new)}, "
                                f"unexpected {sorted(keys_new - keys_old)}")
        for k in sorted(keys_old & keys_new):
            compare_json(f"{where}.{k}", new[k], old[k], rep)
    elif isinstance(old, list):
        if not isinstance(new, list) or len(new) != len(old):
            rep.failures.append(f"{where}: list length {len(new) if isinstance(new, list) else '-'} "
                                f"vs recorded {len(old)}")
            return
        for i, (a, b) in enumerate(zip(new, old)):
            compare_json(f"{where}[{i}]", a, b, rep)
    elif is_number(old) and is_number(new):
        rep.number(where, float(new), float(old))
    else:
        rep.text(where, new, old)


def parse_cell(s):
    try:
        return float(s)
    except ValueError:
        return s


def compare_csv(where, new_path, old_path, rep):
    with open(new_path, newline="", encoding="utf-8") as fh:
        new = list(csv.reader(fh))
    with open(old_path, newline="", encoding="utf-8") as fh:
        old = list(csv.reader(fh))
    if not old or not new or new[0] != old[0]:
        rep.failures.append(f"{where}: header differs")
        return
    if len(new) != len(old):
        rep.failures.append(f"{where}: {len(new) - 1} rows vs recorded {len(old) - 1}")
        return
    header = old[0]
    for r, (a, b) in enumerate(zip(new[1:], old[1:]), start=1):
        if len(a) != len(b):
            rep.failures.append(f"{where} row {r}: column count differs")
            continue
        for col, x, y in zip(header, a, b):
            if col in SKIP_KEYS:
                continue
            xv, yv = parse_cell(x), parse_cell(y)
            if is_number(xv) and is_number(yv):
                rep.number(f"{where} row {r} {col}", xv, yv)
            else:
                rep.text(f"{where} row {r} {col}", x, y)


def compare_paths(new, old, rep):
    if old.endswith(".json"):
        with open(new, encoding="utf-8") as fh:
            a = json.load(fh)
        with open(old, encoding="utf-8") as fh:
            b = json.load(fh)
        compare_json(os.path.basename(old), a, b, rep)
    elif old.endswith(".csv"):
        compare_csv(os.path.basename(old), new, old, rep)
    else:
        rep.failures.append(f"{old}: only .json and .csv files are compared")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("recorded")
    ap.add_argument("new")
    ap.add_argument("--expect-count", type=int, default=None,
                    help="require exactly this many numerical values to be compared")
    ap.add_argument("--profile", choices=sorted(PROFILES), default="metrics",
                    help="tolerance profile: metrics (rtol 1e-6, atol 1e-6) or derived "
                         "(rtol 1e-4, atol 2e-6)")
    args = ap.parse_args(argv)

    rep = Report(*PROFILES[args.profile])
    if os.path.isdir(args.recorded):
        names = sorted(n for n in os.listdir(args.recorded)
                       if n.endswith((".json", ".csv")))
        if not names:
            rep.failures.append(f"{args.recorded}: no .json or .csv files")
        for n in names:
            new = os.path.join(args.new, n)
            if not os.path.exists(new):
                rep.failures.append(f"{n}: missing from {args.new}")
                continue
            compare_paths(new, os.path.join(args.recorded, n), rep)
        extra = sorted(n for n in os.listdir(args.new) if n.endswith((".json", ".csv"))
                       and n not in names) if os.path.isdir(args.new) else []
        for n in extra:
            rep.failures.append(f"{n}: not in the recorded results")
    else:
        compare_paths(args.new, args.recorded, rep)

    if args.expect_count is not None and rep.numbers != args.expect_count:
        rep.failures.append(f"compared {rep.numbers} numerical values, expected {args.expect_count}")
    print(f"{args.recorded}: compared {rep.numbers} numbers and {rep.texts} non-numerical values "
          f"(rtol {rep.rtol:g}, atol {rep.atol:g}; skipped keys {sorted(SKIP_KEYS)}); "
          f"largest relative difference {rep.max_rel:.2e}")
    for f in rep.failures[:40]:
        print("  DIFFERS", f)
    if len(rep.failures) > 40:
        print(f"  ... and {len(rep.failures) - 40} more")
    print("PASS" if not rep.failures else f"FAIL: {len(rep.failures)} differences")
    return 1 if rep.failures else 0


if __name__ == "__main__":
    sys.exit(main())
