"""Compare recorded numerical results with a fresh run, value by value.

    python scripts/compare_results.py RECORDED NEW [--expect-count N]

RECORDED and NEW are two JSON files, two CSV files, or two folders (every
``*.json`` and ``*.csv`` directly inside RECORDED is compared with the file of
the same name in NEW).

* Numbers pass if ``|new - recorded| <= ATOL + RTOL * |recorded|``, with
  ``RTOL = 1e-6`` and ``ATOL = 1e-12``.
* Keys, list lengths, CSV headers and every non-numerical value (labels,
  statuses, phase names, True/False) must match exactly.  A key present on one
  side only is a failure, so a missing output cannot pass.
* Subtrees under keys that hold run-time measurements or run metadata are not
  compared: ``timing`` (wall-clock timings), ``environment`` (versions,
  platform, timestamps) and ``run`` (dates and commands).  Figures (PNG) are
  never compared.

Why these tolerances.  The reported metrics are computed in float64 from
float32 network outputs.  Re-running the six saved models reproduced all 185
recorded metrics bit for bit with the recorded two torch threads; with four
threads 40 of them changed by at most 1.8e-9 relative, because the float32
sums are accumulated in a different order.  ``RTOL = 1e-6`` leaves three orders
of magnitude for such reordering and for different CPU builds of the same
PyTorch release, and still catches any change in a reported digit.
``ATOL = 1e-12`` only matters for values that are zero or nearly zero: the
physics weight ``lambda = 0`` of the data-only models and seed index 0, which
must stay zero to round-off.  It is more than six orders of magnitude below the
smallest non-zero recorded metric (the training-mean baseline's test R^2,
4.4e-6), so it cannot hide a real change.

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
ATOL = 1.0e-12
SKIP_KEYS = {"timing", "environment", "run"}


class Report:
    def __init__(self):
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
        if diff > ATOL + RTOL * abs(old):
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
    args = ap.parse_args(argv)

    rep = Report()
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
          f"(rtol {RTOL:g}, atol {ATOL:g}; skipped keys {sorted(SKIP_KEYS)}); "
          f"largest relative difference {rep.max_rel:.2e}")
    for f in rep.failures[:40]:
        print("  DIFFERS", f)
    if len(rep.failures) > 40:
        print(f"  ... and {len(rep.failures) - 40} more")
    print("PASS" if not rep.failures else f"FAIL: {len(rep.failures)} differences")
    return 1 if rep.failures else 0


if __name__ == "__main__":
    sys.exit(main())
