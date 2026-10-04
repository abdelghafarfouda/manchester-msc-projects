#!/usr/bin/env python3
"""Compare regenerated results with the recorded ones.

    python scripts/compare_results.py results/tables ci/tables
    python scripts/compare_results.py results/depth_blocks ci/depth_blocks

Every ``.csv`` and ``.json`` file in the recorded folder must exist in the new
one, with the same columns, rows and keys.  Integers, strings and booleans must
match exactly.  Floats must agree within

    |new - recorded| <= ATOL + RTOL * |recorded|,  RTOL = 1e-9, ATOL = 1e-12.

Why these values.  With the pinned lock file every table and value reproduces
bit for bit (``results/original_2026-09-27/``).  A fresh install of the newest
compatible NumPy and SciPy changes the last bits only: at most 8.7e-16
relative in the tables, and 1.6e-15 absolute on the in-sample refit bias, a
value near zero (relative 1.7e-12).  A verification residual recorded as 0.0
came out as 3.1e-16.  The tolerances sit several orders of magnitude above
those differences and far below the four significant figures reported.

Skipped: ``generated_utc`` (a timestamp).  Figures are not compared.
Exit status 1 on any difference.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import sys
from pathlib import Path

RTOL = 1e-9
ATOL = 1e-12
SKIP_KEYS = {"generated_utc"}


class Report:
    def __init__(self):
        self.failures = []
        self.values = 0
        self.max_abs = 0.0
        self.byte_identical = 0
        self.files = 0

    def number(self, where, a, b):
        self.values += 1
        if isinstance(a, bool) or isinstance(b, bool) or not isinstance(a, (int, float)) \
                or not isinstance(b, (int, float)):
            if a != b:
                self.failures.append(f"{where}: {a!r} != {b!r}")
            return
        if isinstance(a, int) and isinstance(b, int):
            if a != b:
                self.failures.append(f"{where}: {a} != {b}")
            return
        if math.isnan(a) or math.isnan(b):
            if not (math.isnan(a) and math.isnan(b)):
                self.failures.append(f"{where}: {a!r} != {b!r}")
            return
        diff = abs(a - b)
        self.max_abs = max(self.max_abs, diff)
        if diff > ATOL + RTOL * abs(a):
            self.failures.append(f"{where}: recorded {a!r}, new {b!r} (|diff| {diff:.3e})")


def _walk(rep, where, a, b):
    if isinstance(a, dict):
        if not isinstance(b, dict):
            rep.failures.append(f"{where}: expected an object")
            return
        ka = set(a) - SKIP_KEYS
        kb = set(b) - SKIP_KEYS
        if ka != kb:
            rep.failures.append(f"{where}: keys differ, missing {sorted(ka - kb)}, "
                                f"extra {sorted(kb - ka)}")
        for k in sorted(ka & kb):
            _walk(rep, f"{where}.{k}", a[k], b[k])
    elif isinstance(a, list):
        if not isinstance(b, list) or len(a) != len(b):
            rep.failures.append(f"{where}: list length differs")
            return
        for i, (x, y) in enumerate(zip(a, b)):
            _walk(rep, f"{where}[{i}]", x, y)
    else:
        rep.number(where, a, b)


def _cell(text):
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def compare_csv(rep, name, a_text, b_text):
    a = list(csv.reader(io.StringIO(a_text)))
    b = list(csv.reader(io.StringIO(b_text)))
    if not a or not b or a[0] != b[0]:
        rep.failures.append(f"{name}: header differs")
        return
    if len(a) != len(b):
        rep.failures.append(f"{name}: {len(a) - 1} rows recorded, {len(b) - 1} new")
        return
    for i, (ra, rb) in enumerate(zip(a[1:], b[1:]), start=1):
        if len(ra) != len(rb):
            rep.failures.append(f"{name} row {i}: column count differs")
            continue
        for col, x, y in zip(a[0], ra, rb):
            rep.number(f"{name}[{i}].{col}", _cell(x), _cell(y))


def compare_file(rep, recorded: Path, new: Path):
    rep.files += 1
    if not new.exists():
        rep.failures.append(f"{new}: missing")
        return
    if recorded.read_bytes() == new.read_bytes():
        rep.byte_identical += 1
    if recorded.suffix == ".json":
        _walk(rep, recorded.name, json.loads(recorded.read_text()), json.loads(new.read_text()))
    else:
        compare_csv(rep, recorded.name, recorded.read_text(), new.read_text())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Compare regenerated results with recorded ones.")
    ap.add_argument("recorded", help="recorded folder or file")
    ap.add_argument("new", help="regenerated folder or file")
    ap.add_argument("--expect-files", type=int, default=None,
                    help="fail unless exactly this many .csv/.json files are compared")
    args = ap.parse_args(argv)

    rep = Report()
    rec, new = Path(args.recorded), Path(args.new)
    if rec.is_dir():
        names = sorted(p.name for p in rec.iterdir() if p.suffix in (".csv", ".json"))
        if not names:
            rep.failures.append(f"{rec}: no .csv or .json files")
        for n in names:
            compare_file(rep, rec / n, new / n)
    else:
        compare_file(rep, rec, new)
    if args.expect_files is not None and rep.files != args.expect_files:
        rep.failures.append(f"compared {rep.files} files, expected {args.expect_files}")

    print(f"{rep.files} files, {rep.values} values compared "
          f"(rtol {RTOL:g}, atol {ATOL:g}); largest absolute difference {rep.max_abs:.3e}; "
          f"{rep.byte_identical} of {rep.files} files byte-identical")
    for f in rep.failures[:50]:
        print("  FAIL", f)
    if len(rep.failures) > 50:
        print(f"  ... and {len(rep.failures) - 50} more")
    print("FAIL" if rep.failures else "PASS")
    return 1 if rep.failures else 0


if __name__ == "__main__":
    sys.exit(main())
