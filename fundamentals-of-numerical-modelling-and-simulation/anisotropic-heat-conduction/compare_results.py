"""Compare a run's numerical outputs with the approved recorded results.

    python compare_results.py APPROVED_DIR NEW_DIR [--subset] [--report FILE]

Every CSV, JSON and NPZ file in APPROVED_DIR is compared with the file of the
same name in NEW_DIR, value by value, with numerical tolerances instead of a
byte comparison:

* a number passes if |new - approved| <= ATOL + RTOL * |approved|;
* NaN must stay NaN;
* text (headers, labels, check names, True/False) must match exactly.  Text
  that contains numbers, such as "max |dT| = 4.70e-05 C", is split into its
  words, which must match exactly, and its numbers, which use the tolerance.

RTOL = 1e-9.  ATOL = 1e-9 in the unit of each value (K, J/m, MJ/m, s, m or a
pure number).  The absolute part is there for values that are round-off by
construction - energy-balance closures (about 1e-12), the V4 and V5
differences (about 1e-12 K) - whose digits are not reproducible across
platforms and for which a relative tolerance means nothing.  1e-9 is several
hundred times their recorded size and no larger than the V5 pass tolerance;
the smallest recorded value with physical meaning, the V1 difference from the
course table (4.7e-5 C), is more than four orders of magnitude above it.
Each round-off quantity is also bounded by its own verification check.

Not compared: run_log.txt and environment.txt (timestamps, run time, software
versions), the "environment" block of summary.json for the same reason, and
the PNG figures (pixels change with the Matplotlib version and fonts).

--subset compares only the files and JSON keys present in APPROVED_DIR; it is
used to show that a later run still reproduces the original 2026-09-26
outputs it shares with them.  Without --subset the two sets of files and keys
must be identical.

Exits with status 1 if any value differs by more than the tolerance.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

RTOL = 1e-9
ATOL = 1e-9
SKIP_FILES = {"run_log.txt", "environment.txt"}
SKIP_JSON_KEYS = {"environment"}
NUMBER = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?|(?<![A-Za-z])[-+]?(?:nan|inf)\b",
                    re.IGNORECASE)


class Comparison:
    def __init__(self):
        self.n_numbers = 0
        self.n_text = 0
        self.max_abs = 0.0
        self.max_rel = 0.0
        self.failures: list[str] = []

    def number(self, where: str, new: float, old: float) -> None:
        self.n_numbers += 1
        if math.isnan(old) or math.isnan(new):
            if not (math.isnan(old) and math.isnan(new)):
                self.failures.append(f"{where}: {new!r} vs approved {old!r}")
            return
        if math.isinf(old) or math.isinf(new):
            if new != old:
                self.failures.append(f"{where}: {new!r} vs approved {old!r}")
            return
        diff = abs(new - old)
        self.max_abs = max(self.max_abs, diff)
        if old != 0.0:
            self.max_rel = max(self.max_rel, diff / abs(old))
        if diff > ATOL + RTOL * abs(old):
            self.failures.append(f"{where}: {new!r} vs approved {old!r} (difference {diff:.3e})")

    def text(self, where: str, new, old) -> None:
        """Compare a value that may be text with numbers inside it."""
        if isinstance(old, bool) or isinstance(new, bool):
            self.n_text += 1
            if new is not old:
                self.failures.append(f"{where}: {new!r} vs approved {old!r}")
            return
        if isinstance(old, (int, float)) and isinstance(new, (int, float)):
            self.number(where, float(new), float(old))
            return
        new_s, old_s = str(new), str(old)
        new_words, old_words = NUMBER.split(new_s), NUMBER.split(old_s)
        new_nums, old_nums = NUMBER.findall(new_s), NUMBER.findall(old_s)
        self.n_text += 1
        if new_words != old_words or len(new_nums) != len(old_nums):
            self.failures.append(f"{where}: text {new_s!r} vs approved {old_s!r}")
            return
        for k, (a, b) in enumerate(zip(new_nums, old_nums)):
            self.number(f"{where} [number {k + 1}]", float(a), float(b))


def compare_csv(name: str, new: Path, old: Path, cmp: Comparison) -> None:
    with open(new, newline="", encoding="utf-8") as f:
        new_rows = list(csv.reader(f))
    with open(old, newline="", encoding="utf-8") as f:
        old_rows = list(csv.reader(f))
    if len(new_rows) != len(old_rows):
        cmp.failures.append(f"{name}: {len(new_rows)} rows vs approved {len(old_rows)}")
        return
    for r, (a, b) in enumerate(zip(new_rows, old_rows), start=1):
        if len(a) != len(b):
            cmp.failures.append(f"{name} row {r}: {len(a)} columns vs approved {len(b)}")
            continue
        for c, (x, y) in enumerate(zip(a, b), start=1):
            cmp.text(f"{name} row {r} column {c}", x, y)


def compare_json(where: str, new, old, cmp: Comparison, subset: bool) -> None:
    if isinstance(old, dict):
        if not isinstance(new, dict):
            cmp.failures.append(f"{where}: not an object")
            return
        old_keys = set(old) - SKIP_JSON_KEYS
        new_keys = set(new) - SKIP_JSON_KEYS
        missing = old_keys - new_keys
        extra = set() if subset else new_keys - old_keys
        if missing or extra:
            cmp.failures.append(f"{where}: keys missing {sorted(missing)}, unexpected {sorted(extra)}")
        for k in sorted(old_keys & new_keys):
            compare_json(f"{where}.{k}", new[k], old[k], cmp, subset)
    elif isinstance(old, list):
        if not isinstance(new, list) or len(new) != len(old):
            cmp.failures.append(f"{where}: list length differs")
            return
        for k, (a, b) in enumerate(zip(new, old)):
            compare_json(f"{where}[{k}]", a, b, cmp, subset)
    elif old is None or new is None:
        cmp.n_text += 1
        if old is not new:
            cmp.failures.append(f"{where}: {new!r} vs approved {old!r}")
    else:
        cmp.text(where, new, old)


def compare_npz(name: str, new: Path, old: Path, cmp: Comparison) -> None:
    with np.load(new) as a, np.load(old) as b:
        if set(a.files) != set(b.files):
            cmp.failures.append(f"{name}: arrays {sorted(a.files)} vs approved {sorted(b.files)}")
            return
        for k in b.files:
            x, y = np.asarray(a[k], dtype=float), np.asarray(b[k], dtype=float)
            if x.shape != y.shape:
                cmp.failures.append(f"{name}[{k}]: shape {x.shape} vs approved {y.shape}")
                continue
            for idx in np.ndindex(y.shape):
                cmp.number(f"{name}[{k}]{list(idx)}", float(x[idx]), float(y[idx]))


def compared_files(folder: Path) -> set[str]:
    return {p.name for p in folder.iterdir()
            if p.is_file() and p.suffix in (".csv", ".json", ".npz") and p.name not in SKIP_FILES}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("approved", type=Path, help="folder with the approved results")
    parser.add_argument("new", type=Path, help="folder with the results to check")
    parser.add_argument("--subset", action="store_true",
                        help="compare only the files and JSON keys present in the approved folder")
    parser.add_argument("--report", type=Path, help="also write the outcome to this JSON file")
    args = parser.parse_args(argv)

    approved, new = compared_files(args.approved), compared_files(args.new)
    cmp = Comparison()
    missing = sorted(approved - new)
    extra = [] if args.subset else sorted(new - approved)
    for name in missing:
        cmp.failures.append(f"{name}: missing from {args.new}")
    for name in extra:
        cmp.failures.append(f"{name}: not in the approved results {args.approved}")
    for name in sorted(approved & new):
        a, b = args.new / name, args.approved / name
        if name.endswith(".csv"):
            compare_csv(name, a, b, cmp)
        elif name.endswith(".json"):
            with open(a, encoding="utf-8") as f:
                new_json = json.load(f)
            with open(b, encoding="utf-8") as f:
                old_json = json.load(f)
            compare_json(name, new_json, old_json, cmp, args.subset)
        else:
            compare_npz(name, a, b, cmp)

    files = sorted(approved & new)
    print(f"compared {len(files)} files: {cmp.n_numbers} numbers and {cmp.n_text} text fields "
          f"(rtol {RTOL:g}, atol {ATOL:g}{', subset' if args.subset else ''})")
    print(f"largest absolute difference {cmp.max_abs:.3e}; largest relative difference {cmp.max_rel:.3e}")
    for line in cmp.failures[:50]:
        print("  DIFFERS", line)
    if len(cmp.failures) > 50:
        print(f"  ... and {len(cmp.failures) - 50} more")
    print("PASS" if not cmp.failures else f"FAIL: {len(cmp.failures)} differences")
    if args.report:
        args.report.write_text(json.dumps(dict(
            approved=str(args.approved), new=str(args.new), subset=args.subset, rtol=RTOL, atol=ATOL,
            files=files, numbers_compared=cmp.n_numbers, text_fields_compared=cmp.n_text,
            max_abs_difference=cmp.max_abs, max_rel_difference=cmp.max_rel,
            failures=cmp.failures, passed=not cmp.failures), indent=2) + "\n", encoding="utf-8")
    return 1 if cmp.failures else 0


if __name__ == "__main__":
    sys.exit(main())
