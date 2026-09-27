"""Minimal test runner, so the suite runs without pytest installed.

    python tests/run_tests.py

The test files are ordinary pytest-style functions and also run unchanged
under `pytest tests/`.
"""

from __future__ import annotations

import glob
import importlib.util
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> int:
    passed, failed = 0, []
    for path in sorted(glob.glob(os.path.join(HERE, "test_*.py"))):
        name = os.path.splitext(os.path.basename(path))[0]
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for fn_name in sorted(dir(mod)):
            if not fn_name.startswith("test_"):
                continue
            fn = getattr(mod, fn_name)
            if not callable(fn):
                continue
            try:
                fn()
            except Exception:
                failed.append(f"{name}.{fn_name}")
                print(f"FAIL {name}.{fn_name}")
                traceback.print_exc()
            else:
                passed += 1
                print(f"ok   {name}.{fn_name}")
    print(f"\n{passed} passed, {len(failed)} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
