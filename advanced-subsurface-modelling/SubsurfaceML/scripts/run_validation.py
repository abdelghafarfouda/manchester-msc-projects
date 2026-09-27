#!/usr/bin/env python
"""Run only the numerical verification and validation suite."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from subsurfaceml.config import load_config
from subsurfaceml.validation import run_all

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--config", default=str(ROOT / "config" / "demo.yaml"))
a = ap.parse_args()
cfg = load_config(a.config)
res = run_all(cfg, save=True)
print(json.dumps(res["verdict"], indent=2))
raise SystemExit(0 if res["verdict"]["ALL_PASS"] else 2)
