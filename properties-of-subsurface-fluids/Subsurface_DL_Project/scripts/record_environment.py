"""Record the environment the reported run actually executed in.

Writes results/metrics/environment.json and requirements-lock.txt.
Run it in the same interpreter that produced the results.
"""

from __future__ import annotations

import json
import os
import platform
import sys
from datetime import datetime, timezone

HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# The packages this project actually imports.
PACKAGES = ["numpy", "torch", "matplotlib", "nbformat", "nbconvert"]


def main():
    import importlib

    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = getattr(importlib.import_module(name), "__version__", "?")
        except Exception:
            versions[name] = None

    import torch

    env = {
        "recorded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "executable": sys.executable,
        "packages": versions,
        "torch_threads": torch.get_num_threads(),
        "cuda_available": bool(torch.cuda.is_available()),
        "device_used_for_reported_run": "cpu",
        "where_the_reported_run_executed": (
            "Anthropic cloud Linux container (2 vCPU, no GPU) attached to this "
            "session -- NOT the author's Windows machine.  The NumPy-only "
            "verification script was additionally re-run in the linked "
            "Windows machine's Linux workspace; the PyTorch training was not."
        ),
    }

    os.makedirs(os.path.join(HERE, "results", "metrics"), exist_ok=True)
    with open(os.path.join(HERE, "results", "metrics", "environment.json"), "w") as fh:
        json.dump(env, fh, indent=2)

    lines = [
        "# Exact versions used by the reported run.",
        "# Recorded by scripts/record_environment.py; see results/metrics/environment.json",
        f"# python=={env['python']} on {env['platform']}",
        "",
    ]
    for name in PACKAGES:
        if versions[name]:
            lines.append(f"{name}=={versions[name]}")
    with open(os.path.join(HERE, "requirements-lock.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")

    print(json.dumps(env, indent=2))


if __name__ == "__main__":
    main()
