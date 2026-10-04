"""The development experiments read their inputs from one source.

The pipeline's stored feature table and features rebuilt from the
simulation table agree only to about 2e-13 relative (CSV parsing), which is
enough to tip a near-tied model choice.  The experiments therefore always
rebuild their inputs from ``scenarios.csv`` and record a hash of them.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from subsurfaceml.config import load_config, project_root
from subsurfaceml.experiments import input_table_sha256, load_dev_table
from subsurfaceml.features import FEATURES


def test_experiment_inputs_are_rebuilt_from_the_simulation_table(tmp_path):
    cfg = load_config(project_root() / "config" / "demo.yaml")
    src = Path(cfg.paths.data)
    shutil.copy(src / "scenarios.csv", tmp_path / "scenarios.csv")
    # a stored feature table that disagrees completely must be ignored
    bogus = pd.read_csv(src / "scenarios_features.csv")
    bogus[FEATURES] = 0.0
    bogus.to_csv(tmp_path / "scenarios_features.csv", index=False)
    cfg.paths.data = tmp_path
    a = load_dev_table(cfg)
    b = load_dev_table(cfg)
    assert np.abs(a[FEATURES].to_numpy(float)).sum() > 0
    cols = FEATURES + ["dp_bh_max_MPa", "realisation_id"]
    assert input_table_sha256(a, cols) == input_table_sha256(b, cols)
