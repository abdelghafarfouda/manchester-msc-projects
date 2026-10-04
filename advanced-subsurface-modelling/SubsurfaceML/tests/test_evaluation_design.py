"""Evaluation design and leakage audit.

* every surrogate input can be built with the simulator switched off (no
  input is derived from a simulator output);
* the independent test sets are disjoint from the development reservoirs and
  from each other (ids, seeds and descriptions);
* the split masks give each reservoir exactly one role, and the former test
  reservoirs of the published run (inspected during development) are not
  used as test data any more;
* the hybrid surrogate and the interval calibration see only the data they
  are given.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from subsurfaceml.config import load_config, project_root
from subsurfaceml.features import FEATURES, features_for_schedules
from subsurfaceml.final_eval import check_disjoint, eval_realisations
from subsurfaceml.scenarios import sample_realisations, sample_schedules


@pytest.fixture(scope="module")
def cfg():
    return load_config(project_root() / "config" / "study.yaml")


def test_no_input_needs_the_simulator(cfg, monkeypatch):
    from subsurfaceml import impes

    def boom(*a, **k):
        raise AssertionError("a surrogate input called the simulator")
    monkeypatch.setattr(impes.TwoPhaseModel, "run", boom)
    r = sample_realisations(cfg)[11]
    R = np.vstack([s["rates_kg_s"] for s in sample_schedules(cfg, r)])
    X = features_for_schedules(cfg, r, R)
    assert list(X.columns) == FEATURES and np.isfinite(X.to_numpy()).all()


def test_independent_sets_are_disjoint(cfg):
    dev = sample_realisations(cfg)
    c_t, test = eval_realisations(cfg, "final_test")
    c_s, shift = eval_realisations(cfg, "shift")
    ids = [set(r.realisation_id for r in x) for x in (dev, test, shift)]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])
    seeds = [set(r.seed for r in x) for x in (dev, test, shift)]
    assert not (seeds[0] & seeds[1]) and not (seeds[0] & seeds[2]) and not (seeds[1] & seeds[2])
    frame = lambda rs: pd.DataFrame([r.to_dict() for r in rs])
    d = check_disjoint(frame(dev), frame(test), frame(shift))
    assert d == {"overlapping_ids": 0, "overlapping_descriptions": 0}
    # the shift set differs from the prior in median permeability only
    k = np.array([r.k_median_mD for r in shift])
    assert k.min() >= 10.0 and k.max() <= 30.0
    assert min(r.k_median_mD for r in test) >= cfg.scenarios.k_median_mD[0]


def test_split_roles_and_former_test_reservoirs(cfg):
    from subsurfaceml.pipeline import make_split
    rng = np.random.default_rng(0)
    rows = []
    for rid in range(220):
        for s in range(4):
            rows.append({"realisation_id": rid, "set": "dev", "scenario_id": f"{rid}_{s}",
                         "k_median_mD": rng.uniform(30, 1000)})
    for rid in range(10000, 10020):
        rows.append({"realisation_id": rid, "set": "final_test", "scenario_id": f"t{rid}",
                     "k_median_mD": 100.0})
    for rid in range(20000, 20010):
        rows.append({"realisation_id": rid, "set": "shift", "scenario_id": f"s{rid}",
                     "k_median_mD": 20.0})
    df = pd.DataFrame(rows)
    sp, leak, m = make_split(cfg, df)
    assert all(v == 0 for v in leak["overlaps"].values())
    # every development row has exactly one development role
    dev = (df["set"] == "dev").to_numpy()
    assert np.all(m["train"][dev] ^ m["calib"][dev])
    # the published run's test reservoirs now train, never test
    assert np.all(m["train"][m["dev_former_test"]])
    assert not np.any(m["test"] & dev) and not np.any(m["shift"] & dev)
    # the calibration reservoirs are those of the published run (same seed)
    assert leak["reservoirs"]["calib"] == len(sp["calib_ids"]) == 41


def test_calibration_uses_only_calibration_rows():
    """Changing a test target must not change a calibrated interval."""
    from subsurfaceml.intervals import IntervalModel

    class Sur:
        log_target = True
        def predict(self, X):
            return X["p"].to_numpy()
    rng = np.random.default_rng(3)
    X = pd.DataFrame({"p": np.exp(rng.uniform(0, 2, 400))})
    y = X["p"].to_numpy() * np.exp(rng.normal(0, 0.05, 400))
    g = np.repeat(np.arange(100), 4)
    cal = g < 50
    a = IntervalModel(Sur(), "reservoir_conformal").calibrate(X[cal], y[cal], g[cal])
    y2 = y.copy(); y2[~cal] *= 10.0
    b = IntervalModel(Sur(), "reservoir_conformal").calibrate(X[cal], y2[cal], g[cal])
    assert a.q_ == b.q_
