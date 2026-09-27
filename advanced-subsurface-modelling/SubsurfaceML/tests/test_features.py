"""Defect (b): training and inference must see identical model inputs.

The training data were built with each realisation's own ``krg0`` in the
end-point mobility ratio, while the optimiser substituted ``krg0 = 0.4``.
Every route now goes through ``features.py``; these tests run the same
reservoir and schedule through each route and require bit-identical inputs.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from subsurfaceml.config import load_config, project_root
from subsurfaceml.features import (FEATURES, FeatureError, engineer,
                                   feature_matrix, features_for_schedules,
                                   raw_inputs)
from subsurfaceml.scenarios import (make_schedule, run_scenario,
                                    sample_realisations)


@pytest.fixture(scope="module")
def small():
    cfg = load_config(project_root() / "config" / "demo.yaml")
    cfg.grid.n_r = 12
    cfg.schedule.t_inject_years, cfg.schedule.t_total_years = 0.5, 1.0
    cfg.schedule.n_report = 5
    r = sample_realisations(cfg)[3]
    r = dataclasses.replace(r, krg0=0.61)       # deliberately far from 0.4
    rates = np.array([20.0, 35.0, 10.0, 25.0])
    return cfg, r, rates


def test_every_route_gives_identical_inputs(small, tmp_path):
    cfg, r, rates = small
    # 1. dataset route: simulate, keep the row, engineer
    out = run_scenario(cfg, r, make_schedule(cfg, r, rates), want_series=False)
    assert out["status"] == "ok"
    X_data = engineer(pd.DataFrame([out["row"]]))[FEATURES]
    # 2. saved-CSV route (what training reads back with --skip-dataset)
    f = tmp_path / "s.csv"
    pd.DataFrame([out["row"]]).to_csv(f, index=False)
    X_csv = engineer(pd.read_csv(f))[FEATURES]
    # 3. optimiser / interface route (batch of candidate schedules)
    X_opt = features_for_schedules(cfg, r, np.vstack([rates, rates * 2]))
    # 4. the Predictor used by the CLI and the Streamlit app
    from subsurfaceml.predict import Predictor
    pred = object.__new__(Predictor)
    pred.cfg = cfg
    X_app = pred.features(r, rates[None, :])
    a = X_data.to_numpy(float)[0]
    for other in (X_csv.to_numpy(float)[0], X_opt.to_numpy(float)[0],
                  X_app.to_numpy(float)[0]):
        np.testing.assert_allclose(other, a, rtol=1e-12, atol=0)
    assert list(X_opt.columns) == FEATURES


def test_mobility_ratio_uses_the_realisation_krg0(small):
    cfg, r, rates = small
    row = raw_inputs(cfg, r, rates)
    expected = (r.krg0 / (r.mu_g_cP * 1e-3)) / (1.0 / 6.0e-4)
    assert row["endpoint_mobility_ratio"] == pytest.approx(expected, rel=1e-12)
    old = (0.4 / (r.mu_g_cP * 1e-3)) / (1.0 / 6.0e-4)       # the defect
    assert abs(np.log10(old) - np.log10(expected)) > 0.15


def test_bad_inputs_fail_with_informative_errors(small):
    cfg, r, rates = small
    with pytest.raises(FeatureError, match="period rates"):
        raw_inputs(cfg, r, rates[:3])
    with pytest.raises(FeatureError, match="non-negative"):
        features_for_schedules(cfg, r, np.array([[1.0, -2.0, 3.0, 4.0]]))
    with pytest.raises(FeatureError, match="missing raw columns"):
        feature_matrix(pd.DataFrame({"k_median_mD": [100.0]}))


def test_surrogate_refuses_frames_without_its_features():
    from subsurfaceml.models import FittedSurrogate
    s = FittedSurrogate("x", "t", "", None, ["a", "b"], 0.0, {}, 0.0)
    with pytest.raises(ValueError, match="features.py"):
        s.predict(pd.DataFrame({"a": [1.0]}))
