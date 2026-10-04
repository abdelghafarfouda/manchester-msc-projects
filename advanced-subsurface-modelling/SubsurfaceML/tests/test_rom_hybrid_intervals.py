"""Analytical ROM, hybrid surrogate, interval constructions and the
applicability-domain check."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from subsurfaceml import rom
from subsurfaceml.config import load_config, project_root
from subsurfaceml.domain import DomainCheck
from subsurfaceml.hybrid import ROMOffsetRegressor
from subsurfaceml.intervals import IntervalModel, conformal_quantile
from subsurfaceml.scenarios import sample_realisations
from subsurfaceml.units import MPA, YEAR, md_to_m2


# --------------------------------------------------------------------- ROM
def test_rom_single_layer_matches_pss_exactly():
    """One layer, constant rate: backward Euler is exact for the linear PSS
    growth, so the ROM equals Q t/(c_t V) + Q/J to round-off."""
    k, h, phi, re, rw, rho, q, T = md_to_m2(100.0), 20.0, 0.2, 2000.0, 0.15, 1000.0, 10.0, 3 * YEAR
    dp = rom.bhp_buildup_layers([k], [h], [phi], re, rw, rho, [[q]], T,
                                co2_storage=False)[0]
    V = np.pi * (re ** 2 - rw ** 2) * h * phi
    J = 2 * np.pi * k * h / (rom.MU_A * (np.log(re / rw) - 0.75))
    Q = q / rho
    ana = Q * T / ((rom.C_R + rom.C_A) * V) + Q / J
    assert abs(dp - ana) / ana < 1e-10


def test_rom_time_step_converged():
    cfg = load_config(project_root() / "config" / "study.yaml")
    r = sample_realisations(cfg)[5]
    rates = np.array([30.0, 5.0, 40.0, 10.0])
    a = rom.bhp_buildup(cfg, r, rates, steps_per_period=100)
    b = rom.bhp_buildup(cfg, r, rates, steps_per_period=800)
    assert abs(a - b) / b < 5e-3


def test_rom_batch_equals_single_and_is_monotone_in_rate():
    cfg = load_config(project_root() / "config" / "study.yaml")
    r = sample_realisations(cfg)[7]
    R = np.array([[10.0, 10, 10, 10], [20.0, 20, 20, 20], [30.0, 5, 5, 30]])
    batch = rom.bhp_buildup(cfg, r, R)
    single = np.array([rom.bhp_buildup(cfg, r, q) for q in R])
    np.testing.assert_allclose(batch, single, rtol=1e-12)
    assert batch[1] > batch[0] > 0


def test_rom_injector_does_not_produce():
    """After a high-rate period followed by a low one, a layer whose pressure
    exceeds the BHP is closed, never back-flows: layer volumes never drop."""
    k = md_to_m2(np.array([800.0, 5.0]))
    _, v = rom.bhp_buildup_layers(k, [10.0, 10.0], [0.25, 0.10], 1500.0, 0.15,
                                  700.0, [[80.0, 0.5]], 0.5 * YEAR,
                                  return_layer_volumes=True)
    assert np.all(v >= 0)


# ------------------------------------------------------------------ hybrid
def test_hybrid_recovers_offset_times_correction():
    rng = np.random.default_rng(0)
    n = 300
    rom_mpa = np.exp(rng.uniform(0, 3, n))
    x = rng.uniform(-1, 1, n)
    y = rom_mpa * np.exp(0.2 * x)                 # exact multiplicative correction
    X = pd.DataFrame({"log10_rom_dp_MPa": np.log10(rom_mpa), "x": x})
    from sklearn.linear_model import LinearRegression
    m = ROMOffsetRegressor(LinearRegression()).fit(X, y)
    np.testing.assert_allclose(m.predict(X), y, rtol=1e-10)


def test_hybrid_needs_offset_column():
    from sklearn.linear_model import LinearRegression
    m = ROMOffsetRegressor(LinearRegression())
    with pytest.raises(ValueError):
        m.fit(pd.DataFrame({"x": [1.0, 2.0]}), np.array([1.0, 2.0]))


# --------------------------------------------------------------- intervals
def test_conformal_quantile_rank():
    s = np.arange(1, 20, dtype=float)          # 19 scores
    assert conformal_quantile(s, 0.10) == 18.0  # ceil(20 * 0.9) = 18th
    assert conformal_quantile(s[:5], 0.10) == np.inf


class _Sur:
    log_target = True

    def __init__(self, f):
        self.f = f

    def predict(self, X):
        return self.f(X)


def _grouped(n_groups, per, seed, shift=0.0):
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(n_groups), per)
    group_eff = rng.normal(0, 0.10, n_groups)[g]      # shared reservoir error
    case = rng.normal(0, 0.03, g.size)
    p = np.exp(rng.uniform(0, 2, g.size))
    y = p * np.exp(group_eff + case + shift)
    X = pd.DataFrame({"p": p})
    return X, y, g


def test_reservoir_conformal_covers_whole_reservoirs():
    """Exchangeable reservoirs: the per-reservoir max score covers *all*
    schedules of at least ~90 % of new reservoirs; pooled case-level
    calibration under-covers whole reservoirs when errors are shared."""
    sur = _Sur(lambda X: X["p"].to_numpy())
    Xc, yc, gc = _grouped(200, 4, 1)
    Xt, yt, gt = _grouped(2000, 4, 2)
    res = IntervalModel(sur, "reservoir_conformal").calibrate(Xc, yc, gc).evaluate(Xt, yt, gt)
    case = IntervalModel(sur, "case_conformal").calibrate(Xc, yc, gc).evaluate(Xt, yt, gt)
    assert res["reservoir_all_covered"] >= 0.87
    assert case["reservoir_all_covered"] < res["reservoir_all_covered"]


def test_shift_breaks_coverage():
    """Under a systematic shift no calibrated band keeps its coverage -- the
    guarantee is conditional on the calibration population."""
    sur = _Sur(lambda X: X["p"].to_numpy())
    Xc, yc, gc = _grouped(200, 4, 1)
    Xs, ys, gs = _grouped(500, 4, 3, shift=0.3)
    ev = IntervalModel(sur, "reservoir_conformal").calibrate(Xc, yc, gc).evaluate(Xs, ys, gs)
    assert ev["case_coverage"] < 0.5


# ------------------------------------------------------------------ domain
def test_domain_check_flags_out_of_range():
    rng = np.random.default_rng(0)
    cols = ["a", "b"]
    tr = pd.DataFrame(rng.uniform(0, 1, (100, 2)), columns=cols)
    tr["realisation_id"] = np.arange(100)
    dc = DomainCheck(descriptors=cols).fit(tr)
    te = pd.DataFrame({"a": [0.5, 1.5], "b": [0.5, 0.5]})
    out = dc.check(te)
    assert bool(out["in_domain"].iloc[0]) and not bool(out["in_domain"].iloc[1])
    assert "a above training range" in out["reasons"].iloc[1]
