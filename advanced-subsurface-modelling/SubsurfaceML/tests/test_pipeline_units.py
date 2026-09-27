"""Unit tests for configuration, features and uncertainty."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from subsurfaceml.config import load_config, project_root
from subsurfaceml.features import FEATURES, engineer as add_engineered_features
from subsurfaceml.fluids import FluidProperties, RelPerm, welge_shock
from subsurfaceml.scenarios import (constant_schedule, make_schedule,
                                    reference_rate, sample_realisations,
                                    sample_schedules)
from subsurfaceml.uncertainty import ErrorBand, error_source_table
from subsurfaceml.units import (DARCY, cp_to_pas, m2_to_md, md_to_m2,
                                s_to_years, years_to_s)


# ------------------------------------------------------------------ units
def test_unit_roundtrips():
    assert m2_to_md(md_to_m2(137.0)) == pytest.approx(137.0, rel=1e-12)
    assert s_to_years(years_to_s(3.5)) == pytest.approx(3.5, rel=1e-12)
    assert md_to_m2(1000.0) == pytest.approx(DARCY, rel=1e-12)
    assert cp_to_pas(1.0) == pytest.approx(1e-3, rel=1e-12)


# ----------------------------------------------------------------- config
@pytest.mark.parametrize("name", ["demo", "study"])
def test_shipped_configs_load(name):
    cfg = load_config(project_root() / "config" / f"{name}.yaml")
    assert cfg.grid.n_r > 0 and cfg.grid.n_layers > 0
    assert cfg.schedule.t_total_years >= cfg.schedule.t_inject_years
    assert cfg.optim.p_limit_MPa > cfg.solver.p_init_MPa
    assert cfg.paths.results.is_absolute()


def test_unknown_config_key_is_rejected(tmp_path):
    from subsurfaceml.config import ConfigError
    p = tmp_path / "bad.yaml"
    p.write_text("name: bad\ngrid:\n  not_a_real_option: 3\n")
    with pytest.raises(ConfigError):
        load_config(p)
    p.write_text("name: bad\nphysics:\n  pc_entry_kPa: 3\n")
    with pytest.raises(ConfigError, match="unknown configuration section"):
        load_config(p)


def test_inconsistent_config_values_are_rejected(tmp_path):
    from subsurfaceml.config import ConfigError
    p = tmp_path / "bad.yaml"
    p.write_text("name: bad\nscenarios:\n  V_DP: [0.9, 0.2]\n")
    with pytest.raises(ConfigError, match="V_DP"):
        load_config(p)
    p.write_text("name: bad\noptim:\n  p_limit_MPa: 10.0\n")
    with pytest.raises(ConfigError, match="p_limit"):
        load_config(p)
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "missing.yaml")


# --------------------------------------------------------------- relperm
def test_relperm_endpoints_and_monotonicity():
    rp, fl = RelPerm(), FluidProperties()
    S = np.linspace(0.0, 1.0 - rp.S_ar, 200)
    assert rp.kr_g(np.array([0.0]))[0] == pytest.approx(0.0, abs=1e-12)
    assert rp.kr_a(np.array([0.0]))[0] == pytest.approx(rp.kra0, rel=1e-12)
    assert np.all(np.diff(rp.kr_g(S)) >= -1e-12)
    assert np.all(np.diff(rp.kr_a(S)) <= 1e-12)
    f = rp.f_g(S, fl)
    assert np.all((f >= -1e-12) & (f <= 1 + 1e-12))
    assert np.all(np.diff(f) >= -1e-9)


def test_welge_shock_is_on_the_fractional_flow_curve():
    rp, fl = RelPerm(), FluidProperties()
    S_gf, f_gf, slope = welge_shock(rp, fl)
    assert 0.0 < S_gf < 1.0 - rp.S_ar
    assert f_gf == pytest.approx(float(rp.f_g(np.array([S_gf]), fl)[0]), rel=1e-6)
    # tangency: chord slope from the origin equals the local derivative
    assert slope == pytest.approx(
        float(rp.dfg_dSg(np.array([S_gf]), fl)[0]), rel=5e-2)


# -------------------------------------------------------------- schedules
def test_reference_rate_scales_with_flow_capacity_and_storage():
    """q_ref is the harmonic combination of a flow limit and a storage limit,
    so raising either one must raise it, and neither alone may run away."""
    import dataclasses
    cfg = load_config(project_root() / "config" / "demo.yaml")
    r = sample_realisations(cfg)[0]
    q0 = reference_rate(cfg, r)
    assert q0 > 0
    q_k = reference_rate(cfg, dataclasses.replace(
        r, k_median_mD=r.k_median_mD * 10))
    q_h = reference_rate(cfg, dataclasses.replace(
        r, h_total_m=r.h_total_m * 2))
    assert q_k > q0 and q_h > q0
    # harmonic combination: a 10x flow capacity cannot give a 10x rate,
    # because the storage limit still binds
    assert q_k < 10 * q0


def test_schedule_total_mass_matches_the_rate_integral():
    cfg = load_config(project_root() / "config" / "demo.yaml")
    r = sample_realisations(cfg)[0]
    s = sample_schedules(cfg, r)[0]
    from subsurfaceml.impes import InjectionSchedule
    sch = InjectionSchedule(s["t_edges_s"], np.append(s["rates_kg_s"], 0.0))
    dt = np.diff(s["t_edges_s"])[:-1]
    assert sch.total_mass == pytest.approx(float((s["rates_kg_s"] * dt).sum()),
                                           rel=1e-12)


def test_constant_schedule_delivers_the_requested_mass():
    cfg = load_config(project_root() / "config" / "demo.yaml")
    r = sample_realisations(cfg)[0]
    s = constant_schedule(cfg, r, 1.0e9)
    dt = np.diff(s["t_edges_s"])[:-1]
    assert float((s["rates_kg_s"] * dt).sum()) == pytest.approx(1.0e9, rel=1e-9)


def test_schedule_mismatch_is_rejected():
    from subsurfaceml.impes import InjectionSchedule
    with pytest.raises(ValueError):
        InjectionSchedule([0.0, 1.0, 2.0], [1.0])
    with pytest.raises(ValueError):
        InjectionSchedule([0.0, 1.0], [-5.0])


# ---------------------------------------------------------------- features
def _fake_scenarios(n=30, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "k_median_mD": rng.uniform(30, 1000, n),
        "k_geom_mean_m2": rng.uniform(1e-14, 1e-12, n),
        "V_DP": rng.uniform(0.05, 0.85, n),
        "phi_mean": rng.uniform(0.1, 0.28, n),
        "h_total_m": rng.uniform(20, 80, n),
        "r_e_m": rng.uniform(1200, 4000, n),
        "n_g": rng.uniform(1.5, 3, n), "n_a": rng.uniform(2, 4, n),
        "krg0": rng.uniform(0.2, 0.65, n), "S_ar": rng.uniform(0.15, 0.35, n),
        "mu_g_cP": rng.uniform(0.04, 0.08, n),
        "rho_g": rng.uniform(600, 780, n),
        "q1_kg_s": rng.uniform(1, 60, n), "q2_kg_s": rng.uniform(1, 60, n),
        "q3_kg_s": rng.uniform(1, 60, n), "q4_kg_s": rng.uniform(1, 60, n),
        "q_ref_kg_s": rng.uniform(5, 50, n),
        "planned_mass_kg": rng.uniform(1e8, 2e9, n),
        "endpoint_mobility_ratio": rng.uniform(2, 8, n),
        "V_DP_layers": rng.uniform(0.05, 0.85, n),
    })


def test_engineered_features_work_without_simulator_outputs():
    """The optimiser builds features for schedules that have NOT been
    simulated, so this must not require any output column."""
    out = add_engineered_features(_fake_scenarios())
    for f in FEATURES:
        assert f in out.columns, f
    assert out[FEATURES].notna().all().all()
    assert "dp_bh_max_MPa" not in out.columns


def test_engineered_features_add_targets_when_outputs_are_present():
    df = _fake_scenarios()
    df["dp_bh_max_Pa"] = 5e6
    df["mass_retained_kg"] = 1e9
    out = add_engineered_features(df)
    assert out["dp_bh_max_MPa"].iloc[0] == pytest.approx(5.0)
    assert out["mass_retained_Mt"].iloc[0] == pytest.approx(1.0)


def test_q_ref_is_not_mistaken_for_a_period_rate():
    """`q_ref_kg_s` also matches 'q*_kg_s'; if it leaked into the per-period
    rate matrix every schedule-shape feature would be wrong."""
    df = _fake_scenarios(10)
    out = add_engineered_features(df)
    expected = df[["q1_kg_s", "q2_kg_s", "q3_kg_s", "q4_kg_s"]].mean(axis=1)
    assert np.allclose(out["q_mean_kg_s"], expected)


# ------------------------------------------------------------ uncertainty
class _Dummy:
    def __init__(self, log_target=False):
        self.log_target = log_target

    def predict(self, X):
        return np.asarray(X, float).ravel()


def test_error_band_coverage_is_measured_not_assumed():
    rng = np.random.default_rng(0)
    x = rng.normal(size=2000)
    y = x + rng.normal(scale=0.3, size=2000)
    b = ErrorBand(_Dummy(), 5, 95).calibrate(x[:1000, None], y[:1000])
    ev = b.evaluate(x[1000:, None], y[1000:])
    assert 0.86 < ev["measured_coverage_test"] < 0.94
    assert ev["residual_space"] == "linear"


def test_error_band_is_relative_for_log_targets():
    rng = np.random.default_rng(1)
    x = np.exp(rng.normal(size=1000))
    y = x * np.exp(rng.normal(scale=0.1, size=1000))
    b = ErrorBand(_Dummy(log_target=True)).calibrate(x[:500, None], y[:500])
    lo, p, hi = b.predict_interval(np.array([[1.0], [10.0]]))
    assert (hi[1] - lo[1]) == pytest.approx(10 * (hi[0] - lo[0]), rel=1e-9)


def test_error_source_shares_sum_to_100():
    d = error_source_table("t", prior_std=2.0, surrogate_rmse=1.0,
                           numerical_rel_error=0.05, reference_value=10.0,
                           unmodelled=["gravity"])
    total = d["share_input_pct"] + d["share_surrogate_pct"] + d["share_numerical_pct"]
    assert total == pytest.approx(100.0, rel=1e-9)
