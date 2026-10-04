"""Verification-gated screening: nothing is recommended unless the simulator
has verified it, on the success path and on every failure path."""
from __future__ import annotations

import numpy as np
import pytest

from subsurfaceml import screening as S
from subsurfaceml.config import load_config, project_root
from subsurfaceml.scenarios import sample_realisations


@pytest.fixture(scope="module")
def setup():
    cfg = load_config(project_root() / "config" / "study.yaml")
    r = sample_realisations(cfg)[3]
    return cfg, r


def fake_simulator(cfg, k_dp, k_r=50.0, fail=False):
    """dp = k_dp * mean rate, r95 = k_r * sqrt(mean rate): a linear stand-in
    for the simulator, so outcomes can be forced."""
    dp_lim = cfg.optim.p_limit_MPa - cfg.solver.p_init_MPa
    calls = []

    def sim(rates, label=""):
        rates = np.asarray(rates, float)
        calls.append(rates)
        if fail:
            return S.SimCheck(rates.tolist(), "failed", label=label)
        q = rates.mean()
        dp, r95 = k_dp * q, k_r * np.sqrt(q)
        mass = float(S.planned_mass_Mt(cfg, rates)[0])
        return S.SimCheck(rates.tolist(), "ok", dp, r95, mass,
                          bool(dp <= dp_lim and r95 <= cfg.optim.r_plume_limit_m), label)
    sim.calls = calls
    return sim


def predictor_from(k_dp, k_r=50.0, width=0.1):
    def pred(R):
        q = np.asarray(R, float).mean(axis=1)
        dp, r = k_dp * q, k_r * np.sqrt(q)
        return {"dp": dp, "dp_hi": dp * (1 + width), "r95": r, "r95_hi": r * (1 + width)}
    return pred


def _cands(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return np.exp(rng.uniform(np.log(1.0), np.log(60.0), (n, 4)))


def test_recommendation_cannot_hold_unverified_schedule():
    chk = S.SimCheck([1, 1, 1, 1], "ok", 12.0, 100.0, 0.1, feasible=False)
    with pytest.raises(ValueError):
        S.Recommendation(0, "x", S.VERIFIED, recommended_rates_kg_s=[1, 1, 1, 1],
                         verified=chk)
    with pytest.raises(ValueError):
        S.Recommendation(0, "x", S.VERIFIED)          # verified status, nothing verified
    with pytest.raises(ValueError):
        S.Recommendation(0, "x", S.NO_FEASIBLE, recommended_rates_kg_s=[1, 1, 1, 1],
                         verified=S.SimCheck([1, 1, 1, 1], "ok", 1.0, 1.0, 0.1, True))


def test_success_path_is_verified_and_within_budget(setup):
    cfg, r = setup
    sim = fake_simulator(cfg, k_dp=0.3)
    rec = S.recommend_surrogate(cfg, r, sim, 4, predictor_from(0.3), _cands())
    assert rec.status == S.VERIFIED and rec.verified.feasible
    assert rec.n_simulations <= 4 and len(sim.calls) == rec.n_simulations
    assert rec.verified.dp_MPa <= cfg.optim.p_limit_MPa - cfg.solver.p_init_MPa


def test_optimistic_surrogate_is_caught_and_repaired(setup):
    """The surrogate under-predicts pressure by 2x: its proposal violates when
    simulated; the repair scales it down and only a verified schedule is
    recommended."""
    cfg, r = setup
    sim = fake_simulator(cfg, k_dp=0.6)
    rec = S.recommend_surrogate(cfg, r, sim, 4, predictor_from(0.3), _cands())
    assert not rec.checks[0].feasible
    assert rec.status == S.VERIFIED and "repaired_by_simulator" in rec.flags
    assert rec.verified.feasible


def test_without_repair_an_optimistic_surrogate_gives_no_recommendation(setup):
    cfg, r = setup
    sim = fake_simulator(cfg, k_dp=0.6)
    rec = S.recommend_surrogate(cfg, r, sim, 3, predictor_from(0.3), _cands(),
                                repair=False, n_verify=3, fallback=None)
    assert rec.status == S.NO_FEASIBLE and rec.recommended_rates_kg_s is None
    assert np.isnan(rec.mass_Mt)


def test_no_candidate_predicted_feasible(setup):
    """A limit no candidate can meet: explicit status, no simulation, no
    recommendation."""
    cfg, r = setup
    sim = fake_simulator(cfg, k_dp=10.0)
    rec = S.recommend_surrogate(cfg, r, sim, 4, predictor_from(10.0), _cands(),
                                fallback=None)
    assert rec.status == S.NO_CANDIDATE and rec.n_simulations == 0
    assert rec.recommended_rates_kg_s is None


def test_wide_interval_is_flagged(setup):
    """Point predictions feasible, upper edges not: flagged as an uncertainty
    problem rather than silently screened."""
    cfg, r = setup
    sim = fake_simulator(cfg, k_dp=8.5)
    cands = np.full((5, 4), 1.0)
    rec = S.recommend_surrogate(cfg, r, sim, 4, predictor_from(8.5, width=0.5),
                                cands, fallback=None)
    assert rec.status == S.NO_CANDIDATE and "upper_bound_excludes_all" in rec.flags


def test_failed_simulations_never_recommended(setup):
    cfg, r = setup
    sim = fake_simulator(cfg, k_dp=0.3, fail=True)
    rec = S.recommend_surrogate(cfg, r, sim, 4, predictor_from(0.3), _cands())
    assert rec.status == S.NO_FEASIBLE and rec.recommended_rates_kg_s is None


def test_out_of_domain_does_not_use_surrogate(setup):
    cfg, r = setup
    sim = fake_simulator(cfg, k_dp=0.3)

    def exploding_predictor(R):
        raise AssertionError("the surrogate must not be used out of domain")
    rec = S.recommend_surrogate(cfg, r, sim, 4, exploding_predictor, _cands(),
                                in_domain=False, fallback="simulator_constant")
    assert "surrogate_out_of_domain" in rec.flags and "simulator_fallback" in rec.flags
    assert rec.status == S.VERIFIED and rec.verified.feasible


def test_proportional_search_converges_on_linear_response(setup):
    cfg, r = setup
    dp_lim = cfg.optim.p_limit_MPa - cfg.solver.p_init_MPa
    sim = fake_simulator(cfg, k_dp=0.25, k_r=1.0)
    best, checks = S.proportional_search(np.ones(4), 5.0, sim, 4, dp_lim, 1e9)
    assert best is not None and best.dp_MPa <= dp_lim
    assert best.dp_MPa >= 0.95 * dp_lim           # lands near the binding limit
