"""Upscaling and heterogeneity tests (2-Upscaling.pdf, 5-Uncertainty.pdf)."""
from __future__ import annotations

import numpy as np
import pytest

from subsurfaceml import upscaling as ups
from subsurfaceml.petrophysics import (coefficient_of_variation,
                                       dykstra_parsons, make_layered_rock)
from subsurfaceml.units import md_to_m2


def test_parallel_layers_give_exactly_the_arithmetic_mean():
    k = np.array([[1.0, 1.0], [5.0, 5.0], [20.0, 20.0]])   # layers along flow
    got = ups.upscale_flow_based(k.T, 1.0, 1.0)["k_eff"]
    assert got == pytest.approx(ups.k_arithmetic(k), rel=1e-12)


def test_series_layers_give_exactly_the_harmonic_mean():
    k = np.array([[1.0, 5.0, 20.0], [1.0, 5.0, 20.0]])     # layers across flow
    got = ups.upscale_flow_based(k.T, 1.0, 1.0)["k_eff"]
    assert got == pytest.approx(ups.k_harmonic(k), rel=1e-12)


def test_flow_based_upscaling_conserves_flux():
    rng = np.random.default_rng(0)
    k = np.exp(rng.normal(0.0, 1.0, (12, 12)))
    r = ups.upscale_flow_based(k, 1.0, 1.0)
    assert r["flux_imbalance"] < 1e-9


@pytest.mark.parametrize("sigma", [0.5, 1.0, 1.75])
def test_cardwell_parsons_bounds_hold(sigma):
    rng = np.random.default_rng(int(sigma * 100))
    k = np.exp(rng.normal(0.0, sigma, (24, 24)))
    kh, kha, kah, ka = ups.cardwell_parsons_bounds(k, axis=0)
    kstar = ups.upscale_flow_based(k, 1.0, 1.0)["k_eff"]
    assert kh <= kha + 1e-12 <= kstar + 1e-12 <= kah + 1e-12 <= ka + 1e-12


def test_lecture_2x2_layout_respects_the_bounding_chain():
    """2-Upscaling.pdf p.15 four-block layout (illustrative values; the
    lecture gives none): K_H <= K_HA <= K* <= K_AH <= K_A (p.14)."""
    r = ups.lecture_2x2_case()
    assert r["K_H"] <= r["K_HA"] <= r["k_eff_D"] <= r["K_AH"] <= r["K_A"]
    assert r["flux_imbalance"] < 1e-10


def test_harmonic_face_permeability_matches_the_lecture_formula():
    """1-Transmissibility.pdf p.19: k_bar = (dx_i + dx_i+1) /
    (dx_i/k_i + dx_i+1/k_i+1), i.e. 2 k1 k2 / (k1 + k2) for equal cells."""
    from subsurfaceml.grid import CartesianGrid1D
    g = CartesianGrid1D(n=2, L=1.0, area=1.0)
    k = np.array([1.0, 1.25])
    assert g.harmonic_k(k)[0] == pytest.approx(2 * 1.0 * 1.25 / 2.25, rel=1e-12)


def test_block_upscaling_shape_and_bounds():
    rng = np.random.default_rng(3)
    kf = np.exp(rng.normal(0.0, 1.2, (24, 24)))
    kc = ups.upscale_block_grid(kf, (4, 4), 1.0, 1.0)
    assert kc.shape == (4, 4)
    assert kc.min() >= ups.k_harmonic(kf) * 0.5
    assert kc.max() <= ups.k_arithmetic(kf) * 2.0


# ------------------------------------------------------- heterogeneity
def test_dykstra_parsons_recovers_the_target():
    """V_DP = 1 - exp(-sigma_lnk); generating with that sigma must give it
    back (within sampling error for a finite number of layers)."""
    for target in (0.2, 0.5, 0.8):
        vals = []
        for seed in range(40):
            rock = make_layered_rock(12, 5, 40.0, md_to_m2(100.0), target,
                                     0.2, seed=seed)
            vals.append(dykstra_parsons(rock.k_layer))
        assert np.mean(vals) == pytest.approx(target, abs=0.08)


def test_homogeneous_rock_has_zero_heterogeneity():
    rock = make_layered_rock(6, 10, 30.0, md_to_m2(100.0), 0.0, 0.2, seed=1)
    assert dykstra_parsons(rock.k_layer) == pytest.approx(0.0, abs=1e-9)
    assert coefficient_of_variation(rock.k_layer) == pytest.approx(0.0, abs=1e-9)
