"""Checks on the forward-modelling relations from Seismic/Ex1.ipynb."""

import numpy as np
import pytest

from seisgeomech import seismic as sx
from seisgeomech.units import FT_TO_M, US_TO_S


def test_gardner_coefficients_match_the_notebook():
    assert sx.GARDNER_COEFFICIENT == 0.31
    assert sx.GARDNER_EXPONENT == 0.25


def test_gardner_is_the_stated_expression():
    vp = np.array([2000.0, 3000.0, 4500.0])
    assert np.allclose(sx.gardner_density_gcc(vp), 0.31 * vp ** 0.25)


def test_gardner_is_monotonic_in_velocity():
    vp = np.linspace(1500.0, 6000.0, 100)
    assert np.all(np.diff(sx.gardner_density_gcc(vp)) > 0)


def test_impedance_is_the_product():
    assert sx.acoustic_impedance(2650.0, 4000.0) == pytest.approx(1.06e7)


def test_reflection_coefficient_matches_the_notebook_formula():
    rho = np.array([2.0, 2.5])
    v = np.array([3000.0, 4000.0])
    z = sx.acoustic_impedance(rho, v)
    expected = (rho[1] * v[1] - rho[0] * v[0]) / (rho[1] * v[1] + rho[0] * v[0])
    assert sx.reflection_coefficients(z)[0] == pytest.approx(expected)


def test_reflection_coefficient_is_zero_across_no_contrast():
    z = np.full(10, 1.0e7)
    assert np.allclose(sx.reflection_coefficients(z), 0.0)


def test_reflection_coefficient_reverses_sign_when_the_layers_swap():
    z = np.array([1.0e7, 1.4e7])
    forward = sx.reflection_coefficients(z)[0]
    reverse = sx.reflection_coefficients(z[::-1])[0]
    assert forward == pytest.approx(-reverse)


def test_reflection_coefficient_is_bounded():
    z = np.array([1.0e5, 1.0e9])
    r = sx.reflection_coefficients(z)
    assert np.all(np.abs(r) < 1.0)


def test_reflection_coefficient_needs_two_samples():
    with pytest.raises(ValueError):
        sx.reflection_coefficients(np.array([1.0e7]))


def test_gardner_impedance_depends_on_velocity_alone():
    """With Gardner, Z = 0.31 Vp^1.25, so R has a closed form in Vp."""
    vp = np.array([3000.0, 3500.0, 4200.0, 3900.0])
    rho = sx.gardner_density_gcc(vp)
    r = sx.reflection_coefficients(sx.acoustic_impedance(rho, vp))
    closed = (vp[1:] ** 1.25 - vp[:-1] ** 1.25) / (vp[1:] ** 1.25 + vp[:-1] ** 1.25)
    assert np.allclose(r, closed, atol=1e-15)


def test_ricker_is_zero_phase_with_unit_peak():
    w, t = sx.ricker_wavelet(duration=0.07, dt=0.002, frequency=100.0)
    assert w.size % 2 == 1
    assert np.argmax(w) == w.size // 2
    assert w.max() == pytest.approx(1.0, rel=1e-9)
    assert t[w.size // 2] == pytest.approx(0.0, abs=1e-12)


def test_ricker_is_symmetric():
    w, _ = sx.ricker_wavelet(duration=0.08, dt=0.001, frequency=50.0)
    assert np.allclose(w, w[::-1])


def test_ricker_has_zero_mean_to_a_good_approximation():
    """A Ricker wavelet has no DC component."""
    w, _ = sx.ricker_wavelet(duration=0.5, dt=0.0005, frequency=40.0)
    assert abs(w.sum()) < 1e-6 * np.abs(w).sum()


def test_convolution_with_a_spike_is_the_identity():
    r = np.array([0.0, 0.1, -0.2, 0.05, 0.0, -0.3, 0.12, 0.0, -0.04,
                  0.2, 0.0, 0.07, -0.11, 0.0, 0.03, -0.25, 0.0, 0.09,
                  0.0, -0.02, 0.15])
    spike = np.zeros(9)
    spike[4] = 1.0
    assert np.allclose(sx.convolve_trace(r, spike), r)


def test_convolution_with_a_scaled_spike_scales_the_trace():
    r = np.random.default_rng(0).normal(size=51)
    spike = np.zeros(11)
    spike[5] = 3.0
    assert np.allclose(sx.convolve_trace(r, spike), 3.0 * r)


def test_two_way_time_matches_constant_slowness():
    """Constant DT must give t = 2 * slowness * thickness exactly."""
    z = np.linspace(1000.0, 1100.0, 501)
    dt = np.full_like(z, 80.0)  # us/ft
    twt = sx.two_way_travel_time_along_log(z, dt)
    slowness_s_per_m = 80.0 * US_TO_S / FT_TO_M
    assert twt[-1] == pytest.approx(2.0 * slowness_s_per_m * 100.0, rel=1e-10)


def test_two_way_time_is_consistent_with_velocity():
    """t = 2 * thickness / Vp for a constant-velocity interval."""
    from seisgeomech.units import slowness_us_per_ft_to_velocity_m_s

    z = np.linspace(0.0, 200.0, 1001)
    dt = np.full_like(z, 65.0)
    vp = slowness_us_per_ft_to_velocity_m_s(65.0)
    twt = sx.two_way_travel_time_along_log(z, dt)
    assert twt[-1] == pytest.approx(2.0 * 200.0 / vp, rel=1e-10)


def test_two_way_time_starts_at_zero_and_increases():
    z = np.linspace(0.0, 50.0, 101)
    dt = np.linspace(60.0, 120.0, 101)
    twt = sx.two_way_travel_time_along_log(z, dt)
    assert twt[0] == 0.0
    assert np.all(np.diff(twt) > 0)


def test_similarity_increases_with_frequency():
    """Monotonic within this experiment; not an information-recovery claim."""
    rng = np.random.default_rng(1)
    r = rng.normal(size=4001) * 0.01
    sweep = sx.resolution_sweep(r, 1e-4, (25.0, 50.0, 100.0, 200.0))
    corrs = [row[1] for row in sweep]
    assert corrs == sorted(corrs)


def test_resolution_sweep_returns_four_fields_including_synthetic_rms():
    rng = np.random.default_rng(2)
    r = rng.normal(size=2001) * 0.01
    (f, corr, n, rms), = sx.resolution_sweep(r, 1e-4, (100.0,))
    assert f == 100.0 and n > 0
    assert -1.0 <= corr <= 1.0
    w, _ = sx.ricker_wavelet(2.0 / 100.0, 1e-4, 100.0)
    expected = np.sqrt((sx.convolve_trace(r, w) ** 2).mean())
    assert rms == pytest.approx(expected, rel=1e-12)


def test_resolution_docstring_rejects_the_information_recovered_reading():
    import inspect

    doc = " ".join(inspect.getdoc(sx.resolution_sweep).split())
    assert "not** a fraction of information, variability or structure recovered" in doc
    assert "neither is its square" in doc
    assert "not converted into a resolvable bed thickness" in doc


def test_two_way_time_docstring_denies_a_vertical_or_tie_claim():
    import inspect

    doc = " ".join(inspect.getdoc(sx.two_way_travel_time_along_log).split())
    assert "not vertical two-way time" in doc
    assert "seismic-to-well tie" in doc


def test_resolution_sweep_refuses_a_wavelet_longer_than_the_trace():
    r = np.zeros(100)
    with pytest.raises(ValueError, match="too short"):
        sx.resolution_sweep(r, 1e-4, (10.0,))


def test_min_supportable_frequency():
    assert sx.min_supportable_frequency(1000, 1e-4) == pytest.approx(20.0)


def test_resample_preserves_a_linear_series():
    t = np.array([0.0, 0.001, 0.0025, 0.004])
    y = 3.0 * t + 1.0
    t_reg, y_reg = sx.resample_to_regular_time(t, y, 1e-4)
    assert np.allclose(y_reg, 3.0 * t_reg + 1.0)
