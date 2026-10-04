"""Checks on the gravitational stress relations from Models Q5."""

import inspect

import numpy as np
import pytest

from seisgeomech import stress as st
from seisgeomech.units import STANDARD_GRAVITY


def test_uniform_density_reproduces_rho_g_z():
    """With constant density the integral must be exactly rho*g*s."""
    z = np.linspace(1000.0, 2000.0, 501)
    rho = np.full_like(z, 2500.0)
    sigma = st.rho_g_integral(z, rho)
    expected = 2500.0 * STANDARD_GRAVITY * (z - z[0])
    assert np.allclose(sigma, expected, rtol=1e-12, atol=1e-6)


def test_linear_density_matches_the_analytic_integral():
    """rho = a + b z integrates to g(a dz + b(z^2 - z0^2)/2) exactly."""
    a, b = 2000.0, 0.2
    z = np.linspace(500.0, 1500.0, 2001)
    rho = a + b * z
    sigma = st.rho_g_integral(z, rho)
    analytic = STANDARD_GRAVITY * (
        a * (z - z[0]) + 0.5 * b * (z ** 2 - z[0] ** 2)
    )
    assert np.allclose(sigma, analytic, rtol=1e-8)


def test_increment_starts_at_zero():
    z = np.linspace(0.0, 100.0, 11)
    sigma = st.rho_g_integral(z, np.full_like(z, 2000.0))
    assert sigma[0] == 0.0


def test_gradient_is_datum_independent():
    """Shifting the coordinate origin must not change the mean gradient."""
    z = np.linspace(1000.0, 1200.0, 201)
    rho = 2400.0 + 0.5 * np.sin(z / 10.0) * 100.0
    g1 = st.mean_rho_g_gradient(z, rho)
    g2 = st.mean_rho_g_gradient(z + 3000.0, rho)
    assert g1 == pytest.approx(g2, rel=1e-12)


def test_gradient_equals_mean_density_times_g_for_uniform_spacing():
    z = np.linspace(0.0, 1000.0, 1001)
    rho = np.linspace(2000.0, 2700.0, 1001)
    grad = st.mean_rho_g_gradient(z, rho)
    assert grad == pytest.approx(rho.mean() * STANDARD_GRAVITY, rel=1e-6)


def test_depth_must_increase():
    z = np.array([100.0, 50.0, 200.0])
    with pytest.raises(ValueError):
        st.rho_g_integral(z, np.full_like(z, 2000.0))


@pytest.mark.parametrize("nu,expected", [(0.0, 0.0), (0.25, 1.0 / 3.0), (0.5, 1.0)])
def test_uniaxial_strain_ratio_values(nu, expected):
    """nu = 0.5 gives a ratio of exactly 1, the hydrostatic case of Q5."""
    assert st.horizontal_over_vertical_ratio(nu) == pytest.approx(expected)


def test_hydrostatic_only_at_nu_one_half():
    """The Q5 solution states sigma_xx = sigma_zz requires nu = 1/2."""
    nus = np.linspace(0.01, 0.49, 49)
    ratios = st.horizontal_over_vertical_ratio(nus)
    assert np.all(ratios < 1.0)
    assert st.horizontal_over_vertical_ratio(0.5 - 1e-9) > 0.999


def test_vertical_strain_matches_the_worked_solution_form():
    """eps_zz = (sigma_zz/E)(1 - 2 nu^2/(1-nu))."""
    sigma, E, nu = 40e6, 5e10, 0.25
    expected = (sigma / E) * (1.0 - 2.0 * nu ** 2 / (1.0 - nu))
    assert st.vertical_strain(sigma, E, nu) == pytest.approx(expected, rel=1e-12)


def test_effective_stress_law_subtracts_alpha_times_pore_pressure():
    assert st.effective_stress(50e6, 20e6, alpha=1.0) == pytest.approx(30e6)
    assert st.effective_stress(50e6, 20e6, alpha=0.8) == pytest.approx(34e6)


def test_effective_stress_is_not_used_by_the_workflow():
    """Guard: no pore pressure is supplied, so the workflow must not call it."""
    import inspect

    from seisgeomech import analysis

    src = inspect.getsource(analysis)
    assert "effective_stress" not in src


def test_gradient_ratio_is_invariant_under_uniform_coordinate_scaling():
    """The precise robustness claim: invariance under UNIFORM scaling only.

    Two density profiles integrated over the SAME coordinate array keep the
    same gradient ratio when every interval is multiplied by one constant
    factor, because that factor cancels.  This is what the project claims.
    """
    rng = np.random.default_rng(3)
    s = np.linspace(3614.0, 3835.0, 1105)
    rho_a = 2600.0 + rng.normal(0, 60.0, s.size)
    rho_b = 0.965 * rho_a
    ratio_1 = st.mean_rho_g_gradient(s, rho_b) / st.mean_rho_g_gradient(s, rho_a)
    s_scaled = s[0] + 0.82 * (s - s[0])          # one constant factor
    ratio_2 = st.mean_rho_g_gradient(s_scaled, rho_b) / st.mean_rho_g_gradient(s_scaled, rho_a)
    assert ratio_1 == pytest.approx(ratio_2, rel=1e-12)
    assert ratio_1 == pytest.approx(0.965, rel=1e-12)


def test_uniform_scaling_invariance_does_not_extend_to_a_varying_correction():
    """The limit of the claim above, asserted rather than left implicit.

    A depth-dependent correction reweights the two integrals sample by sample.
    Because the second density profile here is not a constant multiple of the
    first, the gradient ratio then changes, so the project does not claim
    invariance under any such correction and applies none.
    """
    rng = np.random.default_rng(11)
    s = np.linspace(3614.0, 3835.0, 1105)
    rho_a = 2600.0 + rng.normal(0, 60.0, s.size)
    rho_b = rho_a - 90.0 - 0.15 * (s - s[0])     # NOT a constant multiple
    ratio_uniform = st.mean_rho_g_gradient(s, rho_b) / st.mean_rho_g_gradient(s, rho_a)

    # a correction factor that varies smoothly with depth
    ds = np.diff(s) * np.linspace(1.0, 0.6, s.size - 1)
    s_varying = np.concatenate(([s[0]], s[0] + np.cumsum(ds)))
    ratio_varying = (
        st.mean_rho_g_gradient(s_varying, rho_b) / st.mean_rho_g_gradient(s_varying, rho_a)
    )
    assert ratio_varying != pytest.approx(ratio_uniform, rel=1e-9)


def test_stress_module_documents_the_depth_coordinate_caveat():
    doc = " ".join(inspect.getdoc(st).split())
    assert "cannot be shown to be vertical" in doc
    assert "compressive stress is positive" in doc
    assert "coordinate increases downwards" in doc
    assert "no field stress profile is claimed" in doc


def test_functions_are_named_for_what_they_compute():
    """No function in this module is named 'overburden' or 'sigma_v'."""
    for name in dir(st):
        if name.startswith("_"):
            continue
        assert "overburden" not in name.lower()
        assert "sigma_v" not in name.lower()
