"""Checks on the isotropic elasticity relations from Models/1_Elasticity.pdf."""

import numpy as np
import pytest

from seisgeomech import elasticity as el


@pytest.mark.parametrize("E,nu", [(5e10, 0.25), (1e11, 0.15), (2e10, 0.33)])
def test_conversion_table_roundtrip(E, nu):
    """(E, nu) -> (K, G) -> (E, nu) must return the input exactly."""
    K = el.bulk_modulus_from_E_nu(E, nu)
    G = el.shear_modulus_from_E_nu(E, nu)
    assert el.youngs_modulus_from_K_G(K, G) == pytest.approx(E, rel=1e-12)
    assert el.poissons_ratio_from_K_G(K, G) == pytest.approx(nu, rel=1e-12)


@pytest.mark.parametrize("E,nu", [(5e10, 0.25), (1e11, 0.15), (2e10, 0.33)])
def test_identity_E_equals_2G_one_plus_nu(E, nu):
    """E = 2G(1 + nu), stated on p.9 of the notes."""
    G = el.shear_modulus_from_E_nu(E, nu)
    assert 2.0 * G * (1.0 + nu) == pytest.approx(E, rel=1e-12)


@pytest.mark.parametrize("E,nu", [(5e10, 0.25), (1e11, 0.15), (2e10, 0.33)])
def test_identity_E_equals_3K_one_minus_2nu(E, nu):
    K = el.bulk_modulus_from_E_nu(E, nu)
    assert 3.0 * K * (1.0 - 2.0 * nu) == pytest.approx(E, rel=1e-12)


def test_velocities_invert_back_to_moduli():
    """Vp and Vs must return K and G through the same relations."""
    K, G, rho = 6.5e10, 3.9e10, 2650.0
    vp = el.vp_from_K_G_rho(K, G, rho)
    vs = el.vs_from_G_rho(G, rho)
    G_back = rho * vs ** 2
    K_back = rho * vp ** 2 - 4.0 * G_back / 3.0
    assert G_back == pytest.approx(G, rel=1e-12)
    assert K_back == pytest.approx(K, rel=1e-12)


def test_p_wave_modulus_from_log_equals_K_plus_four_thirds_G():
    K, G, rho = 6.5e10, 3.9e10, 2650.0
    vp = el.vp_from_K_G_rho(K, G, rho)
    assert el.p_wave_modulus_from_vp_rho(vp, rho) == pytest.approx(
        K + 4.0 * G / 3.0, rel=1e-12
    )


def test_vp_over_vs_depends_only_on_poissons_ratio():
    """A pure-elasticity consequence: Vp/Vs is a function of nu alone."""
    rho_a, rho_b = 2000.0, 3000.0
    nu, E = 0.25, 5e10
    K = el.bulk_modulus_from_E_nu(E, nu)
    G = el.shear_modulus_from_E_nu(E, nu)
    ratio_a = el.vp_from_K_G_rho(K, G, rho_a) / el.vs_from_G_rho(G, rho_a)
    ratio_b = el.vp_from_K_G_rho(K, G, rho_b) / el.vs_from_G_rho(G, rho_b)
    assert ratio_a == pytest.approx(ratio_b, rel=1e-12)
    # Vp/Vs = sqrt((2 - 2nu)/(1 - 2nu)); at nu = 0.25 this is sqrt(3).
    assert ratio_a == pytest.approx(np.sqrt(3.0), rel=1e-12)


def test_vrh_on_an_isotropic_tensor_returns_the_input_moduli():
    """For a genuinely isotropic C, Voigt and Reuss must coincide."""
    K, G = 7.0e1, 4.0e1  # GPa
    lam = K - 2.0 * G / 3.0
    C = np.array(
        [
            [lam + 2 * G, lam, lam, 0, 0, 0],
            [lam, lam + 2 * G, lam, 0, 0, 0],
            [lam, lam, lam + 2 * G, 0, 0, 0],
            [0, 0, 0, G, 0, 0],
            [0, 0, 0, 0, G, 0],
            [0, 0, 0, 0, 0, G],
        ]
    )
    r = el.voigt_reuss_hill(C)
    assert r.K_voigt == pytest.approx(K, rel=1e-10)
    assert r.K_reuss == pytest.approx(K, rel=1e-10)
    assert r.G_voigt == pytest.approx(G, rel=1e-10)
    assert r.G_reuss == pytest.approx(G, rel=1e-10)
    assert r.K_hill == pytest.approx(K, rel=1e-10)


def test_voigt_is_an_upper_bound_on_reuss():
    from seisgeomech.worked_examples import OLIVINE_STIFFNESS_GPA

    r = el.voigt_reuss_hill(OLIVINE_STIFFNESS_GPA)
    assert r.K_voigt >= r.K_reuss
    assert r.G_voigt >= r.G_reuss


def test_vrh_rejects_a_non_symmetric_tensor():
    C = np.eye(6)
    C[0, 1] = 1.0
    with pytest.raises(ValueError):
        el.voigt_reuss_hill(C)
