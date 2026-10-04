"""The supplied worked examples, used as the verification anchors.

These tests check the project's elasticity code against material whose inputs
and expected behaviour the course itself provides
(``Models/Elasticity_exercise.pdf`` Q3 and Q4).
"""

import numpy as np
import pytest

from seisgeomech import elasticity as el
from seisgeomech import worked_examples as wx


# --------------------------------------------------------------------------
# Q3 - olivine
# --------------------------------------------------------------------------

def test_olivine_tensor_is_transcribed_correctly():
    C = wx.OLIVINE_STIFFNESS_GPA
    assert C.shape == (6, 6)
    assert np.allclose(C, C.T)
    assert C[0, 0] == 320.5
    assert C[1, 1] == 196.5
    assert C[2, 2] == 233.5
    assert C[3, 3] == 64.0
    assert C[4, 4] == 77.0
    assert C[5, 5] == 78.5
    assert C[0, 1] == 68.15
    assert C[0, 2] == 71.6
    assert C[1, 2] == 76.8


def test_olivine_tensor_has_orthorhombic_symmetry():
    """Q3 asks the student to check this: nine independent non-zero terms,
    no coupling between normal and shear components."""
    C = wx.OLIVINE_STIFFNESS_GPA
    assert np.allclose(C[:3, 3:], 0.0)
    assert np.allclose(C[3:, :3], 0.0)
    assert np.allclose(C[3:, 3:], np.diag(np.diag(C[3:, 3:])))
    assert C[0, 0] != C[1, 1] != C[2, 2]


def test_olivine_bounds_are_ordered_and_hill_lies_between_them():
    r = wx.solve_olivine()
    assert r.K_reuss_GPa <= r.K_hill_GPa <= r.K_voigt_GPa
    assert r.G_reuss_GPa <= r.G_hill_GPa <= r.G_voigt_GPa


def test_olivine_velocities_are_consistent_with_the_moduli():
    r = wx.solve_olivine()
    K = r.K_hill_GPa * 1e9
    G = r.G_hill_GPa * 1e9
    assert r.vp_m_s == pytest.approx(
        np.sqrt((K + 4 * G / 3) / r.rho_kg_m3), rel=1e-12
    )
    assert r.vs_m_s == pytest.approx(np.sqrt(G / r.rho_kg_m3), rel=1e-12)


def test_olivine_velocities_are_bounded_by_the_voigt_and_reuss_averages():
    """The Hill velocities must lie between the velocities of the two bounds.

    Q3 also asks 'does this fit with the velocity at the Moho?'.  The supplied
    material does not quote a Moho velocity anywhere, so no numerical Moho
    comparison is asserted here or claimed in the project; see SOURCE_MAP.md
    s.7.  What is checked is that the calculation is internally consistent
    with its own Voigt and Reuss bounds.
    """
    r = wx.solve_olivine()
    vp_v = el.vp_from_K_G_rho(r.K_voigt_GPa * 1e9, r.G_voigt_GPa * 1e9, r.rho_kg_m3)
    vp_r = el.vp_from_K_G_rho(r.K_reuss_GPa * 1e9, r.G_reuss_GPa * 1e9, r.rho_kg_m3)
    vs_v = el.vs_from_G_rho(r.G_voigt_GPa * 1e9, r.rho_kg_m3)
    vs_r = el.vs_from_G_rho(r.G_reuss_GPa * 1e9, r.rho_kg_m3)
    assert vp_r < r.vp_m_s < vp_v
    assert vs_r < r.vs_m_s < vs_v


def test_olivine_reuss_average_uses_the_inverted_tensor():
    """K_Reuss must be reproduced from the compliance matrix directly."""
    S = np.linalg.inv(wx.OLIVINE_STIFFNESS_GPA)
    a = (S[0, 0] + S[1, 1] + S[2, 2]) / 3.0
    b = (S[0, 1] + S[1, 2] + S[0, 2]) / 3.0
    r = wx.solve_olivine()
    assert r.K_reuss_GPa == pytest.approx(1.0 / (3 * a + 6 * b), rel=1e-12)


# --------------------------------------------------------------------------
# Q4 - Merivale granite
# --------------------------------------------------------------------------

def test_merivale_table_is_transcribed_with_the_right_length():
    assert len(wx.MERIVALE_STRESS_MPA) == 16
    assert len(wx.MERIVALE_AXIAL_STRAIN) == 16
    assert len(wx.MERIVALE_CIRCUMFERENTIAL_STRAIN) == 16
    assert len(wx.MERIVALE_VOLUMETRIC_STRAIN) == 16


def test_merivale_volumetric_strain_is_internally_consistent():
    """The supplied fourth column must equal axial + 2 x circumferential."""
    residual = wx.MERIVALE_VOLUMETRIC_STRAIN - (
        wx.MERIVALE_AXIAL_STRAIN + 2.0 * wx.MERIVALE_CIRCUMFERENTIAL_STRAIN
    )
    assert np.abs(residual).max() < 1e-6


def test_merivale_density_from_quoted_mass_and_dimensions():
    """18.73 g in a 15 mm x 40 mm cylinder."""
    volume_cm3 = np.pi * 0.75 ** 2 * 4.0
    expected = 18.73 / volume_cm3 * 1000.0
    assert wx.merivale_density() == pytest.approx(expected, rel=1e-10)
    assert 2600.0 < wx.merivale_density() < 2700.0


def test_merivale_fit_window_is_the_stage_two_interval():
    """Above the first loading step, below half the peak stress."""
    r = wx.solve_merivale()
    lo, hi = r.stress_window_MPa
    assert hi == pytest.approx(0.5 * r.peak_stress_MPa)
    assert lo == pytest.approx(wx.MERIVALE_STRESS_MPA[1])
    assert r.n_points >= 4


def test_merivale_fit_is_linear():
    r = wx.solve_merivale()
    assert r.E_r2 > 0.99
    assert r.nu_r2 > 0.95


def test_merivale_poissons_ratio_is_physically_admissible():
    r = wx.solve_merivale()
    assert 0.0 < r.nu < 0.5


def test_merivale_moduli_are_self_consistent():
    r = wx.solve_merivale()
    assert el.youngs_modulus_from_K_G(r.K_Pa, r.G_Pa) == pytest.approx(
        r.E_Pa, rel=1e-10
    )
    assert el.poissons_ratio_from_K_G(r.K_Pa, r.G_Pa) == pytest.approx(
        r.nu, rel=1e-10
    )


def test_merivale_poissons_ratio_agrees_with_the_volumetric_strain_column():
    """For a linear elastic solid eps_vol/eps_axial = 1 - 2 nu.

    Taking that ratio point by point is an independent route to nu that uses
    no regression at all.  It reads slightly lower than the fitted value
    because the point-wise ratio ignores the non-zero strain offset left by
    stage I crack closure, which the regression absorbs into its intercept
    (Models/2_Hard_rocks.pdf s.3).  Agreement to within 0.05 is the check.
    """
    r = wx.solve_merivale()
    lo, hi = r.stress_window_MPa
    m = (wx.MERIVALE_STRESS_MPA >= lo) & (wx.MERIVALE_STRESS_MPA <= hi)
    ratio = (wx.MERIVALE_VOLUMETRIC_STRAIN[m] / wx.MERIVALE_AXIAL_STRAIN[m]).mean()
    nu_from_volume = 0.5 * (1.0 - ratio)
    assert 0.0 < nu_from_volume < 0.5
    assert nu_from_volume == pytest.approx(r.nu, abs=0.05)


def test_merivale_vp_vs_ratio_follows_from_poissons_ratio():
    r = wx.solve_merivale()
    expected = np.sqrt((2.0 - 2.0 * r.nu) / (1.0 - 2.0 * r.nu))
    assert r.vp_vs == pytest.approx(expected, rel=1e-10)


def test_merivale_dilatancy_begins_near_half_the_peak_stress():
    """Models/2_Hard_rocks.pdf s.3: stage III starts at about half the
    ultimate strength, and the volumetric strain turns over there."""
    turn = int(np.argmax(wx.MERIVALE_VOLUMETRIC_STRAIN))
    stress_at_turn = wx.MERIVALE_STRESS_MPA[turn]
    peak = wx.MERIVALE_STRESS_MPA.max()
    assert 0.45 < stress_at_turn / peak < 0.75


def test_merivale_window_sensitivity_is_modest():
    """Widening the fit window must not change E by more than ~15 per cent."""
    base = wx.solve_merivale()
    wide = wx.solve_merivale(stress_max_MPa=0.6 * base.peak_stress_MPa)
    assert abs(wide.E_Pa - base.E_Pa) / base.E_Pa < 0.15


def test_merivale_rejects_a_window_with_too_few_points():
    with pytest.raises(ValueError):
        wx.solve_merivale(stress_min_MPa=200.0, stress_max_MPa=205.0)
