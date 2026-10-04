"""The two worked examples used to verify the elasticity code.

Both are transcribed verbatim from ``Models/Elasticity_exercise.pdf``.  They
exist so that the velocity-modulus machinery used on the well log is checked
against material whose expected answer the course itself supplies.

Q3 - olivine
------------
The exercise gives the single-crystal stiffness tensor of olivine in GPa and a
density of 3355 kg/m^3, then asks for the Voigt, Reuss and Voigt-Reuss-Hill
averages and the resulting P- and S-wave velocities, with the check "does this
fit with the velocity at the Moho?".

Q4 - Merivale granite
---------------------
The exercise gives a 16-point uniaxial compression record (axial stress, axial
strain, circumferential strain, volumetric strain) for a cylinder 15 mm in
diameter, 40 mm long and 18.73 g in mass, and asks the student to "identify the
linear portion of the graph and calculate E, nu, K, G, Vp and Vs".

Selecting the linear portion is the one judgement in this module, and it is
made with a criterion taken from the supplied notes rather than by eye:
``Models/2_Hard_rocks.pdf`` s.3 divides a compression test into six stages, with
stage I crack closure at the lowest stresses, stage II linear elastic, and
stage III ("onset of crack propagation") beginning "at about half the ultimate
failure strength".  The fit therefore uses the points above the first loading
step and below half the peak stress of the supplied record.  The sensitivity of
the answer to that window is reported alongside it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import elasticity as el

# --------------------------------------------------------------------------
# Q3 data - Elasticity_exercise.pdf, question 3
# --------------------------------------------------------------------------

#: Olivine single-crystal stiffness tensor, contracted notation, GPa.
OLIVINE_STIFFNESS_GPA = np.array(
    [
        [320.50, 68.15, 71.60, 0.0, 0.0, 0.00],
        [68.15, 196.50, 76.80, 0.0, 0.0, 0.00],
        [71.60, 76.80, 233.50, 0.0, 0.0, 0.00],
        [0.00, 0.00, 0.00, 64.0, 0.0, 0.00],
        [0.00, 0.00, 0.00, 0.0, 77.0, 0.00],
        [0.00, 0.00, 0.00, 0.0, 0.0, 78.50],
    ]
)

#: Density of olivine quoted in question 3, kg/m^3.
OLIVINE_DENSITY = 3355.0

# --------------------------------------------------------------------------
# Q4 data - Elasticity_exercise.pdf, question 4
# --------------------------------------------------------------------------

#: Sample geometry and mass quoted in question 4 (SI).
MERIVALE_DIAMETER_M = 0.015
MERIVALE_LENGTH_M = 0.040
MERIVALE_MASS_KG = 18.73e-3

MERIVALE_STRESS_MPA = np.array([
    0.0, 12.20873069, 46.67643903, 65.85650161, 58.68428576, 106.254756,
    146.809304, 159.4029036, 193.3645584, 213.6859072, 221.1086798,
    226.288442, 230.325738, 233.5561661, 233.845639, 233.8302387,
])

MERIVALE_AXIAL_STRAIN = np.array([
    0.0, 0.000203849, 0.000588818, 0.00076036, 0.000699979, 0.001147919,
    0.001549084, 0.001690843, 0.002036272, 0.002284602, 0.002395734,
    0.002476638, 0.002548958, 0.002628878, 0.002673273, 0.002678578,
])

MERIVALE_CIRCUMFERENTIAL_STRAIN = np.array([
    0.0, -4.27e-05, -0.000113058, -0.000160131, -0.00013395, -0.000282147,
    -0.000457269, -0.000552772, -0.000805876, -0.001218712, -0.001449546,
    -0.001724719, -0.002047488, -0.002481291, -0.002839774, -0.002890281,
])

MERIVALE_VOLUMETRIC_STRAIN = np.array([
    0.0, 0.000118369, 0.0003627, 0.000440215, 0.000432003, 0.000583519,
    0.00063444, 0.000585285, 0.000424507, -0.000152893, -0.000503231,
    -0.000972895, -0.001546125, -0.002333714, -0.003006287, -0.003101995,
])


@dataclass(frozen=True)
class OlivineResult:
    K_voigt_GPa: float
    G_voigt_GPa: float
    K_reuss_GPa: float
    G_reuss_GPa: float
    K_hill_GPa: float
    G_hill_GPa: float
    rho_kg_m3: float
    vp_m_s: float
    vs_m_s: float

    @property
    def vp_vs(self) -> float:
        return self.vp_m_s / self.vs_m_s


@dataclass(frozen=True)
class MerivaleResult:
    n_points: int
    stress_window_MPa: tuple
    E_Pa: float
    E_r2: float
    nu: float
    nu_r2: float
    K_Pa: float
    G_Pa: float
    rho_kg_m3: float
    vp_m_s: float
    vs_m_s: float
    peak_stress_MPa: float

    @property
    def vp_vs(self) -> float:
        return self.vp_m_s / self.vs_m_s


def solve_olivine() -> OlivineResult:
    """Voigt/Reuss/Hill averages and velocities for the Q3 olivine tensor."""
    vrh = el.voigt_reuss_hill(OLIVINE_STIFFNESS_GPA)
    K = vrh.K_hill * 1e9
    G = vrh.G_hill * 1e9
    return OlivineResult(
        K_voigt_GPa=vrh.K_voigt,
        G_voigt_GPa=vrh.G_voigt,
        K_reuss_GPa=vrh.K_reuss,
        G_reuss_GPa=vrh.G_reuss,
        K_hill_GPa=vrh.K_hill,
        G_hill_GPa=vrh.G_hill,
        rho_kg_m3=OLIVINE_DENSITY,
        vp_m_s=float(el.vp_from_K_G_rho(K, G, OLIVINE_DENSITY)),
        vs_m_s=float(el.vs_from_G_rho(G, OLIVINE_DENSITY)),
    )


def merivale_density() -> float:
    """Bulk density of the Q4 cylinder from its quoted mass and dimensions."""
    volume = np.pi * (MERIVALE_DIAMETER_M / 2.0) ** 2 * MERIVALE_LENGTH_M
    return MERIVALE_MASS_KG / volume


def _linear_fit(x, y):
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return float(slope), float(intercept), r2


def solve_merivale(stress_min_MPa=None, stress_max_MPa=None) -> MerivaleResult:
    """Fit the linear-elastic stage of the Q4 record and derive the constants.

    The default window is the stage II interval described in the module
    docstring: above the first loading step, below half the peak stress.
    """
    peak = float(MERIVALE_STRESS_MPA.max())
    lo = stress_min_MPa if stress_min_MPa is not None else float(
        MERIVALE_STRESS_MPA[MERIVALE_STRESS_MPA > 0].min()
    )
    hi = stress_max_MPa if stress_max_MPa is not None else 0.5 * peak

    mask = (MERIVALE_STRESS_MPA >= lo) & (MERIVALE_STRESS_MPA <= hi)
    if mask.sum() < 3:
        raise ValueError("linear window contains too few points to fit")

    s = MERIVALE_STRESS_MPA[mask]
    ea = MERIVALE_AXIAL_STRAIN[mask]
    ec = MERIVALE_CIRCUMFERENTIAL_STRAIN[mask]

    slope_E, _, r2_E = _linear_fit(ea, s)
    E = slope_E * 1e6  # MPa per unit strain -> Pa

    slope_nu, _, r2_nu = _linear_fit(ea, ec)
    nu = -slope_nu

    K = float(el.bulk_modulus_from_E_nu(E, nu))
    G = float(el.shear_modulus_from_E_nu(E, nu))
    rho = merivale_density()

    return MerivaleResult(
        n_points=int(mask.sum()),
        stress_window_MPa=(lo, hi),
        E_Pa=E,
        E_r2=r2_E,
        nu=nu,
        nu_r2=r2_nu,
        K_Pa=K,
        G_Pa=G,
        rho_kg_m3=rho,
        vp_m_s=float(el.vp_from_K_G_rho(K, G, rho)),
        vs_m_s=float(el.vs_from_G_rho(G, rho)),
        peak_stress_MPa=peak,
    )
