"""Unit conversions and the one physical constant used by this project.

Every factor here is either an exact definition of a unit or a defined SI
constant.  Nothing in this module is an empirical or rock-physics value.

Source notes
------------
* ``FT_TO_M`` - exact definition of the international foot.  The supplied LAS
  file records depth in feet (``DEPT.F``) and sonic slowness in microseconds
  per foot (``DT.US/F``), so this factor is needed to read the file at all.
* ``STANDARD_GRAVITY`` - the defined SI value of standard gravity,
  9.80665 m/s^2.  ``Models/Elasticity_exercise.pdf`` Q5 writes the overburden
  relation as ``d(sigma_zz)/dz = rho * g`` and calls ``g`` "the gravitational
  acceleration" but gives no number, so the defined SI constant is used.  It is
  a universal constant, not a rock property or an empirical correlation.
"""

from __future__ import annotations

import numpy as np

#: Exact definition of the international foot (m per ft).
FT_TO_M: float = 0.3048

#: Defined SI standard gravity (m s^-2).  See module docstring.
STANDARD_GRAVITY: float = 9.80665

#: Grams per cubic centimetre to kilograms per cubic metre (exact).
GCC_TO_KGM3: float = 1000.0

#: Microseconds to seconds (exact).
US_TO_S: float = 1e-6


def feet_to_metres(depth_ft):
    """Convert a depth in feet to metres."""
    return np.asarray(depth_ft, dtype=float) * FT_TO_M


def slowness_us_per_ft_to_velocity_m_s(dt_us_per_ft):
    """Convert sonic slowness in us/ft to P-wave velocity in m/s.

    ``DT`` is a travel time per unit length.  One foot travelled in ``DT``
    microseconds is ``FT_TO_M`` metres in ``DT * 1e-6`` seconds, hence

        Vp [m/s] = FT_TO_M / (DT * 1e-6) = 1e6 * FT_TO_M / DT

    This is the definition of the logged unit, not a rock-physics model.
    """
    dt = np.asarray(dt_us_per_ft, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return FT_TO_M / (dt * US_TO_S)


def gcc_to_kg_m3(rho_gcc):
    """Convert bulk density from g/cm^3 (LAS ``RHOB.G/C3``) to kg/m^3."""
    return np.asarray(rho_gcc, dtype=float) * GCC_TO_KGM3


def kg_m3_to_gcc(rho_kg_m3):
    """Convert bulk density from kg/m^3 to g/cm^3."""
    return np.asarray(rho_kg_m3, dtype=float) / GCC_TO_KGM3
