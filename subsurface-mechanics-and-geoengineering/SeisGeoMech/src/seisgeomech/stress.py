"""Integrating rho * g along the logged coordinate.

Both relations here come from ``Models/Elasticity_exercise.pdf`` Q5 and its
worked solution in ``Models/Elasticity Exercise Solutions.pdf`` (pages "1 of 2"
and "2 of 2").

Q5 states the overburden relation

    d(sigma_zz) / dz = rho * g

and asks the student to show that, when no lateral strain is permitted, the
gravitationally induced horizontal stress is

    sigma_xx = nu / (1 - nu) * sigma_zz

with the accompanying vertical strain

    eps_zz = (sigma_zz / E) * (1 - 2 nu^2 / (1 - nu))

and that the state is hydrostatic only when nu = 1/2.

Depth coordinate: what this module does and does not claim
---------------------------------------------------------
Q5's ``z`` is a **vertical** coordinate.  The supplied LAS file records a curve
labelled ``DEPT`` and nothing that establishes the convention behind it:

* there is no TVD curve and no deviation, inclination or azimuth curve;
* ``LMF`` ("LOGS MEAS FROM") is ``UNKNOWN``;
* ``EKB``, ``EDF``, ``EPD``, ``EGL``, ``ELZ`` and ``WDMS`` are all 0.0;
* the one statement about depth convention anywhere in the supplied material,
  in ``Seismic/Unsupervised-las_answers.ipynb``, says logs are displayed
  against "measure depth **or** true vertical depth" -- it names the
  distinction and does not resolve it for this file.

So the logged coordinate cannot be shown to be vertical, and this project does
not assume that it is.  The functions below therefore integrate along
**whatever coordinate they are given** and are named for what they compute:
``rho_g_integral`` and ``mean_rho_g_gradient``.  They equal the vertical-stress
increment and the vertical-stress gradient of Q5 only under an explicit
one-dimensional interpretation in which the logged coordinate is vertical.
That interpretation is stated wherever a result is reported; it is not
verified, and no field stress profile is claimed.

One quantity is more robust than the rest, within stated limits.  When two
density profiles are integrated over the **same** coordinate array, the ratio of
their mean gradients is **invariant under uniform scaling of that coordinate**:
multiplying every coordinate interval by one constant factor rescales both
integrals identically and cancels from the ratio.  ``mean_rho_g_gradient`` is
tested for exactly that invariance.

That is the precise claim, and it is narrower than it may look.  It does **not**
establish invariance under a depth-dependent trajectory correction.  If the
coordinate were corrected by a factor that varies with depth, the two integrals
would be reweighted sample by sample, and because the Gardner density is not a
constant multiple of the measured density, the ratio would in general change.
No such correction is applied, tested or claimed here.

Sign convention
---------------
The supplied solution integrates ``sigma_zz = rho g z`` from ``sigma_zz = 0``
at ``z = 0`` with ``z`` increasing away from the surface, so compressive stress
is positive and the coordinate increases downwards.  That convention is kept
throughout this module and the rest of the project.

Total versus effective stress
-----------------------------
``Models/2_Hard_rocks.pdf`` s.8 gives the effective stress law
``(sigma_ij)_eff = sigma_ij - alpha * P_fluid * delta_ij`` with ``alpha ~ 1``
for rock strength.  The supplied well has no pore-pressure measurement and the
LAS header records no mud weight (``DFD`` is 0.000), so **no effective stress
is computed anywhere in this project**.  Everything below is total stress.
``effective_stress`` is provided so the law can be applied if a measured pore
pressure is ever supplied; it is not called by the workflow.
"""

from __future__ import annotations

import numpy as np

from .units import STANDARD_GRAVITY


def rho_g_integral(coordinate_m, rho_kg_m3, g=STANDARD_GRAVITY):
    """Cumulative integral of ``rho * g`` along the supplied coordinate, in Pa.

    Integrates ``d(sigma)/ds = rho g`` by the trapezoidal rule over the supplied
    samples and returns the increase measured **from the first sample of the
    array**.

    ``coordinate_m`` is whatever coordinate the caller supplies.  For the well
    in this project it is the logged ``DEPT`` curve converted to metres, whose
    relation to true vertical depth the supplied files do not establish (see
    the module docstring).  The result equals the Q5 vertical-stress increment
    only under an explicit one-dimensional vertical interpretation.

    It deliberately does not return an absolute stress.  Doing so would require
    the density of everything above the top of the log, which is not measured.

    Parameters
    ----------
    coordinate_m : array_like
        Monotonically increasing coordinate in metres.
    rho_kg_m3 : array_like
        Bulk density at those coordinates, in kg/m^3.
    g : float
        Gravitational acceleration in m/s^2.

    Returns
    -------
    ndarray
        Cumulative ``rho g`` integral in Pa, first element zero.
    """
    s = np.asarray(coordinate_m, dtype=float)
    rho = np.asarray(rho_kg_m3, dtype=float)
    if s.shape != rho.shape:
        raise ValueError("coordinate and density must have the same shape")
    if s.size < 2:
        raise ValueError("need at least two samples to integrate")
    if np.any(np.diff(s) <= 0):
        raise ValueError("coordinate must increase monotonically")

    ds = np.diff(s)
    mean_rho = 0.5 * (rho[1:] + rho[:-1])
    return np.concatenate(([0.0], np.cumsum(mean_rho * g * ds)))


def mean_rho_g_gradient(coordinate_m, rho_kg_m3, g=STANDARD_GRAVITY):
    """Mean ``rho g`` gradient over the interval, in Pa/m.

    The total integral divided by the interval length, i.e. the path-averaged
    density times ``g``.  It is independent of where the coordinate is measured
    from.  It equals the Q5 vertical-stress gradient only under the
    one-dimensional vertical interpretation described in the module docstring.

    When two density profiles are compared over the **same** coordinate array,
    the ratio of their gradients is invariant under uniform scaling of that
    coordinate: one constant factor applied to every interval cancels.  That
    invariance does not extend to a depth-dependent trajectory correction, which
    would reweight the two integrals sample by sample.
    """
    s = np.asarray(coordinate_m, dtype=float)
    integral = rho_g_integral(s, rho_kg_m3, g=g)
    return integral[-1] / (s[-1] - s[0])


def horizontal_over_vertical_ratio(nu):
    """sigma_xx / sigma_zz = nu / (1 - nu) under the no-lateral-strain condition.

    Derived in ``Models/Elasticity Exercise Solutions.pdf`` Q5.  This is a
    *uniaxial-strain* result: it assumes the only load is gravity and that no
    lateral strain is permitted.  It is not a measurement of in-situ horizontal
    stress and says nothing about tectonic loading.  It is reported
    parametrically only; no value of ``nu`` is claimed for the supplied well,
    which has no shear sonic curve.
    """
    nu = np.asarray(nu, dtype=float)
    return nu / (1.0 - nu)


def vertical_strain(sigma_zz, E, nu):
    """eps_zz = (sigma_zz / E) * (1 - 2 nu^2 / (1 - nu)).

    The vertical strain that accompanies the uniaxial-strain state above, from
    the same worked solution.
    """
    sigma_zz = np.asarray(sigma_zz, dtype=float)
    E = np.asarray(E, dtype=float)
    nu = np.asarray(nu, dtype=float)
    return (sigma_zz / E) * (1.0 - 2.0 * nu ** 2 / (1.0 - nu))


def effective_stress(sigma, pore_pressure, alpha=1.0):
    """(sigma_ij)_eff = sigma_ij - alpha * P_fluid * delta_ij.

    ``Models/2_Hard_rocks.pdf`` s.8.  Provided for completeness only: the
    supplied data contain no pore-pressure measurement, so this function is not
    used by the workflow and no default pore pressure is offered.
    """
    sigma = np.asarray(sigma, dtype=float)
    pore_pressure = np.asarray(pore_pressure, dtype=float)
    return sigma - alpha * pore_pressure
