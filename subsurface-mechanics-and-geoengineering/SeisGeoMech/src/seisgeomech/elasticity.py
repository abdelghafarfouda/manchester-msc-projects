"""Isotropic elasticity and its link to seismic velocities.

Every relation in this module is written out explicitly in
``Models/1_Elasticity.pdf``.  Section numbers below refer to that document
(EART35102 "Elasticity" notes); page numbers refer to its printed page
footers.

Relations used
--------------
* s.7, p.10 - conversion table between the five isotropic elastic constants.
  The two pairs used here are ``(E, nu)`` and ``(K, G)``::

      K = E / (3 (1 - 2 nu))            G = E / (2 (1 + nu))
      E = 9 K G / (3 K + G)             nu = (3 K - 2 G) / (2 (3 K + G))

* s.7, p.9 - ``E = 2 G (1 + nu)``.
* s.8, p.11 - Voigt, Reuss and Voigt-Reuss-Hill averages of a single-crystal
  stiffness tensor::

      A = (c11 + c22 + c33) / 3      a = (s11 + s22 + s33) / 3
      B = (c12 + c23 + c13) / 3      b = (s12 + s23 + s13) / 3
      C = (c44 + c55 + c66) / 3      c = (s44 + s55 + s66) / 3

      K_Voigt = (A + 2B) / 3         K_Reuss = 1 / (3a + 6b)
      G_Voigt = (A - B + 3C) / 5     G_Reuss = 5 / (4a - 4b + 3c)

  with ``S = C^-1`` (s.6, p.6) and the Hill average the arithmetic mean of the
  Voigt and Reuss bounds.

* s.10, p.12-13 - body-wave velocities of an isotropic elastic solid::

      Vp = sqrt((K + 4G/3) / rho)        Vs = sqrt(G / rho)

  The notes state directly that "by measuring P- and S-wave velocities the
  elastic properties may be determined, and vice versa".

Nothing else is added.  In particular no dynamic-to-static correction, no
anisotropy model and no empirical Vp-Vs relation is applied anywhere, because
none is supplied.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# --------------------------------------------------------------------------
# Conversions between isotropic elastic constants (1_Elasticity.pdf s.7, p.10)
# --------------------------------------------------------------------------

def bulk_modulus_from_E_nu(E, nu):
    """K = E / (3 (1 - 2 nu))."""
    E = np.asarray(E, dtype=float)
    nu = np.asarray(nu, dtype=float)
    return E / (3.0 * (1.0 - 2.0 * nu))


def shear_modulus_from_E_nu(E, nu):
    """G = E / (2 (1 + nu))."""
    E = np.asarray(E, dtype=float)
    nu = np.asarray(nu, dtype=float)
    return E / (2.0 * (1.0 + nu))


def youngs_modulus_from_K_G(K, G):
    """E = 9 K G / (3 K + G)."""
    K = np.asarray(K, dtype=float)
    G = np.asarray(G, dtype=float)
    return 9.0 * K * G / (3.0 * K + G)


def poissons_ratio_from_K_G(K, G):
    """nu = (3K - 2G) / (2 (3K + G))."""
    K = np.asarray(K, dtype=float)
    G = np.asarray(G, dtype=float)
    return (3.0 * K - 2.0 * G) / (2.0 * (3.0 * K + G))


# --------------------------------------------------------------------------
# Velocities (1_Elasticity.pdf s.10, p.12-13)
# --------------------------------------------------------------------------

def p_wave_modulus(K, G):
    """M = K + 4G/3, the modulus that appears inside the Vp expression."""
    return np.asarray(K, dtype=float) + 4.0 * np.asarray(G, dtype=float) / 3.0


def vp_from_K_G_rho(K, G, rho):
    """Vp = sqrt((K + 4G/3) / rho)."""
    return np.sqrt(p_wave_modulus(K, G) / np.asarray(rho, dtype=float))


def vs_from_G_rho(G, rho):
    """Vs = sqrt(G / rho)."""
    return np.sqrt(np.asarray(G, dtype=float) / np.asarray(rho, dtype=float))


def p_wave_modulus_from_vp_rho(vp, rho):
    """M = rho * Vp^2.

    This is the only rearrangement of ``Vp = sqrt((K + 4G/3)/rho)`` that can be
    evaluated from a sonic and a density log alone.  It returns the combination
    ``K + 4G/3``; **K and G cannot be separated without a shear velocity**, and
    the supplied well has no shear sonic curve.
    """
    vp = np.asarray(vp, dtype=float)
    rho = np.asarray(rho, dtype=float)
    return rho * vp ** 2


# --------------------------------------------------------------------------
# Voigt / Reuss / Hill averaging (1_Elasticity.pdf s.8, p.11)
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class VRHResult:
    """Voigt, Reuss and Hill averages of a 6x6 stiffness matrix."""

    K_voigt: float
    G_voigt: float
    K_reuss: float
    G_reuss: float

    @property
    def K_hill(self) -> float:
        return 0.5 * (self.K_voigt + self.K_reuss)

    @property
    def G_hill(self) -> float:
        return 0.5 * (self.G_voigt + self.G_reuss)


def voigt_reuss_hill(C):
    """Voigt, Reuss and Hill averages from a 6x6 stiffness matrix ``C``.

    ``C`` must be given in contracted (Voigt) notation.  Units are carried
    through unchanged: give ``C`` in GPa and the averages come back in GPa.
    """
    C = np.asarray(C, dtype=float)
    if C.shape != (6, 6):
        raise ValueError("stiffness matrix must be 6x6 in contracted notation")
    if not np.allclose(C, C.T):
        raise ValueError("stiffness matrix must be symmetric (1_Elasticity.pdf s.6)")

    S = np.linalg.inv(C)

    A = (C[0, 0] + C[1, 1] + C[2, 2]) / 3.0
    B = (C[0, 1] + C[1, 2] + C[0, 2]) / 3.0
    Cc = (C[3, 3] + C[4, 4] + C[5, 5]) / 3.0

    a = (S[0, 0] + S[1, 1] + S[2, 2]) / 3.0
    b = (S[0, 1] + S[1, 2] + S[0, 2]) / 3.0
    c = (S[3, 3] + S[4, 4] + S[5, 5]) / 3.0

    return VRHResult(
        K_voigt=(A + 2.0 * B) / 3.0,
        G_voigt=(A - B + 3.0 * Cc) / 5.0,
        K_reuss=1.0 / (3.0 * a + 6.0 * b),
        G_reuss=5.0 / (4.0 * a - 4.0 * b + 3.0 * c),
    )
