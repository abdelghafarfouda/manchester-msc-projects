"""Fluid and rock-fluid property models for a CO2 (gaseous/supercritical) -
brine (aqueous) system at constant reservoir temperature.

Phase labelling follows ``4-CO2 BL.pdf``:

* ``g`` -- CO2-rich **gaseous** phase (injected, non-wetting)
* ``a`` -- brine **aqueous** phase (resident, wetting)

Mapping to the oil-water notation of ``3-IMPES.pdf``:
``oil -> a (displaced/wetting)``, ``water -> g (injected/non-wetting)``.

Assumptions (all explicit, all documented in ``docs/ASSUMPTIONS.md``)
--------------------------------------------------------------------
A1  Isothermal.  ``1-Transmissibility.pdf`` p.2 states the isothermal
    simplification that removes the energy equation.
A2  **Immiscible**: no CO2 dissolution into brine and no water vaporisation.
    ``4-CO2 BL.pdf`` p.19-29 does treat compositional partitioning
    (omega_CO2,a = 0.011); we deliberately drop it and state the consequence:
    the trailing (drying) shock of the two-shock CO2/brine solution is not
    reproduced, and dissolution trapping is *not* modelled.  Retained mass is
    therefore free-phase mass only, which is conservative for plume extent.
A3  Formation volume factors B_a = B_g = 1 (all volumes are reservoir
    volumes).  Densities are constant at reservoir P,T -- the constant-P,T
    fluid-property assumption of ``4-CO2 BL.pdf`` p.21.
A4  Slight compressibility is retained *only* in the pressure-storage term
    (c_t = c_r + S_a c_a + S_g c_g).  This is the classical slightly
    compressible approximation and is mildly inconsistent with A3; the
    inconsistency is O(c_t * dP) ~ 1e-2 for dP ~ 10 MPa.
A5  Capillary pressure is zero (``4-CO2 BL.pdf`` p.19 "no capillary and/or
    gravity forces").  No capillary-pressure curve is supplied in the course
    material for CO2-brine, so none is modelled (see ``impes`` docstring).
A6  Gravity is neglected: the model is a single horizontal layer set with
    1-D radial flow, so buoyant override is *not* represented.  This is the
    single largest unmodelled physics term; see ``docs/ASSUMPTIONS.md``.

Default numerical values are typical deep saline-aquifer values
(~1500 m, ~40 C, ~15 MPa).  They are **assumed**, not measured, and not
taken from the course archives; the archives do not supply a property table
at these conditions.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field

import numpy as np


@dataclass(frozen=True)
class FluidProperties:
    """Constant-P,T two-phase fluid description (SI units)."""

    rho_g: float = 700.0        #: CO2 density [kg/m^3]
    rho_a: float = 1050.0       #: brine density [kg/m^3]
    mu_g: float = 5.5e-5        #: CO2 viscosity [Pa.s]
    mu_a: float = 6.0e-4        #: brine viscosity [Pa.s]
    c_g: float = 1.0e-8         #: CO2 compressibility [1/Pa]
    c_a: float = 4.5e-10        #: brine compressibility [1/Pa]

    @property
    def endpoint_mobility_ratio(self) -> float:
        """M = (krg0/mu_g)/(kra0/mu_a) evaluated at krg0=kra0=1.

        The saturation-endpoint-corrected value is
        :meth:`RelPerm.endpoint_mobility_ratio`.
        """
        return self.mu_a / self.mu_g

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RockProperties:
    c_r: float = 4.5e-10  #: rock (pore) compressibility [1/Pa]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RelPerm:
    """Power-law (Corey-type) relative permeabilities on the normalised
    saturation of ``4-CO2 BL.pdf`` p.18 (endpoint values krg0, kra0).

    Normalised wetting-phase (brine) saturation, following the ``S_nw``
    definition sketched on ``4-CO2 BL.pdf`` p.18::

        Se = (S_a - S_ar) / (1 - S_ar - S_gr)

        kr_a = kra0 * Se**na
        kr_g = krg0 * (1 - Se)**ng

    ``S_gr`` is the residual (trapped) gas saturation used only as a
    *drainage* endpoint here -- no imbibition hysteresis is modelled, so the
    model does not represent residual trapping during post-injection
    redistribution.  ``3-IMPES.pdf`` p.4 notes that both drainage and
    imbibition curves may be required; we implement drainage only and list
    hysteresis as an extension.
    """

    S_ar: float = 0.20   #: irreducible brine saturation [-]
    S_gr: float = 0.05   #: residual gas saturation (drainage endpoint) [-]
    n_a: float = 3.0     #: brine Corey exponent [-]
    n_g: float = 2.0     #: gas Corey exponent [-]
    kra0: float = 1.0    #: brine endpoint relative permeability [-]
    krg0: float = 0.4    #: gas endpoint relative permeability [-]

    # -- normalisation -----------------------------------------------------
    def Se(self, S_g: np.ndarray) -> np.ndarray:
        """Normalised brine saturation from gas saturation."""
        S_a = 1.0 - np.asarray(S_g, dtype=float)
        denom = 1.0 - self.S_ar - self.S_gr
        return np.clip((S_a - self.S_ar) / denom, 0.0, 1.0)

    # -- relative permeability --------------------------------------------
    def kr_a(self, S_g: np.ndarray) -> np.ndarray:
        return self.kra0 * self.Se(S_g) ** self.n_a

    def kr_g(self, S_g: np.ndarray) -> np.ndarray:
        return self.krg0 * (1.0 - self.Se(S_g)) ** self.n_g

    # -- mobilities --------------------------------------------------------
    def lam_a(self, S_g: np.ndarray, fl: FluidProperties) -> np.ndarray:
        return self.kr_a(S_g) / fl.mu_a

    def lam_g(self, S_g: np.ndarray, fl: FluidProperties) -> np.ndarray:
        return self.kr_g(S_g) / fl.mu_g

    def lam_t(self, S_g: np.ndarray, fl: FluidProperties) -> np.ndarray:
        return self.lam_a(S_g, fl) + self.lam_g(S_g, fl)

    # -- fractional flow ---------------------------------------------------
    def f_g(self, S_g: np.ndarray, fl: FluidProperties) -> np.ndarray:
        """Gas fractional flow f_g = lam_g / (lam_g + lam_a).

        This is the ``f_w`` of ``4-CO2 BL.pdf`` p.10 with water replaced
        by the injected CO2 phase, capillary and gravity terms dropped (A5,A6).
        """
        lg = self.lam_g(S_g, fl)
        lt = lg + self.lam_a(S_g, fl)
        out = np.zeros_like(np.asarray(S_g, dtype=float))
        np.divide(lg, lt, out=out, where=lt > 0)
        return out

    def dfg_dSg(self, S_g: np.ndarray, fl: FluidProperties,
                h: float = 1e-6) -> np.ndarray:
        """Central-difference derivative of the fractional-flow curve."""
        S = np.asarray(S_g, dtype=float)
        return (self.f_g(np.clip(S + h, 0, 1), fl)
                - self.f_g(np.clip(S - h, 0, 1), fl)) / (2 * h)

    # -- diagnostics --------------------------------------------------------
    def endpoint_mobility_ratio(self, fl: FluidProperties) -> float:
        """M = (krg0/mu_g) / (kra0/mu_a) -- the end-point mobility ratio of
        ``4-CO2 BL.pdf`` p.18.  M >> 1 means an unstable, low-efficiency
        displacement with an early, low-saturation shock."""
        return (self.krg0 / fl.mu_g) / (self.kra0 / fl.mu_a)

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------
# Buckley-Leverett / Welge helpers  (4-CO2 BL , p.11-17)
# --------------------------------------------------------------------------
def welge_shock(rp: RelPerm, fl: FluidProperties, n: int = 20001
                ) -> tuple[float, float, float]:
    """Welge tangent construction for the leading shock.

    Finds ``S_gf`` such that the chord from the initial state
    ``(S_g = 0, f_g = 0)`` is tangent to the fractional-flow curve, i.e.

        f_g(S_gf) / (S_gf - S_g,init) = df_g/dS_g |_{S_gf}

    (``4-CO2 BL.pdf`` p.14-16).

    Returns
    -------
    (S_gf, f_gf, dfdS_shock)
        Shock saturation, its fractional flow, and the shock (chord) speed
        ``f_gf / S_gf`` which equals the tangent slope.
    """
    S = np.linspace(1e-9, 1.0 - rp.S_ar - 1e-9, n)
    f = rp.f_g(S, fl)
    chord = f / S                       # slope of chord from (0, 0)
    tangent = rp.dfg_dSg(S, fl)
    diff = chord - tangent
    # first sign change from + to - after the inflection
    idx = np.where(np.sign(diff[:-1]) != np.sign(diff[1:]))[0]
    if len(idx) == 0:
        i = int(np.argmin(np.abs(diff)))
    else:
        i = int(idx[-1])
    # linear interpolation on the sign change for a sharper root
    if i + 1 < len(S) and diff[i] != diff[i + 1]:
        w = diff[i] / (diff[i] - diff[i + 1])
        S_gf = S[i] + w * (S[i + 1] - S[i])
    else:
        S_gf = S[i]
    f_gf = float(rp.f_g(np.array([S_gf]), fl)[0])
    return float(S_gf), f_gf, float(f_gf / S_gf)


def bl_profile_1d(rp: RelPerm, fl: FluidProperties, x: np.ndarray,
                  t: float, q_t: float, area: float, phi: float,
                  L: float | None = None) -> np.ndarray:
    """Analytical 1-D Buckley-Leverett saturation profile.

    Method of characteristics (``4-CO2 BL.pdf`` p.11-13):
    a saturation ``S_g`` travels at ``v = (q_t / (A phi)) df_g/dS_g``; the
    profile is truncated at the Welge shock.

    Parameters
    ----------
    x : positions [m]
    t : time [s]
    q_t : total volumetric rate [m^3/s] at reservoir conditions
    area : cross-sectional area [m^2]
    phi : porosity [-]
    """
    S_gf, _, v_shock_slope = welge_shock(rp, fl)
    v0 = q_t / (area * phi)
    x_shock = v0 * v_shock_slope * t
    S_grid = np.linspace(S_gf, 1.0 - rp.S_ar - 1e-9, 4001)
    x_of_S = v0 * rp.dfg_dSg(S_grid, fl) * t
    order = np.argsort(x_of_S)
    Sg = np.interp(np.asarray(x, float), x_of_S[order], S_grid[order],
                   left=1.0 - rp.S_ar, right=0.0)
    Sg = np.where(np.asarray(x, float) > x_shock, 0.0, Sg)
    if L is not None:
        Sg = np.where(np.asarray(x, float) > L, 0.0, Sg)
    return Sg
