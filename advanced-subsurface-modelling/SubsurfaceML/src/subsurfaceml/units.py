"""Unit constants and conversions.

The whole package works internally in **SI units**:

===================  ======  ===============================================
Quantity             Unit    Notes
===================  ======  ===============================================
length               m
time                 s
pressure             Pa
permeability         m^2     (1 mD = 9.869233e-16 m^2)
viscosity            Pa.s
density              kg/m^3
volumetric rate      m^3/s   at *reservoir* conditions unless stated
mass                 kg
compressibility      1/Pa
===================  ======  ===============================================

Conversion helpers are provided for reporting only.  No conversion is applied
silently inside the solvers.

Source of the Darcy conversion factor: ``2-Upscaling.pdf`` p.19
("Converted to SI units, 1 Darcy is equivalent to 9.869233e-13 m2").
"""
from __future__ import annotations

# --- base conversions -------------------------------------------------------
DARCY = 9.869233e-13  # m^2  (see 2-Upscaling.pdf, p.19)
MILLIDARCY = DARCY * 1e-3
CENTIPOISE = 1e-3  # Pa.s
BAR = 1e5  # Pa
ATM = 101325.0  # Pa
MPA = 1e6  # Pa
DAY = 86400.0  # s
YEAR = 365.25 * DAY
MEGATONNE = 1e9  # kg


def md_to_m2(k_md: float) -> float:
    """Millidarcy -> m^2."""
    return k_md * MILLIDARCY


def m2_to_md(k_m2: float) -> float:
    """m^2 -> millidarcy."""
    return k_m2 / MILLIDARCY


def cp_to_pas(mu_cp: float) -> float:
    return mu_cp * CENTIPOISE


def mpa_to_pa(p_mpa: float) -> float:
    return p_mpa * MPA


def pa_to_mpa(p_pa: float) -> float:
    return p_pa / MPA


def years_to_s(t_yr: float) -> float:
    return t_yr * YEAR


def s_to_years(t_s: float) -> float:
    return t_s / YEAR


def kg_to_mt(m_kg: float) -> float:
    """kg -> megatonne (Mt)."""
    return m_kg / MEGATONNE


#: Display units used by figures / tables / the dashboard.
DISPLAY_UNITS = {
    "p_max": "MPa",
    "p_bh_max": "MPa",
    "dp_max": "MPa",
    "r_plume": "m",
    "mass_injected": "Mt",
    "mass_retained": "Mt",
    "mass_lost": "Mt",
    "retention_fraction": "-",
    "time": "years",
    "k": "mD",
    "rate": "kg/s",
}
