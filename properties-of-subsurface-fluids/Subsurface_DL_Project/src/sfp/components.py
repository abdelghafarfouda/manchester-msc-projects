"""The component table, taken verbatim from the supplied material.

Source
------
``Models/3 - Two-phase flash calculation.pdf``

* **p. 14** -- "Example [from .../PETE310/goodies/VLE_Pete310.xls]",
  headed *Whitson's K-value Correlation*.  The table gives, for seven
  components, the moles ``ni``, the mole fraction ``zi``, the **critical
  pressure** ``Pc (psia)``, the **critical temperature** ``Tc (R)`` and the
  **acentric factor** ``w`` (printed "Accentric Factor").
* **p. 15** -- "Model: Flash vaporization calculation using K-factor
  correlation" repeats the same seven components with the same constants,
  adds ``Tc (F)``, and works a full flash at 796.43821 psia and 180 F.

The same two pages appear again as scanned OneNote prints in
``Models/Part 3.pdf`` pp. 14-15.  A page-by-page OCR sweep of every PDF in
``Models/`` found the string "Accentric"/"acentric" **only** on those pages
(and on p. 4 of the same file, where the factor is defined in words), so this
is the only acentric-factor data anywhere in the supplied material.

The numbers below are transcribed exactly as printed, in the units printed
(psia and degrees Rankine).  They are **not** checked against, corrected by,
or supplemented from any outside source -- doing so is exactly what the
source restriction forbids.  ``scripts/verify_flash.py`` checks the
transcription by reproducing the two calculations the notes perform with it:
the bubble and dew points on p. 14 and the flash on p. 15.

Note on units: the source sheet converts Fahrenheit to Rankine with
``T[R] = T[F] + 460`` (e.g. CO2, 87.91 F -> 547.91 R), not 459.67.  Since the
Rankine values are the ones printed and used, they are the ones stored here,
and every temperature in this project is in Rankine on that same convention.
"""

from __future__ import annotations

import numpy as np

#: Component names exactly as printed on pp. 14-15.
NAMES = ("CO2", "C1", "C2", "C3", "C4", "C5", "C10")

#: Critical pressure, psia -- ``Pc (psia)`` column, p. 14.
PC_PSIA = np.array([1071.00, 666.40, 706.50, 539.25, 481.00, 488.60, 359.84])

#: Critical temperature, degrees Rankine -- ``Tc (R)`` column, p. 14.
TC_RANKINE = np.array([547.91, 343.33, 549.92, 750.04, 818.68, 845.80, 1129.20])

#: Acentric factor -- ``Accentric Factor`` column, p. 14.
OMEGA = np.array([0.26670, 0.01040, 0.09790, 0.19235, 0.22523, 0.25140, 0.38900])

#: Moles ``ni`` of the worked example, p. 14 (``zi = ni / sum(ni)``).
EXAMPLE_MOLES = np.array([5.0, 25.0, 10.0, 15.0, 14.0, 20.0, 40.0])

#: Mole fractions ``zi`` exactly as printed on pp. 14-15.
EXAMPLE_Z = np.array([0.0387597, 0.1937984, 0.0775194, 0.1162791,
                      0.1085271, 0.1550388, 0.3100775])

N_COMPONENTS = len(NAMES)

# ---------------------------------------------------------------------------
# States that appear in the supplied material, used to bound the sampling
# domain in ``sfp.data`` rather than inventing one.
# ---------------------------------------------------------------------------

#: (T [R], p [psia], where it appears) -- every numeric flash state printed in
#: ``Models/3 - Two-phase flash calculation.pdf``.
SUPPLIED_STATES = (
    (610.0, 1590.8769, "p. 14, bubble point of the example mixture at 150 F"),
    (610.0, 1.9995691, "p. 14, dew point of the example mixture at 150 F"),
    (640.0, 796.43821, "p. 15, flash of the example mixture at 180 F"),
    (680.0, 4000.0, "p. 16, C1/nC10 example, 220 F"),
    (620.0, 1000.0, "p. 17, C1/nC10 example, 160 F"),
)

#: Temperature span of the states above, degrees Rankine (150 F to 220 F).
T_MIN_RANKINE, T_MAX_RANKINE = 610.0, 680.0

#: Pressure span of the states above, psia (the p. 14 dew point to the
#: p. 16 example's 4000 psia).
P_MIN_PSIA, P_MAX_PSIA = 2.0, 4000.0

# Handy unit conversions for reporting only -- they introduce no physics.
PSIA_PER_MPA = 145.0377377
RANKINE_PER_KELVIN = 1.8


def psia_to_MPa(p):
    return np.asarray(p, float) / PSIA_PER_MPA


def rankine_to_K(T):
    return np.asarray(T, float) / RANKINE_PER_KELVIN
