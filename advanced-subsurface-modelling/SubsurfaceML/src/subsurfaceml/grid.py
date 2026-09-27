"""Grids and geometric transmissibility factors.

Two 1-D geometries are supported by the same solver core:

``RadialGrid``
    Logarithmically-spaced radial cells around a central injector.  Used for
    the CO2 storage model.  Derived from the cylindrical form given in
    ``4-CO2 BL.pdf`` p.20::

        -(1/r) d/dr [ (r k / (mu B)) dP/dr ] = d/dt (phi / B)

``CartesianGrid1D``
    Uniform linear cells.  Used only to reproduce the Buckley-Leverett
    benchmark of ``4-CO2 BL.pdf`` p.11-17 and the linear
    transmissibility of ``1-Transmissibility.pdf`` p.18-19.

Both expose the same interface, so :mod:`subsurfaceml.impes` is geometry
agnostic:

``n``            number of cells
``pore_volume``  phi-weighted cell volume [m^3]  (set by the caller)
``bulk_volume``  cell volume [m^3]
``geom_factor``  array of length ``n-1``: the purely geometric part of the
                 inter-cell transmissibility, i.e. ``T = geom * k_harm * lam``

Derivation of the geometric factors
-----------------------------------
*Linear* (``1-Transmissibility.pdf`` p.18):  ``q = -(kA/mu) dP/dx`` discretised
between neighbouring cell centres gives ``T = k_h A / (mu dx)``; with the
mobility ``lam = kr/mu`` factored out, ``geom = A / dx``.

*Radial* (series radial flow, ``2-Upscaling.pdf`` p.8): integrating Darcy's
law between two radii at constant rate gives
``q = 2 pi h k lam (P2 - P1) / ln(r2/r1)``, hence ``geom = 2 pi h / ln(r2/r1)``
and the two half-cells combine **harmonically weighted by log-radius**:

    k_h = ln(r_{i+1}/r_i) / [ ln(r_{i+1/2}/r_i)/k_i + ln(r_{i+1}/r_{i+1/2})/k_{i+1} ]

which is the radial analogue of the harmonic mean quoted in
``1-Transmissibility.pdf`` p.19.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CartesianGrid1D:
    """Uniform 1-D linear grid of ``n`` cells, length ``L``, area ``A``."""

    n: int
    L: float
    area: float

    def __post_init__(self) -> None:
        self.dx = self.L / self.n
        self.faces = np.linspace(0.0, self.L, self.n + 1)
        self.centres = 0.5 * (self.faces[:-1] + self.faces[1:])
        self.bulk_volume = np.full(self.n, self.dx * self.area)

    def geom_factor(self) -> np.ndarray:
        """A/dx between neighbouring cell centres (length n-1)."""
        return np.full(self.n - 1, self.area / self.dx)

    def harmonic_k(self, k: np.ndarray) -> np.ndarray:
        """Harmonic mean of neighbouring permeabilities (equal dx).

        ``1-Transmissibility.pdf`` p.19:
        ``k_bar = (dx_i + dx_{i+1}) / (dx_i/k_i + dx_{i+1}/k_{i+1})``.
        """
        k = np.asarray(k, float)
        return 2.0 * k[:-1] * k[1:] / (k[:-1] + k[1:])

    @property
    def outer_distance(self) -> float:
        """Distance from the last cell centre to the outer boundary face."""
        return self.dx / 2.0

    def outer_geom_factor(self) -> float:
        return self.area / self.outer_distance


@dataclass
class RadialGrid:
    """Logarithmically graded radial grid from ``r_w`` to ``r_e``.

    Cell centres are the geometric means of the bounding face radii, which is
    the pressure-equivalent radius for radial Darcy flow.
    """

    n: int
    r_w: float
    r_e: float
    h: float          #: layer thickness [m]
    r_near: float = 20.0
    """Outer radius of the **well block**, cell 0 = ``[r_w, r_near]``.

    A purely logarithmic grid down to a 0.15 m wellbore makes the innermost
    pore volume ~0.06 m^3, i.e. a fluid residence time of a few seconds at
    field injection rates.  An explicit saturation update then requires
    second-scale time steps for the whole simulation, which is unusable and is
    *not* what the physics of interest requires.  Instead the near-well region
    is represented by a single well block whose pressure drop to the wellbore
    is carried **analytically** by the well index (:meth:`well_index_geom`).
    Because cell centres are the geometric means of the bounding face radii
    and the radial transmissibility between two radii is exact, the
    steady-state single-phase solution -- *including the bottom-hole
    pressure* -- is reproduced exactly on this grid; that is asserted in
    ``tests/test_single_phase.py::test_well_index_bhp``.

    Set ``r_near = r_w`` to recover a fully logarithmic grid.
    """

    def __post_init__(self) -> None:
        if not (0 < self.r_w < self.r_e):
            raise ValueError("require 0 < r_w < r_e")
        r1 = float(np.clip(self.r_near, self.r_w, self.r_e * 0.5))
        if r1 <= self.r_w * (1 + 1e-12):
            self.faces = np.geomspace(self.r_w, self.r_e, self.n + 1)
        else:
            self.faces = np.concatenate(
                ([self.r_w], np.geomspace(r1, self.r_e, self.n)))
        self.centres = np.sqrt(self.faces[:-1] * self.faces[1:])
        self.bulk_volume = np.pi * (self.faces[1:] ** 2
                                    - self.faces[:-1] ** 2) * self.h

    def geom_factor(self) -> np.ndarray:
        """2 pi h / ln(r_{i+1}/r_i) between neighbouring cell centres."""
        return 2.0 * np.pi * self.h / np.log(self.centres[1:] / self.centres[:-1])

    def harmonic_k(self, k: np.ndarray) -> np.ndarray:
        """Log-radius-weighted harmonic mean across the shared face."""
        k = np.asarray(k, float)
        rf = self.faces[1:-1]              # shared faces, length n-1
        w_l = np.log(rf / self.centres[:-1])
        w_r = np.log(self.centres[1:] / rf)
        return (w_l + w_r) / (w_l / k[:-1] + w_r / k[1:])

    # -- well ---------------------------------------------------------------
    def well_index_geom(self) -> float:
        """Geometric part of the injector well index for the innermost cell.

        ``q = WI * lam * (p_bh - p_0)`` with

            WI = 2 pi k h / ln(r_0 / r_w)

        i.e. the radial well equation of ``3-IMPES.pdf`` p.16 with the
        drainage radius replaced by the first cell centre.  Because the grid is
        genuinely radial and its inner face *is* the wellbore, no Peaceman
        equivalent-radius correction is required.
        """
        return 2.0 * np.pi * self.h / np.log(self.centres[0] / self.r_w)

    # -- outer boundary ------------------------------------------------------
    def outer_geom_factor(self) -> float:
        """Geometric factor between the last cell centre and ``r_e``."""
        return 2.0 * np.pi * self.h / np.log(self.r_e / self.centres[-1])
