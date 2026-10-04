"""SeisGeoMech: Gardner's density prediction, tested at a well, and its cost.

A deliberately small, fully source-traceable project.  Every scientific
relation used here is written out in one of two supplied teaching folders:

* ``Models``  - EART35102 Hydrogeology and Geomechanics notes and exercises
* ``Seismic`` - seismic imaging practical notebooks and the LAS file they use

``SOURCE_MAP.md`` lists the exact document, section and cell behind every
equation, parameter and dataset.  Nothing outside those two folders is used.
"""

from __future__ import annotations

__version__ = "3.1.0"

from . import analysis, depth_blocks, elasticity, las_io, seismic, stress, units, worked_examples

__all__ = [
    "analysis",
    "depth_blocks",
    "elasticity",
    "las_io",
    "seismic",
    "stress",
    "units",
    "worked_examples",
    "__version__",
]
