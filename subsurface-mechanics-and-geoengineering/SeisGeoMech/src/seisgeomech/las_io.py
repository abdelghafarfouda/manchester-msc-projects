"""Reading and quality-controlling the supplied well log.

The file is ``Seismic/48_10b-_9_jwl_JWL_FILE_1682139.las`` (UK well 48/10b-9,
spudded 1990-08-26, operator BP).  ``Seismic/Unsupervised-las_answers.ipynb``
cell 33 reads it with ``lasio.read(...)``, so ``lasio`` is used here too.

Quality control policy
----------------------
The only rejection applied is the NULL value declared in the LAS header
(``NULL. -999.250``).  No caliper, DRHO or gas-effect cut-off is applied,
because the supplied material states no threshold for any of them.  Their
values are reported in the QC table so a reader can see the hole condition,
but they are not used to discard samples.  Applying an unsupported numeric
cut-off would be exactly the kind of imported convention this project excludes.

Depth convention
----------------
The depth curve is labelled ``DEPT``.  That label alone does not establish
true vertical depth, and nothing else in the supplied files does either:
``depth_convention_evidence()`` collects the relevant header fields and curve
names so the reader can check this directly.  ``DEPT_M`` is therefore treated
throughout as *the logged depth coordinate in metres*, not as TVD.  See the
``stress`` module docstring for what follows from that.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .units import feet_to_metres, gcc_to_kg_m3, slowness_us_per_ft_to_velocity_m_s

#: Curves declared in the ~Curve Information Block of the supplied LAS file.
CURVES = (
    "DEPT", "CALI", "ILM", "ILD", "DT", "TENS", "CGR", "DTL",
    "RHOB", "DRHO", "PEF", "NPHI", "THOR", "URAN", "GR", "POTA",
)

DEFAULT_LAS = (
    Path(__file__).resolve().parents[2]
    / "data" / "raw" / "48_10b-_9_jwl_JWL_FILE_1682139.las"
)


@dataclass(frozen=True)
class WellLog:
    """The supplied log, with derived depth in metres and velocity in m/s."""

    frame: pd.DataFrame
    path: Path
    null_value: float

    @property
    def n_samples(self) -> int:
        return len(self.frame)

    def coverage(self) -> pd.DataFrame:
        """Per-curve sample count and depth range, in metres."""
        rows = []
        for name in self.frame.columns:
            if name in ("DEPT", "DEPT_M"):
                continue
            present = self.frame[name].notna()
            n = int(present.sum())
            if n:
                d = self.frame.loc[present, "DEPT_M"]
                rows.append((name, n, 100.0 * n / self.n_samples,
                             float(d.min()), float(d.max())))
            else:
                rows.append((name, 0, 0.0, np.nan, np.nan))
        return pd.DataFrame(
            rows,
            columns=["curve", "n_valid", "coverage_pct", "top_m", "base_m"],
        )

    def overlap(self, curves=("DT", "RHOB")) -> pd.DataFrame:
        """Rows where every named curve is present, sorted by depth."""
        sub = self.frame.dropna(subset=list(curves)).copy()
        return sub.sort_values("DEPT_M").reset_index(drop=True)

    def depth_convention_evidence(self) -> pd.DataFrame:
        """What the supplied file says about its own depth convention.

        Reads the relevant header fields straight out of the LAS text and
        checks for any curve that would establish true vertical depth.  The
        point of this table is that it comes up empty: nothing in the file
        distinguishes measured depth from true vertical depth, which is why
        this project does not assume they are the same.
        """
        text = self.path.read_text(errors="replace").splitlines()
        header = [ln for ln in text if not ln.strip().startswith("~")]
        header = header[: next(
            (i for i, ln in enumerate(text) if ln.strip().startswith("~A")), len(header)
        )]

        def field(mnem):
            for ln in header:
                if ln.strip().upper().startswith(mnem.upper()) and "." in ln:
                    parts = ln.split(":", 1)
                    value = parts[0].split(".", 1)[1].strip() if "." in parts[0] else ""
                    descr = parts[1].strip() if len(parts) > 1 else ""
                    return value, descr
            return "", "(field absent)"

        rows = []
        for mnem in ("LMF", "EKB", "EDF", "EPD", "EGL", "ELZ", "WDMS"):
            value, descr = field(mnem)
            rows.append((mnem, descr, value))

        present = {c.upper() for c in self.frame.columns}
        for name, what in (
            ("TVD", "true vertical depth curve"),
            ("TVDSS", "TVD subsea curve"),
            ("DEVI", "deviation / inclination curve"),
            ("INCL", "inclination curve"),
            ("AZIM", "azimuth curve"),
        ):
            rows.append((name, what, "present" if name in present else "ABSENT"))

        return pd.DataFrame(rows, columns=["field_or_curve", "meaning", "value"])


def read_las(path=None) -> WellLog:
    """Read the supplied LAS file into a :class:`WellLog`.

    Uses ``lasio`` (the reader used in ``Seismic/Unsupervised-las_answers.ipynb``)
    when it is installed, and falls back to a direct parse of the ``~A`` section
    otherwise.  Both paths produce identical numbers; the fallback exists only
    so the project runs without an optional dependency.
    """
    path = Path(path) if path is not None else DEFAULT_LAS
    if not path.exists():
        raise FileNotFoundError(f"supplied LAS file not found: {path}")

    null_value = -999.25
    try:
        import lasio

        las = lasio.read(str(path))
        df = las.df().reset_index()
        df.columns = [str(c).upper() for c in df.columns]
        if "DEPTH" in df.columns and "DEPT" not in df.columns:
            df = df.rename(columns={"DEPTH": "DEPT"})
        hdr_null = las.well.get("NULL")
        if hdr_null is not None and hdr_null.value is not None:
            null_value = float(hdr_null.value)
    except ImportError:
        df = _parse_ascii_section(path)

    df = df.replace(null_value, np.nan)
    for name in CURVES:
        if name not in df.columns:
            df[name] = np.nan
    df = df[list(CURVES)].astype(float)

    df["DEPT_M"] = feet_to_metres(df["DEPT"].to_numpy())
    df["VP"] = slowness_us_per_ft_to_velocity_m_s(df["DT"].to_numpy())
    df["RHOB_SI"] = gcc_to_kg_m3(df["RHOB"].to_numpy())

    return WellLog(frame=df, path=path, null_value=null_value)


def _parse_ascii_section(path: Path) -> pd.DataFrame:
    """Minimal ~A-section parser used when lasio is unavailable."""
    text = path.read_text(errors="replace").splitlines()
    start = next(i for i, line in enumerate(text) if line.strip().startswith("~A"))
    rows = []
    for line in text[start + 1:]:
        parts = line.split()
        if len(parts) == len(CURVES):
            rows.append([float(p) for p in parts])
    return pd.DataFrame(rows, columns=list(CURVES))
