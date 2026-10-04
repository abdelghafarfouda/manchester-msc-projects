"""Zero-offset forward modelling, exactly as taught in the seismic practicals.

Every relation here is written out in ``Seismic/Ex1.ipynb``
("Practical 1: Synthetic Seismic models").  Cell numbers are the positional
index of the cell in the supplied notebook.

Relations used
--------------
* Gardner relation, ``Ex1.ipynb`` markdown cell 35::

      rho = 0.31 * Vp ** 0.25

  The notebook gives the expression without stating units.  The only unit
  convention under which it returns geologically sensible bulk densities is
  ``Vp`` in m/s giving ``rho`` in g/cm^3, and that convention is confirmed
  against the measured density log in ``results/tables/gardner_vs_rhob.csv``.

* Acoustic impedance, ``Ex1.ipynb`` s.1.3 and code cell 8::

      Z = rho * Vp

* Normal-incidence reflection coefficient, ``Ex1.ipynb`` s.1.4 markdown cell
  14 and code cell 15::

      R = (rho2 V2 - rho1 V1) / (rho2 V2 + rho1 V1)

* Ricker wavelet and convolution, ``Ex1.ipynb`` s.1.6, code cells 23 and 25.
  The notebook calls ``bruges.filters.ricker(duration, dt, f)`` and
  ``np.convolve(..., mode='same')``; both are used unchanged here.

* Wavelet-frequency experiment, ``Ex1.ipynb`` learning objective 3 and
  markdown cells 28/30, which pose it as an experiment: change the wavelet
  frequency and see how the convolved trace compares with the
  reflection-coefficient series.  ``resolution_sweep`` below reports a Pearson
  correlation between the two as a similarity measure for that particular
  synthetic experiment.  No quarter-wavelength or tuning-thickness rule is
  used, because none is given in the supplied material, and the correlation is
  not converted into a bed thickness or a resolvability statement.

Not implemented
---------------
No NMO, semblance, migration, attribute or SEG-Y handling is included.  Those
practicals (Ex2-Ex4) operate on data volumes that are not part of the supplied
``Seismic`` folder, so there is nothing here for them to run on.
"""

from __future__ import annotations

import numpy as np

from .units import FT_TO_M, US_TO_S

#: Coefficient and exponent of the Gardner relation as written in Ex1.ipynb.
GARDNER_COEFFICIENT: float = 0.31
GARDNER_EXPONENT: float = 0.25


def gardner_density_gcc(vp_m_s):
    """rho [g/cm^3] = 0.31 * Vp[m/s] ** 0.25  (Ex1.ipynb cell 35)."""
    vp = np.asarray(vp_m_s, dtype=float)
    return GARDNER_COEFFICIENT * vp ** GARDNER_EXPONENT


def acoustic_impedance(rho, vp):
    """Z = rho * Vp  (Ex1.ipynb s.1.3).

    Units are carried through: pass SI and you get kg m^-2 s^-1.
    """
    return np.asarray(rho, dtype=float) * np.asarray(vp, dtype=float)


def reflection_coefficients(impedance):
    """Normal-incidence reflection coefficients between adjacent samples.

    ``R_i = (Z_{i+1} - Z_i) / (Z_{i+1} + Z_i)``  (Ex1.ipynb s.1.4, cell 15).
    Returns an array one element shorter than ``impedance``.
    """
    z = np.asarray(impedance, dtype=float)
    if z.ndim != 1:
        raise ValueError("expected a 1-D impedance series")
    if z.size < 2:
        raise ValueError("need at least two impedance samples")
    return (z[1:] - z[:-1]) / (z[1:] + z[:-1])


def ricker_wavelet(duration, dt, frequency):
    """Ricker wavelet from ``bruges.filters.ricker`` (Ex1.ipynb cell 23).

    Returns ``(amplitude, time)``.  ``bruges`` is the library the supplied
    practical uses, so its definition is taken as the supplied one.
    """
    import bruges as bg

    wavelet, t = bg.filters.ricker(duration=duration, dt=dt, f=frequency)
    return np.asarray(wavelet, dtype=float), np.asarray(t, dtype=float)


def convolve_trace(reflectivity, wavelet):
    """Synthetic trace = reflectivity * wavelet, ``mode='same'``.

    ``Ex1.ipynb`` cell 25 uses ``np.convolve(..., mode='same')``.
    """
    return np.convolve(
        np.asarray(reflectivity, dtype=float),
        np.asarray(wavelet, dtype=float),
        mode="same",
    )


def two_way_travel_time_along_log(coordinate_m, dt_us_per_ft):
    """Two-way travel time along the logged path, from the sonic log itself.

    ``DT`` is logged in microseconds per foot: it *is* a one-way travel time
    per unit length.  Integrating it along the logged coordinate therefore
    gives one-way time along that path directly, and twice that is the
    round-trip time along the same path.  This is arithmetic on the logged
    unit, not a velocity model.

        t_ow(s) = integral of DT ds,  t = 2 * t_ow

    **This is travel time along the logged path, not vertical two-way time.**
    The supplied files do not establish that the logged coordinate is vertical
    (no TVD curve, no deviation survey, ``LMF`` is ``UNKNOWN``), and no
    checkshot, datum or drift correction is supplied.  It is used here only to
    place the reflectivity series on a time axis so that a wavelet with a
    frequency in hertz can be convolved with it.  Nothing in this project is a
    seismic-to-well tie, and no field seismic data are involved.

    Returns time in seconds, measured from the first sample.
    """
    z = np.asarray(coordinate_m, dtype=float)
    dt = np.asarray(dt_us_per_ft, dtype=float)
    if z.shape != dt.shape:
        raise ValueError("coordinate and slowness must have the same shape")
    if z.size < 2:
        raise ValueError("need at least two samples to integrate")

    slowness_s_per_m = dt * US_TO_S / FT_TO_M  # us/ft -> s/m
    dz = np.diff(z)
    mean_slowness = 0.5 * (slowness_s_per_m[1:] + slowness_s_per_m[:-1])
    one_way = np.concatenate(([0.0], np.cumsum(mean_slowness * dz)))
    return 2.0 * one_way


def resample_to_regular_time(t_irregular, series, dt_s):
    """Linear resampling of a series onto a regular two-way-time axis.

    Convolution with a wavelet defined on a fixed sample interval requires a
    regularly sampled trace.  The log is regular in depth but not in time, so
    the reflectivity is interpolated onto a regular time axis before
    convolution.  Linear interpolation is a presentation choice; it introduces
    no physical assumption beyond the measured time-depth pairs.
    """
    t = np.asarray(t_irregular, dtype=float)
    y = np.asarray(series, dtype=float)
    t_reg = np.arange(t[0], t[-1], dt_s)
    return t_reg, np.interp(t_reg, t, y)


def resolution_sweep(reflectivity, dt_s, frequencies, duration=None):
    """Similarity between the synthetic trace and its input reflectivity.

    ``Ex1.ipynb`` markdown cells 28 and 30 ask the student to change the
    wavelet frequency and compare the convolved trace with the
    reflection-coefficient series.  For each frequency this function convolves
    the reflectivity with a Ricker wavelet and reports the **Pearson
    correlation** between the resulting synthetic trace and that same input
    series, together with the RMS amplitude of the trace.

    How to read the correlation
    ---------------------------
    It is a linear-similarity measure between two specific series in this
    synthetic experiment, and nothing more.  In particular it is **not** a
    fraction of information, variability or structure recovered, and neither is
    its square: the comparison is between a broadband input series and a
    band-limited output of that same series, so the number is not a
    decomposition of anything.  It is reported so that frequencies can be
    ranked against each other within this experiment.  It is not converted into
    a resolvable bed thickness, and it supports no statement about what a field
    seismic survey would or would not image.

    Returns a list of ``(frequency, correlation, n_wavelet_samples,
    synthetic_rms)`` tuples.
    """
    r = np.asarray(reflectivity, dtype=float)
    out = []
    for f in frequencies:
        # A Ricker wavelet is effectively zero beyond about +/- 1/f; a duration
        # of 2/f therefore contains the wavelet without truncating it.
        dur = duration if duration is not None else 2.0 / f
        w, _ = ricker_wavelet(duration=dur, dt=dt_s, frequency=f)
        if w.size > r.size:
            raise ValueError(
                f"a {f:g} Hz Ricker wavelet spans {w.size} samples "
                f"({w.size * dt_s * 1e3:.1f} ms) but the trace is only "
                f"{r.size} samples ({r.size * dt_s * 1e3:.1f} ms) long. "
                "The logged interval is too short to assess this frequency; "
                "see min_supportable_frequency()."
            )
        s = convolve_trace(r, w)
        corr = float(np.corrcoef(r, s)[0, 1])
        out.append((float(f), corr, int(w.size), float(np.sqrt((s ** 2).mean()))))
    return out


def min_supportable_frequency(n_samples, dt_s):
    """Lowest Ricker centre frequency a trace of this length can carry.

    A wavelet of duration ``2/f`` must fit inside the trace, so
    ``f >= 2 / (n_samples * dt_s)``.  This is a limit imposed by the length of
    the logged interval, not by the subsurface.
    """
    return 2.0 / (n_samples * dt_s)
