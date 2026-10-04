"""Two-phase vapour-liquid flash, exactly as taught in the module.

Every equation in this module comes from one supplied file:

``Models/3 - Two-phase flash calculation.pdf`` (CHEN60492, Dr M. Babaei)

    p. 2   K_i = y_i / x_i ;  z_i = x_i F_L + y_i F_V ;  F_V + F_L = 1
    p. 4   Wilson correlation
               K_i = exp[ 5.37 (1 + w_i) (1 - 1/T_ri) ] / p_ri
           with T_ri = T / T_ci and p_ri = p / p_ci
    p. 7   x_i = z_i / [ F_V (K_i - 1) + 1 ]
           y_i = z_i K_i / [ F_V (K_i - 1) + 1 ]
    p. 8   Rachford-Rice   h(F_V) = SUM z_i (K_i - 1) / [ F_V (K_i - 1) + 1 ] = 0
           physical-root test:  SUM z_i K_i > 1  and  SUM z_i / K_i > 1
    p. 11  the phase table that turns those two sums into a phase label

Nothing else is used.  There is no equation of state here: the module's notes
introduce the EoS route (p. 6) only in words -- no cubic EoS, no mixing rule
and no table of component constants is given anywhere in the supplied files --
so this project stays on the Wilson route, which is written out in full.

The root is found by bisection rather than by a library solver so that the
iteration the network is asked to replace is visible in one screen of code.
"""

from __future__ import annotations

import numpy as np

# Rachford-Rice is solved on the open interval (0, 1); the two end points are
# the bubble point (F_V = 0) and the dew point (F_V = 1), notes p. 12-13.
_EPS = 1.0e-12


def wilson_k(p, T, Tc, pc, omega):
    """Wilson K-values -- notes p. 4.

    Parameters are broadcast against each other, so ``p`` and ``T`` may be
    ``(n, 1)`` while ``Tc, pc, omega`` are ``(n, n_c)`` or ``(n_c,)``.

    ``p`` and ``pc`` must share units; ``T`` and ``Tc`` must share units.
    """
    Tr = np.asarray(T, float) / np.asarray(Tc, float)
    pr = np.asarray(p, float) / np.asarray(pc, float)
    return np.exp(5.37 * (1.0 + np.asarray(omega, float)) * (1.0 - 1.0 / Tr)) / pr


def phase_sums(z, K):
    """The two sums of the physical-root test -- notes p. 8 and p. 11."""
    z = np.asarray(z, float)
    K = np.asarray(K, float)
    return (z * K).sum(-1), (z / K).sum(-1)


def is_two_phase(z, K):
    """True where SUM z_i K_i > 1 and SUM z_i / K_i > 1 -- notes p. 8, p. 11."""
    s_zk, s_z_over_k = phase_sums(z, K)
    return (s_zk > 1.0) & (s_z_over_k > 1.0)


def phase_label(z, K):
    """'liquid' / 'two-phase' / 'vapour' from the table on notes p. 11."""
    s_zk, s_z_over_k = phase_sums(np.atleast_2d(z), np.atleast_2d(K))
    out = np.empty(s_zk.shape, dtype=object)
    out[:] = "two-phase"
    out[s_zk <= 1.0] = "liquid"
    out[s_z_over_k <= 1.0] = "vapour"
    return out if out.size > 1 else out.item()


def rachford_rice(FV, z, K):
    """h(F_V) = SUM z_i (K_i - 1) / [F_V (K_i - 1) + 1] -- notes p. 8.

    ``FV`` has shape ``(n,)`` or is a scalar; ``z`` and ``K`` have shape
    ``(n, n_c)``.  Returns shape ``(n,)``.
    """
    z = np.atleast_2d(np.asarray(z, float))
    K = np.atleast_2d(np.asarray(K, float))
    FV = np.asarray(FV, float).reshape(-1, 1)
    Km1 = K - 1.0
    return (z * Km1 / (FV * Km1 + 1.0)).sum(-1)


def solve_fv(z, K, tol=1.0e-12, max_iter=200):
    """Vectorised bisection for the Rachford-Rice root -- notes p. 8.

    ``h`` is monotonically decreasing in ``F_V`` on (0, 1) for positive
    ``K_i``, with ``h(0) = SUM z_i K_i - 1 > 0`` and
    ``h(1) = 1 - SUM z_i / K_i < 0`` exactly when the mixture passes the
    physical-root test, so a sign-change bracket always exists there.

    Returns ``(FV, n_iter)``.  Rows that are not two-phase return ``NaN``.
    """
    z = np.atleast_2d(np.asarray(z, float))
    K = np.atleast_2d(np.asarray(K, float))
    n = z.shape[0]

    ok = is_two_phase(z, K)
    lo = np.full(n, _EPS)
    hi = np.full(n, 1.0 - _EPS)

    it = 0
    for it in range(1, max_iter + 1):
        mid = 0.5 * (lo + hi)
        h = rachford_rice(mid, z, K)
        pos = h > 0.0            # root lies above mid (h decreasing)
        lo = np.where(pos, mid, lo)
        hi = np.where(pos, hi, mid)
        if np.all(hi - lo < tol):
            break

    FV = 0.5 * (lo + hi)
    FV[~ok] = np.nan
    return FV, it


def phase_compositions(FV, z, K):
    """x_i and y_i from F_V -- notes p. 7."""
    z = np.atleast_2d(np.asarray(z, float))
    K = np.atleast_2d(np.asarray(K, float))
    FV = np.asarray(FV, float).reshape(-1, 1)
    denom = FV * (K - 1.0) + 1.0
    x = z / denom
    y = z * K / denom
    return x, y


def wilson_saturation_pressures(z, T, Tc, pc, omega):
    """Bubble- and dew-point pressures with Wilson K-values -- notes pp. 12-13.

    Wilson's ``K_i`` is inversely proportional to ``p`` (p. 4), so
    ``SUM z_i K_i = p_b / p`` and ``SUM z_i / K_i = p / p_d`` with

        p_b = SUM z_i p_ci exp[5.37 (1 + w_i)(1 - T_ci/T)]          (F_V = 0, p. 12)
        p_d = 1 / SUM z_i / (p_ci exp[5.37 (1 + w_i)(1 - T_ci/T)])  (F_V = 1, p. 13)

    -- the same sums as the ``Bpi`` and ``Dpi`` columns of the p. 14 sheet.  The
    phase test of p. 8 is therefore exactly ``p_d < p < p_b``.  ``z`` is
    ``(n, n_c)``, ``T`` is ``(n,)`` or a scalar; returns ``(p_b, p_d)``, each
    ``(n,)``, in the units of ``pc``.
    """
    z = np.atleast_2d(np.asarray(z, float))
    T = np.asarray(T, float).reshape(-1, 1)
    a = np.asarray(pc, float) * np.exp(
        5.37 * (1.0 + np.asarray(omega, float)) * (1.0 - np.asarray(Tc, float) / T))
    return (z * a).sum(-1), 1.0 / (z / a).sum(-1)


def window_position(p, p_bubble, p_dew):
    """Position of ``p`` inside the two-phase pressure window, in log pressure.

        xi = ln(p / p_d) / ln(p_b / p_d)

    ``xi = 0`` at the dew point (``F_V = 1``) and ``xi = 1`` at the bubble point
    (``F_V = 0``); it increases with pressure, from the all-vapour end of the
    window to the all-liquid end.  Log pressure is used because Wilson ``K`` is
    proportional to ``1/p`` and the windows span decades (the p. 14 mixture's
    runs from 2.0 to 1590.9 psia).  Defined for ``p_d < p_b``.
    """
    p, pb, pd = (np.asarray(v, float) for v in (p, p_bubble, p_dew))
    return np.log(p / pd) / np.log(pb / pd)
