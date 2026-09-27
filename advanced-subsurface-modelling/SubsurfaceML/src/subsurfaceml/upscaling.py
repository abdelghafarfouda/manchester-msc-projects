"""Analytical and numerical (flow-based) permeability upscaling.

Implements ``2-Upscaling.pdf``.  (An earlier version also encoded numbers
from an ``Upscaling practical session.pdf`` that is not among the supplied
materials; that case has been replaced by the lecture's own 2x2 layout.)

Analytical averages (``2-Upscaling.pdf`` pp.5-11)
-----------------------------------------------------
* arithmetic  ``K_A``   -- parallel flow (upper Wiener bound)
* harmonic    ``K_H``   -- series flow  (lower Wiener bound)
* ``K_HA``    -- harmonic within columns transverse to flow, then arithmetic
* ``K_AH``    -- arithmetic within rows along flow, then harmonic

with the bounding chain of p.14

    K_H <= K_HA <= K_mHA <= K* <= K_mAH <= K_AH <= K_A

Numerical (flow-based) upscaling (``2-Upscaling.pdf`` pp.15-16)
--------------------------------------------------------------------
A single-phase, incompressible pressure solve is run on the fine block with
a fixed pressure drop across the flow direction and **no-flow** on the
transverse faces (the "sealed side" boundary condition used in the lecture
figure).  The total outflow ``Q`` then defines

    K* = Q mu L / (A dP)

Two-phase relevance
-------------------
:func:`upscale_two_phase_check` re-runs the *two-phase* IMPES model on the
fine and coarse descriptions of the same layered unit and reports the error in
the two-phase quantities of interest.  Single-phase upscaling is not expected
to preserve them, and the demonstration quantifies by how much rather than
asserting that it does.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


# --------------------------------------------------------------------------
# analytical averages
# --------------------------------------------------------------------------
def k_arithmetic(k, weights=None):
    k = np.asarray(k, float).ravel()
    w = np.ones_like(k) if weights is None else np.asarray(weights, float).ravel()
    return float(np.sum(w * k) / np.sum(w))


def k_harmonic(k, weights=None):
    k = np.asarray(k, float).ravel()
    w = np.ones_like(k) if weights is None else np.asarray(weights, float).ravel()
    return float(np.sum(w) / np.sum(w / k))


def k_ha(k2d, axis=0):
    """Harmonic **along** the flow direction, then arithmetic across it.

    ``axis`` is the flow direction (0 = first index).  Assumes no cross-flow
    between the streamtubes, which is the lower of the two Cardwell-Parsons
    bounds.
    """
    k2d = np.asarray(k2d, float)
    along = 1.0 / np.mean(1.0 / k2d, axis=axis)     # harmonic along flow
    return float(np.mean(along))                     # arithmetic across


def k_ah(k2d, axis=0):
    """Arithmetic **across** the flow direction, then harmonic along it.

    Assumes perfect transverse pressure equilibration -- the upper bound.
    """
    k2d = np.asarray(k2d, float)
    across = np.mean(k2d, axis=1 - axis)             # arithmetic across flow
    return float(1.0 / np.mean(1.0 / across))        # harmonic along


def cardwell_parsons_bounds(k2d, axis=0):
    """Return ``(K_H, K_HA, K_AH, K_A)`` for the given flow direction."""
    return (k_harmonic(k2d), k_ha(k2d, axis), k_ah(k2d, axis), k_arithmetic(k2d))


# --------------------------------------------------------------------------
# flow-based (numerical) upscaling
# --------------------------------------------------------------------------
def _solve_pressure_2d(k, dx, dy, dz, mu, p_in, p_out, axis):
    """Incompressible single-phase pressure solve on a Cartesian block.

    Dirichlet ``p_in``/``p_out`` on the two faces normal to ``axis``; no-flow
    on the transverse faces.  Returns ``(p, T_ax, T_tr, T_in, T_out)``.
    """
    nx, ny = k.shape
    d = (dx, dy)[axis]
    dt_ = (dy, dx)[axis]
    A_face_along = dt_ * dz          # area of a face normal to the flow axis
    A_face_trans = d * dz

    # face transmissibilities: harmonic k * A / (mu * spacing)
    kx_f = 2 * k[:-1, :] * k[1:, :] / (k[:-1, :] + k[1:, :])
    ky_f = 2 * k[:, :-1] * k[:, 1:] / (k[:, :-1] + k[:, 1:])
    if axis == 0:
        Tx = kx_f * (dy * dz) / (mu * dx)
        Ty = ky_f * (dx * dz) / (mu * dy)
        T_bnd = 2.0 * k[[0, -1], :] * (dy * dz) / (mu * dx)
    else:
        Tx = kx_f * (dy * dz) / (mu * dx)
        Ty = ky_f * (dx * dz) / (mu * dy)
        T_bnd = 2.0 * k[:, [0, -1]] * (dx * dz) / (mu * dy)

    n = nx * ny
    idx = np.arange(n).reshape(nx, ny)
    rows, cols, vals = [], [], []
    diag = np.zeros((nx, ny))

    def add(i, j, v):
        rows.append(i); cols.append(j); vals.append(v)

    for a in range(nx - 1):
        for b in range(ny):
            t = Tx[a, b]
            add(idx[a, b], idx[a + 1, b], -t); add(idx[a + 1, b], idx[a, b], -t)
            diag[a, b] += t; diag[a + 1, b] += t
    for a in range(nx):
        for b in range(ny - 1):
            t = Ty[a, b]
            add(idx[a, b], idx[a, b + 1], -t); add(idx[a, b + 1], idx[a, b], -t)
            diag[a, b] += t; diag[a, b + 1] += t

    rhs = np.zeros((nx, ny))
    if axis == 0:
        diag[0, :] += T_bnd[0]; rhs[0, :] += T_bnd[0] * p_in
        diag[-1, :] += T_bnd[1]; rhs[-1, :] += T_bnd[1] * p_out
    else:
        diag[:, 0] += T_bnd[:, 0]; rhs[:, 0] += T_bnd[:, 0] * p_in
        diag[:, -1] += T_bnd[:, 1]; rhs[:, -1] += T_bnd[:, 1] * p_out

    for a in range(nx):
        for b in range(ny):
            add(idx[a, b], idx[a, b], diag[a, b])
    A = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsc()
    p = spla.spsolve(A, rhs.ravel()).reshape(nx, ny)
    return p, T_bnd


def upscale_flow_based(k, dx, dy, dz=1.0, mu=1.0, dp=1.0, axis=0):
    """Flow-based effective permeability of a fine Cartesian block.

    Parameters
    ----------
    k : (nx, ny) fine-scale permeability
    dx, dy, dz : fine cell dimensions
    mu : viscosity
    dp : applied pressure drop across the block
    axis : 0 for x-direction, 1 for y-direction

    Returns
    -------
    dict with ``k_eff``, ``Q``, ``L``, ``A``
    """
    k = np.asarray(k, float)
    nx, ny = k.shape
    p_in, p_out = dp, 0.0
    p, T_bnd = _solve_pressure_2d(k, dx, dy, dz, mu, p_in, p_out, axis)

    if axis == 0:
        Q = float(np.sum(T_bnd[0] * (p_in - p[0, :])))
        Q_out = float(np.sum(T_bnd[1] * (p[-1, :] - p_out)))
        L = nx * dx
        A = ny * dy * dz
    else:
        Q = float(np.sum(T_bnd[:, 0] * (p_in - p[:, 0])))
        Q_out = float(np.sum(T_bnd[:, 1] * (p[:, -1] - p_out)))
        L = ny * dy
        A = nx * dx * dz
    k_eff = Q * mu * L / (A * dp)
    return {"k_eff": k_eff, "Q_in": Q, "Q_out": Q_out, "L": L, "A": A,
            "flux_imbalance": abs(Q - Q_out) / max(abs(Q), 1e-30),
            "p": p}


def upscale_block_grid(k_fine, coarse_shape, dx, dy, dz=1.0, mu=1.0, axis=0):
    """Flow-based upscaling of every coarse block of a fine field.

    Returns a coarse permeability array of shape ``coarse_shape``.
    """
    k_fine = np.asarray(k_fine, float)
    nx, ny = k_fine.shape
    cx, cy = coarse_shape
    if nx % cx or ny % cy:
        raise ValueError("fine grid must divide evenly into coarse blocks")
    bx, by = nx // cx, ny // cy
    out = np.empty((cx, cy))
    for i in range(cx):
        for j in range(cy):
            blk = k_fine[i * bx:(i + 1) * bx, j * by:(j + 1) * by]
            out[i, j] = upscale_flow_based(blk, dx, dy, dz, mu, axis=axis)["k_eff"]
    return out


# --------------------------------------------------------------------------
# the lecture's 2x2 flow-based example
# --------------------------------------------------------------------------
def lecture_2x2_case(k_D=((1.0, 1.25), (2.0, 1.75))) -> dict:
    """The four-block flow-based upscaling set-up of ``2-Upscaling.pdf``
    p.15 (``K1..K4``, flow along x, sealed transverse faces).  The lecture
    gives no numerical values, so the default permeabilities [D] are
    illustrative choices of this project.  Returns ``K*`` with the analytical
    bounds of p.14 that it must satisfy."""
    k = np.asarray(k_D, float)            # rows = y (layers), cols = x (flow)
    res = upscale_flow_based(k.T, 0.5, 0.5, 1.0, 1.0, dp=1.0, axis=0)
    k_hastar = float(np.mean([k_harmonic(row) for row in k]))
    k_ahstar = float(k_harmonic([k_arithmetic(col) for col in k.T]))
    return {"k_eff_D": res["k_eff"], "K_H": k_harmonic(k), "K_HA": k_hastar,
            "K_AH": k_ahstar, "K_A": k_arithmetic(k),
            "flux_imbalance": res["flux_imbalance"]}
