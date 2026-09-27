"""Single-phase, slightly-compressible finite-volume flow solver.

This is the *validated foundation* required before any two-phase work.  It
implements exactly the discretisation of ``1-Transmissibility.pdf`` p.17-20,

    T_{i+1/2} (P_{i+1} - P_i) + T_{i-1/2} (P_{i-1} - P_i) + Q_i
        = Cp_i (P_i - P_i^t),          Cp_i = Vp_i c_t / dt

with the *source-positive* sign convention (``Q_i > 0`` = fluid injected into
cell ``i``); the lecture uses a sink-positive ``q_i`` on the left-hand side, so
``Q_i = -q_i * V_i``.

Transmissibility (``1-Transmissibility.pdf`` p.18-19)::

    T = geometric_factor * k_harmonic * lambda,     lambda = 1/(mu B),  B = 1

with the geometric factor supplied by the grid object (linear ``A/dx`` or
radial ``2 pi h / ln(r2/r1)``).

Boundary conditions (``1-Transmissibility.pdf`` p.11-13)
--------------------------------------------------------
``"closed"``   Neumann, zero flux across ``r_e``.  Mass is conserved only if
               storage (compressibility) is present, so ``c_t > 0`` is
               enforced for this option -- an incompressible closed domain
               with injection has no solution.
``"constant_pressure"``  Dirichlet ``p = p_outer`` applied at the outer face
               through a half-cell transmissibility, giving the compensating
               outflow that an incompressible injection case requires.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


@dataclass
class SinglePhaseResult:
    t: np.ndarray            #: times [s], length nt
    p: np.ndarray            #: (nt, n) pressure [Pa]
    q_outer: np.ndarray      #: (nt,) volumetric outflow across r_e [m^3/s]
    mass_balance_error: float  #: relative volume-balance error [-]
    meta: dict = field(default_factory=dict)


def solve_single_phase(grid, k, phi, mu, c_t, p_init, *,
                       q_well=0.0, t_end=1.0, n_steps=50,
                       outer_bc="constant_pressure", p_outer=None,
                       store_every=1):
    """Fully-implicit single-phase pressure solve.

    Parameters
    ----------
    grid : RadialGrid or CartesianGrid1D
    k : (n,) absolute permeability [m^2]
    phi : (n,) porosity [-]
    mu : viscosity [Pa.s]
    c_t : total compressibility [1/Pa]  (rock + fluid)
    p_init : initial pressure [Pa] (scalar or array)
    q_well : volumetric injection rate into cell 0 [m^3/s], positive = injection
    t_end : end time [s]
    n_steps : number of uniform time steps
    outer_bc : ``"closed"`` | ``"constant_pressure"``
    p_outer : Dirichlet pressure [Pa]; defaults to ``p_init``
    """
    n = grid.n
    k = np.broadcast_to(np.asarray(k, float), (n,)).copy()
    phi = np.broadcast_to(np.asarray(phi, float), (n,)).copy()
    p = np.broadcast_to(np.asarray(p_init, float), (n,)).astype(float).copy()
    if outer_bc == "closed" and c_t <= 0:
        raise ValueError("a closed domain with injection requires c_t > 0 "
                         "(storage term); see docstring")
    p_outer = float(p_init if p_outer is None else p_outer) if np.isscalar(p_init) \
        else float(np.mean(p) if p_outer is None else p_outer)

    lam = 1.0 / mu
    T = grid.geom_factor() * grid.harmonic_k(k) * lam       # (n-1,)
    Vp = grid.bulk_volume * phi
    dt = t_end / n_steps

    # constant part of the matrix
    main = np.zeros(n)
    main[:-1] += T
    main[1:] += T
    T_out = 0.0
    if outer_bc == "constant_pressure":
        T_out = grid.outer_geom_factor() * k[-1] * lam
        main[-1] += T_out
    elif outer_bc != "closed":
        raise ValueError(f"unknown outer_bc {outer_bc!r}")

    C = Vp * c_t / dt
    A = sp.diags([-T, main + C, -T], [-1, 0, 1], format="csc")
    lu = spla.splu(A.tocsc())

    ts, ps, qouts = [0.0], [p.copy()], [0.0]
    cum_in = 0.0
    cum_out = 0.0
    for step in range(1, n_steps + 1):
        rhs = C * p
        rhs[0] += q_well
        if outer_bc == "constant_pressure":
            rhs[-1] += T_out * p_outer
        p = lu.solve(rhs)
        q_out = T_out * (p[-1] - p_outer) if outer_bc == "constant_pressure" else 0.0
        cum_in += q_well * dt
        cum_out += q_out * dt
        if step % store_every == 0 or step == n_steps:
            ts.append(step * dt)
            ps.append(p.copy())
            qouts.append(q_out)

    # volume balance: injected - produced = storage change
    dV_storage = float(np.sum(Vp * c_t * (ps[-1] - np.broadcast_to(
        np.asarray(p_init, float), (n,)))))
    imbalance = cum_in - cum_out - dV_storage
    scale = max(abs(cum_in), abs(cum_out), abs(dV_storage), 1e-30)
    return SinglePhaseResult(
        t=np.array(ts), p=np.array(ps), q_outer=np.array(qouts),
        mass_balance_error=float(abs(imbalance) / scale),
        meta={"dt": dt, "outer_bc": outer_bc, "p_outer": p_outer,
              "cum_in_m3": cum_in, "cum_out_m3": cum_out,
              "storage_change_m3": dV_storage, "T_outer": T_out},
    )


def steady_state_radial(grid, k, mu, q_well, p_outer):
    """Analytical steady-state radial pressure for a constant-rate injector in
    a homogeneous disc with a constant-pressure outer boundary::

        p(r) = p_e + (q mu) / (2 pi k h) * ln(r_e / r)

    Standard radial Darcy solution; consistent with the cylindrical flow
    equation of ``4-CO2 BL.pdf`` p.20.
    """
    return p_outer + q_well * mu / (2.0 * np.pi * k * grid.h) * \
        np.log(grid.r_e / grid.centres)


def tank_material_balance(t, q_well, Vp_total, c_t, p_init):
    """Closed-tank (pseudo-steady) average pressure::

        p_avg(t) = p_i + q t / (c_t Vp)

    Exact volume balance for a closed, slightly compressible domain.
    """
    return p_init + q_well * np.asarray(t, float) / (c_t * Vp_total)
