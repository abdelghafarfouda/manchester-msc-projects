"""Analytical reduced-order model (ROM) of the bottom-hole pressure build-up.

Purpose
-------
A closed-form, simulator-free estimate of ``dp_bh_max`` for one reservoir
realisation and one (or many) injection schedules.  It is used

* as a **physics feature** for the pressure surrogate (it encodes how the
  realised layers share the injected volume, which the sampled prior
  parameters ``k_median`` / ``V_DP`` do not), and
* as a **baseline** the surrogate must beat (``docs/TECHNICAL_REPORT.md``).

It uses only quantities known before simulating: the realised layer
permeabilities, porosities and thicknesses of :func:`petrophysics.make_layered_rock`,
the compartment radius, the fluid compressibilities and the schedule.  It
never reads a simulator output, so it is a legitimate model input.

Model
-----
Each layer ``l`` is treated as a sealed, slightly compressible tank of pore
volume ``V_l = pi (r_e^2 - r_w^2) h_l phi_l`` connected to the common
wellbore by the pseudo-steady-state (PSS) productivity index of a bounded
circular drainage area,

.. math::

    q_l = J_l (p_w - \\bar p_l), \\qquad
    J_l = \\frac{2 \\pi k_l h_l \\lambda}{\\ln(r_e/r_w) - 3/4},

(the radial well equation of ``3-IMPES.pdf`` p.16 with the PSS average-
pressure form of the radial solution; the closed-tank material balance of
``single_phase.tank_material_balance`` gives ``V_l c_t d\\bar p_l/dt = q_l``).
All layers share one bottom-hole pressure ``p_w`` and
``sum_l q_l = Q(t)``, exactly as in the simulator's well model, including
the rule that an injector does not produce: a layer whose pressure exceeds
``p_w`` is closed (active set).  During shut-in every layer is closed.

The linear system is integrated with backward Euler on a fixed sub-step grid
(default 100 steps per schedule period; the layer time constants
``V_l c_t / J_l`` are months to years, so the time-discretisation error is
small and is tested in ``tests/test_rom.py``).

Simplifications (stated, not hidden)
------------------------------------
* single-phase brine mobility ``lambda = 1/mu_a`` everywhere, i.e. the
  lower-mobility CO2 front and the higher-mobility CO2-filled near-well
  region are not represented (``mobility="composite"`` adds a crude
  two-region correction);
* uniform pressure inside each layer apart from the PSS profile, so the
  early transient after a rate change is ignored;
* constant total compressibility ``c_r + c_a`` (CO2 compressibility enters
  only through the ``co2_storage`` correction of the storage term).

The ROM is therefore *not* a replacement for the simulator; the residual it
leaves is what the machine-learning surrogate is asked to learn.
"""
from __future__ import annotations

import numpy as np

from .units import YEAR, md_to_m2

#: Properties the simulator uses for every realisation (``fluids.py`` defaults).
MU_A = 6.0e-4        # brine viscosity [Pa s]
C_R = 4.5e-10        # rock compressibility [1/Pa]
C_A = 4.5e-10        # brine compressibility [1/Pa]
C_G = 1.0e-8         # CO2 compressibility [1/Pa]


def layer_properties(cfg, realisation) -> dict:
    """Realised layer permeability [m^2], porosity, thickness [m] and pore
    volume [m^3] of one realisation -- the same rock the simulator builds."""
    from .petrophysics import make_layered_rock
    r = realisation
    rock = make_layered_rock(cfg.grid.n_layers, cfg.grid.n_r, r.h_total_m,
                             md_to_m2(r.k_median_mD), r.V_DP, r.phi_mean,
                             seed=r.seed)
    k = rock.k[:, 0].copy()             # layers are homogeneous in r
    phi = rock.phi[:, 0].copy()
    h = rock.h.copy()
    Vp = np.pi * (r.r_e_m ** 2 - cfg.grid.r_w_m ** 2) * h * phi
    return {"k": k, "phi": phi, "h": h, "Vp": Vp}


def bhp_buildup(cfg, realisation, rates_kg_s, *, steps_per_period: int = 100,
                mobility: str = "brine", co2_storage: bool = True,
                return_history: bool = False):
    """Peak bottom-hole pressure build-up [Pa] predicted by the ROM for one
    realisation.

    Parameters
    ----------
    cfg : Config
    realisation : scenarios.Realisation
    rates_kg_s : array ``(n_periods,)`` or ``(n_schedules, n_periods)`` of
        injection-window rates [kg/s]; the post-injection shut-in is not
        needed because ``dp_bh_max`` is taken over injecting steps only.
    steps_per_period, mobility, co2_storage, return_history :
        see :func:`bhp_buildup_layers`.

    Returns
    -------
    ``dp_max`` with the leading shape of ``rates_kg_s`` (scalar for one
    schedule); with ``return_history`` also ``(t, p_w - p_i)``.
    """
    R = np.atleast_2d(np.asarray(rates_kg_s, float))
    if R.shape[1] != cfg.schedule.n_periods:
        raise ValueError(f"expected {cfg.schedule.n_periods} period rates, "
                         f"got {R.shape[1]}")
    r = realisation
    lay = layer_properties(cfg, r)
    lam_g = None
    if mobility == "composite":
        from .scenarios import fluid_models
        fl, rp = fluid_models(r)
        lam_g = rp.krg0 / fl.mu_g
    elif mobility != "brine":
        raise ValueError("mobility must be 'brine' or 'composite'")
    out = bhp_buildup_layers(
        lay["k"], lay["h"], lay["phi"], r.r_e_m, cfg.grid.r_w_m, r.rho_g, R,
        cfg.schedule.t_inject_years * YEAR / R.shape[1],
        steps_per_period=steps_per_period, lam_g=lam_g, S_ar=r.S_ar,
        co2_storage=co2_storage, return_history=return_history)
    if return_history:
        dp, hist = out
        return (dp[0] if np.ndim(rates_kg_s) == 1 else dp), hist
    return out[0] if np.ndim(rates_kg_s) == 1 else out


def bhp_buildup_layers(k, h, phi, r_e, r_w, rho_g, rates_kg_s, t_period, *,
                       steps_per_period: int = 100, lam_g: float | None = None,
                       S_ar: float = 0.2, co2_storage: bool = True,
                       mu_a: float = MU_A, c_r: float = C_R, c_a: float = C_A,
                       c_g: float = C_G, return_history: bool = False,
                       return_layer_volumes: bool = False):
    """Core of the ROM on explicit layer arrays.

    ``k`` [m^2], ``h`` [m], ``phi`` [-] have one entry per layer;
    ``rates_kg_s`` is ``(n_schedules, n_periods)``; ``t_period`` [s] is the
    length of one schedule period.  ``lam_g`` switches on the two-region
    (composite) mobility with the CO2 end-point mobility ``lam_g`` inside the
    volume-balance plume radius.  Returns ``dp_max`` of shape ``(n_schedules,)``
    [Pa] (and the history with ``return_history``; with
    ``return_layer_volumes`` the CO2 volume [m^3] each layer has received at
    the end of the injection window, shape ``(n_schedules, n_layers)``).
    """
    k, h, phi = (np.asarray(x, float) for x in (k, h, phi))
    R = np.atleast_2d(np.asarray(rates_kg_s, float))
    if np.any(R < 0) or not np.all(np.isfinite(R)):
        raise ValueError("rates must be finite and non-negative")
    n_s, n_p = R.shape
    Vp = np.pi * (r_e ** 2 - r_w ** 2) * h * phi
    ln_term = np.log(r_e / r_w) - 0.75
    J_brine = 2.0 * np.pi * k * h / (mu_a * ln_term)          # (L,)
    c_t = c_r + c_a
    dt = t_period / steps_per_period
    L = k.size
    p = np.zeros((n_s, L))            # layer pressure increments [Pa]
    v_co2 = np.zeros((n_s, L))        # cumulative CO2 volume per layer [m^3]
    dp_max = np.zeros(n_s)
    hist_t, hist_p = [0.0], [np.zeros(n_s)]
    t = 0.0
    for ip in range(n_p):
        Q = R[:, ip] / rho_g                                   # (n_s,) m^3/s
        for _ in range(steps_per_period):
            if lam_g is None:
                J = np.broadcast_to(J_brine, (n_s, L))
            else:
                sat = max(1.0 - S_ar, 1e-6)
                rp_l = np.sqrt(v_co2 / (np.pi * h * phi * sat) + r_w ** 2)
                rp_l = np.clip(rp_l, r_w, 0.9 * r_e)
                res = (np.log(rp_l / r_w) / lam_g
                       + (np.log(r_e / rp_l) - 0.75) * mu_a)
                J = 2.0 * np.pi * k * h / res
            stor = Vp * c_t
            if co2_storage:
                stor = stor + v_co2 * (c_g - c_a)
            C = stor / dt                                      # (n_s, L)
            a = J * C / (C + J)
            # backward Euler with one common BHP and injector-only layers:
            # q_l = a_l (p_w - p_l^n), sum_l q_l = Q  (active set)
            active = np.ones((n_s, L), bool)
            for _it in range(L + 1):
                aa = np.where(active, a, 0.0)
                pw = (Q + np.sum(aa * p, axis=1)) / np.maximum(aa.sum(axis=1), 1e-300)
                back = active & (p > pw[:, None])
                if not back.any():
                    break
                active &= ~back
            q_l = np.where(active, a * (pw[:, None] - p), 0.0)
            p = p + q_l * dt / stor
            v_co2 = v_co2 + q_l * dt
            t += dt
            dp_max = np.maximum(dp_max, np.where(Q > 0, pw, 0.0))
            if return_history:
                hist_t.append(t); hist_p.append(pw.copy())
    out = (dp_max,)
    if return_history:
        out += ((np.array(hist_t), np.array(hist_p)),)
    if return_layer_volumes:
        out += (v_co2,)
    return out if len(out) > 1 else dp_max


def physics_features(cfg, realisation, rates_matrix) -> dict:
    """ROM-derived surrogate inputs for a batch of schedules.

    ``rom_dp_bh_max_Pa``   the ROM peak build-up (:func:`bhp_buildup`)
    ``rom_r_fill_max_m``   largest layer radius that the CO2 volume the ROM
                           sends into that layer would fill at the end-point
                           saturation ``1 - S_ar`` (volume balance,
                           ``4-CO2 BL.pdf`` p.30) -- a layer-resolved version
                           of ``r_fill_est_m``
    ``rom_max_layer_share`` largest fraction of the injected CO2 volume taken
                           by one layer
    Each entry is an array of length ``n_schedules``.
    """
    R = np.atleast_2d(np.asarray(rates_matrix, float))
    r = realisation
    lay = layer_properties(cfg, r)
    dp, v = bhp_buildup_layers(
        lay["k"], lay["h"], lay["phi"], r.r_e_m, cfg.grid.r_w_m, r.rho_g, R,
        cfg.schedule.t_inject_years * YEAR / R.shape[1],
        return_layer_volumes=True)
    sat = max(1.0 - r.S_ar, 1e-6)
    r_fill = np.sqrt(v / (np.pi * lay["h"] * lay["phi"] * sat))
    tot = np.maximum(v.sum(axis=1), 1e-300)
    return {"rom_dp_bh_max_Pa": dp,
            "rom_r_fill_max_m": r_fill.max(axis=1),
            "rom_max_layer_share": v.max(axis=1) / tot}
