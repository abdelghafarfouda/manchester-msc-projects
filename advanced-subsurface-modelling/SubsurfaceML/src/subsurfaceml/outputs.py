"""Quantities of interest (QoIs) extracted from a simulation.

Everything here is computed **directly from conservation laws or from the
saturation field** -- never predicted by a machine-learning model.  The ML
surrogates predict these QoIs; they do not define them.

Definitions (all stated, all reported with units)
-------------------------------------------------
``dp_bh_max``
    Maximum bottom-hole pressure buildup while the well is injecting,
    ``max_t p_bh(t) - p_init`` [Pa], tracked at **every** time step, not only
    at report times.  ``p_bh`` is the single common well pressure of the
    coupled multi-layer well (``impes`` docstring).  ``p_bh`` comes from the radial well index
    (``3-IMPES.pdf`` p.16), so it is a well-model estimate; the near-well
    discretisation sensitivity is quantified in the convergence study.

``r_plume``
    Largest radius at which ``S_g > SG_THRESHOLD`` in **any** layer, using
    linear interpolation between cell centres.  The threshold is an explicit
    modelling choice (default 0.05) and is reported with every result.

``mass_injected``
    Time integral of the scheduled CO2 mass rate [kg].

``mass_retained``
    Free-phase CO2 mass **inside the modelled domain** at the final time [kg].
    This is *not* the same as the injected mass: CO2 that crosses the outer
    boundary at ``r_e`` has left the modelled storage compartment and is
    counted in ``mass_lost``.  ``mass_retained + mass_lost = mass_injected``
    to within the reported mass-balance error.

    ``mass_retained`` is free-phase mass only.  Dissolution and residual
    trapping are **not modelled** (see ``fluids`` A2 and the ``RelPerm``
    docstring), so this is a conservative measure of what stays in the
    compartment, not a claim about long-term secure storage.

Pressure limits
---------------
``p_limit`` is an **explicit scenario assumption** supplied by the
configuration.  Satisfying it demonstrates only that the *modelled* pressure
stays below a *stated* number.  It is not evidence about fracture gradients,
caprock integrity or leakage, none of which are modelled here.
"""
from __future__ import annotations

import numpy as np

SG_THRESHOLD_DEFAULT = 0.05


def plume_radius(centres: np.ndarray, Sg_layer: np.ndarray,
                 threshold: float = SG_THRESHOLD_DEFAULT) -> float:
    """Outermost radius where ``S_g`` exceeds ``threshold``, interpolated.

    Returns 0.0 if no cell exceeds the threshold.
    """
    Sg = np.asarray(Sg_layer, float)
    idx = np.where(Sg > threshold)[0]
    if idx.size == 0:
        return 0.0
    i = int(idx[-1])
    if i + 1 >= Sg.size or Sg[i + 1] >= Sg[i]:
        return float(centres[i])
    # linear interpolation of the threshold crossing
    w = (Sg[i] - threshold) / (Sg[i] - Sg[i + 1])
    return float(centres[i] + w * (centres[i + 1] - centres[i]))


def plume_radius_all_layers(grids, Sg, threshold=SG_THRESHOLD_DEFAULT):
    """Per-layer plume radii for one time level ``Sg`` of shape (L, n)."""
    return np.array([plume_radius(grids[l].centres, Sg[l], threshold)
                     for l in range(len(grids))])


def plume_radius_mass_fraction(grids, Vp: np.ndarray, Sg: np.ndarray,
                               fraction: float = 0.95) -> float:
    """Radius enclosing ``fraction`` of the free-phase CO2 currently in the
    domain, obtained by interpolating the cumulative radial mass profile.

    This is an **integral** measure and is far less grid sensitive than a
    saturation-contour radius: front smearing moves CO2 between adjacent cells
    but barely moves the cumulative distribution.  The convergence study
    (``results/metrics/validation.json``, ``V7``) reports both.

    Layers are pooled: cell volumes from every layer are sorted by outer face
    radius and the cumulative CO2 volume is accumulated over that ordering.
    """
    Vp = np.asarray(Vp, float)
    Sg = np.asarray(Sg, float)
    r_out = np.concatenate([g.faces[1:] for g in grids])
    m = (Vp * Sg).ravel()
    order = np.argsort(r_out, kind="stable")
    r_s, m_s = r_out[order], m[order]

    # Layers share the same face radii, so the pooled array holds each radius
    # once per layer.  Interpolating over it directly makes np.interp snap the
    # answer to a face radius whenever the target fraction falls inside one of
    # those repeated groups -- the radius then only ever takes grid values and
    # stops responding to changes in injected mass.  Aggregate the mass at each
    # distinct radius first, then interpolate.
    r_u, inv = np.unique(r_s, return_inverse=True)
    m_u = np.bincount(inv, weights=m_s, minlength=r_u.size)
    tot = m_u.sum()
    if tot <= 0:
        return 0.0
    cum = np.cumsum(m_u) / tot
    # prepend the inner boundary so a plume confined to the first cell still
    # interpolates instead of collapsing onto that cell's outer face
    r_ref = np.concatenate(([grids[0].faces[0]], r_u))
    cum = np.concatenate(([0.0], cum))
    keep = np.concatenate(([True], np.diff(cum) > 0))
    return float(np.interp(fraction, cum[keep], r_ref[keep]))


def sweep_efficiency(Vp: np.ndarray, Sg: np.ndarray,
                     threshold: float = SG_THRESHOLD_DEFAULT) -> float:
    """Fraction of the total pore volume with ``S_g > threshold``."""
    Vp = np.asarray(Vp, float)
    return float(np.sum(Vp[Sg > threshold]) / np.sum(Vp))


def storage_efficiency(Vp: np.ndarray, Sg: np.ndarray) -> float:
    """Pore-volume-weighted mean CO2 saturation of the whole domain."""
    Vp = np.asarray(Vp, float)
    return float(np.sum(Vp * Sg) / np.sum(Vp))


def extract_qois(result, grids, model, *, p_limit: float | None = None,
                 threshold: float = SG_THRESHOLD_DEFAULT) -> dict:
    """Scalar QoIs for one completed simulation.

    Parameters
    ----------
    result : ImpesResult
    grids : list of layer grids
    model : the TwoPhaseModel (for pore volumes and initial pressure)
    p_limit : optional pressure-limit assumption [Pa]
    threshold : plume saturation cut-off [-]
    """
    d = result.diagnostics
    p_init = model.p_init
    Sg_end = result.Sg[-1]
    r_layers = plume_radius_all_layers(grids, Sg_end, threshold)

    # plume radius history (per report time, max over layers)
    r_hist = np.array([np.max(plume_radius_all_layers(grids, S, threshold))
                       for S in result.Sg])

    mass_inj = float(result.mass_injected[-1])
    mass_ret = float(result.mass_in_place[-1])
    mass_lost = float(result.mass_out[-1])

    qoi = {
        "r_plume_m95_m": plume_radius_mass_fraction(grids, model.Vp, Sg_end, 0.95),
        "r_plume_m50_m": plume_radius_mass_fraction(grids, model.Vp, Sg_end, 0.50),
        "dp_bh_max_Pa": float(d["p_bh_max_Pa"] - p_init),
        "p_bh_max_Pa": float(d["p_bh_max_Pa"]),
        "dp_cell_max_Pa": float(d["p_cell_max_Pa"] - p_init),
        "r_plume_max_m": float(np.max(r_layers)),
        "r_plume_end_m": float(r_hist[-1]),
        "r_plume_layer_max_m": float(np.max(r_layers)),
        "r_plume_layer_min_m": float(np.min(r_layers)),
        "r_plume_spread_m": float(np.max(r_layers) - np.min(r_layers)),
        "mass_injected_kg": mass_inj,
        "mass_retained_kg": mass_ret,
        "mass_lost_kg": mass_lost,
        "retention_fraction": float(mass_ret / mass_inj) if mass_inj > 0 else np.nan,
        "sweep_efficiency": sweep_efficiency(model.Vp, Sg_end, threshold),
        "storage_efficiency": storage_efficiency(model.Vp, Sg_end),
        "sg_max": float(np.max(Sg_end)),
        "sg_min": float(np.min(Sg_end)),
        "mass_balance_error": float(result.mass_balance_error),
        "n_steps": int(d["n_steps"]),
        "n_clipped": int(d["n_clipped"]),
        "max_clip": float(d["max_clip"]),
        "wall_time_s": float(d["wall_time_s"]),
        "max_bhp_spread_Pa": float(d.get("max_bhp_spread_Pa", np.nan)),
        "max_rate_rel_err": float(d.get("max_rate_rel_err", np.nan)),
        "n_steps_layer_closed": int(d.get("n_steps_layer_closed", 0)),
        "sg_threshold": float(threshold),
    }
    if p_limit is not None:
        qoi["p_limit_Pa"] = float(p_limit)
        qoi["p_limit_exceeded"] = bool(d["p_bh_max_Pa"] > p_limit)
        qoi["p_margin_Pa"] = float(p_limit - d["p_bh_max_Pa"])
    qoi["_r_plume_history_m"] = r_hist
    return qoi


def time_series_frame(result, grids, model,
                      threshold: float = SG_THRESHOLD_DEFAULT):
    """Per-report-time trajectory table (plots, notebooks, interface).

    Columns: ``t_s``, ``q_kg_s``, ``dp_bh_Pa``, ``r_plume_m``,
    ``mass_in_place_kg``, ``mass_out_kg``, ``mean_sg``.
    """
    import pandas as pd
    rows = []
    for i, t in enumerate(result.t):
        Sg = result.Sg[i]
        rows.append({
            "t_s": float(t),
            "q_kg_s": float(result.q_mass[i]),
            "dp_bh_Pa": float(result.p_bh[i] - model.p_init),
            "r_plume_m": float(np.max(plume_radius_all_layers(grids, Sg, threshold))),
            "r_plume_m95_m": plume_radius_mass_fraction(grids, model.Vp, Sg, 0.95),
            "mass_in_place_kg": float(result.mass_in_place[i]),
            "mass_out_kg": float(result.mass_out[i]),
            "mean_sg": storage_efficiency(model.Vp, Sg),
        })
    return pd.DataFrame(rows)
