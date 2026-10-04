"""The ONE feature-generation path.

Every consumer -- dataset generation, model training and evaluation, the
schedule screening in :mod:`optimise`, and the Streamlit interface -- builds
surrogate inputs through the functions in this module, so a given reservoir
and schedule always produce identical model inputs.
``tests/test_features.py`` asserts this for every route.

Two steps
---------
:func:`raw_inputs`
    Quantities known *before* simulating: the sampled reservoir description,
    the layer summary of the generated rock, the per-period rates, the planned
    mass, the reference rate and the end-point mobility ratio.  The mobility
    ratio is computed from the **same** ``RelPerm``/``FluidProperties``
    objects the simulator uses (:func:`scenarios.fluid_models`), i.e. with the
    realisation's own ``krg0`` and ``mu_g``.  (An earlier optimiser path
    substituted a fixed ``krg0 = 0.4``, so its inputs differed from those the
    model was trained on.)
:func:`engineer`
    Physically motivated transforms of those raw columns (log transforms,
    flow capacity, pore volume, schedule-shape descriptors, dimensionless
    ratios) -- the feature-engineering step of
    ``E01_FeatureEngineering_Encoding (1).ipynb`` / ``Lecture03.ipynb``.

Nothing here uses a simulator output.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .units import YEAR

#: The 29 inputs of the published (2026-09-20) surrogates.  They describe
#: the reservoir through the parameters of its sampling prior (``k_median_mD``,
#: ``V_DP``) plus ``V_DP_layers``; kept unchanged so that the published
#: approach can be re-run and compared like for like.
FEATURES_BASELINE = [
    "log10_k_mD", "V_DP", "phi_mean", "h_total_m", "r_e_m",
    "n_g", "n_a", "krg0", "S_ar", "mu_g_cP", "rho_g",
    "log10_kh", "log10_pv", "q_mean_kg_s", "q_max_kg_s", "q_cv",
    "q_front_load", "q_ramp", "log10_mobility_ratio", "dimensionless_mass",
    "log10_injectivity", "aspect_ratio", "V_DP_layers",
    "planned_mass_kg", "log10_q_ref", "log10_q_mult_mean", "q_mult_max",
    "r_fill_est_m", "r_fill_over_re",
]

#: Summaries of the *realised* layers the simulator actually uses (the rock
#: is generated deterministically from the realisation's seed before any
#: simulation).  With only four layers per realisation the realised layers
#: can differ several-fold from what ``(k_median, V_DP)`` implies -- the cause
#: of the largest pressure errors of the published surrogate
#: (``docs/TECHNICAL_REPORT.md``).
ROCK_FEATURES = ["log10_k_arith_mD", "log10_k_harm_mD", "log10_k_min_layer_mD",
                 "log10_k_max_layer_mD", "log10_kh_realised"]

#: Outputs of the analytical reduced-order model (:mod:`rom`), a closed-form
#: function of the inputs above and the schedule -- no simulator output.
ROM_FEATURES = ["log10_rom_dp_MPa", "rom_r_fill_max_m", "rom_max_layer_share"]

#: Surrogate inputs: strictly quantities known before running the simulator.
FEATURES = FEATURES_BASELINE + ROCK_FEATURES + ROM_FEATURES

#: Raw columns :func:`engineer` needs (the realised-rock and ROM columns are
#: optional, so that tables written by the published version still engineer
#: the baseline features).
RAW_REQUIRED = ["k_median_mD", "V_DP", "phi_mean", "h_total_m", "r_e_m",
                "n_g", "n_a", "krg0", "S_ar", "mu_g_cP", "rho_g",
                "V_DP_layers", "q_ref_kg_s", "planned_mass_kg",
                "endpoint_mobility_ratio"]

class FeatureError(ValueError):
    """Raised with an informative message when inputs are unusable."""


def raw_inputs(cfg, realisation, rates_kg_s) -> dict:
    """Pre-simulation description of one (realisation, schedule) pair.

    ``rates_kg_s`` are the per-period rates of the injection window (length
    ``cfg.schedule.n_periods``).
    """
    from .petrophysics import make_layered_rock
    from .scenarios import fluid_models, reference_rate
    from .units import md_to_m2

    rates = np.asarray(rates_kg_s, float).ravel()
    n_p = cfg.schedule.n_periods
    if rates.size != n_p:
        raise FeatureError(f"expected {n_p} period rates, got {rates.size}")
    if not np.all(np.isfinite(rates)) or np.any(rates < 0):
        raise FeatureError("rates must be finite and non-negative")
    r = realisation
    rock = make_layered_rock(cfg.grid.n_layers, cfg.grid.n_r, r.h_total_m,
                             md_to_m2(r.k_median_mD), r.V_DP, r.phi_mean,
                             seed=r.seed)
    fl, rp = fluid_models(r)
    row = dict(r.to_dict())
    row.update(rock.summary())
    row["q_ref_kg_s"] = reference_rate(cfg, r)
    row["endpoint_mobility_ratio"] = rp.endpoint_mobility_ratio(fl)
    row.update(_rate_fields(cfg, rates))
    row.update(_rom_fields(cfg, r, rates[None, :], 0))
    return row


def _rom_fields(cfg, realisation, rates_matrix, i=None) -> dict:
    """ROM outputs for one schedule (``i``) or a batch (arrays)."""
    from .rom import physics_features
    ph = physics_features(cfg, realisation, rates_matrix)
    if i is None:
        return ph
    return {k: float(v[i]) for k, v in ph.items()}


def _rate_fields(cfg, rates) -> dict:
    """The schedule-dependent raw columns."""
    dt = cfg.schedule.t_inject_years * YEAR / cfg.schedule.n_periods
    out = {f"q{i+1}_kg_s": float(q) for i, q in enumerate(rates)}
    out["planned_mass_kg"] = float(np.sum(rates) * dt)
    return out


def _rate_columns(df: pd.DataFrame) -> list[str]:
    # exactly q1_kg_s .. qN_kg_s -- NOT q_ref_kg_s
    cols = [c for c in df.columns if re.fullmatch(r"q\d+_kg_s", c)]
    return sorted(cols, key=lambda c: int(re.findall(r"\d+", c)[0]))


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Add the engineered features (and unit-converted targets when the
    simulator outputs are present).  Pure function of the raw columns."""
    missing = [c for c in RAW_REQUIRED if c not in df.columns]
    qcols = _rate_columns(df)
    if missing or not qcols:
        raise FeatureError(
            "cannot build features; missing raw columns: "
            + ", ".join(missing + ([] if qcols else ["q1_kg_s..qN_kg_s"])))
    df = df.copy()
    Q = df[qcols].to_numpy(float)
    n_half = max(1, Q.shape[1] // 2)
    qsum = np.maximum(Q.sum(axis=1), 1e-12)
    qmean = np.maximum(Q.mean(axis=1), 1e-12)

    df["log10_k_mD"] = np.log10(df["k_median_mD"])
    df["kh_mD_m"] = df["k_median_mD"] * df["h_total_m"]
    df["log10_kh"] = np.log10(df["kh_mD_m"])
    df["pore_volume_m3"] = np.pi * df["r_e_m"] ** 2 * df["h_total_m"] * df["phi_mean"]
    df["log10_pv"] = np.log10(df["pore_volume_m3"])
    df["q_mean_kg_s"] = Q.mean(axis=1)
    df["q_max_kg_s"] = Q.max(axis=1)
    df["q_cv"] = Q.std(axis=1, ddof=0) / qmean
    df["q_front_load"] = Q[:, :n_half].sum(axis=1) / qsum
    df["q_ramp"] = (Q[:, -1] - Q[:, 0]) / qmean
    df["mobility_ratio"] = df["endpoint_mobility_ratio"]
    df["log10_mobility_ratio"] = np.log10(df["mobility_ratio"])
    df["dimensionless_mass"] = df["planned_mass_kg"] / (df["pore_volume_m3"] * df["rho_g"])
    df["injectivity_index"] = df["kh_mD_m"] / df["mu_g_cP"]
    df["log10_injectivity"] = np.log10(df["injectivity_index"])
    df["aspect_ratio"] = df["r_e_m"] / df["h_total_m"]
    df["log10_q_ref"] = np.log10(df["q_ref_kg_s"])
    df["q_mult_mean"] = df["q_mean_kg_s"] / df["q_ref_kg_s"]
    df["q_mult_max"] = df["q_max_kg_s"] / df["q_ref_kg_s"]
    df["log10_q_mult_mean"] = np.log10(np.maximum(df["q_mult_mean"], 1e-12))
    # radius the planned mass would fill at the end-point saturation (a
    # closed-form volume balance, 4-CO2 BL.pdf p.30: displaced brine volume =
    # injected CO2 volume)
    df["r_fill_est_m"] = np.sqrt(df["planned_mass_kg"] / (
        df["rho_g"] * np.pi * df["h_total_m"] * df["phi_mean"]
        * np.maximum(1.0 - df["S_ar"], 1e-6)))
    df["r_fill_over_re"] = df["r_fill_est_m"] / df["r_e_m"]

    # realised layers (present in every table written since 2026-10)
    md = 9.869233e-16
    for src, dst in (("k_arith_mean_m2", "log10_k_arith_mD"),
                     ("k_harm_mean_m2", "log10_k_harm_mD"),
                     ("k_min_layer_m2", "log10_k_min_layer_mD"),
                     ("k_max_layer_m2", "log10_k_max_layer_mD")):
        if src in df.columns:
            df[dst] = np.log10(df[src] / md)
    if "kh_total" in df.columns:
        df["log10_kh_realised"] = np.log10(df["kh_total"] / md)
    if "rom_dp_bh_max_Pa" in df.columns:
        df["log10_rom_dp_MPa"] = np.log10(np.maximum(df["rom_dp_bh_max_Pa"], 1.0) / 1e6)

    for src, dst, scale in (("dp_bh_max_Pa", "dp_bh_max_MPa", 1e6),
                            ("p_bh_max_Pa", "p_bh_max_MPa", 1e6),
                            ("mass_retained_kg", "mass_retained_Mt", 1e9),
                            ("mass_injected_kg", "mass_injected_Mt", 1e9),
                            ("mass_lost_kg", "mass_lost_Mt", 1e9)):
        if src in df.columns:
            df[dst] = df[src] / scale
    return df


def feature_matrix(df_raw: pd.DataFrame, features=None) -> pd.DataFrame:
    """Engineer and select the model inputs, validating the result."""
    features = list(features or FEATURES)
    out = engineer(df_raw)
    missing = [f for f in features if f not in out.columns]
    if missing:
        raise FeatureError(f"features not produced: {missing}")
    X = out[features]
    bad = [c for c in features if not np.all(np.isfinite(X[c].to_numpy(float)))]
    if bad:
        raise FeatureError(f"non-finite values in features: {bad}")
    return X


def features_for_schedules(cfg, realisation, rates_matrix, features=None
                           ) -> pd.DataFrame:
    """Model inputs for one realisation and a batch of candidate schedules
    (shape ``(n, n_periods)``) -- the path used by the optimiser and the
    interface."""
    R = np.atleast_2d(np.asarray(rates_matrix, float))
    base = raw_inputs(cfg, realisation, R[0])       # validates, builds rock once
    if not np.all(np.isfinite(R)) or np.any(R < 0):
        raise FeatureError("candidate rates must be finite and non-negative")
    ph = _rom_fields(cfg, realisation, R)           # one vectorised ROM call
    rows = []
    for i, q in enumerate(R):
        row = dict(base)
        row.update(_rate_fields(cfg, q))
        row.update({k: float(v[i]) for k, v in ph.items()})
        rows.append(row)
    return feature_matrix(pd.DataFrame(rows), features)
