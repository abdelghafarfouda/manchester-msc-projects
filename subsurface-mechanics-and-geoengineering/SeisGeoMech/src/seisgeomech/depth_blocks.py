"""A two-direction depth-block test of the Gardner functional form.

The original refit (``analysis.stage_gardner``) fits ``rho = a Vp^b`` to all
1,105 paired samples and evaluates it on the same samples, so it describes
those samples and predicts nothing.  Paired sonic and density data exist only
in this one well, so independent validation on another well is not possible.
A within-well test is: fit on one contiguous depth block, predict the other.

Design (``configs/depth_blocks.json``, committed before any held-out score)
--------------------------------------------------------------------------
1. The paired samples, sorted by the logged coordinate ``DEPT_M``.
2. Two contiguous blocks either side of a 20 m exclusion gap centred on the
   midpoint of the overlap.  The gap follows an earlier estimate of how far the
   Gardner residuals stay correlated along the log.  It is a design choice,
   not a guarantee that the blocks are independent, and it is measured along
   the logged coordinate, whose vertical convention the file does not
   establish.
3. Fit the Gardner form on block A only, with the method of the original
   refit, and predict block B.
4. The reverse: fit on B, predict A.
5. Score the supplied relation, ``rho = 0.31 Vp^0.25``, on exactly the same
   evaluation samples.

Nothing here is tuned: one midpoint, one gap, two directions.  The fitting
method, units and functional form are those of the original refit, and no
other correlation is introduced.

The geomechanical consequence is reported with the same qualification as the
original gradients: the mean ``rho g`` gradient over each evaluation block is
an explicitly one-dimensional application of ``d(sigma_zz)/dz = rho g`` to the
logged coordinate, not a verified vertical stress gradient.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import seismic as sx
from . import stress as st
from .las_io import read_las
from .units import gcc_to_kg_m3

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "depth_blocks.json"
DEFAULT_OUT = PROJECT_ROOT / "results" / "depth_blocks"

BLOCKS = ("A", "B")


def load_config(path=None) -> dict:
    """The frozen design."""
    return json.loads(Path(path or DEFAULT_CONFIG).read_text())


# --------------------------------------------------------------------------
# Samples and split (no scoring)
# --------------------------------------------------------------------------

def paired_samples(log=None) -> pd.DataFrame:
    """The paired DT/RHOB samples, selected exactly as ``analysis.stage_load``.

    Adds ``las_row``, the 0-based row of each sample in the supplied LAS data
    section, so every sample can be identified without reference to a float
    depth.
    """
    log = log if log is not None else read_las()
    frame = log.frame
    ov = log.overlap(("DT", "RHOB"))
    ov = ov[np.isfinite(ov["VP"]) & (ov["VP"] > 0)].reset_index(drop=True)

    rows = np.searchsorted(frame["DEPT"].to_numpy(), ov["DEPT"].to_numpy())
    if not np.array_equal(frame["DEPT"].to_numpy()[rows], ov["DEPT"].to_numpy()):
        raise ValueError("could not locate every paired sample in the LAS file")
    out = pd.DataFrame(
        {
            "las_row": rows.astype(int),
            "dept_ft": ov["DEPT"].to_numpy(),
            "dept_m": ov["DEPT_M"].to_numpy(),
            "dt_us_per_ft": ov["DT"].to_numpy(),
            "vp_m_s": ov["VP"].to_numpy(),
            "rhob_gcc": ov["RHOB"].to_numpy(),
        }
    )
    if np.any(np.diff(out["dept_m"].to_numpy()) <= 0):
        raise ValueError("paired samples are not strictly increasing in depth")
    return out


def assign_blocks(depth_m, gap_m):
    """Label each sample ``A``, ``B`` or ``excluded``; return labels and midpoint.

    ``depth_m`` must be sorted.  The midpoint is that of the coordinate range,
    and a sample within ``gap_m / 2`` of it (inclusive) is excluded.
    """
    d = np.asarray(depth_m, dtype=float)
    if d.ndim != 1 or d.size < 3:
        raise ValueError("need a 1-D array of at least three depths")
    if np.any(np.diff(d) <= 0):
        raise ValueError("depths must be strictly increasing")
    if not gap_m >= 0:
        raise ValueError("gap must be non-negative")
    midpoint = 0.5 * (d[0] + d[-1])
    half = 0.5 * gap_m
    labels = np.where(np.abs(d - midpoint) <= half, "excluded",
                      np.where(d < midpoint, "A", "B"))
    return labels, float(midpoint)


def _ids_sha256(frame: pd.DataFrame) -> str:
    """SHA-256 of ``las_row:dept_ft`` lines, one per sample, in depth order."""
    text = "".join(f"{int(r)}:{repr(float(f))}\n"
                   for r, f in zip(frame["las_row"], frame["dept_ft"]))
    return hashlib.sha256(text.encode()).hexdigest()


def _describe(frame: pd.DataFrame) -> dict:
    return {
        "n": int(len(frame)),
        "las_row_first": int(frame["las_row"].iloc[0]),
        "las_row_last": int(frame["las_row"].iloc[-1]),
        "las_rows_contiguous": bool(np.all(np.diff(frame["las_row"].to_numpy()) == 1)),
        "dept_ft_first": float(frame["dept_ft"].iloc[0]),
        "dept_ft_last": float(frame["dept_ft"].iloc[-1]),
        "depth_m_first": float(frame["dept_m"].iloc[0]),
        "depth_m_last": float(frame["dept_m"].iloc[-1]),
        "thickness_m": float(frame["dept_m"].iloc[-1] - frame["dept_m"].iloc[0]),
        "vp_min_m_s": float(frame["vp_m_s"].min()),
        "vp_max_m_s": float(frame["vp_m_s"].max()),
        "vp_mean_m_s": float(frame["vp_m_s"].mean()),
        "ids_sha256": _ids_sha256(frame),
    }


def derive_split(samples: pd.DataFrame, config: dict | None = None):
    """Apply the frozen design.  Returns ``(record, labelled_samples)``.

    Only depths, identifiers and velocity ranges are described; no density is
    fitted or compared here.
    """
    config = config or load_config()
    expected = config["samples"]["expected_count"]
    if len(samples) != expected:
        raise ValueError(f"expected {expected} paired samples, found {len(samples)}")
    gap = float(config["split"]["exclusion_gap_m"])
    labels, midpoint = assign_blocks(samples["dept_m"].to_numpy(), gap)
    labelled = samples.assign(block=labels)

    parts = {name: labelled[labelled["block"] == name] for name in (*BLOCKS, "excluded")}
    record = {
        "config": "configs/depth_blocks.json",
        "n_paired_samples": int(len(samples)),
        "all_ids_sha256": _ids_sha256(samples),
        "overlap_top_m": float(samples["dept_m"].iloc[0]),
        "overlap_base_m": float(samples["dept_m"].iloc[-1]),
        "midpoint_m": midpoint,
        "exclusion_gap_m": gap,
        "excluded_interval_m": [midpoint - 0.5 * gap, midpoint + 0.5 * gap],
        "separation_last_A_to_first_B_m": float(
            parts["B"]["dept_m"].iloc[0] - parts["A"]["dept_m"].iloc[-1]
        ),
        "blocks": {name: _describe(frame) for name, frame in parts.items()},
    }
    return record, labelled


def write_split(out_dir=None, log=None, config=None):
    """Derive the split and write ``split.json`` and ``split_samples.csv``."""
    out = Path(out_dir) if out_dir is not None else DEFAULT_OUT
    out.mkdir(parents=True, exist_ok=True)
    record, labelled = derive_split(paired_samples(log), config)
    (out / "split.json").write_text(json.dumps(record, indent=2) + "\n")
    labelled[["las_row", "dept_ft", "dept_m", "block"]].to_csv(
        out / "split_samples.csv", index=False
    )
    return record, labelled


# --------------------------------------------------------------------------
# Fitting and scoring
# --------------------------------------------------------------------------

def fit_gardner_form(vp_m_s, rho_gcc):
    """Fit ``rho = a Vp^b`` exactly as the original refit does.

    ``numpy.polyfit(log Vp, log rho, 1)``: least squares in log space, slope
    ``b`` and ``a = exp(intercept)``.  Returns ``(a, b)``.
    """
    vp = np.asarray(vp_m_s, dtype=float)
    rho = np.asarray(rho_gcc, dtype=float)
    if vp.shape != rho.shape or vp.ndim != 1 or vp.size < 2:
        raise ValueError("need two 1-D arrays of equal length >= 2")
    b, log_a = np.polyfit(np.log(vp), np.log(rho), 1)
    return float(np.exp(log_a)), float(b)


def predict_gardner_form(vp_m_s, a, b):
    """``rho = a Vp^b`` in g/cm^3, Vp in m/s."""
    return a * np.asarray(vp_m_s, dtype=float) ** b


def error_stats(predicted, measured) -> dict:
    """Bias (mean of predicted - measured), RMSE and MAE, in g/cm^3."""
    p = np.asarray(predicted, dtype=float)
    m = np.asarray(measured, dtype=float)
    if p.shape != m.shape or p.size == 0:
        raise ValueError("predicted and measured must be non-empty and equal in shape")
    r = p - m
    return {
        "n": int(r.size),
        "bias_gcc": float(r.mean()),
        "bias_pct_of_mean_measured": float(100.0 * r.mean() / m.mean()),
        "rmse_gcc": float(np.sqrt((r ** 2).mean())),
        "mae_gcc": float(np.abs(r).mean()),
    }


def _gradients(depth_m, measured, supplied, fitted) -> dict:
    """Mean rho g gradient over one block with each density, in MPa/km."""
    g_m = st.mean_rho_g_gradient(depth_m, gcc_to_kg_m3(measured)) / 1e3
    g_s = st.mean_rho_g_gradient(depth_m, gcc_to_kg_m3(supplied)) / 1e3
    g_f = st.mean_rho_g_gradient(depth_m, gcc_to_kg_m3(fitted)) / 1e3
    return {
        "measured_MPa_per_km": float(g_m),
        "supplied_gardner_MPa_per_km": float(g_s),
        "block_fit_MPa_per_km": float(g_f),
        "supplied_minus_measured_MPa_per_km": float(g_s - g_m),
        "block_fit_minus_measured_MPa_per_km": float(g_f - g_m),
        "supplied_minus_measured_pct": float(100.0 * (g_s - g_m) / g_m),
        "block_fit_minus_measured_pct": float(100.0 * (g_f - g_m) / g_m),
    }


def score_direction(labelled: pd.DataFrame, fit_on: str, evaluate_on: str) -> dict:
    """Fit on one block, evaluate on the other, against the supplied relation."""
    train = labelled[labelled["block"] == fit_on]
    test = labelled[labelled["block"] == evaluate_on]
    if train.empty or test.empty or fit_on == evaluate_on:
        raise ValueError("need two different, non-empty blocks")

    a, b = fit_gardner_form(train["vp_m_s"], train["rhob_gcc"])
    vp_t, rho_t = test["vp_m_s"].to_numpy(), test["rhob_gcc"].to_numpy()
    fitted = predict_gardner_form(vp_t, a, b)
    supplied = sx.gardner_density_gcc(vp_t)

    held = error_stats(fitted, rho_t)
    ref = error_stats(supplied, rho_t)
    outside = (vp_t < train["vp_m_s"].min()) | (vp_t > train["vp_m_s"].max())
    return {
        "fit_on": fit_on,
        "evaluate_on": evaluate_on,
        "training": _describe(train),
        "evaluation": _describe(test),
        "evaluation_vp_outside_training_range_n": int(outside.sum()),
        "fitted_coefficient_a": a,
        "fitted_exponent_b": b,
        "training_fit": error_stats(
            predict_gardner_form(train["vp_m_s"], a, b), train["rhob_gcc"]
        ),
        "held_out_block_fit": held,
        "held_out_supplied_gardner": ref,
        "rmse_change_block_fit_vs_supplied_pct": float(
            100.0 * (held["rmse_gcc"] - ref["rmse_gcc"]) / ref["rmse_gcc"]
        ),
        "mae_change_block_fit_vs_supplied_pct": float(
            100.0 * (held["mae_gcc"] - ref["mae_gcc"]) / ref["mae_gcc"]
        ),
        "rho_g_gradient_over_evaluation_block": _gradients(
            test["dept_m"].to_numpy(), rho_t, supplied, fitted
        ),
    }


def residual_lag_correlation(depth_m, residual, lags_m) -> pd.DataFrame:
    """Pearson correlation of a residual series with itself shifted along the log.

    Context for the 20 m gap only; the gap was fixed before this was computed
    and is not chosen from it.  Assumes regular sampling, which the overlap has
    (0.2 m); the lag in samples is rounded from the median spacing.
    """
    d = np.asarray(depth_m, dtype=float)
    r = np.asarray(residual, dtype=float)
    step = float(np.median(np.diff(d)))
    rows = []
    for lag in lags_m:
        k = int(round(lag / step))
        if k < 1 or k >= r.size - 2:
            continue
        rows.append({"lag_m": float(lag), "lag_samples": k,
                     "pairs": int(r.size - k),
                     "correlation": float(np.corrcoef(r[:-k], r[k:])[0, 1])})
    return pd.DataFrame(rows)


LAG_CONTEXT_M = (0.2, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 40.0)


def score(labelled: pd.DataFrame, config: dict | None = None) -> dict:
    """Score both frozen directions and add the residual-correlation context."""
    config = config or load_config()
    directions = {
        d["name"]: score_direction(labelled, d["fit_on"], d["evaluate_on"])
        for d in config["directions"]
    }
    vp = labelled["vp_m_s"].to_numpy()
    rho = labelled["rhob_gcc"].to_numpy()
    a_all, b_all = fit_gardner_form(vp, rho)
    lags = pd.DataFrame({"lag_m": [float(x) for x in LAG_CONTEXT_M]})
    lag_supplied = residual_lag_correlation(
        labelled["dept_m"], sx.gardner_density_gcc(vp) - rho, LAG_CONTEXT_M
    )
    lag_refit = residual_lag_correlation(
        labelled["dept_m"], predict_gardner_form(vp, a_all, b_all) - rho, LAG_CONTEXT_M
    )
    lags = lags.merge(
        lag_supplied.rename(columns={"correlation": "supplied_gardner_residual"}),
        on="lag_m",
    ).merge(
        lag_refit[["lag_m", "correlation"]].rename(
            columns={"correlation": "in_sample_refit_residual"}
        ),
        on="lag_m",
    )
    return {
        "directions": directions,
        "in_sample_refit_all_samples": {"coefficient_a": a_all, "exponent_b": b_all},
        "residual_lag_correlation": lags.to_dict(orient="records"),
    }


def results_table(scores: dict) -> pd.DataFrame:
    """One row per direction and relation, on the evaluation block."""
    rows = []
    for name, d in scores["directions"].items():
        for relation, key, a, b in (
            ("block fit", "held_out_block_fit",
             d["fitted_coefficient_a"], d["fitted_exponent_b"]),
            ("supplied Gardner", "held_out_supplied_gardner",
             sx.GARDNER_COEFFICIENT, sx.GARDNER_EXPONENT),
        ):
            s = d[key]
            g = d["rho_g_gradient_over_evaluation_block"]
            rows.append({
                "direction": name,
                "fit_on": d["fit_on"],
                "evaluate_on": d["evaluate_on"],
                "relation": relation,
                "coefficient_a": a,
                "exponent_b": b,
                "evaluation_depth_m_first": d["evaluation"]["depth_m_first"],
                "evaluation_depth_m_last": d["evaluation"]["depth_m_last"],
                "n": s["n"],
                "bias_gcc": s["bias_gcc"],
                "rmse_gcc": s["rmse_gcc"],
                "mae_gcc": s["mae_gcc"],
                "rho_g_gradient_minus_measured_MPa_per_km": (
                    g["block_fit_minus_measured_MPa_per_km"] if relation == "block fit"
                    else g["supplied_minus_measured_MPa_per_km"]
                ),
            })
    return pd.DataFrame(rows)


def run(out_dir=None, log=None, config=None) -> dict:
    """Derive the frozen split, score it and write every output."""
    out = Path(out_dir) if out_dir is not None else DEFAULT_OUT
    config = config or load_config()
    split, labelled = write_split(out, log, config)
    scores = score(labelled, config)
    payload = {"config": "configs/depth_blocks.json", "split": "split.json", **scores}
    (out / "depth_block_results.json").write_text(json.dumps(payload, indent=2) + "\n")
    results_table(scores).to_csv(out / "depth_block_table.csv", index=False)
    pd.DataFrame(scores["residual_lag_correlation"]).to_csv(
        out / "residual_lag_correlation.csv", index=False
    )
    return {"split": split, "scores": scores, "labelled": labelled, "out_dir": out}
