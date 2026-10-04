"""The single analysis workflow: Gardner error and what it costs.

Question
--------
Over the interval of UK well 48/10b-9 where the sonic and density logs
overlap, how closely does the Gardner relation taught in the seismic practical
reproduce the measured bulk density, and what does the residual do to

  (a) the ``rho g`` gradient of the supplied overburden model, and
  (b) the zero-offset reflectivity and the synthetic trace built from it?

Stage 1  load the log, and record what it says about its own depth convention
Stage 2  test Gardner against the measured density
Stage 3  propagate the density error into the rho g gradient
Stage 4  propagate it into impedance, reflectivity and the synthetic trace
Stage 5  wavelet-frequency sweep on the reflectivity
Stage 6  elastic properties recoverable from the log, and what is not
Stage 7  verification against the two supplied worked examples

Two framing points that the reported numbers depend on:

* The logged coordinate is not shown to be vertical (see ``stress``), so the
  absolute gradient is reported as an explicitly one-dimensional application of
  the supplied model.  The measured-versus-Gardner *difference* is invariant
  under uniform scaling of that coordinate, because both profiles are
  integrated over the same array; it is **not** shown to be invariant under a
  depth-dependent trajectory correction, and none is applied.
* The reflection-coefficient RMS ratio and the synthetic-trace RMS ratio are
  different quantities and are reported separately.  The first is computed
  before wavelet convolution; the second after it, and they do not agree.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import elasticity as el
from . import seismic as sx
from . import stress as st
from . import worked_examples as wx
from .las_io import read_las
from .units import STANDARD_GRAVITY, gcc_to_kg_m3

#: Wavelet frequencies used in the wavelet-frequency experiment, in Hz.
#: These are an experimental sweep, not a property of the subsurface.
#: Ex1.ipynb cell 23 demonstrates the wavelet at 100 Hz and cells 28/30 ask the
#: student to "try different frequencies"; the sweep brackets that value.
#: The lower end is set by the length of the logged interval, not by choice:
#: the overlap is only about 97 ms of two-way time, so a Ricker wavelet below
#: roughly 20 Hz is longer than the whole trace.  See
#: ``seismic.min_supportable_frequency``.
RESOLUTION_FREQUENCIES_HZ = (25.0, 30.0, 40.0, 50.0, 75.0, 100.0, 150.0, 200.0, 300.0)

#: Poisson's ratio values at which the uniaxial-strain stress ratio is
#: tabulated.  A plain sweep of the admissible range for an isotropic solid;
#: no value is claimed for this well, which has no shear sonic log.
NU_SWEEP = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Results:
    """Everything the workflow produces."""

    tables: dict = field(default_factory=dict)
    scalars: dict = field(default_factory=dict)
    arrays: dict = field(default_factory=dict)

    def write(self, root: Path | None = None) -> Path:
        root = Path(root) if root is not None else PROJECT_ROOT / "results"
        (root / "tables").mkdir(parents=True, exist_ok=True)
        for name, frame in self.tables.items():
            frame.to_csv(root / "tables" / f"{name}.csv", index=False)
        payload = {
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "scalars": self.scalars,
        }
        (root / "tables" / "summary.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )
        return root


# --------------------------------------------------------------------------
# Stage 1
# --------------------------------------------------------------------------

def stage_load(res: Results, las_path=None):
    log = read_las(las_path)
    res.tables["las_curve_coverage"] = log.coverage()
    res.tables["depth_convention_evidence"] = log.depth_convention_evidence()

    overlap = log.overlap(("DT", "RHOB"))
    overlap = overlap[np.isfinite(overlap["VP"]) & (overlap["VP"] > 0)]
    overlap = overlap.reset_index(drop=True)

    res.scalars["las_n_samples"] = int(log.n_samples)
    res.scalars["las_null_value"] = float(log.null_value)
    res.scalars["overlap_n"] = int(len(overlap))
    res.scalars["overlap_top_m"] = float(overlap["DEPT_M"].min())
    res.scalars["overlap_base_m"] = float(overlap["DEPT_M"].max())
    res.scalars["overlap_thickness_m"] = (
        res.scalars["overlap_base_m"] - res.scalars["overlap_top_m"]
    )

    dt_valid = log.frame["DT"].notna()
    res.scalars["dt_top_m"] = float(log.frame.loc[dt_valid, "DEPT_M"].min())
    res.scalars["dt_base_m"] = float(log.frame.loc[dt_valid, "DEPT_M"].max())
    res.scalars["dt_thickness_m"] = res.scalars["dt_base_m"] - res.scalars["dt_top_m"]

    # Hole-condition curves are reported, not used as filters.
    res.tables["hole_condition"] = pd.DataFrame(
        [
            ("CALI_in", float(overlap["CALI"].min()), float(overlap["CALI"].mean()),
             float(overlap["CALI"].max())),
            ("DRHO_gcc", float(overlap["DRHO"].min()), float(overlap["DRHO"].mean()),
             float(overlap["DRHO"].max())),
        ],
        columns=["curve", "min", "mean", "max"],
    )

    # The file establishes no vertical-depth convention: no TVD or deviation
    # curve, LMF UNKNOWN, every elevation field 0.0.  Recorded as a scalar so
    # downstream reporting cannot quietly assume otherwise.
    evidence = res.tables["depth_convention_evidence"]
    res.scalars["vertical_depth_established"] = bool(
        (evidence["value"] == "present").any()
    )

    res.arrays["overlap"] = overlap
    res.arrays["log"] = log
    return overlap


# --------------------------------------------------------------------------
# Stage 2
# --------------------------------------------------------------------------

def stage_gardner(res: Results):
    ov = res.arrays["overlap"]
    vp = ov["VP"].to_numpy()
    rhob = ov["RHOB"].to_numpy()                 # g/cm^3, measured
    rho_g = sx.gardner_density_gcc(vp)           # g/cm^3, Gardner

    residual = rho_g - rhob
    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum((rhob - rhob.mean()) ** 2))

    res.scalars["vp_mean_m_s"] = float(vp.mean())
    res.scalars["vp_min_m_s"] = float(vp.min())
    res.scalars["vp_max_m_s"] = float(vp.max())
    res.scalars["rhob_mean_gcc"] = float(rhob.mean())
    res.scalars["gardner_mean_gcc"] = float(rho_g.mean())
    res.scalars["gardner_bias_gcc"] = float(residual.mean())
    res.scalars["gardner_bias_pct"] = float(100.0 * residual.mean() / rhob.mean())
    res.scalars["gardner_rmse_gcc"] = float(np.sqrt((residual ** 2).mean()))
    res.scalars["gardner_sd_gcc"] = float(residual.std(ddof=1))
    res.scalars["gardner_r2"] = float(1.0 - ss_res / ss_tot)
    res.scalars["corr_vp_rhob"] = float(np.corrcoef(vp, rhob)[0, 1])

    # Refit the same functional form to these measured points.  This is an
    # IN-SAMPLE CALIBRATION: no samples were held out; the coefficients were
    # fitted and evaluated on the same 1105 paired samples.  The reduction in
    # RMSE is therefore a descriptive fit statistic on the calibration data
    # and is not evidence of predictive skill elsewhere, nor evidence that the
    # functional form is correct and only its coefficients wrong.  The refit is
    # not used anywhere else in the workflow.
    coef_b, log_a = np.polyfit(np.log(vp), np.log(rhob), 1)
    coef_a = float(np.exp(log_a))
    rho_fit = coef_a * vp ** coef_b
    res.scalars["gardner_refit_coefficient"] = coef_a
    res.scalars["gardner_refit_exponent"] = float(coef_b)
    res.scalars["gardner_refit_rmse_gcc"] = float(
        np.sqrt(((rho_fit - rhob) ** 2).mean())
    )
    res.scalars["gardner_refit_bias_gcc"] = float((rho_fit - rhob).mean())
    res.scalars["gardner_refit_is_in_sample"] = True
    res.scalars["gardner_refit_n_calibration_samples"] = int(vp.size)
    res.scalars["gardner_refit_rmse_reduction_pct_in_sample"] = float(
        100.0
        * (res.scalars["gardner_rmse_gcc"] - res.scalars["gardner_refit_rmse_gcc"])
        / res.scalars["gardner_rmse_gcc"]
    )

    res.tables["gardner_vs_rhob"] = pd.DataFrame(
        {
            "depth_m": ov["DEPT_M"],
            "vp_m_s": vp,
            "rhob_measured_gcc": rhob,
            "rho_gardner_gcc": rho_g,
            "residual_gcc": residual,
            "rho_refit_gcc": rho_fit,
        }
    )
    res.arrays["vp"] = vp
    res.arrays["rho_measured_si"] = gcc_to_kg_m3(rhob)
    res.arrays["rho_gardner_si"] = gcc_to_kg_m3(rho_g)


# --------------------------------------------------------------------------
# Stage 3
# --------------------------------------------------------------------------

def stage_rho_g_gradient(res: Results):
    """Integrate rho g along the logged coordinate, measured vs Gardner density.

    The absolute gradient is an explicitly one-dimensional application of the
    supplied overburden model (Models Q5) to a coordinate the file does not
    establish as vertical.

    The measured-versus-Gardner difference is invariant under uniform scaling of
    that coordinate: both profiles are integrated over the same array, so one
    constant factor applied to every interval cancels from their ratio.  It is
    not shown to be invariant under a depth-dependent trajectory correction,
    which would reweight the two integrals sample by sample; no such correction
    is applied.
    """
    ov = res.arrays["overlap"]
    s = ov["DEPT_M"].to_numpy()          # logged depth coordinate, metres
    rho_m = res.arrays["rho_measured_si"]
    rho_g = res.arrays["rho_gardner_si"]

    grad_m = st.mean_rho_g_gradient(s, rho_m)
    grad_g = st.mean_rho_g_gradient(s, rho_g)

    res.scalars["rho_g_gradient_measured_Pa_per_m"] = float(grad_m)
    res.scalars["rho_g_gradient_gardner_Pa_per_m"] = float(grad_g)
    res.scalars["rho_g_gradient_measured_MPa_per_km"] = float(grad_m / 1e3)
    res.scalars["rho_g_gradient_gardner_MPa_per_km"] = float(grad_g / 1e3)
    res.scalars["rho_g_gradient_difference_MPa_per_km"] = float((grad_g - grad_m) / 1e3)
    res.scalars["rho_g_gradient_difference_pct"] = float(
        100.0 * (grad_g - grad_m) / grad_m
    )
    # Invariant under uniform scaling of the logged coordinate (both profiles
    # share one array, so a single constant factor cancels from the ratio).
    # NOT established for a depth-dependent trajectory correction.
    res.scalars["rho_g_gradient_difference_invariant_under_uniform_scaling"] = True

    inc_m = st.rho_g_integral(s, rho_m)
    inc_g = st.rho_g_integral(s, rho_g)
    res.scalars["rho_g_integral_measured_MPa"] = float(inc_m[-1] / 1e6)
    res.scalars["rho_g_integral_gardner_MPa"] = float(inc_g[-1] / 1e6)

    res.tables["rho_g_profile"] = pd.DataFrame(
        {
            "logged_depth_m": s,
            "rho_g_integral_measured_MPa": inc_m / 1e6,
            "rho_g_integral_gardner_MPa": inc_g / 1e6,
            "difference_MPa": (inc_g - inc_m) / 1e6,
        }
    )

    # Uniaxial-strain horizontal/vertical ratio, purely parametric in nu.
    res.tables["uniaxial_strain_stress_ratio"] = pd.DataFrame(
        {
            "poissons_ratio": list(NU_SWEEP),
            "sigma_h_over_sigma_v": [
                float(st.horizontal_over_vertical_ratio(n)) for n in NU_SWEEP
            ],
        }
    )
    res.arrays["rho_g_integral_measured"] = inc_m
    res.arrays["rho_g_integral_gardner"] = inc_g


# --------------------------------------------------------------------------
# Stage 4
# --------------------------------------------------------------------------

def stage_reflectivity(res: Results):
    ov = res.arrays["overlap"]
    vp = res.arrays["vp"]
    z_m = ov["DEPT_M"].to_numpy()

    imp_m = sx.acoustic_impedance(res.arrays["rho_measured_si"], vp)
    imp_g = sx.acoustic_impedance(res.arrays["rho_gardner_si"], vp)

    rc_m = sx.reflection_coefficients(imp_m)
    rc_g = sx.reflection_coefficients(imp_g)

    res.scalars["impedance_measured_mean"] = float(imp_m.mean())
    res.scalars["impedance_gardner_mean"] = float(imp_g.mean())
    res.scalars["rc_measured_rms"] = float(np.sqrt((rc_m ** 2).mean()))
    res.scalars["rc_gardner_rms"] = float(np.sqrt((rc_g ** 2).mean()))
    # NOTE: this ratio is computed on the reflection coefficients BEFORE any
    # wavelet convolution.  It is not a synthetic-trace amplitude ratio; that
    # is reported separately, per frequency, in stage_resolution.
    res.scalars["rc_rms_ratio_gardner_over_measured_pre_convolution"] = float(
        np.sqrt((rc_g ** 2).mean()) / np.sqrt((rc_m ** 2).mean())
    )
    res.scalars["rc_max_abs_measured"] = float(np.abs(rc_m).max())
    res.scalars["rc_max_abs_gardner"] = float(np.abs(rc_g).max())
    res.scalars["rc_correlation"] = float(np.corrcoef(rc_m, rc_g)[0, 1])
    res.scalars["rc_rms_difference"] = float(np.sqrt(((rc_m - rc_g) ** 2).mean()))

    res.tables["impedance_and_reflectivity"] = pd.DataFrame(
        {
            "depth_m": z_m[:-1],
            "impedance_measured": imp_m[:-1],
            "impedance_gardner": imp_g[:-1],
            "rc_measured": rc_m,
            "rc_gardner": rc_g,
        }
    )

    # Travel time along the logged path, from the sonic log itself, then a
    # regular time axis.  Not vertical two-way time, and not a seismic tie.
    twt = sx.two_way_travel_time_along_log(z_m, ov["DT"].to_numpy())
    res.scalars["interval_two_way_travel_time_along_log_ms"] = float(twt[-1] * 1e3)

    dt_native = float(np.median(np.diff(twt)))
    dt_regular = 1e-4  # 0.1 ms; coarser than the native log sampling below
    res.scalars["travel_time_native_median_step_s"] = dt_native
    res.scalars["travel_time_regular_step_s"] = dt_regular

    t_mid = 0.5 * (twt[1:] + twt[:-1])
    t_reg, rc_m_reg = sx.resample_to_regular_time(t_mid, rc_m, dt_regular)
    _, rc_g_reg = sx.resample_to_regular_time(t_mid, rc_g, dt_regular)

    res.arrays["t_reg"] = t_reg
    res.arrays["rc_measured_time"] = rc_m_reg
    res.arrays["rc_gardner_time"] = rc_g_reg
    res.arrays["rc_measured_depth"] = rc_m
    res.arrays["rc_gardner_depth"] = rc_g
    res.arrays["impedance_measured"] = imp_m
    res.arrays["impedance_gardner"] = imp_g
    res.arrays["twt"] = twt


# --------------------------------------------------------------------------
# Stage 5
# --------------------------------------------------------------------------

def stage_resolution(res: Results):
    """Ricker frequency sweep: similarity and amplitude of the synthetic trace.

    Two distinct quantities are reported per frequency and must not be
    conflated with the pre-convolution reflection-coefficient ratio:

    * the Pearson correlation between the synthetic trace and its input
      reflectivity -- a similarity measure for this synthetic experiment only,
      not a fraction of anything recovered;
    * the RMS amplitude of the synthetic trace itself, and the Gardner/measured
      ratio of those amplitudes after convolution.
    """
    dt_s = res.scalars["travel_time_regular_step_s"]
    rc_m = res.arrays["rc_measured_time"]
    rc_g = res.arrays["rc_gardner_time"]

    res.scalars["min_supportable_frequency_Hz"] = float(
        sx.min_supportable_frequency(rc_m.size, dt_s)
    )
    res.scalars["trace_n_samples_time"] = int(rc_m.size)

    sweep_m = sx.resolution_sweep(rc_m, dt_s, RESOLUTION_FREQUENCIES_HZ)
    sweep_g = sx.resolution_sweep(rc_g, dt_s, RESOLUTION_FREQUENCIES_HZ)

    rows = []
    for (f, c_m, n, rms_m), (_, c_g, _, rms_g) in zip(sweep_m, sweep_g):
        w, _ = sx.ricker_wavelet(duration=2.0 / f, dt=dt_s, frequency=f)
        syn_m = sx.convolve_trace(rc_m, w)
        syn_g = sx.convolve_trace(rc_g, w)
        rows.append(
            {
                "frequency_Hz": f,
                "wavelet_samples": n,
                "corr_synthetic_vs_reflectivity_measured": c_m,
                "corr_synthetic_vs_reflectivity_gardner": c_g,
                "synthetic_rms_measured": rms_m,
                "synthetic_rms_gardner": rms_g,
                "synthetic_rms_ratio_gardner_over_measured": rms_g / rms_m,
                "corr_synthetic_measured_vs_gardner": float(
                    np.corrcoef(syn_m, syn_g)[0, 1]
                ),
            }
        )
    sweep = pd.DataFrame(rows)
    res.tables["resolution_sweep"] = sweep

    ratio = sweep["synthetic_rms_ratio_gardner_over_measured"]
    res.scalars["synthetic_rms_ratio_min"] = float(ratio.min())
    res.scalars["synthetic_rms_ratio_max"] = float(ratio.max())
    res.scalars["synthetic_rms_ratio_all_below_one"] = bool((ratio < 1.0).all())

    # A spike input must return the reflectivity unchanged: identity check.
    spike = np.zeros(21)
    spike[10] = 1.0
    identity = sx.convolve_trace(rc_m, spike)
    res.scalars["spike_convolution_max_error"] = float(np.abs(identity - rc_m).max())

    # Store the synthetic traces at the highest and lowest sweep frequency.
    for label, f in (("low", RESOLUTION_FREQUENCIES_HZ[0]),
                     ("high", RESOLUTION_FREQUENCIES_HZ[-1])):
        w, _ = sx.ricker_wavelet(duration=2.0 / f, dt=dt_s, frequency=f)
        res.arrays[f"synthetic_{label}_measured"] = sx.convolve_trace(rc_m, w)
        res.arrays[f"synthetic_{label}_gardner"] = sx.convolve_trace(rc_g, w)
        res.arrays[f"synthetic_{label}_frequency"] = f


# --------------------------------------------------------------------------
# Stage 6
# --------------------------------------------------------------------------

def stage_elastic_profile(res: Results):
    ov = res.arrays["overlap"]
    vp = res.arrays["vp"]
    rho = res.arrays["rho_measured_si"]

    M = el.p_wave_modulus_from_vp_rho(vp, rho)
    res.scalars["p_wave_modulus_mean_GPa"] = float(M.mean() / 1e9)
    res.scalars["p_wave_modulus_min_GPa"] = float(M.min() / 1e9)
    res.scalars["p_wave_modulus_max_GPa"] = float(M.max() / 1e9)

    res.tables["p_wave_modulus_profile"] = pd.DataFrame(
        {
            "logged_depth_m": ov["DEPT_M"],
            "vp_m_s": vp,
            "rho_kg_m3": rho,
            "p_wave_modulus_GPa": M / 1e9,
        }
    )
    res.arrays["p_wave_modulus"] = M


# --------------------------------------------------------------------------
# Stage 7
# --------------------------------------------------------------------------

def stage_verification(res: Results):
    ol = wx.solve_olivine()
    mv = wx.solve_merivale()

    res.tables["worked_example_olivine"] = pd.DataFrame(
        [
            ("K_Voigt_GPa", ol.K_voigt_GPa),
            ("K_Reuss_GPa", ol.K_reuss_GPa),
            ("K_VRH_GPa", ol.K_hill_GPa),
            ("G_Voigt_GPa", ol.G_voigt_GPa),
            ("G_Reuss_GPa", ol.G_reuss_GPa),
            ("G_VRH_GPa", ol.G_hill_GPa),
            ("rho_kg_m3", ol.rho_kg_m3),
            ("Vp_m_s", ol.vp_m_s),
            ("Vs_m_s", ol.vs_m_s),
            ("Vp_over_Vs", ol.vp_vs),
        ],
        columns=["quantity", "value"],
    )

    res.tables["worked_example_merivale"] = pd.DataFrame(
        [
            ("n_points_in_fit", float(mv.n_points)),
            ("window_low_MPa", mv.stress_window_MPa[0]),
            ("window_high_MPa", mv.stress_window_MPa[1]),
            ("peak_stress_MPa", mv.peak_stress_MPa),
            ("E_GPa", mv.E_Pa / 1e9),
            ("E_fit_r2", mv.E_r2),
            ("poissons_ratio", mv.nu),
            ("nu_fit_r2", mv.nu_r2),
            ("K_GPa", mv.K_Pa / 1e9),
            ("G_GPa", mv.G_Pa / 1e9),
            ("rho_kg_m3", mv.rho_kg_m3),
            ("Vp_m_s", mv.vp_m_s),
            ("Vs_m_s", mv.vs_m_s),
            ("Vp_over_Vs", mv.vp_vs),
        ],
        columns=["quantity", "value"],
    )

    # Window sensitivity for the Merivale fit.
    rows = []
    for hi in (0.4, 0.5, 0.6):
        m = wx.solve_merivale(stress_max_MPa=hi * mv.peak_stress_MPa)
        rows.append(
            {
                "window_high_fraction_of_peak": hi,
                "n_points": m.n_points,
                "E_GPa": m.E_Pa / 1e9,
                "poissons_ratio": m.nu,
                "Vp_m_s": m.vp_m_s,
                "Vs_m_s": m.vs_m_s,
            }
        )
    res.tables["merivale_window_sensitivity"] = pd.DataFrame(rows)

    # Round-trip identities from the conversion table (1_Elasticity.pdf p.10).
    E_rt = float(el.youngs_modulus_from_K_G(mv.K_Pa, mv.G_Pa))
    nu_rt = float(el.poissons_ratio_from_K_G(mv.K_Pa, mv.G_Pa))
    res.scalars["identity_E_roundtrip_rel_error"] = abs(E_rt - mv.E_Pa) / mv.E_Pa
    res.scalars["identity_nu_roundtrip_abs_error"] = abs(nu_rt - mv.nu)
    res.scalars["identity_E_eq_2G1plusnu_rel_error"] = abs(
        2.0 * mv.G_Pa * (1.0 + mv.nu) - mv.E_Pa
    ) / mv.E_Pa

    # Volumetric strain must equal eps_axial + 2 eps_circumferential.
    res.scalars["merivale_volumetric_strain_max_residual"] = float(
        np.abs(
            wx.MERIVALE_VOLUMETRIC_STRAIN
            - (wx.MERIVALE_AXIAL_STRAIN + 2.0 * wx.MERIVALE_CIRCUMFERENTIAL_STRAIN)
        ).max()
    )

    # Voigt bound must be at or above the Reuss bound.
    res.scalars["vrh_bounds_ordered"] = bool(
        ol.K_voigt_GPa >= ol.K_reuss_GPa and ol.G_voigt_GPa >= ol.G_reuss_GPa
    )

    # Analytical check of the Gardner reflectivity ratio.  With Gardner,
    # Z = 0.31 Vp^1.25, so the reflection coefficient depends on velocity
    # alone.  Compare the computed rc_gardner with that closed form.
    vp = res.arrays["vp"]
    rc_closed = (vp[1:] ** 1.25 - vp[:-1] ** 1.25) / (vp[1:] ** 1.25 + vp[:-1] ** 1.25)
    res.scalars["gardner_rc_closed_form_max_error"] = float(
        np.abs(rc_closed - res.arrays["rc_gardner_depth"]).max()
    )

    res.arrays["olivine"] = ol
    res.arrays["merivale"] = mv


# --------------------------------------------------------------------------

def run(las_path=None, results_root=None) -> Results:
    """Run every stage and write the tables."""
    res = Results()
    stage_load(res, las_path)
    stage_gardner(res)
    stage_rho_g_gradient(res)
    stage_reflectivity(res)
    stage_resolution(res)
    stage_elastic_profile(res)
    stage_verification(res)
    res.scalars["standard_gravity_m_s2"] = STANDARD_GRAVITY
    res.write(results_root)
    return res
