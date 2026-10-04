"""Figures for the workflow.  Presentation only; no calculation happens here."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _save(fig, root: Path, name: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def make_all(res, root=None):
    root = Path(root) if root is not None else PROJECT_ROOT / "results" / "figures"
    paths = []
    paths.append(fig_log_coverage(res, root))
    paths.append(fig_gardner(res, root))
    paths.append(fig_rho_g_gradient(res, root))
    paths.append(fig_reflectivity(res, root))
    paths.append(fig_resolution(res, root))
    paths.append(fig_verification(res, root))
    return paths


def fig_log_coverage(res, root):
    log = res.arrays["log"].frame
    fig, axes = plt.subplots(1, 4, figsize=(11, 8), sharey=True)
    for ax, curve, label in zip(
        axes,
        ["GR", "DT", "RHOB", "NPHI"],
        ["GR (API)", "DT (us/ft)", "RHOB (g/cm3)", "NPHI (v/v)"],
    ):
        m = log[curve].notna()
        ax.plot(log.loc[m, curve], log.loc[m, "DEPT_M"], lw=0.4, color="tab:blue")
        ax.set_xlabel(label)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("logged depth coordinate (m)")
    axes[0].invert_yaxis()
    for ax in axes:
        ax.axhspan(
            res.scalars["overlap_top_m"],
            res.scalars["overlap_base_m"],
            color="tab:orange",
            alpha=0.15,
        )
    fig.suptitle(
        "Well 48/10b-9 supplied curves.  Shaded band: the "
        f"{res.scalars['overlap_thickness_m']:.0f} m where DT and RHOB overlap",
        fontsize=10,
    )
    return _save(fig, root, "fig01_log_coverage.png")


def fig_gardner(res, root):
    t = res.tables["gardner_vs_rhob"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 7))

    ax = axes[0]
    ax.plot(t["rhob_measured_gcc"], t["depth_m"], lw=0.5, label="measured RHOB")
    ax.plot(t["rho_gardner_gcc"], t["depth_m"], lw=0.5, label="Gardner from Vp")
    ax.invert_yaxis()
    ax.set_xlabel("bulk density (g/cm3)")
    ax.set_ylabel("logged depth coordinate (m)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.scatter(t["vp_m_s"], t["rhob_measured_gcc"], s=2, alpha=0.3, label="measured")
    v = np.linspace(t["vp_m_s"].min(), t["vp_m_s"].max(), 200)
    ax.plot(v, 0.31 * v ** 0.25, color="tab:red", lw=2,
            label=r"Gardner as supplied  $\rho = 0.31\,V_p^{0.25}$")
    a = res.scalars["gardner_refit_coefficient"]
    b = res.scalars["gardner_refit_exponent"]
    ax.plot(v, a * v ** b, color="tab:green", lw=2, ls="--",
            label=f"same form refitted here: ${a:.3f}\\,V_p^{{{b:.3f}}}$")
    ax.set_xlabel("Vp (m/s)")
    ax.set_ylabel("bulk density (g/cm3)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.hist(t["residual_gcc"], bins=50, color="tab:grey")
    ax.axvline(0, color="k", lw=1)
    ax.axvline(res.scalars["gardner_bias_gcc"], color="tab:red", lw=2,
               label=f"bias {res.scalars['gardner_bias_gcc']:+.3f} g/cm3")
    ax.set_xlabel("Gardner minus measured (g/cm3)")
    ax.set_ylabel("count")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle(
        f"Gardner tested against the measured density log, n = {len(t)} samples "
        f"({res.scalars['overlap_top_m']:.0f}-{res.scalars['overlap_base_m']:.0f} m)",
        fontsize=11,
    )
    return _save(fig, root, "fig02_gardner_vs_measured.png")


def fig_rho_g_gradient(res, root):
    t = res.tables["rho_g_profile"]
    u = res.tables["uniaxial_strain_stress_ratio"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 6))

    ax = axes[0]
    ax.plot(t["rho_g_integral_measured_MPa"], t["logged_depth_m"], label="measured density")
    ax.plot(t["rho_g_integral_gardner_MPa"], t["logged_depth_m"], label="Gardner density")
    ax.invert_yaxis()
    ax.set_xlabel(r"$\int \rho g\,\mathrm{d}s$ below top of interval (MPa)")
    ax.set_ylabel("logged depth coordinate (m)")
    ax.set_title(
        f"gradient {res.scalars['rho_g_gradient_measured_MPa_per_km']:.2f} vs "
        f"{res.scalars['rho_g_gradient_gardner_MPa_per_km']:.2f} MPa/km "
        f"({res.scalars['rho_g_gradient_difference_pct']:+.2f} %)",
        fontsize=9,
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(u["poissons_ratio"], u["sigma_h_over_sigma_v"], marker="o")
    ax.set_xlabel("Poisson's ratio (not measured at this well)")
    ax.set_ylabel(r"$\sigma_{xx}/\sigma_{zz} = \nu/(1-\nu)$")
    ax.set_title("uniaxial-strain ratio, parametric only", fontsize=9)
    ax.grid(alpha=0.3)

    fig.suptitle(
        r"$\rho g$ integrated along the logged coordinate  "
        "(Models/Elasticity_exercise.pdf Q5, applied one-dimensionally; "
        "the file does not establish true vertical depth)",
        fontsize=10,
    )
    return _save(fig, root, "fig03_rho_g_gradient_and_ratio.png")


def fig_reflectivity(res, root):
    t = res.tables["impedance_and_reflectivity"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 7), sharey=True)

    ax = axes[0]
    ax.plot(t["impedance_measured"] / 1e6, t["depth_m"], lw=0.5, label="measured")
    ax.plot(t["impedance_gardner"] / 1e6, t["depth_m"], lw=0.5, label="Gardner")
    ax.set_xlabel("acoustic impedance (1e6 kg m-2 s-1)")
    ax.set_ylabel("logged depth coordinate (m)")
    ax.invert_yaxis()
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(t["rc_measured"], t["depth_m"], lw=0.4, color="tab:blue")
    ax.set_xlabel("reflection coefficient, measured density")
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(t["rc_gardner"], t["depth_m"], lw=0.4, color="tab:orange")
    ax.set_xlabel("reflection coefficient, Gardner density")
    ax.grid(alpha=0.3)

    fig.suptitle(
        "Zero-offset reflection coefficients, BEFORE wavelet convolution.  "
        f"Gardner RMS is {res.scalars['rc_rms_ratio_gardner_over_measured_pre_convolution']:.3f}"
        f" times the measured, correlation {res.scalars['rc_correlation']:.3f}.  "
        "After convolution the ratio is different (fig05).",
        fontsize=10,
    )
    return _save(fig, root, "fig04_impedance_reflectivity.png")


def fig_resolution(res, root):
    t = res.tables["resolution_sweep"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    ax = axes[0]
    ax.plot(t["frequency_Hz"], t["corr_synthetic_vs_reflectivity_measured"],
            marker="o", label="measured density")
    ax.plot(t["frequency_Hz"], t["corr_synthetic_vs_reflectivity_gardner"],
            marker="s", label="Gardner density")
    ax.set_xscale("log")
    ax.set_xlabel("Ricker centre frequency (Hz)")
    ax.set_ylabel("Pearson r (synthetic trace vs input reflectivity)")
    ax.set_title("similarity measure for this synthetic experiment only;\nnot a fraction of information recovered", fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    twt = res.arrays["t_reg"] * 1e3
    lo = res.arrays["synthetic_low_measured"]
    hi = res.arrays["synthetic_high_measured"]
    rc = res.arrays["rc_measured_time"]
    ax.plot(rc / np.abs(rc).max(), twt, lw=0.4, label="reflectivity (normalised)")
    ax.plot(lo / np.abs(lo).max() + 2.5, twt, lw=0.6,
            label=f"{res.arrays['synthetic_low_frequency']:.0f} Hz")
    ax.plot(hi / np.abs(hi).max() + 5.0, twt, lw=0.6,
            label=f"{res.arrays['synthetic_high_frequency']:.0f} Hz")
    ax.invert_yaxis()
    ax.set_xlabel("normalised amplitude (traces offset)")
    ax.set_ylabel("travel time along the logged path (ms)")
    ax.set_title("synthetic traces at the sweep extremes", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(t["frequency_Hz"], t["synthetic_rms_ratio_gardner_over_measured"],
            marker="D", color="tab:purple", label="after convolution (synthetic trace)")
    ax.axhline(res.scalars["rc_rms_ratio_gardner_over_measured_pre_convolution"],
               color="tab:red", ls="--",
               label="before convolution (reflection coefficients)")
    ax.axhline(1.0, color="k", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("Ricker centre frequency (Hz)")
    ax.set_ylabel("RMS ratio, Gardner / measured")
    ax.set_title("two different quantities:\nthe pre-convolution ratio does not\ncarry over to the synthetic trace",
                 fontsize=8)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")

    fig.suptitle("Ricker frequency experiment (Seismic/Ex1.ipynb s.1.6)", fontsize=11)
    plt.tight_layout()
    return _save(fig, root, "fig05_resolution.png")


def fig_verification(res, root):
    mv = res.arrays["merivale"]
    ol = res.arrays["olivine"]
    from . import worked_examples as wx

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))

    ax = axes[0]
    ax.plot(wx.MERIVALE_AXIAL_STRAIN, wx.MERIVALE_STRESS_MPA, "o-",
            ms=4, label="axial")
    ax.plot(wx.MERIVALE_CIRCUMFERENTIAL_STRAIN, wx.MERIVALE_STRESS_MPA, "s-",
            ms=4, label="circumferential")
    ax.plot(wx.MERIVALE_VOLUMETRIC_STRAIN, wx.MERIVALE_STRESS_MPA, "^-",
            ms=4, label="volumetric")
    ax.axhline(mv.stress_window_MPa[1], color="tab:red", ls="--", lw=1,
               label="half peak stress")
    ax.set_xlabel("strain")
    ax.set_ylabel("axial stress (MPa)")
    ax.set_title("Merivale granite, Q4 record", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1]
    m = (wx.MERIVALE_STRESS_MPA >= mv.stress_window_MPa[0]) & (
        wx.MERIVALE_STRESS_MPA <= mv.stress_window_MPa[1]
    )
    ax.plot(wx.MERIVALE_AXIAL_STRAIN[m], wx.MERIVALE_STRESS_MPA[m], "o", ms=6)
    xx = np.linspace(0, wx.MERIVALE_AXIAL_STRAIN[m].max(), 10)
    ax.plot(xx, mv.E_Pa / 1e6 * xx, color="tab:red",
            label=f"E = {mv.E_Pa/1e9:.1f} GPa (R2 = {mv.E_r2:.4f})")
    ax.set_xlabel("axial strain")
    ax.set_ylabel("axial stress (MPa)")
    ax.set_title("linear-elastic stage fit", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[2]
    labels = ["K_Voigt", "K_Reuss", "K_VRH", "G_Voigt", "G_Reuss", "G_VRH"]
    vals = [ol.K_voigt_GPa, ol.K_reuss_GPa, ol.K_hill_GPa,
            ol.G_voigt_GPa, ol.G_reuss_GPa, ol.G_hill_GPa]
    ax.bar(labels, vals, color=["tab:blue"] * 3 + ["tab:green"] * 3)
    ax.set_ylabel("GPa")
    ax.set_title(
        f"olivine VRH -> Vp {ol.vp_m_s:.0f}, Vs {ol.vs_m_s:.0f} m/s", fontsize=9
    )
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("Verification against the two supplied worked examples", fontsize=11)
    return _save(fig, root, "fig06_verification.png")


# --------------------------------------------------------------------------
# October 2026 revision: the depth-block test.  Not part of make_all, which
# reproduces the six original figures unchanged.
# --------------------------------------------------------------------------

#: Block colours (validated categorical pair); the supplied relation is drawn
#: in neutral ink because it is the fixed reference, not a block.
BLOCK_COLOURS = {"A": "#2a78d6", "B": "#eb6834"}
BLOCKS_ORDER = ("A", "B")
REFERENCE_INK = "#52514e"
EXCLUDED_INK = "#a8a7a2"


def fig_depth_blocks(result, root):
    """Split, fits and held-out residuals of the two-direction depth-block test.

    Draws the stored predictions (``result["predictions"]``); nothing is fitted
    or predicted here.  Colour identifies the depth block throughout.
    """
    from . import depth_blocks as db
    from . import seismic as sx

    lab = result["labelled"]
    split = result["split"]
    dirs = result["scores"]["directions"]
    pred = result["predictions"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 7))
    lo, hi = split["excluded_interval_m"]

    ax = axes[0]
    ax.axhspan(lo, hi, color=EXCLUDED_INK, alpha=0.25, lw=0)
    for name in ("A", "B", "excluded"):
        part = lab[lab["block"] == name]
        colour = BLOCK_COLOURS.get(name, EXCLUDED_INK)
        label = (f"block {name} (n = {len(part)})" if name in BLOCK_COLOURS else
                 f"excluded gap, {split['exclusion_gap_m']:.0f} m (n = {len(part)})")
        ax.plot(part["rhob_gcc"], part["dept_m"], lw=0.6, color=colour, label=label)
    ax.invert_yaxis()
    ax.set_xlabel("measured RHOB (g/cm3)")
    ax.set_ylabel("logged depth coordinate (m)")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.3)
    ax.set_title("the frozen split", fontsize=10)

    ax = axes[1]
    for name in BLOCKS_ORDER:
        part = lab[lab["block"] == name]
        ax.scatter(part["vp_m_s"], part["rhob_gcc"], s=4, alpha=0.35,
                   color=BLOCK_COLOURS[name], linewidths=0)
    v = np.linspace(lab["vp_m_s"].min(), lab["vp_m_s"].max(), 200)
    for d in dirs.values():
        a, b = d["fitted_coefficient_a"], d["fitted_exponent_b"]
        ax.plot(v, db.predict_gardner_form(v, a, b), lw=2, color=BLOCK_COLOURS[d["fit_on"]],
                label=f"fitted on {d['fit_on']}: {a:.3f} Vp^{b:.3f}")
    ax.plot(v, sx.gardner_density_gcc(v), lw=2, ls="--", color=REFERENCE_INK,
            label=f"supplied: {sx.GARDNER_COEFFICIENT:g} Vp^{sx.GARDNER_EXPONENT:g}")
    ax.set_xlabel("Vp (m/s)")
    ax.set_ylabel("bulk density (g/cm3)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    ax.set_title("points: measured, coloured by block", fontsize=10)

    ax = axes[2]
    for d in dirs.values():
        p = pred[pred["evaluate_on"] == d["evaluate_on"]]
        h, s = d["held_out_block_fit"], d["held_out_supplied_gardner"]
        colour = BLOCK_COLOURS[d["evaluate_on"]]
        ax.plot(p["residual_supplied_gardner_gcc"], p["dept_m"], lw=0.6, ls=":",
                color=colour, label=f"block {d['evaluate_on']}, supplied: RMSE {s['rmse_gcc']:.3f}")
        ax.plot(p["residual_block_fit_gcc"], p["dept_m"], lw=0.6, color=colour,
                label=f"block {d['evaluate_on']}, fitted on {d['fit_on']}: RMSE {h['rmse_gcc']:.3f}")
    ax.axvline(0.0, color="black", lw=0.8)
    ax.axhspan(lo, hi, color=EXCLUDED_INK, alpha=0.25, lw=0)
    ax.invert_yaxis()
    ax.set_xlabel("predicted - measured density (g/cm3)")
    ax.set_ylabel("logged depth coordinate (m)")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.3)
    ax.set_title("held-out residuals: each block predicted from the other", fontsize=10)

    fig.suptitle(
        "Depth-block test of the Gardner form at 48/10b-9: fit on one block, predict the other, "
        "compare with the supplied relation on the same samples", fontsize=10,
    )
    fig.tight_layout()
    return _save(fig, Path(root), "fig07_depth_blocks.png")
