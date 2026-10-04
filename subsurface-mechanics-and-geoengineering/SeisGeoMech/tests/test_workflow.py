"""End-to-end checks on the workflow, the log and the source-compliance rules."""

import inspect
from pathlib import Path

import numpy as np
import pytest

from seisgeomech import analysis, las_io
from seisgeomech import seismic as sx
from seisgeomech import stress as st

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def log():
    return las_io.read_las()


@pytest.fixture(scope="module")
def res(tmp_path_factory):
    return analysis.run(results_root=tmp_path_factory.mktemp("results"))


# --------------------------------------------------------------------------
# The supplied log
# --------------------------------------------------------------------------

def test_las_file_is_present_in_the_repository():
    assert (ROOT / "data" / "raw" / "48_10b-_9_jwl_JWL_FILE_1682139.las").exists()


def test_log_sample_count_and_null(log):
    assert log.n_samples == 18975
    assert log.null_value == pytest.approx(-999.25)


def test_null_value_has_been_removed(log):
    for curve in ("DT", "RHOB", "GR", "NPHI"):
        assert not (log.frame[curve] == -999.25).any()


def test_depth_is_monotonic_and_in_metres(log):
    d = log.frame["DEPT_M"].to_numpy()
    assert np.all(np.diff(d) > 0)
    assert d.min() == pytest.approx(181.1024 * 0.3048, rel=1e-9)
    assert d.max() == pytest.approx(12631.2336 * 0.3048, rel=1e-9)


def test_density_is_only_logged_over_a_short_interval(log):
    """The central data limitation of this project."""
    cov = log.coverage().set_index("curve")
    assert cov.loc["RHOB", "coverage_pct"] < 10.0
    assert cov.loc["DT", "coverage_pct"] > 75.0


def test_there_is_no_shear_sonic_curve(log):
    """DTS is absent, which is why no K, G, E or nu is reported for the well."""
    assert "DTS" not in log.frame.columns
    assert not any(c.startswith("DTS") for c in las_io.CURVES)


def test_velocity_is_physically_plausible_where_the_sonic_is_logged(log):
    vp = log.frame["VP"].dropna()
    assert vp.min() > 1000.0
    assert vp.max() < 30000.0


# --------------------------------------------------------------------------
# Workflow results
# --------------------------------------------------------------------------

def test_workflow_produces_every_expected_table(res):
    expected = {
        "las_curve_coverage", "depth_convention_evidence", "hole_condition",
        "gardner_vs_rhob", "rho_g_profile", "uniaxial_strain_stress_ratio",
        "impedance_and_reflectivity", "resolution_sweep",
        "p_wave_modulus_profile", "worked_example_olivine",
        "worked_example_merivale", "merivale_window_sensitivity",
    }
    assert expected <= set(res.tables)


def test_overlap_interval_is_the_one_the_log_supports(res):
    assert res.scalars["overlap_n"] == 1105
    assert res.scalars["overlap_top_m"] == pytest.approx(3614.2, abs=0.1)
    assert res.scalars["overlap_base_m"] == pytest.approx(3835.0, abs=0.1)


def test_gardner_bias_is_reproduced(res):
    assert res.scalars["gardner_bias_gcc"] == pytest.approx(-0.0932, abs=5e-4)
    assert res.scalars["gardner_rmse_gcc"] == pytest.approx(0.1179, abs=5e-4)


def test_gardner_bias_carries_into_the_gradient_proportionally(res):
    """The rho g gradient is linear in density, so the percentage must match."""
    assert res.scalars["rho_g_gradient_difference_pct"] == pytest.approx(
        res.scalars["gardner_bias_pct"], abs=0.02
    )


def test_gradient_equals_mean_density_times_g(res):
    s = res.scalars
    rho_mean = s["rhob_mean_gcc"] * 1000.0
    assert s["rho_g_gradient_measured_Pa_per_m"] == pytest.approx(
        rho_mean * 9.80665, rel=2e-3
    )


def test_gardner_reflectivity_has_a_closed_form_in_velocity(res):
    assert res.scalars["gardner_rc_closed_form_max_error"] < 1e-12


def test_gardner_raises_reflection_coefficient_rms_before_convolution(res):
    """Z_gardner ~ Vp^1.25 removes the density degree of freedom.

    This is a statement about the reflection coefficients only.  See the next
    test for what happens once a wavelet is applied.
    """
    assert res.scalars["rc_rms_ratio_gardner_over_measured_pre_convolution"] > 1.0
    assert res.scalars["rc_correlation"] > 0.9


def test_synthetic_trace_rms_ratio_is_not_the_reflection_coefficient_ratio(res):
    """The corrected distinction: the pre-convolution ratio does not carry over.

    The reflection-coefficient RMS ratio is about 1.20, but every synthetic
    trace in the sweep has a Gardner/measured RMS ratio BELOW one.  The two
    quantities must never be reported as the same thing.
    """
    s = res.scalars
    assert s["rc_rms_ratio_gardner_over_measured_pre_convolution"] > 1.15
    assert s["synthetic_rms_ratio_all_below_one"] is True
    assert 0.85 < s["synthetic_rms_ratio_min"] < s["synthetic_rms_ratio_max"] < 1.0


def test_gardner_refit_is_recorded_as_in_sample(res):
    """No samples were held out; fitted and evaluated on the same 1,105 pairs."""
    s = res.scalars
    assert s["gardner_refit_is_in_sample"] is True
    assert s["gardner_refit_n_calibration_samples"] == s["overlap_n"]
    assert s["gardner_refit_rmse_reduction_pct_in_sample"] == pytest.approx(
        41.5, abs=1.0
    )


def test_depth_convention_is_not_established_by_the_supplied_file(res):
    """No TVD curve, no deviation survey, LMF UNKNOWN, all elevations zero."""
    assert res.scalars["vertical_depth_established"] is False
    ev = res.tables["depth_convention_evidence"].set_index("field_or_curve")
    assert ev.loc["LMF", "value"] == "UNKNOWN"
    for curve in ("TVD", "TVDSS", "DEVI", "INCL", "AZIM"):
        assert ev.loc[curve, "value"] == "ABSENT"


def test_gradient_difference_is_flagged_invariant_under_uniform_scaling(res):
    """The flag records the precise claim, not a broader geometry independence."""
    s = res.scalars
    assert s["rho_g_gradient_difference_invariant_under_uniform_scaling"] is True
    assert "geometry_independent" not in " ".join(s)


def test_spike_convolution_is_exact(res):
    assert res.scalars["spike_convolution_max_error"] == 0.0


def test_elastic_identities_hold_to_machine_precision(res):
    assert res.scalars["identity_E_roundtrip_rel_error"] < 1e-12
    assert res.scalars["identity_nu_roundtrip_abs_error"] < 1e-12
    assert res.scalars["identity_E_eq_2G1plusnu_rel_error"] < 1e-12


def test_voigt_reuss_bounds_are_ordered(res):
    assert res.scalars["vrh_bounds_ordered"] is True


def test_resolution_correlation_increases_with_frequency(res):
    t = res.tables["resolution_sweep"].sort_values("frequency_Hz")
    c = t["corr_synthetic_vs_reflectivity_measured"].to_numpy()
    assert np.all(np.diff(c) > 0)


def test_every_sweep_frequency_is_supportable_by_the_interval(res):
    t = res.tables["resolution_sweep"]
    assert t["frequency_Hz"].min() > res.scalars["min_supportable_frequency_Hz"]


def test_p_wave_modulus_is_in_a_physically_sensible_range(res):
    s = res.scalars
    assert 10.0 < s["p_wave_modulus_min_GPa"] < s["p_wave_modulus_max_GPa"] < 200.0


def test_results_are_deterministic(tmp_path):
    a = analysis.run(results_root=tmp_path / "a")
    b = analysis.run(results_root=tmp_path / "b")
    for key, value in a.scalars.items():
        if isinstance(value, float):
            assert b.scalars[key] == pytest.approx(value, rel=1e-12, nan_ok=True)


# --------------------------------------------------------------------------
# Source-compliance guards
# --------------------------------------------------------------------------

BANNED_TERMS = (
    "biot", "thermal_expansion", "alpha_T", "kirsch", "byerlee",
    "cohesion", "ucs", "mud_weight", "hydrostatic_gradient",
    "quarter_wavelength", "tuning_thickness", "castagna", "greenberg",
    "dynamic_to_static",
)


@pytest.mark.parametrize("term", BANNED_TERMS)
def test_no_unsupported_rock_physics_enters_the_package(term):
    """Guard against re-introducing parameters the supplied folders do not give."""
    import seisgeomech

    package_dir = Path(seisgeomech.__file__).parent
    for path in package_dir.glob("*.py"):
        text = path.read_text().lower()
        # Allow the word only inside an explicit note that it is NOT used.
        for line in text.splitlines():
            if term in line:
                assert ("not" in line or "no " in line or "none" in line), (
                    f"{path.name}: '{term}' appears in an active line: {line!r}"
                )


def test_no_pore_pressure_value_is_hard_coded():
    import seisgeomech

    package_dir = Path(seisgeomech.__file__).parent
    for path in package_dir.glob("*.py"):
        text = path.read_text()
        assert "1025" not in text, f"{path.name} contains a seawater density"
        assert "1000.0 * 9.8" not in text


def test_gardner_constants_are_defined_once():
    """The coefficients must not be duplicated with different values."""
    src = inspect.getsource(sx)
    assert src.count("0.31") <= 3
    assert sx.GARDNER_COEFFICIENT == 0.31


def test_no_information_recovered_language_anywhere_in_the_package():
    """Guard for correction 1: the correlation is a similarity measure only."""
    import seisgeomech

    banned = (
        "fraction of the log",
        "variability recovered",
        "structure recovered",
        "information recovered",
        "% of information",
        "survives the wavelet",
        "layering survives",
    )
    for path in Path(seisgeomech.__file__).parent.glob("*.py"):
        text = " ".join(path.read_text().split()).lower()
        for phrase in banned:
            if phrase in text:
                idx = text.index(phrase)
                window = text[max(0, idx - 120): idx + 60]
                assert ("not" in window or "never" in window), (
                    f"{path.name}: '{phrase}' used affirmatively: ...{window}..."
                )


def test_no_field_seismic_amplitude_calibration_claim():
    """Guard for correction 2: no field seismic data are involved anywhere."""
    import seisgeomech

    for path in Path(seisgeomech.__file__).parent.glob("*.py"):
        text = " ".join(path.read_text().split()).lower()
        for phrase in ("amplitude calibration", "well tie", "seismic tie"):
            if phrase in text:
                idx = text.index(phrase)
                window = text[max(0, idx - 140): idx + 40]
                assert ("not" in window or "no " in window or "nothing" in window), (
                    f"{path.name}: '{phrase}' used affirmatively: ...{window}..."
                )
