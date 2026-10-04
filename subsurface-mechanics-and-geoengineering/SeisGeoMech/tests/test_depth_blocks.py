"""The two-direction depth-block test: frozen split, fitting method, scoring."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from seisgeomech import analysis, depth_blocks as db
from seisgeomech import seismic as sx

ROOT = Path(__file__).resolve().parents[1]
RECORDED = ROOT / "results" / "depth_blocks"


@pytest.fixture(scope="module")
def config():
    return db.load_config()


@pytest.fixture(scope="module")
def samples():
    return db.paired_samples()


@pytest.fixture(scope="module")
def split(samples, config):
    return db.derive_split(samples, config)


# --------------------------------------------------------------------------
# The frozen design and the split it produces
# --------------------------------------------------------------------------

def test_config_fixes_one_gap_and_two_directions(config):
    assert config["split"]["exclusion_gap_m"] == 20.0
    assert [(d["fit_on"], d["evaluate_on"]) for d in config["directions"]] == [
        ("A", "B"), ("B", "A")]
    assert config["samples"]["expected_count"] == 1105


def test_paired_samples_are_the_original_overlap(samples, tmp_path_factory):
    res = analysis.run(results_root=tmp_path_factory.mktemp("orig"))
    ov = res.arrays["overlap"]
    assert len(samples) == res.scalars["overlap_n"] == 1105
    assert np.array_equal(samples["dept_m"].to_numpy(), ov["DEPT_M"].to_numpy())
    assert np.array_equal(samples["vp_m_s"].to_numpy(), ov["VP"].to_numpy())
    assert np.array_equal(samples["rhob_gcc"].to_numpy(), ov["RHOB"].to_numpy())


def test_las_rows_identify_the_samples(samples):
    from seisgeomech import las_io
    frame = las_io.read_las().frame
    rows = samples["las_row"].to_numpy()
    assert np.array_equal(frame["DEPT"].to_numpy()[rows], samples["dept_ft"].to_numpy())
    assert np.array_equal(frame["RHOB"].to_numpy()[rows], samples["rhob_gcc"].to_numpy())


def _close(a, b, rel=1e-12):
    """Equal structure; equal ints, strings and booleans; floats within ``rel``."""
    if isinstance(a, dict):
        assert set(a) == set(b)
        for k in a:
            _close(a[k], b[k], rel)
    elif isinstance(a, list):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            _close(x, y, rel)
    elif isinstance(a, float):
        assert b == pytest.approx(a, rel=rel, abs=0)
    else:
        assert a == b


def test_split_matches_the_recorded_frozen_split(split):
    record, labelled = split
    recorded = json.loads((RECORDED / "split.json").read_text())
    _close(recorded, record)
    for name in ("A", "B", "excluded"):
        assert record["blocks"][name]["ids_sha256"] == recorded["blocks"][name]["ids_sha256"]
        assert record["blocks"][name]["n"] == recorded["blocks"][name]["n"]
    csv = pd.read_csv(RECORDED / "split_samples.csv")
    assert csv["las_row"].tolist() == labelled["las_row"].tolist()
    assert csv["block"].tolist() == labelled["block"].tolist()


def test_blocks_are_contiguous_ordered_and_separated(split):
    record, labelled = split
    order = labelled["block"].tolist()
    # A ... A excluded ... excluded B ... B, each run contiguous
    runs = [k for i, k in enumerate(order) if i == 0 or order[i - 1] != k]
    assert runs == ["A", "excluded", "B"]
    blocks = record["blocks"]
    assert sum(blocks[k]["n"] for k in ("A", "B", "excluded")) == 1105
    assert all(blocks[k]["las_rows_contiguous"] for k in ("A", "B", "excluded"))
    assert blocks["A"]["depth_m_last"] < record["excluded_interval_m"][0]
    assert blocks["B"]["depth_m_first"] > record["excluded_interval_m"][1]
    assert record["separation_last_A_to_first_B_m"] >= 20.0
    assert record["separation_last_A_to_first_B_m"] < 20.5


def test_every_excluded_sample_is_within_the_gap(split):
    record, labelled = split
    mid, half = record["midpoint_m"], record["exclusion_gap_m"] / 2
    d = labelled["dept_m"].to_numpy()
    exc = labelled["block"].to_numpy() == "excluded"
    assert np.all(np.abs(d[exc] - mid) <= half)
    assert np.all(np.abs(d[~exc] - mid) > half)


def test_midpoint_is_the_middle_of_the_overlap(split, samples):
    record, _ = split
    assert record["midpoint_m"] == pytest.approx(
        0.5 * (samples["dept_m"].iloc[0] + samples["dept_m"].iloc[-1]), abs=0)


def test_assign_blocks_rejects_bad_input():
    with pytest.raises(ValueError):
        db.assign_blocks([1.0, 2.0], 0.5)
    with pytest.raises(ValueError):
        db.assign_blocks([1.0, 3.0, 2.0], 0.5)
    with pytest.raises(ValueError):
        db.assign_blocks([1.0, 2.0, 3.0], -1.0)


def test_assign_blocks_on_a_small_example():
    labels, mid = db.assign_blocks(np.arange(11.0), 2.0)
    assert mid == 5.0
    assert labels.tolist() == ["A"] * 4 + ["excluded"] * 3 + ["B"] * 4


# --------------------------------------------------------------------------
# The fitting method is the original one
# --------------------------------------------------------------------------

def test_fit_reproduces_the_original_in_sample_refit_exactly(samples, tmp_path):
    """Same method: bit-identical to stage_gardner in this environment."""
    res = analysis.run(results_root=tmp_path)
    a, b = db.fit_gardner_form(samples["vp_m_s"], samples["rhob_gcc"])
    assert a == res.scalars["gardner_refit_coefficient"]
    assert b == res.scalars["gardner_refit_exponent"]
    summary = json.loads((ROOT / "results" / "tables" / "summary.json").read_text())
    assert a == pytest.approx(summary["scalars"]["gardner_refit_coefficient"], rel=1e-12)
    assert b == pytest.approx(summary["scalars"]["gardner_refit_exponent"], rel=1e-12)


def test_fit_recovers_a_known_power_law():
    vp = np.linspace(3000.0, 6000.0, 50)
    a, b = db.fit_gardner_form(vp, 0.2 * vp ** 0.3)
    assert a == pytest.approx(0.2, rel=1e-10)
    assert b == pytest.approx(0.3, rel=1e-10)


def test_supplied_relation_is_the_fixed_one():
    assert db.predict_gardner_form(4000.0, sx.GARDNER_COEFFICIENT, sx.GARDNER_EXPONENT) == \
        pytest.approx(float(sx.gardner_density_gcc(4000.0)), rel=1e-15)


# --------------------------------------------------------------------------
# Scoring, on synthetic data
# --------------------------------------------------------------------------

def _synthetic(n=60, gap=10, a=0.2, b=0.3, offset_b=0.0):
    depth = np.arange(n, dtype=float)
    vp = np.linspace(3500.0, 5500.0, n)
    rho = a * vp ** b
    labels, _ = db.assign_blocks(depth, gap)
    rho = np.where(labels == "B", rho + offset_b, rho)
    return pd.DataFrame({"las_row": np.arange(n), "dept_ft": depth / 0.3048,
                         "dept_m": depth, "vp_m_s": vp, "rhob_gcc": rho, "block": labels})


def _sd(lab, fit_on, evaluate_on):
    return db.score_direction(lab, fit_on, evaluate_on)[0]


def test_error_stats_on_known_residuals():
    s = db.error_stats([1.1, 0.9, 1.3], [1.0, 1.0, 1.0])
    assert s["n"] == 3
    assert s["bias_gcc"] == pytest.approx(0.1)
    assert s["bias_pct_of_mean_measured"] == pytest.approx(10.0)
    assert s["mae_gcc"] == pytest.approx((0.1 + 0.1 + 0.3) / 3)
    assert s["rmse_gcc"] == pytest.approx(np.sqrt((0.01 + 0.01 + 0.09) / 3))
    with pytest.raises(ValueError):
        db.error_stats([], [])


def test_a_perfect_power_law_is_predicted_exactly_across_blocks():
    d = _sd(_synthetic(), "A", "B")
    assert d["held_out_block_fit"]["rmse_gcc"] < 1e-12
    assert d["fitted_coefficient_a"] == pytest.approx(0.2, rel=1e-9)
    assert d["held_out_supplied_gardner"]["n"] == d["held_out_block_fit"]["n"]
    assert d["reading"] == "block fit better"


def test_an_offset_in_the_held_out_block_appears_as_held_out_bias():
    d = _sd(_synthetic(offset_b=0.05), "A", "B")
    assert d["held_out_block_fit"]["bias_gcc"] == pytest.approx(-0.05, abs=1e-12)
    assert d["held_out_block_fit"]["rmse_gcc"] == pytest.approx(0.05, abs=1e-12)
    assert d["training_fit"]["rmse_gcc"] < 1e-12


def test_rho_g_consequence_of_a_constant_density_offset():
    """A -0.05 g/cm3 residual is -0.05 * 1000 * 9.80665 Pa/m = -0.490 MPa/km."""
    lab = _synthetic(offset_b=0.05)
    d = _sd(lab, "A", "B")
    g = d["rho_g_gradient_over_evaluation_block"]
    assert g["block_fit_minus_measured_MPa_per_km"] == pytest.approx(-0.05 * 9.80665, rel=1e-9)
    ev = lab[lab["block"] == "B"]
    from seisgeomech import stress as st
    expect = (st.mean_rho_g_gradient(ev["dept_m"], 1000 * sx.gardner_density_gcc(ev["vp_m_s"]))
              - st.mean_rho_g_gradient(ev["dept_m"], 1000 * ev["rhob_gcc"])) / 1e3
    assert g["supplied_minus_measured_MPa_per_km"] == pytest.approx(expect, rel=1e-12)
    assert g["measured_MPa_per_km"] == pytest.approx(
        st.mean_rho_g_gradient(ev["dept_m"], 1000 * ev["rhob_gcc"]) / 1e3, rel=1e-12)


def test_reductions_are_positive_when_the_block_fit_has_lower_error():
    d = _sd(_synthetic(), "B", "A")
    h, s = d["held_out_block_fit"], d["held_out_supplied_gardner"]
    assert d["rmse_reduction_block_fit_vs_supplied_pct"] == pytest.approx(
        100 * (s["rmse_gcc"] - h["rmse_gcc"]) / s["rmse_gcc"])
    assert d["mae_reduction_block_fit_vs_supplied_pct"] == pytest.approx(
        100 * (s["mae_gcc"] - h["mae_gcc"]) / s["mae_gcc"])
    assert d["rmse_reduction_block_fit_vs_supplied_pct"] > 0


def test_reading_rule():
    s = {"rmse_gcc": 0.10, "mae_gcc": 0.08}
    assert db.reading({"rmse_gcc": 0.05, "mae_gcc": 0.04}, s) == "block fit better"
    assert db.reading({"rmse_gcc": 0.20, "mae_gcc": 0.09}, s) == "block fit worse"
    assert db.reading({"rmse_gcc": 0.05, "mae_gcc": 0.09}, s) == "mixed"
    assert db.reading({"rmse_gcc": 0.10, "mae_gcc": 0.04}, s) == "mixed"
    assert db.overall_reading(["block fit better"] * 2) == "block fit better"
    assert db.overall_reading(["block fit better", "mixed"]) == "direction-dependent"


def test_velocity_outside_the_training_range_is_counted():
    lab = _synthetic()
    lab.loc[lab["block"] == "B", "vp_m_s"] += 5000.0
    d = _sd(lab, "A", "B")
    assert d["evaluation_vp_outside_training_range_n"] == d["evaluation"]["n"]
    assert _sd(_synthetic(), "A", "B")["evaluation_vp_outside_training_range_n"] == \
        int((_synthetic().query("block == 'B'")["vp_m_s"] > _synthetic().query("block == 'A'")["vp_m_s"].max()).sum())


def test_the_evaluation_block_never_enters_the_fit():
    base = _synthetic()
    changed = base.copy()
    changed.loc[changed["block"] == "B", "rhob_gcc"] *= 1.5
    d0 = _sd(base, "A", "B")
    d1 = _sd(changed, "A", "B")
    assert d0["fitted_coefficient_a"] == d1["fitted_coefficient_a"]
    assert d0["fitted_exponent_b"] == d1["fitted_exponent_b"]


def test_the_excluded_gap_never_enters_fit_or_evaluation():
    base = _synthetic()
    changed = base.copy()
    changed.loc[changed["block"] == "excluded", ["vp_m_s", "rhob_gcc"]] = [9999.0, 9.0]
    for fit_on, ev in (("A", "B"), ("B", "A")):
        assert _sd(base, fit_on, ev) == _sd(changed, fit_on, ev)


def test_both_relations_are_scored_on_the_same_samples():
    lab = _synthetic()
    d, pred = db.score_direction(lab, "B", "A")
    ev = lab[lab["block"] == "A"]
    assert d["evaluation"]["ids_sha256"] == db.ids_sha256(ev)
    assert pred["las_row"].tolist() == ev["las_row"].tolist()
    ref = db.error_stats(sx.gardner_density_gcc(ev["vp_m_s"]), ev["rhob_gcc"])
    assert d["held_out_supplied_gardner"] == ref
    assert np.allclose(pred["residual_supplied_gardner_gcc"],
                       sx.gardner_density_gcc(ev["vp_m_s"]) - ev["rhob_gcc"])


def test_score_direction_rejects_a_block_scored_on_itself():
    with pytest.raises(ValueError):
        db.score_direction(_synthetic(), "A", "A")


def test_residual_lag_correlation_of_a_smooth_series_is_high_at_short_lags():
    d = np.arange(0.0, 100.0, 0.2)
    t = db.residual_lag_correlation(d, np.sin(d / 10.0), (0.2, 1.0))
    assert t["lag_samples"].tolist() == [1, 5]
    assert (t["correlation"] > 0.99).all()


# --------------------------------------------------------------------------
# The whole scoring path, end to end, on a synthetic log
# --------------------------------------------------------------------------

def _synthetic_log(n=301, offset_b=0.03):
    """A WellLog whose paired interval is a perfect power law plus an offset at depth."""
    from seisgeomech import las_io
    real = las_io.read_las()
    frame = real.frame.copy()
    frame.loc[:, ["DT", "RHOB"]] = np.nan
    rows = np.arange(1000, 1000 + n)
    vp = np.linspace(3500.0, 5500.0, n)
    frame.loc[rows, "DT"] = 1e6 * 0.3048 / vp
    vp_exact = 0.3048 / (frame.loc[rows, "DT"].to_numpy() * 1e-6)
    rho = 0.2 * vp_exact ** 0.3
    rho[n // 2 + 20:] += offset_b
    frame.loc[rows, "RHOB"] = rho
    frame["VP"] = 0.3048 / (frame["DT"] * 1e-6)
    frame["RHOB_SI"] = frame["RHOB"] * 1000.0
    return las_io.WellLog(frame=frame, path=real.path, null_value=real.null_value)


def _frozen_config_for(log, config):
    cfg = json.loads(json.dumps(config))
    cfg["samples"]["expected_count"] = len(db.paired_samples(log))
    cfg["split"]["exclusion_gap_m"] = 10.0
    record, _ = db.derive_split(db.paired_samples(log), cfg)
    cfg["status"] = "frozen (synthetic test)"
    cfg["frozen_split"] = {
        "all_ids_sha256": record["all_ids_sha256"],
        "blocks": {k: {"n": v["n"], "ids_sha256": v["ids_sha256"]}
                   for k, v in record["blocks"].items()},
    }
    return cfg


def test_run_end_to_end_on_a_synthetic_log(tmp_path, config):
    from seisgeomech import figures
    log = _synthetic_log()
    cfg = _frozen_config_for(log, config)
    result = db.run(tmp_path, log=log, config=cfg)
    for name in ("split.json", "split_samples.csv", "depth_block_results.json",
                 "depth_block_table.csv", "depth_block_predictions.csv",
                 "residual_lag_correlation.csv"):
        assert (tmp_path / name).exists(), name
    payload = json.loads((tmp_path / "depth_block_results.json").read_text())
    assert payload["config_sha256"] == db.config_sha256(cfg)
    assert payload["qualifications"]["vertical_depth_established"] is False
    assert payload["qualifications"]["gap_guarantees_independence"] is False
    split = json.loads((tmp_path / "split.json").read_text())
    for name, d in payload["directions"].items():
        assert d["training"]["ids_sha256"] == split["blocks"][d["fit_on"]]["ids_sha256"]
        assert d["evaluation"]["ids_sha256"] == split["blocks"][d["evaluate_on"]]["ids_sha256"]
    # the deeper block carries an offset the shallower one does not: A-fitted
    # coefficients under-predict B by about that offset
    assert payload["directions"]["A_to_B"]["held_out_block_fit"]["bias_gcc"] < -0.02
    table = pd.read_csv(tmp_path / "depth_block_table.csv")
    assert len(table) == 4
    sup = table[table["relation"] == "supplied Gardner"]
    assert (sup["coefficients_fitted_on"] == "none (fixed coefficients)").all()
    assert (sup["coefficient_a"] == 0.31).all() and (sup["exponent_b"] == 0.25).all()
    fit = table[table["relation"] == "block fit"]
    assert fit["coefficients_fitted_on"].tolist() == ["A", "B"]
    pred = pd.read_csv(tmp_path / "depth_block_predictions.csv")
    assert len(pred) == result["split"]["blocks"]["A"]["n"] + result["split"]["blocks"]["B"]["n"]
    lags = pd.read_csv(tmp_path / "residual_lag_correlation.csv")
    assert lags["lag_m"].tolist() == [float(x) for x in db.LAG_CONTEXT_M]
    png = figures.fig_depth_blocks(result, tmp_path)
    assert png.exists() and png.stat().st_size > 10_000


def test_scoring_is_refused_unless_the_design_is_frozen(tmp_path, config):
    log = _synthetic_log()
    cfg = _frozen_config_for(log, config)
    cfg["status"] = "under review"
    with pytest.raises(db.NotFrozenError):
        db.run(tmp_path, log=log, config=cfg)
    assert not (tmp_path / "depth_block_results.json").exists()


def test_scoring_is_refused_if_the_split_differs_from_the_frozen_one(tmp_path, config):
    log = _synthetic_log()
    cfg = _frozen_config_for(log, config)
    cfg["split"]["exclusion_gap_m"] = 12.0          # edited after the freeze
    with pytest.raises(db.NotFrozenError):
        db.run(tmp_path, log=log, config=cfg)
    cfg = _frozen_config_for(log, config)
    cfg["frozen_split"]["blocks"]["A"]["ids_sha256"] = "0" * 64
    with pytest.raises(db.NotFrozenError):
        db.run(tmp_path, log=log, config=cfg)


def test_config_hash_does_not_depend_on_formatting(config):
    assert db.config_sha256(config) == db.config_sha256(json.loads(json.dumps(config, indent=7)))


# --------------------------------------------------------------------------
# Wording guards for the new files
# --------------------------------------------------------------------------

NEW_FILES = (ROOT / "src" / "seisgeomech" / "depth_blocks.py",
             ROOT / "scripts" / "depth_blocks.py",
             ROOT / "configs" / "depth_blocks.json")


@pytest.mark.parametrize("path", NEW_FILES, ids=lambda p: p.name)
def test_no_claim_of_independence_or_of_validation_elsewhere(path):
    text = " ".join(path.read_text().split()).lower()
    for stem, allowed in (("independen", ("not", "no ", "unavailable")),
                          ("validat", ("unavailable", "another well", "not "))):
        start = 0
        while (i := text.find(stem, start)) >= 0:
            window = text[max(0, i - 160): i + 80]
            assert any(a in window for a in allowed), f"{path.name}: ...{window}..."
            start = i + 1


# --------------------------------------------------------------------------
# The committed design is the frozen one
# --------------------------------------------------------------------------

#: Canonical-JSON SHA-256 of configs/depth_blocks.json as frozen on 2026-10-04.
#: Any later edit to the design, including its wording, fails this test.
FROZEN_CONFIG_SHA256 = "b994cf6c2b4e4bc3d8662cdbe8a40a3bd94d6fa2cc69696ff93fc7769ece4e42"


def test_the_committed_design_is_frozen_and_unchanged(config, split):
    record, _ = split
    assert config["status"].startswith("frozen")
    assert config["frozen_on"] == "2026-10-04"
    db.check_frozen(record, config)
    assert db.config_sha256(config) == FROZEN_CONFIG_SHA256


# --------------------------------------------------------------------------
# The recorded scores (written after the freeze commit) reproduce
# --------------------------------------------------------------------------

def test_recorded_scores_reproduce(tmp_path):
    result = db.run(tmp_path)
    for name in ("depth_block_results.json",):
        _close(json.loads((RECORDED / name).read_text()),
               json.loads((tmp_path / name).read_text()), rel=1e-9)
    for name in ("depth_block_table.csv", "depth_block_predictions.csv",
                 "residual_lag_correlation.csv"):
        a, b = pd.read_csv(RECORDED / name), pd.read_csv(tmp_path / name)
        pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-9, atol=1e-12)
    assert result["scores"]["overall_reading"] == "block fit better"


def test_held_out_scores_follow_the_frozen_reading_rule():
    r = json.loads((RECORDED / "depth_block_results.json").read_text())
    for d in r["directions"].values():
        assert d["reading"] == db.reading(d["held_out_block_fit"], d["held_out_supplied_gardner"])
        assert d["held_out_block_fit"]["n"] == d["held_out_supplied_gardner"]["n"] == d["evaluation"]["n"]
    assert r["overall_reading"] == db.overall_reading(d["reading"] for d in r["directions"].values())
    assert r["qualifications"]["vertical_depth_established"] is False
