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


def test_error_stats_on_known_residuals():
    s = db.error_stats([1.1, 0.9, 1.3], [1.0, 1.0, 1.0])
    assert s["n"] == 3
    assert s["bias_gcc"] == pytest.approx(0.1)
    assert s["mae_gcc"] == pytest.approx((0.1 + 0.1 + 0.3) / 3)
    assert s["rmse_gcc"] == pytest.approx(np.sqrt((0.01 + 0.01 + 0.09) / 3))
    with pytest.raises(ValueError):
        db.error_stats([], [])


def test_a_perfect_power_law_is_predicted_exactly_across_blocks():
    d = db.score_direction(_synthetic(), "A", "B")
    assert d["held_out_block_fit"]["rmse_gcc"] < 1e-12
    assert d["fitted_coefficient_a"] == pytest.approx(0.2, rel=1e-9)
    assert d["held_out_supplied_gardner"]["n"] == d["held_out_block_fit"]["n"]


def test_an_offset_in_the_held_out_block_appears_as_held_out_bias():
    d = db.score_direction(_synthetic(offset_b=0.05), "A", "B")
    assert d["held_out_block_fit"]["bias_gcc"] == pytest.approx(-0.05, abs=1e-12)
    assert d["held_out_block_fit"]["rmse_gcc"] == pytest.approx(0.05, abs=1e-12)
    assert d["training_fit"]["rmse_gcc"] < 1e-12


def test_the_evaluation_block_never_enters_the_fit():
    base = _synthetic()
    changed = base.copy()
    changed.loc[changed["block"] == "B", "rhob_gcc"] *= 1.5
    d0 = db.score_direction(base, "A", "B")
    d1 = db.score_direction(changed, "A", "B")
    assert d0["fitted_coefficient_a"] == d1["fitted_coefficient_a"]
    assert d0["fitted_exponent_b"] == d1["fitted_exponent_b"]


def test_the_excluded_gap_never_enters_fit_or_evaluation():
    base = _synthetic()
    changed = base.copy()
    changed.loc[changed["block"] == "excluded", ["vp_m_s", "rhob_gcc"]] = [9999.0, 9.0]
    for fit_on, ev in (("A", "B"), ("B", "A")):
        assert db.score_direction(base, fit_on, ev) == db.score_direction(changed, fit_on, ev)


def test_both_relations_are_scored_on_the_same_samples():
    lab = _synthetic()
    d = db.score_direction(lab, "B", "A")
    ev = lab[lab["block"] == "A"]
    assert d["evaluation"]["ids_sha256"] == db._ids_sha256(ev)
    ref = db.error_stats(sx.gardner_density_gcc(ev["vp_m_s"]), ev["rhob_gcc"])
    assert d["held_out_supplied_gardner"] == ref


def test_score_direction_rejects_a_block_scored_on_itself():
    with pytest.raises(ValueError):
        db.score_direction(_synthetic(), "A", "A")


def test_residual_lag_correlation_of_a_smooth_series_is_high_at_short_lags():
    d = np.arange(0.0, 100.0, 0.2)
    t = db.residual_lag_correlation(d, np.sin(d / 10.0), (0.2, 1.0))
    assert t["lag_samples"].tolist() == [1, 5]
    assert (t["correlation"] > 0.99).all()
