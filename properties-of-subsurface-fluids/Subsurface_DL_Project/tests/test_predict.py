"""Tests for the guarded prediction path (src/sfp/predict.py) and its training domain."""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from sfp import components as C  # noqa: E402
from sfp import flash  # noqa: E402
from sfp import predict as P  # noqa: E402

HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
Z_EX = C.EXAMPLE_MOLES / C.EXAMPLE_MOLES.sum()          # the p. 14 mixture
_SURROGATE = []


def surrogate():
    if not _SURROGATE:
        _SURROGATE.append(P.FlashSurrogate())
    return _SURROGATE[0]


def raises_input_error(fn, *needles):
    try:
        fn()
    except P.InputError as exc:
        msg = str(exc)
        for n in needles:
            assert n in msg, f"{n!r} not in message: {msg}"
        return msg
    raise AssertionError("InputError not raised")


def saturation(z, T):
    pb, pd = flash.wilson_saturation_pressures(np.atleast_2d(z), T, C.TC_RANKINE,
                                               C.PC_PSIA, C.OMEGA)
    return float(pb[0]), float(pd[0])


# --------------------------------------------------------------------------
# saturation pressures and window position
# --------------------------------------------------------------------------
def test_saturation_pressures_reproduce_p14_and_close_the_phase_test():
    pb, pd = saturation(Z_EX, 610.0)
    assert abs(pb - 1590.8769) / 1590.8769 < 1e-5
    assert abs(pd - 1.9995691) / 1.9995691 < 1e-4
    for p, which in ((pb, 0), (pd, 1)):
        K = flash.wilson_k(p, 610.0, C.TC_RANKINE, C.PC_PSIA, C.OMEGA)
        sums = flash.phase_sums(Z_EX[None], K[None])
        assert abs(sums[which][0] - 1.0) < 1e-12


def test_window_position_runs_from_dew_to_bubble():
    pb, pd = saturation(Z_EX, 640.0)
    xi = flash.window_position(np.array([pd, np.sqrt(pb * pd), pb]), pb, pd)
    assert np.allclose(xi, [0.0, 0.5, 1.0], atol=1e-12)


def test_two_phase_exactly_between_dew_and_bubble_points():
    rng = np.random.default_rng(4)
    z = rng.integers(1, 41, size=(200, 7)).astype(float)
    z /= z.sum(1, keepdims=True)
    T = rng.uniform(610, 680, 200)
    p = np.exp(rng.uniform(np.log(1.0), np.log(5000.0), 200))
    pb, pd = flash.wilson_saturation_pressures(z, T, C.TC_RANKINE, C.PC_PSIA, C.OMEGA)
    K = flash.wilson_k(p[:, None], T[:, None], C.TC_RANKINE, C.PC_PSIA, C.OMEGA)
    assert np.array_equal(flash.is_two_phase(z, K), (pd < p) & (p < pb))


# --------------------------------------------------------------------------
# input validation: malformed input is rejected, never repaired
# --------------------------------------------------------------------------
def test_wrong_number_of_components_is_rejected_with_the_binary_explained():
    msg = raises_input_error(lambda: P.validate_inputs([0.6, 0.4], 1000.0, 620.0),
                             "seven-component", "binary")
    assert "sfp.flash" in msg


def test_binary_worked_example_is_solved_by_the_reference_flash_only():
    """Notes pp. 16-18: the binary verifies the reference calculation, but it is
    not a seven-component state, so the guarded predictor refuses it."""
    FV, _ = flash.solve_fv(np.array([[0.60, 0.40]]), np.array([[3.8, 0.0029]]))
    assert abs(FV[0] - 0.4588879) < 1e-6
    raises_input_error(lambda: surrogate().predict([0.6, 0.4], 1000.0, 620.0), "binary")


def test_composition_must_sum_to_one_and_is_not_normalised():
    z = Z_EX * 0.98
    msg = raises_input_error(lambda: P.validate_inputs(z, 500.0, 640.0), "sum to 1")
    assert "not normalised" in msg
    z_ok, *_ = P.validate_inputs(Z_EX, 500.0, 640.0)
    assert np.array_equal(z_ok[0], Z_EX)               # passed through unchanged


def test_negative_nan_and_non_numeric_compositions_are_rejected():
    z = Z_EX.copy()
    z[0], z[1] = -0.01, z[1] + 0.01 + z[0]
    raises_input_error(lambda: P.validate_inputs(z, 500.0, 640.0), "non-negative")
    raises_input_error(lambda: P.validate_inputs([np.nan] * 7, 500.0, 640.0), "NaN")
    raises_input_error(lambda: P.validate_inputs(["a"] * 7, 500.0, 640.0), "real numbers")


def test_pressure_and_temperature_must_be_finite_and_positive():
    for p, T in ((0.0, 640.0), (-5.0, 640.0), (np.inf, 640.0), (500.0, 0.0), (500.0, np.nan)):
        raises_input_error(lambda: P.validate_inputs(Z_EX, p, T))


def test_units_other_than_psia_and_rankine_are_rejected_not_converted():
    raises_input_error(lambda: P.validate_inputs(Z_EX, 5.5, 640.0, pressure_unit="MPa"), "psia")
    raises_input_error(lambda: P.validate_inputs(Z_EX, 500.0, 180.0, temperature_unit="F"),
                       "Rankine", "+ 460")
    raises_input_error(lambda: P.validate_inputs(Z_EX, 500.0, 355.0, temperature_unit="K"),
                       "Rankine")


def test_component_order_is_checked_and_a_named_mapping_is_accepted():
    raises_input_error(lambda: P.validate_inputs(Z_EX, 500.0, 640.0,
                                                 components_order=list(reversed(C.NAMES))),
                       "order")
    z, *_ = P.validate_inputs(dict(zip(C.NAMES, Z_EX)), 500.0, 640.0)
    assert np.array_equal(z[0], Z_EX)
    bad = dict(zip(C.NAMES, Z_EX))
    bad["nC4"] = bad.pop("C4")
    raises_input_error(lambda: P.validate_inputs(bad, 500.0, 640.0), "missing", "C4")


def test_batch_shapes_are_checked():
    z = np.repeat(Z_EX[None], 3, axis=0)
    zz, p, T, single = P.validate_inputs(z, [100.0, 200.0, 300.0], 640.0)
    assert zz.shape == (3, 7) and p.shape == T.shape == (3,) and not single
    raises_input_error(lambda: P.validate_inputs(z, [100.0, 200.0], 640.0), "3 compositions")
    raises_input_error(lambda: P.validate_inputs(Z_EX, [100.0, 200.0], 640.0))


def test_unknown_unsupported_mode_is_rejected():
    raises_input_error(lambda: surrogate().predict(Z_EX, 500.0, 640.0, on_unsupported="guess"),
                       "on_unsupported")


# --------------------------------------------------------------------------
# the phase test comes before the network
# --------------------------------------------------------------------------
def test_single_phase_states_use_the_phase_test_not_the_network():
    s = surrogate()
    pb, pd = saturation(Z_EX, 610.0)
    liquid = s.predict(Z_EX, 1.5 * pb, 610.0)
    vapour = s.predict(Z_EX, 0.5 * pd, 610.0)
    assert (liquid.phase, liquid.method, liquid.FV) == ("liquid", "single_phase", 0.0)
    assert (vapour.phase, vapour.method, vapour.FV) == ("vapour", "single_phase", 1.0)
    assert liquid.model is None and vapour.model is None


def test_raw_network_answers_a_single_phase_state_but_the_guard_does_not():
    """The hazard being fixed: the sigmoid gives a plausible fraction anyway."""
    s = surrogate()
    pb, _ = saturation(Z_EX, 640.0)
    p = 0.5 * (pb + 2000.0) if pb < 2000.0 else 1.2 * pb
    raw = s._network(Z_EX[None], np.array([p]), np.array([640.0]))[0]
    assert 0.0 < raw < 1.0
    guarded = s.predict(Z_EX, p, 640.0)
    assert guarded.phase == "liquid" and guarded.FV == 0.0 and guarded.method == "single_phase"


def test_bubble_and_dew_points_are_handled_explicitly():
    s = surrogate()
    pb, pd = saturation(Z_EX, 610.0)
    b, d = s.predict(Z_EX, pb, 610.0), s.predict(Z_EX, pd, 610.0)
    assert (b.phase, b.method, b.FV, b.status) == ("bubble_point", "phase_boundary", 0.0, "ok")
    assert (d.phase, d.method, d.FV, d.status) == ("dew_point", "phase_boundary", 1.0, "ok")
    # just inside the window the state is two-phase again and reaches the network
    inside = s.predict(Z_EX, pb * (1.0 - 1e-6), 610.0)
    assert inside.phase == "two_phase" and inside.method == "network"


def test_pure_component_at_its_vapour_pressure_is_indeterminate():
    """Both phase-test sums equal 1 only when every K_i of a component present is 1."""
    s = surrogate()
    z = np.zeros(7)
    z[1] = 1.0                                        # pure C1
    T = 640.0
    psat = C.PC_PSIA[1] * np.exp(5.37 * (1 + C.OMEGA[1]) * (1 - C.TC_RANKINE[1] / T))
    r = s.predict(z, psat, T)
    assert (r.phase, r.status, r.method, r.FV) == ("indeterminate", "unsupported", "none", None)
    assert s.predict(z, 1.1 * psat, T).phase == "liquid"
    assert s.predict(z, 0.9 * psat, T).phase == "vapour"


# --------------------------------------------------------------------------
# the training domain
# --------------------------------------------------------------------------
def test_domain_file_matches_the_saved_training_split():
    with open(os.path.join(HERE, "configs", "prediction_domain.json"), encoding="utf-8") as fh:
        cfg = json.load(fh)
    blob = np.load(os.path.join(HERE, "data", "flash_dataset.npz"))
    itr = blob["idx_train"]
    lim = cfg["limits"]
    assert lim["p_psia"] == [2.0, 2000.0] and lim["T_R"] == [610.0, 680.0]
    assert np.array_equal(lim["z_min"], blob["z"][itr].min(0))
    assert np.array_equal(lim["z_max"], blob["z"][itr].max(0))
    assert blob["p_psia"][itr].max() <= 2000.0 and blob["T_R"][itr].min() >= 610.0
    # the configuration would allow 1/241 .. 40/46; the training rows are narrower
    implied = cfg["configuration_implied_z_range"]
    assert abs(implied["z_min"] - 1 / 241) < 1e-15 and abs(implied["z_max"] - 40 / 46) < 1e-15
    assert max(lim["z_max"]) < 0.5 < implied["z_max"]


def test_every_training_row_is_inside_the_domain():
    blob = np.load(os.path.join(HERE, "data", "flash_dataset.npz"))
    itr = blob["idx_train"]
    dom = P.TrainingDomain.from_file()
    v = dom.violations(blob["z"][itr], blob["p_psia"][itr], blob["T_R"][itr])
    assert not any(v)


def test_out_of_domain_two_phase_states_fall_back_to_the_solver_by_default():
    s = surrogate()
    blob = np.load(os.path.join(HERE, "data", "flash_dataset.npz"))
    i = 0                                             # a 2000-4000 psia extrapolation row
    z, p, T, FV = blob["z_ood"][i], float(blob["p_ood_psia"][i]), float(blob["T_ood_R"][i]), \
        float(blob["FV_ood"][i])
    r = s.predict(z, p, T)
    assert (r.status, r.phase, r.method, r.in_training_domain) == \
        ("ok", "two_phase", "reference_solver_fallback", False)
    assert abs(r.FV - FV) < 1e-10 and r.model is None
    assert any("2000-4000 psia" in v for v in r.domain_violations)
    u = s.predict(z, p, T, on_unsupported="status")
    assert (u.status, u.method, u.FV) == ("unsupported", "none", None)
    e = s.predict(z, p, T, on_unsupported="extrapolate")
    assert e.method == "network_extrapolation" and "EXTRAPOLATION" in e.message
    assert 0.0 < e.FV < 1.0 and e.model == s.model_name


def test_temperature_composition_and_high_pressure_violations_are_named():
    s = surrogate()
    hot = s.predict(Z_EX, 796.43821, 700.0)
    assert hot.method == "reference_solver_fallback"
    assert any("T = 700 R" in v for v in hot.domain_violations)
    z = np.array([0.60, 0.10, 0.05, 0.05, 0.05, 0.05, 0.10])   # z_CO2 above the training range
    r = s.predict(z, 300.0, 640.0)
    assert r.phase == "two_phase" and r.method == "reference_solver_fallback"
    assert any(v.startswith("z_CO2") for v in r.domain_violations)
    dom = P.TrainingDomain.from_file()
    v = dom.violations(Z_EX[None], [5000.0], [640.0])[0]
    assert any("above every pressure tested" in m for m in v)


def test_in_domain_two_phase_uses_the_network_and_reports_its_model():
    s = surrogate()
    r = s.predict(Z_EX, 796.43821, 640.0)             # the p. 15 flash
    assert (r.status, r.phase, r.method, r.in_training_domain) == ("ok", "two_phase", "network", True)
    assert r.model == "ffn_phys1_s2" and abs(r.FV - 0.1917) < 0.01
    assert r.p_dew_psia < 796.43821 < r.p_bubble_psia


def test_batch_mixes_every_route_and_labels_each_row():
    s = surrogate()
    pb, pd = saturation(Z_EX, 610.0)
    z = np.repeat(Z_EX[None], 5, axis=0)
    out = s.predict_many(z, [1.5 * pb, 0.5 * pd, pb, 796.43821, 796.43821],
                         [610.0, 610.0, 610.0, 640.0, 700.0])
    assert [r.method for r in out] == ["single_phase", "single_phase", "phase_boundary",
                                       "network", "reference_solver_fallback"]
    for r in out:
        assert r.status in P.STATUSES and r.phase in P.PHASES and r.method in P.METHODS
        assert set(r.as_dict()) >= {"FV", "status", "phase", "method", "in_training_domain",
                                    "domain_violations", "model", "message"}


# --------------------------------------------------------------------------
# added after an adversarial review (mutation testing of predict.py)
# --------------------------------------------------------------------------
def test_units_and_component_order_are_enforced_on_the_public_path():
    s = surrogate()
    raises_input_error(lambda: s.predict(Z_EX, 180.0, 640.0, temperature_unit="F"), "Rankine")
    raises_input_error(lambda: s.predict_many(Z_EX[None], [5.5], [640.0], pressure_unit="MPa"),
                       "psia", "145.0377")
    raises_input_error(lambda: s.predict_many(Z_EX[None], [500.0], [640.0],
                                              components_order=list(reversed(C.NAMES))), "order")
    for bad in ("CO2", 5):
        raises_input_error(lambda: P.validate_inputs(Z_EX, 500.0, 640.0, components_order=bad),
                           "components_order")
    raises_input_error(lambda: P.validate_inputs(Z_EX, 500.0, 300.0, temperature_unit="C"), "1.8")
    raises_input_error(lambda: P.validate_inputs(Z_EX, 485.3, 640.0, pressure_unit="psig"), "14.7")


def test_named_mapping_in_any_key_order_gives_the_same_answer():
    s = surrogate()
    ref = s.predict(Z_EX, 796.43821, 640.0)
    shuffled = dict(reversed(list(zip(C.NAMES, Z_EX))))
    assert s.predict(shuffled, 796.43821, 640.0).FV == ref.FV


def test_mapping_with_one_value_per_state_is_a_batch_not_a_transpose():
    s = surrogate()
    zb = np.array([np.roll(Z_EX, k) for k in range(7)])     # rows and columns both sum to 1
    p, T = np.full(7, 300.0), np.full(7, 640.0)
    by_rows = s.predict_many(zb, p, T)
    by_name = s.predict_many({n: zb[:, i] for i, n in enumerate(C.NAMES)}, p, T)
    assert [r.FV for r in by_name] == [r.FV for r in by_rows]
    assert [r.method for r in by_name] == [r.method for r in by_rows]
    raises_input_error(lambda: s.predict({n: zb[:, i] for i, n in enumerate(C.NAMES)}, 300.0, 640.0),
                       "one state")
    raises_input_error(lambda: P.validate_inputs({n: [0.1] * (2 + (i == 0)) for i, n in
                                                  enumerate(C.NAMES)}, 300.0, 640.0), "same length")


def test_booleans_text_complex_none_and_ragged_input_are_refused():
    raises_input_error(lambda: P.validate_inputs(Z_EX, True, 640.0), "booleans")
    raises_input_error(lambda: P.validate_inputs(Z_EX > 0.1, 500.0, 640.0), "booleans")
    raises_input_error(lambda: P.validate_inputs(Z_EX, "796.4", 640.0), "text")
    raises_input_error(lambda: P.validate_inputs([str(v) for v in Z_EX], 500.0, 640.0), "text")
    raises_input_error(lambda: P.validate_inputs(Z_EX, 796.4 + 1j, 640.0), "complex")
    raises_input_error(lambda: P.validate_inputs(Z_EX, None, 640.0), "None")
    msg = raises_input_error(lambda: P.validate_inputs([list(Z_EX), list(Z_EX[:6])], 500.0, 640.0),
                             "rectangular")
    big = raises_input_error(lambda: P.validate_inputs([list(Z_EX)] * 5000 + [["x"] * 7], 500.0, 640.0))
    assert len(msg) < 300 and len(big) < 300
    raises_input_error(lambda: P.validate_inputs(Z_EX[:, None], 500.0, 640.0), "column")
    raises_input_error(lambda: P.validate_inputs([], 500.0, 640.0), "no states")
    raises_input_error(lambda: surrogate().predict(0.5, 500.0, 640.0), "shape")


def test_composition_sum_tolerance_is_one_in_a_million():
    raises_input_error(lambda: P.validate_inputs(Z_EX * 0.995, 500.0, 640.0), "sum to 1")
    raises_input_error(lambda: P.validate_inputs(Z_EX * (1 + 2e-6), 500.0, 640.0), "sum to 1")
    P.validate_inputs(Z_EX * (1 + 5e-7), 500.0, 640.0)            # accepted, unchanged


def test_phase_test_does_not_depend_on_rounding_of_the_composition():
    """A composition accepted within the 1e-6 sum tolerance is classified as its
    normalised counterpart, also within that tolerance of a phase boundary."""
    s = surrogate()
    pb, pd = saturation(Z_EX, 640.0)
    for factor in (1 - 9e-7, 1 + 9e-7):
        z = Z_EX * factor
        assert s.predict(z, pb * (1 + 5e-7), 640.0).phase == "liquid"
        assert s.predict(z, pb * (1 - 5e-7), 640.0).phase == "two_phase"
        assert s.predict(z, pd * (1 - 5e-7), 640.0).phase == "vapour"
        assert s.predict(z, pd, 640.0).phase == "dew_point"
    zc1 = np.zeros(7)
    zc1[1] = 1 - 5e-7                                              # accepted, near-pure C1
    psat = C.PC_PSIA[1] * np.exp(5.37 * (1 + C.OMEGA[1]) * (1 - C.TC_RANKINE[1] / 640.0))
    assert s.predict(zc1, psat * 1.0000004, 640.0).phase == "liquid"


def test_extreme_conditions_are_refused_or_answered_by_the_solver_not_the_network():
    s = surrogate()
    # K-values underflow to zero for components that are present: refused
    raises_input_error(lambda: s.predict(Z_EX, 500.0, 1.0), "cannot be evaluated")
    # absent components with underflowed K: still a valid, labelled solver answer
    z = np.array([0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0])
    r = s.predict(z, 1e-77, 10.0)
    assert r.method == "reference_solver_fallback" and np.isfinite(r.FV) and 0.0 < r.FV < 1.0
    assert r.model is None and any(v.startswith("T = 10 R") for v in r.domain_violations)


def test_lower_and_upper_domain_edges():
    dom = P.TrainingDomain.from_file()
    z = Z_EX[None]
    assert dom.violations(z, [2000.0], [610.0]) == [[]]               # both edges inside
    assert dom.violations(z, [2.0], [680.0]) == [[]]
    assert any("above the training maximum" in v for v in dom.violations(z, [2000.001], [640.0])[0])
    assert any("below the training minimum" in v for v in dom.violations(z, [1.999], [640.0])[0])
    assert any(v.startswith("T = 609.99999 R") for v in dom.violations(z, [500.0], [609.99999])[0])
    low = Z_EX.copy()
    low[2], low[6] = dom.z_min[2] * 0.9, low[6] + low[2] - dom.z_min[2] * 0.9
    assert any(v.startswith("z_C2") for v in dom.violations(low[None], [500.0], [640.0])[0])
    # a scalar p and T apply to every row, and mismatched lengths are refused
    bad = np.array([0.60, 0.10, 0.05, 0.05, 0.05, 0.05, 0.10])
    v = dom.violations(np.vstack([Z_EX, bad, bad]), 500.0, 640.0)
    assert len(v) == 3 and v[0] == [] and v[1] and v[2]
    try:
        dom.violations(Z_EX, [500.0, 5000.0, 9000.0], [640.0] * 3)
    except ValueError:
        pass
    else:
        raise AssertionError("mismatched lengths accepted")


def test_each_route_gets_its_value_from_the_right_source():
    s = surrogate()
    blob = np.load(os.path.join(HERE, "data", "flash_dataset.npz"))
    z, p, T = blob["z_ood"][0], float(blob["p_ood_psia"][0]), float(blob["T_ood_R"][0])
    K = flash.wilson_k(p, T, C.TC_RANKINE, C.PC_PSIA, C.OMEGA)[None]
    solver = flash.solve_fv(z[None], K)[0][0]
    net = s._network(z[None], np.array([p]), np.array([T]))[0]
    assert abs(net - solver) > 1e-3                                  # the two sources differ here
    assert s.predict(z, p, T).FV == solver
    assert s.predict(z, p, T, on_unsupported="extrapolate").FV == net
    inside = s.predict(Z_EX, 796.43821, 640.0)
    assert inside.FV == s._network(Z_EX[None], np.array([796.43821]), np.array([640.0]))[0]


def test_network_is_never_called_for_single_phase_or_boundary_states():
    s = P.FlashSurrogate()
    calls = []
    original = s._network
    s._network = lambda z, p, T: (calls.append(len(z)), original(z, p, T))[1]
    pb, pd = saturation(Z_EX, 610.0)
    s.predict_many(np.repeat(Z_EX[None], 4, 0), [1.5 * pb, 0.5 * pd, pb, pd], [610.0] * 4)
    assert calls == []
    s.predict(Z_EX, 796.43821, 640.0)
    assert calls == [1]


def test_batch_rows_match_single_predictions_and_carry_their_own_metadata():
    s = surrogate()
    blob = np.load(os.path.join(HERE, "data", "flash_dataset.npz"))
    pb, pd = saturation(Z_EX, 610.0)
    z = np.vstack([Z_EX, Z_EX, blob["z"][blob["idx_test"][0]], blob["z_ood"][0], Z_EX, Z_EX])
    p = [1.5 * pb, 796.43821, float(blob["p_psia"][blob["idx_test"][0]]),
         float(blob["p_ood_psia"][0]), 796.43821, 0.5 * pd]
    T = [610.0, 640.0, float(blob["T_R"][blob["idx_test"][0]]), float(blob["T_ood_R"][0]), 700.0, 610.0]
    for mode in P.ON_UNSUPPORTED:
        batch = s.predict_many(z, p, T, on_unsupported=mode)
        for i, r in enumerate(batch):
            one = s.predict(z[i], p[i], T[i], on_unsupported=mode)
            a, b = r.as_dict(), one.as_dict()
            fa, fb = a.pop("FV"), b.pop("FV")
            assert a == b, (mode, i)
            # a float32 forward pass over a batch may round differently from one row
            assert (fa is None and fb is None) or abs(fa - fb) <= 1e-6, (mode, i, fa, fb)


def test_metadata_of_every_route():
    s = surrogate()
    blob = np.load(os.path.join(HERE, "data", "flash_dataset.npz"))
    pb, _ = saturation(Z_EX, 610.0)
    liquid = s.predict(Z_EX, 1.5 * pb, 610.0)
    assert (liquid.in_training_domain, liquid.domain_violations, liquid.model) == (None, (), None)
    assert liquid.sum_zK < 1.0 < liquid.sum_z_over_K and "single phase" in liquid.message
    z, p, T = blob["z_ood"][0], float(blob["p_ood_psia"][0]), float(blob["T_ood_R"][0])
    for mode, method, model in (("solver", "reference_solver_fallback", None),
                                ("status", "none", None),
                                ("extrapolate", "network_extrapolation", s.model_name)):
        r = s.predict(z, p, T, on_unsupported=mode)
        assert r.method == method and r.in_training_domain is False and r.model == model
        assert r.domain_violations and r.p_dew_psia < p < r.p_bubble_psia
    ok = s.predict(Z_EX, 796.43821, 640.0)
    assert ok.in_training_domain is True and ok.domain_violations == ()
