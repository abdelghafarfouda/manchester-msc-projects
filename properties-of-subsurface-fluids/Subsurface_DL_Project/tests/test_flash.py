"""Tests for the flash solver, the component table, the split and the physics term."""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from sfp import components as C  # noqa: E402
from sfp import data as sdata  # noqa: E402
from sfp import flash  # noqa: E402
from sfp import nn as snn  # noqa: E402


# --------------------------------------------------------------------------
# the component table, against the two calculations the notes do with it
# --------------------------------------------------------------------------
def test_component_table_shapes_and_printed_z():
    assert len(C.NAMES) == C.N_COMPONENTS == 7
    for a in (C.PC_PSIA, C.TC_RANKINE, C.OMEGA, C.EXAMPLE_MOLES, C.EXAMPLE_Z):
        assert a.shape == (7,)
    z = C.EXAMPLE_MOLES / C.EXAMPLE_MOLES.sum()
    assert np.abs(z - C.EXAMPLE_Z).max() < 5e-8     # p. 14 zi column
    assert abs(C.EXAMPLE_Z.sum() - 1.0) < 1e-6


def test_saturation_pressures_of_the_p14_example():
    """Models/3 - ..., p. 14: p_b = 1590.8769 psia, p_d = 1.9995691 psia."""
    z = C.EXAMPLE_MOLES / C.EXAMPLE_MOLES.sum()
    T = 610.0
    e = np.exp(5.37 * (1.0 + C.OMEGA) * (1.0 - C.TC_RANKINE / T))
    pb = float((z * C.PC_PSIA * e).sum())
    pd = float(1.0 / (z / (C.PC_PSIA * e)).sum())
    assert abs(pb - 1590.8769) / 1590.8769 < 1e-5
    assert abs(pd - 1.9995691) / 1.9995691 < 1e-4


def test_flash_of_the_p15_example():
    """Models/3 - ..., p. 15: K_i, f_v = 0.1917145, x_i and y_i."""
    z = (C.EXAMPLE_MOLES / C.EXAMPLE_MOLES.sum())[None, :]
    K = flash.wilson_k(796.43821, 640.0, C.TC_RANKINE, C.PC_PSIA, C.OMEGA)[None, :]
    K_notes = np.array([3.57859, 10.34885, 2.03398, 0.22518,
                        0.09622, 0.07069, 0.00151])
    assert np.abs(np.round(K[0], 5) - K_notes).max() <= 1.5e-5
    FV, _ = flash.solve_fv(z, K)
    assert abs(FV[0] - 0.1917145) < 5e-5        # sheet stops at a non-zero residual
    x, y = flash.phase_compositions(FV, z, K)
    assert np.abs(x[0] - np.array([0.025937, 0.069404, 0.064695, 0.136565,
                                   0.131273, 0.188649, 0.383486])).max() < 2e-5
    assert abs(x.sum() - 1.0) < 1e-12 and abs(y.sum() - 1.0) < 1e-12


# --------------------------------------------------------------------------
# the binary worked example, and the arithmetic behind the notes' 0.457
# --------------------------------------------------------------------------
def test_binary_worked_example_pp16_18():
    z = np.array([[0.60, 0.40]])
    K = np.array([[3.8, 0.0029]])
    FV, _ = flash.solve_fv(z, K)
    K1, K2 = K[0]
    form1 = (1.0 - z[0, 0] * K1 - z[0, 1] * K2) / ((K1 - 1.0) * (K2 - 1.0))
    form2 = (z[0, 0] * (K1 - K2) / (1.0 - K2) - 1.0) / (K1 - 1.0)
    assert abs(FV[0] - form1) < 1e-9            # notes p. 17
    assert abs(FV[0] - form2) < 1e-9            # notes p. 18
    x, y = flash.phase_compositions(FV, z, K)
    assert abs(x[0, 0] - 0.263) < 1e-3 and abs(x[0, 1] - 0.737) < 1e-3
    assert abs(x.sum() - 1.0) < 1e-12 and abs(y.sum() - 1.0) < 1e-12


def test_neglecting_K2_reproduces_the_printed_0457():
    """Neglecting K2 = 0.0029 against 1 reproduces the printed 0.457.

    The notes do not state this approximation; it is offered as the numerical
    explanation that is consistent with every figure they print.
    """
    z1, K1, K2 = 0.60, 3.8, 0.0029
    exact = (1.0 - z1 * K1 - 0.40 * K2) / ((K1 - 1.0) * (K2 - 1.0))
    dropped = (z1 * K1 - 1.0) / (K1 - 1.0)
    assert abs(dropped - 0.457142857142857) < 1e-12
    assert round(dropped, 3) == 0.457
    assert round(exact, 3) == 0.459             # the full expression does not round to 0.457
    assert abs(exact - dropped) > 1.7e-3


def test_phase_label_liquid_case():
    z = np.array([[0.60, 0.40]])
    K = np.array([[1.4, 0.13]])
    s1, s2 = flash.phase_sums(z, K)
    assert s1[0] < 1.0 and s2[0] > 1.0
    assert flash.phase_label(z, K) == "liquid"


# --------------------------------------------------------------------------
# the generated dataset
# --------------------------------------------------------------------------
def test_residual_is_zero_at_the_root():
    ds = sdata.build_dataset(40, sdata.Domain(), seed=7, states_per_realisation=10)
    h = flash.rachford_rice(ds["FV"], ds["z"], ds["K"])
    assert np.abs(h).max() < 1e-8
    assert np.isfinite(ds["FV"]).all()


def test_material_balance_of_the_solution():
    """z_i = x_i F_L + y_i F_V must hold exactly (notes p. 2)."""
    ds = sdata.build_dataset(30, sdata.Domain(), seed=11, states_per_realisation=10)
    x, y = flash.phase_compositions(ds["FV"], ds["z"], ds["K"])
    FV = ds["FV"][:, None]
    assert np.abs(x * (1.0 - FV) + y * FV - ds["z"]).max() < 1e-10
    assert np.abs(x.sum(1) - 1.0).max() < 1e-10
    assert np.abs(y.sum(1) - 1.0).max() < 1e-10


def test_compositions_come_from_integer_moles():
    """z_i = n_i / sum(n_j) -- flash notes p. 2, and the ni/zi columns on p. 14."""
    d = sdata.Domain()
    ds = sdata.build_dataset(25, d, seed=13, states_per_realisation=8)
    n = ds["moles"]
    assert n.dtype.kind in "iu"
    assert n.min() >= d.moles_min and n.max() <= d.moles_max
    assert np.abs(ds["z"] - n / n.sum(axis=1, keepdims=True)).max() < 1e-15
    assert (ds["z"] >= 0).all()
    assert np.abs(ds["z"].sum(1) - 1.0).max() < 1e-12
    # the notes' own mixture is reproduced by the same rule
    assert np.abs(C.EXAMPLE_MOLES / C.EXAMPLE_MOLES.sum() - C.EXAMPLE_Z).max() < 5e-8


def test_dataset_stays_inside_the_supplied_pT_window():
    d = sdata.Domain()
    ds = sdata.build_dataset(30, d, seed=5, states_per_realisation=10)
    assert ds["T_R"].min() >= d.T_min_R and ds["T_R"].max() <= d.T_max_R
    assert ds["p_psia"].min() >= d.p_min_psia
    assert ds["p_psia"].max() <= d.p_max_train_psia <= d.p_max_psia
    assert np.abs(ds["z"].sum(1) - 1.0).max() < 1e-12


def test_torch_residual_matches_numpy():
    ds = sdata.build_dataset(20, sdata.Domain(), seed=3, states_per_realisation=8)
    h_np = flash.rachford_rice(ds["FV"], ds["z"], ds["K"])
    h_t = snn.rachford_rice_torch(
        torch.as_tensor(ds["FV"], dtype=torch.float64),
        torch.as_tensor(ds["z"], dtype=torch.float64),
        torch.as_tensor(ds["K"], dtype=torch.float64),
    ).numpy()
    assert np.abs(h_np - h_t).max() < 1e-10


def test_group_split_keeps_mixtures_together():
    rid = np.repeat(np.arange(50), 7)
    a, b, c = sdata.group_split(rid, (0.6, 0.2, 0.2), seed=1)
    assert a.size + b.size + c.size == rid.size
    for i, j in ((a, b), (a, c), (b, c)):
        assert np.intersect1d(rid[i], rid[j]).size == 0


def test_standardiser_uses_training_statistics_only():
    rng = np.random.default_rng(0)
    Xtr = rng.normal(3.0, 2.0, size=(500, 4))
    Xte = rng.normal(-1.0, 5.0, size=(100, 4))
    sc = sdata.Standardiser().fit(Xtr)
    assert np.abs(sc.transform(Xtr).mean(0)).max() < 1e-12
    assert np.abs(sc.transform(Xte).mean(0)).max() > 0.5


def test_sigmoid_output_is_inside_the_open_unit_interval():
    snn.set_seed(0)
    model = snn.simpleFFN(9, num_hidden=(8, 8))
    out = snn.predict(model, np.random.default_rng(0).normal(size=(64, 9)))
    assert out.min() > 0.0 and out.max() < 1.0
