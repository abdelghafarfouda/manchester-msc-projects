"""Check the flash solver against every worked example in the module's notes.

All three cases come from ``Models/3 - Two-phase flash calculation.pdf``.

Case A (p. 14)  bubble and dew point of the seven-component example mixture at
                150 F, using the printed Pc, Tc and acentric factors.
                The notes give p_b = 1590.8769 psia and p_d = 1.9995691 psia.

Case B (p. 15)  a full flash of the same mixture at 796.43821 psia and 180 F.
                The notes give every K_i, f_v = 0.1917145, and x_i, y_i.

Case C (pp. 16-18)  the C1/nC10 binary at two states.  The notes give the
                phase labels and, for the two-phase state, F_V = 0.457 with
                x = (0.263, 0.737) and y = (0.999, 0.001).  This case also
                carries the arithmetic showing that neglecting K2 reproduces
                the printed 0.457, where the full expression gives 0.458888.

Run:  python scripts/verify_flash.py
Writes results/metrics/verify_flash.json
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from sfp import components as C  # noqa: E402
from sfp import flash  # noqa: E402

HERE = os.path.join(os.path.dirname(__file__), "..")


def rel(a, b):
    return abs(a - b) / abs(b)


def main() -> int:
    report = {"source": "Models/3 - Two-phase flash calculation.pdf"}
    fail = []

    # =====================================================================
    # Case A -- p. 14: saturation pressures of the example mixture at 150 F
    # =====================================================================
    z = C.EXAMPLE_MOLES / C.EXAMPLE_MOLES.sum()
    T = 610.0  # R  (150 F + 460, the sheet's own conversion)
    expo = np.exp(5.37 * (1.0 + C.OMEGA) * (1.0 - C.TC_RANKINE / T))
    Bpi = z * C.PC_PSIA * expo          # the sheet's "Bpi" column
    Dpi = z / (C.PC_PSIA * expo)        # the sheet's "Dpi" column
    pb, pd = float(Bpi.sum()), float(1.0 / Dpi.sum())

    report["case_A_saturation_pressures_p14"] = {
        "T_R": T, "T_F": T - 460.0,
        "z_from_moles": z.tolist(),
        "z_printed_in_notes": C.EXAMPLE_Z.tolist(),
        "max_abs_diff_in_z": float(np.abs(z - C.EXAMPLE_Z).max()),
        "p_b_psia_computed": pb, "p_b_psia_notes": 1590.8769,
        "p_d_psia_computed": pd, "p_d_psia_notes": 1.9995691,
        "p_b_relative_error": rel(pb, 1590.8769),
        "p_d_relative_error": rel(pd, 1.9995691),
        "sum_zK_at_pb": float((z * flash.wilson_k(pb, T, C.TC_RANKINE,
                                                  C.PC_PSIA, C.OMEGA)).sum()),
        "sum_z_over_K_at_pd": float((z / flash.wilson_k(pd, T, C.TC_RANKINE,
                                                        C.PC_PSIA, C.OMEGA)).sum()),
    }
    if rel(pb, 1590.8769) > 1e-5 or rel(pd, 1.9995691) > 1e-4:
        fail.append("case A saturation pressures")
    if np.abs(z - C.EXAMPLE_Z).max() > 5e-8:
        fail.append("case A composition transcription")

    # =====================================================================
    # Case B -- p. 15: flash of the same mixture at 796.43821 psia, 180 F
    # =====================================================================
    p, T = 796.43821, 640.0
    K = flash.wilson_k(p, T, C.TC_RANKINE, C.PC_PSIA, C.OMEGA)
    K_notes = np.array([3.57859, 10.34885, 2.03398, 0.22518,
                        0.09622, 0.07069, 0.00151])
    FV, n_it = flash.solve_fv(z[None, :], K[None, :])
    x, y = flash.phase_compositions(FV, z[None, :], K[None, :])
    x_notes = np.array([0.025937, 0.069404, 0.064695, 0.136565,
                        0.131273, 0.188649, 0.383486])
    y_notes = np.array([0.092819, 0.718255, 0.131588, 0.030751,
                        0.012630, 0.013335, 0.000579])
    h_at_notes_fv = float(flash.rachford_rice(np.array([0.1917145]),
                                              z[None, :], K[None, :])[0])

    report["case_B_flash_p15"] = {
        "p_psia": p, "T_R": T, "T_F": T - 460.0,
        "K_computed": K.tolist(),
        "K_notes": K_notes.tolist(),
        "K_max_abs_diff_against_5dp_print": float(
            np.abs(np.round(K, 5) - K_notes).max()),
        "FV_computed": float(FV[0]),
        "FV_notes": 0.1917145,
        "FV_difference": float(FV[0] - 0.1917145),
        "bisection_iterations": int(n_it),
        "residual_at_our_root": float(flash.rachford_rice(FV, z[None, :],
                                                          K[None, :])[0]),
        "residual_at_notes_FV": h_at_notes_fv,
        "notes_stated_residual": 9.89e-06,
        "x_computed": x[0].tolist(), "x_notes": x_notes.tolist(),
        "y_computed": y[0].tolist(), "y_notes": y_notes.tolist(),
        "x_max_abs_diff": float(np.abs(x[0] - x_notes).max()),
        "y_max_abs_diff": float(np.abs(y[0] - y_notes).max()),
        "comment": ("the sheet's own Target cell prints a non-zero residual "
                    "(9.89e-06), so it stopped short of the root; our bisection "
                    "converges to ~1e-12.  That is consistent with the 1.6e-05 "
                    "difference in f_v, but it does not fully explain it: the "
                    "residual we measure at the sheet's f_v is -5.19e-05 and "
                    "does not reproduce the printed 9.89e-06 under any "
                    "combination of the printed (5 d.p.) or exact K and the "
                    "printed (8 d.p.) or exact z.  The sheet's own y column "
                    "also sums to 0.99996 rather than 1.  The cause of the "
                    "remaining difference is therefore NOT established."),
        "residual_at_notes_FV_with_printed_5dp_K": None,
        "notes_printed_y_column_sum": 0.99996,
    }
    # allow one unit in the last printed place (the sheet prints K to 5 d.p.)
    if np.abs(np.round(K, 5) - K_notes).max() > 1.5e-5:
        fail.append("case B K-values against the printed 5 d.p.")
    if abs(FV[0] - 0.1917145) > 5e-5:
        fail.append("case B F_V")
    if np.abs(x[0] - x_notes).max() > 2e-5 or np.abs(y[0] - y_notes).max() > 5e-5:
        fail.append("case B phase compositions")

    # =====================================================================
    # Case C -- pp. 16-18: the C1/nC10 binary
    # =====================================================================
    zb = np.array([[0.60, 0.40]])

    Ka = np.array([[1.4, 0.13]])
    s1, s2 = flash.phase_sums(zb, Ka)
    label = flash.phase_label(zb, Ka)

    Kb = np.array([[3.8, 0.0029]])
    s1b, s2b = flash.phase_sums(zb, Kb)
    FVb, n_itb = flash.solve_fv(zb, Kb)
    xb, yb = flash.phase_compositions(FVb, zb, Kb)

    K1, K2 = Kb[0]
    z1, z2 = zb[0]
    exact_form1 = (1.0 - z1 * K1 - z2 * K2) / ((K1 - 1.0) * (K2 - 1.0))
    exact_form2 = (z1 * (K1 - K2) / (1.0 - K2) - 1.0) / (K1 - 1.0)
    # the same two closed forms with K2 neglected against 1
    drop_form1 = (1.0 - z1 * K1) / ((K1 - 1.0) * (0.0 - 1.0))
    drop_form2 = (z1 * K1 - 1.0) / (K1 - 1.0)
    drop_den_only = (1.0 - z1 * K1 - z2 * K2) / ((K1 - 1.0) * (0.0 - 1.0))
    drop_num_only = (1.0 - z1 * K1) / ((K1 - 1.0) * (K2 - 1.0))

    def comps(fv, neglect_K2):
        k2 = 0.0 if neglect_K2 else K2
        x1 = z1 / (fv * (K1 - 1.0) + 1.0)
        x2 = z2 / (fv * (k2 - 1.0) + 1.0)
        return x1, x2

    x_exact = comps(exact_form1, False)
    x_drop = comps(drop_form2, True)

    report["case_C_binary_pp16_18"] = {
        "state_a": {
            "K": Ka[0].tolist(),
            "sum_zK": float(s1[0]), "notes_sum_zK": 0.89,
            "sum_z_over_K": float(s2[0]), "notes_sum_z_over_K": 3.50,
            "phase": str(label), "notes_phase": "compressed (undersaturated) liquid",
        },
        "state_b": {
            "K": Kb[0].tolist(),
            "sum_zK": float(s1b[0]), "notes_sum_zK": 2.28,
            "sum_z_over_K": float(s2b[0]), "notes_sum_z_over_K": 138.0,
            "FV_bisection": float(FVb[0]),
            "bisection_iterations": int(n_itb),
            "residual_at_root": float(flash.rachford_rice(FVb, zb, Kb)[0]),
            "x": xb[0].tolist(), "y": yb[0].tolist(),
            "notes_FV": 0.457, "notes_x": [0.263, 0.737], "notes_y": [0.999, 0.001],
        },
        "why_the_notes_print_0_457": {
            "exact_closed_form_1_p17": exact_form1,
            "exact_closed_form_2_p18": exact_form2,
            "the_two_exact_forms_agree_to": abs(exact_form1 - exact_form2),
            "exact_value_rounds_to_3dp": round(exact_form1, 3),
            "K2_neglected_in_form_1": drop_form1,
            "K2_neglected_in_form_2": drop_form2,
            "K2_neglected_rounds_to_3dp": round(drop_form2, 3),
            "K2_neglected_in_denominator_only": drop_den_only,
            "K2_neglected_in_numerator_only": drop_num_only,
            "x_from_exact_FV": list(x_exact),
            "x_from_neglected_FV_also_neglecting_K2_in_x2": list(x_drop),
            "y1_from_rounded_x1_times_K1": round(0.263 * K1, 4),
            "conclusion": (
                "K2 = 0.0029 is about 0.3 % of 1.  Neglecting it against 1 in "
                "the binary closed form gives (z1*K1 - 1)/(K1 - 1) = "
                "(2.28 - 1)/2.8 = 0.4571428..., which prints as 0.457 and so "
                "reproduces the printed value.  The full expression gives "
                "0.4588879, which prints as 0.459, so the printed figure is "
                "not the full expression rounded.  The same neglect is also "
                "consistent with the printed x = (0.263, 0.737), and their "
                "y2 = 0.001 is 1 - y1 with y1 recomputed from the "
                "already-rounded x1 = 0.263.  The notes do not state this "
                "approximation; it is offered as the numerical explanation "
                "consistent with every figure they print, not as a claim "
                "about what was done."
            ),
        },
    }
    if abs(s1[0] - 0.892) > 5e-3 or abs(s2[0] - 3.5055) > 5e-3:
        fail.append("case C state (a) phase sums")
    if label != "liquid":
        fail.append("case C state (a) phase label")
    if abs(FVb[0] - exact_form1) > 1e-9 or abs(FVb[0] - exact_form2) > 1e-9:
        fail.append("case C bisection vs the notes' two closed forms")
    if abs(drop_form2 - 0.457142857142857) > 1e-12:
        fail.append("case C neglected-term reconstruction of 0.457")
    if abs(xb[0, 0] - 0.263) > 1e-3 or abs(xb[0, 1] - 0.737) > 1e-3:
        fail.append("case C phase compositions vs notes")

    report["failures"] = fail
    report["passed"] = not fail

    out = os.path.join(HERE, "results", "metrics", "verify_flash.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(report, fh, indent=2)

    # ------------------------------------------------------------------ print
    A = report["case_A_saturation_pressures_p14"]
    print("CASE A  p.14  seven-component mixture, 150 F (610 R)")
    print(f"  z transcription max |diff| vs printed z : {A['max_abs_diff_in_z']:.1e}")
    print(f"  bubble point p_b = {pb:12.4f} psia   notes 1590.8769   "
          f"rel err {A['p_b_relative_error']:.2e}")
    print(f"  dew point    p_d = {pd:12.7f} psia   notes    1.9995691   "
          f"rel err {A['p_d_relative_error']:.2e}")
    print(f"  sum z_i K_i at p_b = {A['sum_zK_at_pb']:.6f}  (must be 1)")
    print(f"  sum z_i/K_i at p_d = {A['sum_z_over_K_at_pd']:.6f}  (must be 1)")

    B = report["case_B_flash_p15"]
    print("\nCASE B  p.15  same mixture, 796.43821 psia, 180 F (640 R)")
    print(f"  K-values agree with the printed 5 d.p. to {B['K_max_abs_diff_against_5dp_print']:.0e}")
    print(f"  F_V bisection = {B['FV_computed']:.7f}   notes 0.1917145   "
          f"diff {B['FV_difference']:+.2e}")
    print(f"  residual at our root  {B['residual_at_our_root']:+.2e}")
    print(f"  residual at notes f_v {B['residual_at_notes_FV']:+.2e}  "
          f"(notes print Target = {B['notes_stated_residual']:.2e})")
    print(f"  max |x - x_notes| = {B['x_max_abs_diff']:.1e},  "
          f"max |y - y_notes| = {B['y_max_abs_diff']:.1e}")

    D = report["case_C_binary_pp16_18"]
    print("\nCASE C  pp.16-18  C1/nC10 binary")
    print(f"  (a) sum z_i K_i = {s1[0]:.4f} (notes 0.89), "
          f"sum z_i/K_i = {s2[0]:.4f} (notes 3.50) -> {label}")
    print(f"  (b) F_V bisection          = {FVb[0]:.7f}")
    print(f"      notes' closed form 1   = {exact_form1:.7f}")
    print(f"      notes' closed form 2   = {exact_form2:.7f}")
    print(f"      same forms with K2 = 0 = {drop_form2:.7f}  -> prints as 0.457")
    print(f"      exact value would print as {round(exact_form1, 3)}")
    print(f"      x exact = ({x_exact[0]:.4f}, {x_exact[1]:.4f}); "
          f"x with K2 = 0 = ({x_drop[0]:.4f}, {x_drop[1]:.4f}); notes (0.263, 0.737)")
    print(f"      y1 from the notes' rounded x1: 0.263 x 3.8 = "
          f"{0.263 * K1:.4f} -> 0.999, so their y2 = 1 - 0.999 = 0.001")

    print("\n" + ("PASS" if not fail else "FAIL: " + ", ".join(fail)))
    print(f"written: {os.path.relpath(out, HERE)}")
    return 0 if not fail else 1


if __name__ == "__main__":
    raise SystemExit(main())
