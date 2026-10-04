# Verification of the reference calculation against the notes' worked examples

`scripts/verify_flash.py` → `results/metrics/verify_flash.json`. Differences
are reported as measured.

| case | what the notes print | measured here |
|---|---|---|
| **A** p. 14 — bubble/dew point of the 7-component mixture at 150 °F | `p_b = 1590.8769` psia, `p_d = 1.9995691` psia | `1590.8767` psia (rel. diff **1.3e−07**), `1.9995763` psia (rel. diff **3.6e−06**); closure `Σ z_iK_i = 1.000000` at `p_b`, `Σ z_i/K_i = 1.000000` at `p_d` |
| **B** p. 15 — full flash at 796.43821 psia, 180 °F | seven `K_i`; `f_v = 0.1917145`; `x_i`, `y_i` | every `K_i` within 1e−05 of the printed 5 d.p. (one component differs by one unit in the last place); `F_V = 0.1916985`, difference **−1.6e−05**; `max\|x − x_notes\| = 7.1e−06`, `max\|y − y_notes\| = 3.9e−05` |
| **C** pp. 16–18 — C1/nC10 binary | phase labels; `F_V = 0.457`, `x = (0.263, 0.737)`, `y = (0.999, 0.001)` | labels agree; bisection root `0.4588879`, matching **both** closed forms the notes derive (pp. 17, 18) to 1e−09; `x = (0.2626, 0.7374)` |

## Case B: the cause of the 1.6e−05 difference is not established

The sheet's own `Target` cell prints a non-zero residual, `9.89E-06`, so it
stopped short of the root — consistent with a difference of this size. But the
residual measured here at the sheet's `f_v` is `−5.19e−05`, and that value is
not reproduced by any combination of the printed (5 d.p.) or recomputed `K`
with the printed (8 d.p.) or recomputed `z`. The sheet's own `y` column also
sums to 0.99996 rather than 1. Something beyond the stopping tolerance
differs; the evidence here does not identify it, so the difference is reported
rather than attributed.

## Case C: neglecting K₂ reproduces the printed 0.457

The two closed forms the notes derive (pp. 17, 18), evaluated with both
K-values, agree with the bisection at `0.4588879`. `K₂ = 0.0029` is about
0.3 % of 1. Neglecting it against 1 in those forms gives

```
(z₁K₁ − 1)/(K₁ − 1) = (0.6×3.8 − 1)/(3.8 − 1) = 1.28/2.8 = 0.4571428…
```

which to three decimals is **0.457**, the value the notes print. The full
expression gives 0.4588879, which to three decimals is 0.459, so the printed
figure is not the full expression rounded. The same neglect is also consistent
with their `x = (0.263, 0.737)`, and their `y₁ = 0.999` is `0.263 × 3.8`
recomputed from the already-rounded `x₁`, with `y₂ = 1 − 0.999`.

**The notes do not state this approximation.** It is offered as the numerical
explanation consistent with every figure they print, not as a claim about what
was done. Nothing was adjusted to force agreement: the inputs are the notes'
own `z` and `K`.
`tests/test_flash.py::test_neglecting_K2_reproduces_the_printed_0457` asserts
the arithmetic.

These three cases verify the **reference calculation** (Wilson K-values and
Rachford-Rice as taught). They are not tests of the network: cases A and B use
the seven-component table, and the binary of case C lies outside the network's
domain altogether ([`GUARDED_PREDICTION.md`](GUARDED_PREDICTION.md) §6).
