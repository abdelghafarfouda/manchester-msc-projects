"""Guarded prediction of the vapour fraction ``F_V`` -- the public prediction path.

The network in :mod:`sfp.nn` was trained only on **two-phase** states of one
seven-component mixture family inside one pressure-temperature window.  Its
sigmoid output returns a number between 0 and 1 for *any* input -- a
single-phase state, a pressure ten times outside the window, a composition
that does not sum to one -- so calling it directly is unsafe.
:class:`FlashSurrogate` puts three gates in front of it, in this order:

1. **Input validation** (:func:`validate_inputs`).  The composition must list
   the seven components in the order of the notes' table (pp. 14-15), with
   finite, non-negative mole fractions that sum to one; pressure and
   temperature must be finite, positive, and in psia and degrees Rankine.
   Malformed input raises :class:`InputError` with a short explanation.
   Nothing is normalised, reordered or converted silently.
2. **The course phase test** with Wilson K-values (notes p. 4, p. 8, p. 11).
   A single-phase state gets its phase result from the flash itself
   (``F_V = 0`` for a liquid, ``F_V = 1`` for a vapour).  A state on a phase
   boundary -- the bubble point (``SUM z_i K_i = 1``, ``F_V = 0``, p. 12) or
   the dew point (``SUM z_i / K_i = 1``, ``F_V = 1``, p. 13) -- is labelled as
   such, and the degenerate state in which every ``K_i = 1`` (both sums equal
   to one, so no split can be defined) is reported as indeterminate.  The
   network is never called for any of these.
3. **The training domain** (``configs/prediction_domain.json``, derived from
   the saved training split by ``scripts/derive_domain.py``) for two-phase
   states: pressure, temperature and every mole fraction against the ranges
   in the training data.  Inside, the network predicts.  Outside, by default
   the reference solver (bisection on Rachford-Rice, :mod:`sfp.flash`)
   answers and the result is labelled as a fallback; alternatively the call
   returns an ``unsupported`` status, or the network's answer clearly marked
   as an extrapolation.

These range checks are **marginal**: each variable is checked on its own.
They reject what is certainly outside the training data; they do not certify
that a combination of in-range values was covered by it.

Every result says which of these routes produced it (``method``), so a
network prediction, a single-phase calculation and a solver fallback can never
be confused.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field

import numpy as np

from . import components, flash

HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_DOMAIN_FILE = os.path.join(HERE, "configs", "prediction_domain.json")

#: Tolerance on ``|SUM z_i - 1|`` for a valid composition (not normalised).
COMPOSITION_SUM_TOL = 1.0e-6
#: Relative tolerance on the phase-test sums for a state *on* a boundary:
#: ``|SUM z_i K_i - 1| <= tol`` is a bubble point, ``|SUM z_i/K_i - 1| <= tol``
#: a dew point.  With Wilson K-values these sums are ``p_b/p`` and ``p/p_d``,
#: so the tolerance is a relative distance in pressure.
BOUNDARY_TOL = 1.0e-9

PRESSURE_UNITS = ("psia",)
TEMPERATURE_UNITS = ("R", "degR", "Rankine")
_UNIT_HINTS = {
    "psig": "p[psia] = p[psig] + the atmospheric pressure (about 14.7 psia)",
    "MPa": f"p[psia] = p[MPa] x {components.PSIA_PER_MPA}",
    "kPa": f"p[psia] = p[kPa] x {components.PSIA_PER_MPA / 1000:.7f}",
    "bar": "p[psia] = p[bar] x 14.50377",
    "Pa": f"p[psia] = p[Pa] x {components.PSIA_PER_MPA / 1e6:.4e}",
    "F": "T[R] = T[degF] + 460 (the notes' convention)",
    "degF": "T[R] = T[degF] + 460 (the notes' convention)",
    "C": "T[R] = 1.8 T[degC] + 32 + 460 (via degF, the notes' convention)",
    "degC": "T[R] = 1.8 T[degC] + 32 + 460 (via degF, the notes' convention)",
    "K": "T[R] = 1.8 T[K] (absolute scales; the notes' degF convention differs by 0.33 R)",
}

# Values of the ``phase``, ``method`` and ``status`` fields.
PHASES = ("two_phase", "liquid", "vapour", "bubble_point", "dew_point", "indeterminate")
METHODS = ("network", "network_extrapolation", "single_phase", "phase_boundary",
           "reference_solver_fallback", "none")
STATUSES = ("ok", "unsupported")
ON_UNSUPPORTED = ("solver", "status", "extrapolate")


class InputError(ValueError):
    """Raised for malformed input: wrong component set or order, non-finite
    values, negative or non-normalised mole fractions, non-positive pressure
    or temperature, or units other than psia and degrees Rankine."""


# --------------------------------------------------------------------------
# 1. input validation
# --------------------------------------------------------------------------
def _short(value):
    text = repr(value)
    return text if len(text) <= 80 else text[:77] + "..."


def _as_float_array(value, what):
    """Real numbers only: booleans, strings, complex numbers, None and ragged
    nested sequences are refused rather than converted."""
    if value is None:
        raise InputError(f"{what} is missing (None)")
    try:
        arr = np.asarray(value)
    except ValueError:
        raise InputError(f"{what} must be a rectangular array of numbers "
                         "(every state with the same number of values)") from None
    if arr.dtype.kind == "O":
        if any(not isinstance(v, (int, float, np.integer, np.floating)) or isinstance(v, bool)
               for v in arr.ravel()):
            raise InputError(f"{what} must contain real numbers only, in a rectangular array; "
                             f"got {_short(value)}")
        arr = arr.astype(float)
    if arr.dtype.kind not in "iuf":
        kind = {"b": "booleans", "U": "text", "S": "text", "c": "complex numbers"}.get(
            arr.dtype.kind, f"values of type {arr.dtype}")
        raise InputError(f"{what} must be real numbers, not {kind}; got {_short(value)}")
    return arr.astype(float)


def validate_inputs(z, p_psia, T_R, *, components_order=None,
                    pressure_unit="psia", temperature_unit="R"):
    """Check and shape one state or a batch of states.

    ``z``  a sequence of seven mole fractions in the order
           ``CO2, C1, C2, C3, C4, C5, C10`` (shape ``(7,)`` or ``(n, 7)``), or
           a mapping from those seven names to mole fractions.
    ``p_psia``, ``T_R``  scalars or length-``n`` sequences.
    ``components_order``  optional sequence of names describing the order of
           ``z``; it must equal the notes' order exactly.

    Returns ``(z, p, T, single)`` with ``z`` of shape ``(n, 7)`` and ``p``,
    ``T`` of shape ``(n,)``.  Raises :class:`InputError` otherwise.
    """
    names = components.NAMES
    if pressure_unit not in PRESSURE_UNITS:
        hint = _UNIT_HINTS.get(str(pressure_unit), "convert it to absolute pounds per square inch")
        raise InputError(f"pressure must be given in psia (got unit {pressure_unit!r}); "
                         f"convert it first: {hint}")
    if temperature_unit not in TEMPERATURE_UNITS:
        hint = _UNIT_HINTS.get(str(temperature_unit), "the notes use T[R] = T[degF] + 460")
        raise InputError(f"temperature must be given in degrees Rankine (got unit "
                         f"{temperature_unit!r}); convert it first: {hint}")

    if isinstance(z, Mapping):
        keys = set(z)
        if keys != set(names):
            missing = [n for n in names if n not in keys]
            extra = sorted(str(k) for k in keys - set(names))
            raise InputError(
                f"composition must give exactly the seven components {list(names)}; "
                f"missing {missing}, unexpected {extra}")
        if components_order is not None:
            raise InputError("components_order applies to sequences, not to a mapping")
        columns = [_as_float_array(z[n], f"mole fraction of {n}") for n in names]
        shapes = {c.shape for c in columns}
        if len(shapes) != 1 or columns[0].ndim > 1:
            raise InputError("a composition mapping must give every component either one "
                             "value or a 1-D sequence of the same length (one value per state)")
        z = np.stack(columns, axis=-1)          # (7,) for one state, (n, 7) for n states
    elif components_order is not None:
        if isinstance(components_order, (str, bytes)) or not isinstance(components_order, Iterable):
            raise InputError(f"components_order must be a sequence of the seven component names "
                             f"{list(names)}; got {_short(components_order)}")
        order = [str(c) for c in components_order]
        if order != list(names):
            raise InputError(
                f"components must be in the order of the notes' table {list(names)}; "
                f"got {order}. Reorder the mole fractions explicitly before calling")

    z = _as_float_array(z, "composition")
    if z.size == 0:
        raise InputError("no states given")
    if z.ndim not in (1, 2):
        raise InputError(f"composition must have shape (7,) or (n, 7); got {z.shape}")
    if z.ndim == 2 and z.shape[1] == 1 and z.shape[0] == components.N_COMPONENTS:
        raise InputError("composition was given as a column of shape (7, 1); pass one state as "
                         "seven values, shape (7,), or a batch as rows, shape (n, 7)")
    single = z.ndim == 1
    z = np.atleast_2d(z)
    if z.shape[-1] != components.N_COMPONENTS:
        raise InputError(
            f"composition has {z.shape[-1]} mole fractions; the network covers only the "
            f"seven-component set {list(names)} of the notes (pp. 14-15). Other "
            "mixtures, such as the C1/nC10 binary of pp. 16-18, can be flashed "
            "directly with sfp.flash")
    n = z.shape[0]
    if not np.isfinite(z).all():
        raise InputError("composition contains NaN or infinite values")
    if (z < 0.0).any():
        rows = np.flatnonzero((z < 0.0).any(axis=1))[:5].tolist()
        raise InputError(f"mole fractions must be non-negative (rows {rows})")
    total = z.sum(axis=1)
    bad = np.abs(total - 1.0) > COMPOSITION_SUM_TOL
    if bad.any():
        i = int(np.flatnonzero(bad)[0])
        raise InputError(
            f"mole fractions must sum to 1 within {COMPOSITION_SUM_TOL:g}; row {i} sums to "
            f"{total[i]:.9g}. They are not normalised automatically: use z_i = n_i / sum(n_j)")

    out = []
    for value, what, unit in ((p_psia, "pressure", "psia"), (T_R, "temperature", "R")):
        arr = _as_float_array(value, what)
        if arr.ndim > 1:
            raise InputError(f"{what} must be a scalar or a 1-D sequence; got shape {arr.shape}")
        if arr.ndim == 1 and arr.size not in (1, n):
            raise InputError(f"{what} has {arr.size} values for {n} compositions")
        arr = np.broadcast_to(arr.reshape(-1), (n,)).astype(float)
        if not np.isfinite(arr).all():
            raise InputError(f"{what} contains NaN or infinite values")
        if (arr <= 0.0).any():
            raise InputError(f"{what} must be positive (absolute {unit}); got {arr[arr <= 0.0][0]:g}")
        out.append(arr)
    if single and (np.ndim(p_psia) > 0 and np.size(p_psia) > 1 or
                   np.ndim(T_R) > 0 and np.size(T_R) > 1):
        raise InputError("one composition was given with several pressures or temperatures; "
                         "pass a composition per state")
    return z, out[0], out[1], single


# --------------------------------------------------------------------------
# 2. the course phase test
# --------------------------------------------------------------------------
def classify_phase(z, p_psia, T_R):
    """Wilson K-values and the phase test of notes p. 8 / p. 11, row by row.

    The test is made on ``SUM z_i K_i / SUM z_i`` and ``SUM (z_i / K_i) / SUM z_i``,
    i.e. on the Rachford-Rice function at its two ends, ``h(0)`` and ``h(1)``
    (p. 8), so it does not depend on how the mole fractions were rounded to sum
    to one.  Components with ``z_i = 0`` contribute nothing.

    Returns a dict of arrays: ``phase`` (one of :data:`PHASES`), ``sum_zK``,
    ``sum_z_over_K``, ``p_bubble_psia``, ``p_dew_psia`` and ``K``.  Inputs must
    already be validated (:func:`validate_inputs`).
    """
    T = np.asarray(T_R, float).reshape(-1, 1)
    p = np.asarray(p_psia, float).reshape(-1, 1)
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        K = flash.wilson_k(p, T, components.TC_RANKINE[None, :],
                           components.PC_PSIA[None, :], components.OMEGA[None, :])
        present = z > 0.0
        s_zk = np.where(present, z * K, 0.0).sum(-1)
        s_zok = np.where(present, z / np.where(present, K, 1.0), 0.0).sum(-1)
        pb, pd = flash.wilson_saturation_pressures(z, T[:, 0], components.TC_RANKINE,
                                                   components.PC_PSIA, components.OMEGA)
    bad = ~(np.isfinite(s_zk) & np.isfinite(s_zok) & (s_zk > 0) & (s_zok > 0))
    if bad.any():
        i = int(np.flatnonzero(bad)[0])
        raise InputError(
            f"Wilson K-values cannot be evaluated at p = {p[i, 0]:.10g} psia, T = {T[i, 0]:.10g} R "
            "(they overflow or underflow); these conditions are far outside any range the "
            "correlation or the network covers")
    total = z.sum(-1)
    r_zk, r_zok = s_zk / total, s_zok / total
    on_bubble = np.abs(r_zk - 1.0) <= BOUNDARY_TOL
    on_dew = np.abs(r_zok - 1.0) <= BOUNDARY_TOL
    phase = np.full(s_zk.shape, "two_phase", dtype=object)
    # Cauchy-Schwarz: (sum z K)(sum z/K) >= (sum z)^2, so r_zk and r_zok cannot both be < 1
    phase[r_zk < 1.0] = "liquid"           # no vapour can form
    phase[r_zok < 1.0] = "vapour"          # no liquid can form
    phase[on_bubble] = "bubble_point"
    phase[on_dew] = "dew_point"
    # both at once: the bubble and dew points coincide within the tolerance -- every
    # K_i of a component present is 1, e.g. an (effectively) pure component at its
    # own vapour pressure -- so p and T do not fix a vapour fraction
    phase[on_bubble & on_dew] = "indeterminate"
    return {"phase": phase, "sum_zK": s_zk, "sum_z_over_K": s_zok,
            "p_bubble_psia": pb, "p_dew_psia": pd, "K": K}


# --------------------------------------------------------------------------
# 3. the training domain
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class TrainingDomain:
    """Marginal ranges of the training split.  ``violations`` lists, per state,
    every variable outside its range; an empty list means only that no single
    variable is outside -- not that the combination was covered."""

    p_min_psia: float
    p_max_psia: float
    T_min_R: float
    T_max_R: float
    z_min: tuple
    z_max: tuple
    p_max_tested_psia: float = 4000.0
    source: str = ""

    @classmethod
    def from_file(cls, path=DEFAULT_DOMAIN_FILE):
        with open(path, encoding="utf-8") as fh:
            cfg = json.load(fh)
        lim = cfg["limits"]
        return cls(p_min_psia=lim["p_psia"][0], p_max_psia=lim["p_psia"][1],
                   T_min_R=lim["T_R"][0], T_max_R=lim["T_R"][1],
                   z_min=tuple(lim["z_min"]), z_max=tuple(lim["z_max"]),
                   p_max_tested_psia=lim.get("p_max_tested_psia", 4000.0),
                   source=os.path.relpath(path, HERE))

    def violations(self, z, p_psia, T_R):
        z = np.atleast_2d(np.asarray(z, float))
        n = z.shape[0]
        p_all = np.broadcast_to(np.ravel(np.asarray(p_psia, float)), (n,))
        T_all = np.broadcast_to(np.ravel(np.asarray(T_R, float)), (n,))
        out = []
        for zi, p, T in zip(z, p_all, T_all):
            v = []
            if p > self.p_max_psia:
                where = ("inside the reported pressure-extrapolation test (2000-4000 psia), "
                         "where the network's error was measured but it was not trained"
                         if p <= self.p_max_tested_psia else
                         "above every pressure tested, including the extrapolation test")
                v.append(f"p = {p:.10g} psia is above the training maximum "
                         f"{self.p_max_psia:g} psia ({where})")
            elif p < self.p_min_psia:
                v.append(f"p = {p:.10g} psia is below the training minimum {self.p_min_psia:g} psia")
            if not self.T_min_R <= T <= self.T_max_R:
                v.append(f"T = {T:.10g} R is outside the training range "
                         f"{self.T_min_R:g}-{self.T_max_R:g} R")
            for name, x, lo, hi in zip(components.NAMES, zi, self.z_min, self.z_max):
                if not lo <= x <= hi:
                    v.append(f"z_{name} = {x:.10g} is outside the training range "
                             f"{lo:.6g}-{hi:.6g}")
            out.append(v)
        return out


# --------------------------------------------------------------------------
# results
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Prediction:
    """One state's answer and how it was produced.

    ``FV`` is ``None`` when ``status == "unsupported"``.  ``method`` is one of
    :data:`METHODS`: ``network`` (inside the training domain),
    ``network_extrapolation`` (outside it, on request), ``single_phase``,
    ``phase_boundary``, ``reference_solver_fallback`` or ``none``.
    """

    FV: float | None
    status: str
    phase: str
    method: str
    in_training_domain: bool | None
    domain_violations: tuple = field(default_factory=tuple)
    sum_zK: float = math.nan
    sum_z_over_K: float = math.nan
    p_bubble_psia: float = math.nan
    p_dew_psia: float = math.nan
    model: str | None = None
    message: str = ""

    def as_dict(self):
        d = asdict(self)
        d["domain_violations"] = list(self.domain_violations)
        return d


_MESSAGES = {
    "liquid": "single phase (liquid): sum z_i K_i < 1, so F_V = 0 from the phase test; "
              "the network was not used",
    "vapour": "single phase (vapour): sum z_i / K_i < 1, so F_V = 1 from the phase test; "
              "the network was not used",
    "bubble_point": "on the bubble point (sum z_i K_i = 1): F_V = 0; the network was not used",
    "dew_point": "on the dew point (sum z_i / K_i = 1): F_V = 1; the network was not used",
    "indeterminate": "the bubble and dew points coincide (both phase-test sums equal 1 within "
                     "the tolerance): every K_i of a component present is 1, as for an "
                     "effectively pure component at its own vapour pressure, so p and T do not "
                     "fix a vapour fraction; the network was not used",
}


# --------------------------------------------------------------------------
# the guarded predictor
# --------------------------------------------------------------------------
class FlashSurrogate:
    """The trained network behind the three gates described in the module docstring.

    ``checkpoint`` defaults to the model named in ``configs/prediction_domain.json``
    (chosen there by a validation rule, without test data).
    """

    def __init__(self, checkpoint=None, domain_file=DEFAULT_DOMAIN_FILE):
        import torch  # imported here so the input and phase checks work without torch

        from . import data as sdata
        from . import nn as snn

        with open(domain_file, encoding="utf-8") as fh:
            cfg = json.load(fh)
        self.domain = TrainingDomain.from_file(domain_file)
        rel = checkpoint or cfg["default_checkpoint"]["path"]
        path = rel if os.path.isabs(rel) else os.path.join(HERE, rel)
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self._model = snn.simpleFFN(ck["n_inputs"], num_hidden=tuple(ck["hidden"]))
        self._model.load_state_dict(ck["model_state_dict"])
        self._model.eval()
        self._scaler = sdata.Standardiser().load_state_dict(ck["scaler"])
        self._predict = snn.predict
        self.model_name = os.path.splitext(os.path.basename(path))[0]

    def _network(self, z, p, T):
        X = np.concatenate([z, p[:, None], T[:, None]], axis=1)
        return self._predict(self._model, self._scaler.transform(X)).astype(float)

    def predict_many(self, z, p_psia, T_R, *, on_unsupported="solver",
                     components_order=None, pressure_unit="psia", temperature_unit="R"):
        """Guarded prediction for a batch of states; returns a list of :class:`Prediction`.

        ``on_unsupported`` decides what happens to a two-phase state outside the
        training domain: ``"solver"`` (default) answers with the reference
        solver and labels it ``reference_solver_fallback``; ``"status"``
        returns ``status = "unsupported"`` and no value; ``"extrapolate"``
        returns the network's answer labelled ``network_extrapolation``.
        """
        if on_unsupported not in ON_UNSUPPORTED:
            raise InputError(f"on_unsupported must be one of {ON_UNSUPPORTED}; "
                             f"got {on_unsupported!r}")
        z, p, T, _ = validate_inputs(z, p_psia, T_R, components_order=components_order,
                                     pressure_unit=pressure_unit,
                                     temperature_unit=temperature_unit)
        ph = classify_phase(z, p, T)
        phase = ph["phase"]
        two = phase == "two_phase"
        viol = self.domain.violations(z, p, T)
        inside = np.array([not v for v in viol])

        FV = np.full(z.shape[0], np.nan)
        use_net = two & (inside | (on_unsupported == "extrapolate"))
        if use_net.any():
            FV[use_net] = self._network(z[use_net], p[use_net], T[use_net])
        use_solver = two & ~inside & (on_unsupported == "solver")
        if use_solver.any():
            # an absent component (z_i = 0) contributes nothing to Rachford-Rice; give it
            # K = 1 so that an underflowed K_i cannot turn the solver's phase test into 0/0
            K_eff = np.where(z > 0.0, ph["K"], 1.0)
            FV[use_solver], _ = flash.solve_fv(z[use_solver], K_eff[use_solver])

        results = []
        for i in range(z.shape[0]):
            common = dict(sum_zK=float(ph["sum_zK"][i]),
                          sum_z_over_K=float(ph["sum_z_over_K"][i]),
                          p_bubble_psia=float(ph["p_bubble_psia"][i]),
                          p_dew_psia=float(ph["p_dew_psia"][i]))
            if phase[i] in ("liquid", "bubble_point", "vapour", "dew_point"):
                fv = 0.0 if phase[i] in ("liquid", "bubble_point") else 1.0
                method = "single_phase" if phase[i] in ("liquid", "vapour") else "phase_boundary"
                results.append(Prediction(fv, "ok", phase[i], method, None, (),
                                          message=_MESSAGES[phase[i]], **common))
            elif phase[i] == "indeterminate":
                results.append(Prediction(None, "unsupported", phase[i], "none", None, (),
                                          message=_MESSAGES["indeterminate"], **common))
            elif inside[i]:
                results.append(Prediction(float(FV[i]), "ok", "two_phase", "network", True, (),
                                          model=self.model_name,
                                          message="two-phase state inside the training domain: "
                                                  "network prediction", **common))
            else:
                v = tuple(viol[i])
                if on_unsupported == "solver":
                    results.append(Prediction(
                        float(FV[i]), "ok", "two_phase", "reference_solver_fallback", False, v,
                        message="two-phase state outside the training domain: answered by the "
                                "reference solver (bisection on Rachford-Rice), not the network",
                        **common))
                elif on_unsupported == "status":
                    results.append(Prediction(
                        None, "unsupported", "two_phase", "none", False, v,
                        message="two-phase state outside the training domain: no prediction "
                                "returned", **common))
                else:
                    results.append(Prediction(
                        float(FV[i]), "ok", "two_phase", "network_extrapolation", False, v,
                        model=self.model_name,
                        message="EXTRAPOLATION: two-phase state outside the training domain; "
                                "network answer returned on request, accuracy not established",
                        **common))
        return results

    def predict(self, z, p_psia, T_R, **kwargs):
        """Guarded prediction for one state; returns one :class:`Prediction`."""
        if isinstance(z, Mapping):
            if any(np.ndim(v) for v in z.values()):
                raise InputError("predict() takes one state: give one mole fraction per "
                                 "component, or use predict_many() for a batch")
            zz = z
        else:
            zz = _as_float_array(z, "composition")
            if zz.ndim != 1:
                raise InputError(f"predict() takes one state of seven mole fractions, shape (7,); "
                                 f"got shape {zz.shape}. Use predict_many() for a batch")
            zz = zz[None, :]
        if np.ndim(p_psia) or np.ndim(T_R):
            raise InputError("predict() takes scalar pressure and temperature; "
                             "use predict_many() for a batch")
        return self.predict_many(zz, [p_psia], [T_R], **kwargs)[0]
