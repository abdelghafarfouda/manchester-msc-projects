"""Building the dataset of flash calculations, and splitting it honestly.

The ground truth is the module's own flash model (``sfp.flash``): Wilson /
Whitson K-values (``Models/3 - Two-phase flash calculation.pdf`` p. 4 and the
worked sheets on pp. 14-15) fed into Rachford-Rice (p. 8).

**Every physical input is taken from the supplied material.** The seven
components and their critical constants and acentric factors are the table on
pp. 14-15, transcribed in :mod:`sfp.components`. The temperature and pressure
window is the span of the flash states printed in the same file
(150-220 F, 2-4000 psia; see ``components.SUPPLIED_STATES``).

**Composition generation follows the supplied material's own construction.**
The notes define ``z_i = n_i / n`` with ``n`` the total moles (flash notes
p. 2) and build the worked mixture that way: the table on p. 14 lists integer
moles ``ni = 5, 25, 10, 15, 14, 20, 40`` and the mole fractions ``zi`` are
those divided by 129.  This module does the same -- it draws an integer mole
charge for each of the seven components and normalises by the total, using
``numpy``'s uniform integer draw, the kind of draw demonstrated in the
supplied notebooks (``np.random.randint`` /
``torch.randint`` in ``Day01``, ``Day02`` and ``Day04``).  Mole fractions are
non-negative and sum to one by construction, with no distribution assumed over
the composition simplex.  A Dirichlet draw was used in an earlier version of
this project; it appears nowhere in either supplied source and has been
removed.

**What is an ordinary project choice, prescribed by nobody:** the range the
integer mole charges are drawn from, how many mixtures and how many states per
mixture are generated, that pressure is drawn log-uniformly inside the
supplied window, and where inside that window the training band stops and the
extrapolation band begins.  These set the size and coverage of the experiment.
They introduce no physical property value and no physical law.

Vocabulary
----------
*realisation* -- one mixture: one feed composition ``z`` over the seven
components. A realisation is evaluated at many (p, T) states, so many rows of
the dataset share the same underlying mixture. Those rows are **never** split
across train and test: the split is by realisation (``group_split`` below).
Splitting by row would let the network see the same mixture at a neighbouring
pressure during training and would flatter the test score.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from . import components, flash


@dataclass(frozen=True)
class Domain:
    """The sampling window.

    Temperatures and pressures are in the units of the source table -- degrees
    Rankine and psia. The defaults are the span of the flash states printed in
    ``Models/3 - Two-phase flash calculation.pdf`` (see
    ``components.SUPPLIED_STATES``); ``p_max_train`` splits that span into a
    training band and a higher-pressure extrapolation band, which is a design
    choice, not a physical boundary.
    """

    T_min_R: float = components.T_MIN_RANKINE       # 610 R = 150 F
    T_max_R: float = components.T_MAX_RANKINE       # 680 R = 220 F
    p_min_psia: float = components.P_MIN_PSIA       # 2 psia
    p_max_psia: float = components.P_MAX_PSIA       # 4000 psia
    p_max_train_psia: float = 2000.0                # training band ends here
    # integer mole charges per component: z_i = n_i / sum(n_j), the
    # construction of flash notes p. 2 and of the table on p. 14.  The range
    # is a project choice; the example's own charges run from 5 to 40 moles.
    moles_min: int = 1
    moles_max: int = 40

    def as_dict(self):
        d = asdict(self)
        d["components"] = list(components.NAMES)
        d["Tc_R"] = components.TC_RANKINE.tolist()
        d["pc_psia"] = components.PC_PSIA.tolist()
        d["omega"] = components.OMEGA.tolist()
        d["composition_rule"] = (
            "z_i = n_i / sum(n_j) from integer mole charges drawn uniformly in "
            f"[{self.moles_min}, {self.moles_max}] -- the construction of "
            "Models/3 - ... p. 2 and of the ni/zi columns on p. 14"
        )
        d["source_of_component_constants"] = (
            "Models/3 - Two-phase flash calculation.pdf, pp. 14-15"
        )
        d["source_of_pT_window"] = [
            f"{T:.0f} R / {p:.4f} psia -- {where}"
            for T, p, where in components.SUPPLIED_STATES
        ]
        return d


def wilson_k_grid(p_psia, T_R):
    """K-values for the fixed component set at the given states.

    ``p_psia`` and ``T_R`` have shape ``(n, 1)``; returns ``(n, 7)``.
    """
    return flash.wilson_k(
        p_psia, T_R,
        components.TC_RANKINE[None, :],
        components.PC_PSIA[None, :],
        components.OMEGA[None, :],
    )


def build_dataset(
    n_real,
    domain,
    seed,
    states_per_realisation=20,
    p_range_psia=None,
    max_attempts=200,
):
    """Draw mixtures, evaluate them at (p, T) states, keep the two-phase ones.

    Single-phase states are discarded by the module's own physical-root test
    (p. 8): there is no vapour fraction to predict there. The number of states
    kept per realisation is fixed so that no mixture dominates the dataset.

    Compositions come from integer mole charges normalised by their total,
    the construction of flash notes p. 2 and of the p. 14 table.  Pressure is
    drawn log-uniformly across the window, because the printed two-phase
    window of the example mixture spans three decades (dew point 2.0 psia to
    bubble point 1590.9 psia, p. 14).  Both the mole range and the log draw are
    project choices.
    """
    rng = np.random.default_rng(seed)
    n_c = components.N_COMPONENTS
    # z_i = n_i / sum(n_j), exactly as the p. 14 table builds its zi column.
    moles_all = rng.integers(domain.moles_min, domain.moles_max + 1,
                             size=(n_real, n_c))
    z_all = moles_all / moles_all.sum(axis=1, keepdims=True)
    p_lo, p_hi = (p_range_psia if p_range_psia is not None
                  else (domain.p_min_psia, domain.p_max_train_psia))
    log_lo, log_hi = np.log(p_lo), np.log(p_hi)

    rows_z, rows_p, rows_T, rows_K, rows_rid, rows_n = [], [], [], [], [], []
    kept = np.zeros(n_real, dtype=int)

    for r in range(n_real):
        z = z_all[r]
        found = 0
        for _ in range(max_attempts):
            need = states_per_realisation - found
            if need <= 0:
                break
            m = max(need * 4, 32)
            T = rng.uniform(domain.T_min_R, domain.T_max_R, size=(m, 1))
            p = np.exp(rng.uniform(log_lo, log_hi, size=(m, 1)))
            K = wilson_k_grid(p, T)
            zz = np.repeat(z[None, :], m, axis=0)
            idx = np.flatnonzero(flash.is_two_phase(zz, K))[:need]
            if idx.size == 0:
                continue
            rows_z.append(zz[idx])
            rows_p.append(p[idx, 0])
            rows_T.append(T[idx, 0])
            rows_K.append(K[idx])
            rows_rid.append(np.full(idx.size, r))
            rows_n.append(np.repeat(moles_all[r][None, :], idx.size, axis=0))
            found += idx.size
        kept[r] = found

    if not rows_z:
        raise RuntimeError("no two-phase states found in this window")

    z = np.concatenate(rows_z)
    K = np.concatenate(rows_K)
    FV, _ = flash.solve_fv(z, K)

    return {
        "z": z,
        "moles": np.concatenate(rows_n),
        "p_psia": np.concatenate(rows_p),
        "T_R": np.concatenate(rows_T),
        "K": K,
        "realisation": np.concatenate(rows_rid),
        "FV": FV,
        "kept_per_realisation": kept,
        "n_realisations": n_real,
        "p_range_psia": (float(p_lo), float(p_hi)),
        "moles_range": (int(domain.moles_min), int(domain.moles_max)),
    }


def features(ds):
    """Network inputs: the seven mole fractions, the pressure and the
    temperature -- what an engineer would type into the spreadsheet on p. 15.

    Nothing is pre-transformed. In particular the network is given raw
    pressure, not ``1/p`` or ``log p``, so it has to learn the strong inverse
    pressure dependence of the K-values for itself rather than being handed it.
    """
    return np.concatenate(
        [ds["z"], ds["p_psia"][:, None], ds["T_R"][:, None]], axis=1
    )


def feature_names():
    return [f"z_{n}" for n in components.NAMES] + ["p_psia", "T_R"]


def group_split(realisation_id, fractions, seed):
    """Split *by realisation*, not by row.

    ``fractions`` is e.g. ``(0.6, 0.2, 0.2)``. Returns one index array per
    fraction. Every row of a given mixture lands in exactly one part.
    """
    rng = np.random.default_rng(seed)
    groups = np.unique(realisation_id)
    rng.shuffle(groups)
    cuts = np.cumsum(np.asarray(fractions, float))
    cuts = (cuts / cuts[-1] * groups.size).astype(int)
    parts, start = [], 0
    for c in cuts:
        chosen = groups[start:c]
        parts.append(np.flatnonzero(np.isin(realisation_id, chosen)))
        start = c
    return parts


class Standardiser:
    """Zero-mean unit-variance scaling, fitted on the training rows only.

    Follows the ``apply_standardization`` step of
    ``Deep Learning/Day01-Intro_DL_FFNs_and_Colab_morning_solutions.ipynb``
    (cell 29), where the statistics come from the training data and the same
    transform is then applied to validation and test.
    """

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ < 1e-12] = 1.0
        return self

    def transform(self, X):
        if self.mean_ is None:
            raise RuntimeError("Standardiser used before fit()")
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def state_dict(self):
        return {"mean": self.mean_.tolist(), "std": self.std_.tolist()}

    def load_state_dict(self, state):
        self.mean_ = np.asarray(state["mean"], float)
        self.std_ = np.asarray(state["std"], float)
        return self
