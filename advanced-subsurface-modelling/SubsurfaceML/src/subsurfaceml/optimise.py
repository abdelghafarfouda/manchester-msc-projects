"""Candidate schedules for the screening.

The screening itself -- surrogate proposals that become recommendations
only after simulator verification -- is in :mod:`screening`.  This module
keeps the two pieces it shares with the rest of the package:

* :func:`sample_candidates` draws candidate rate vectors from the same
  level x shape design the training schedules were drawn from
  (:func:`scenarios.sample_schedules`), so the screening never extrapolates
  in schedule space;
* :func:`exact_planned_mass` computes the objective exactly from the
  schedule (under the sealed boundary the retained mass equals the planned
  injected mass, verified in the pipeline), so no surrogate is used for it.

An earlier version (``optimise_schedule`` / ``resimulate``, 2026-09-20)
ranked candidates that were only *predicted* to meet the limits and
re-simulated a shortlist afterwards; it is replaced by :mod:`screening`,
which cannot return an unverified schedule as a recommendation.
"""
from __future__ import annotations

import numpy as np

from .config import Config
from .scenarios import Realisation, reference_rate, shape_vector
from .units import YEAR


def exact_planned_mass(cfg: Config, rates_kg_s) -> float:
    """Planned injected mass [kg] -- exact, from the schedule alone."""
    dt = cfg.schedule.t_inject_years * YEAR / cfg.schedule.n_periods
    return float(np.sum(np.asarray(rates_kg_s, float)) * dt)


def sample_candidates(cfg: Config, r: Realisation, n: int,
                      rng: np.random.Generator) -> np.ndarray:
    """Random candidate rate vectors ``(n, n_periods)`` [kg/s] from the same
    level x shape design as :func:`scenarios.sample_schedules`."""
    n_p = cfg.schedule.n_periods
    q_ref = reference_rate(cfg, r)
    lo, hi = cfg.schedule.q_mult_min, cfg.schedule.q_mult_max
    m = np.exp(rng.uniform(np.log(lo), np.log(hi), size=n))
    tilt = rng.uniform(-1.0, 1.0, size=n)
    curve = rng.uniform(-0.5, 0.5, size=n)
    sh = np.stack([shape_vector(n_p, tilt[i], curve[i]) for i in range(n)])
    return np.clip(m[:, None] * q_ref * sh, cfg.schedule.q_min_kg_s,
                   cfg.schedule.q_max_kg_s)
