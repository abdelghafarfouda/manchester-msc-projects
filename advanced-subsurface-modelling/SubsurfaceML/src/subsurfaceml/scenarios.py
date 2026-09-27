"""Scenario definition, sampling and simulation driver.

A **scenario** = one reservoir *realisation* (rock + fluid description) plus
one *injection schedule*.  Realisations are the grouping unit for every
train/test split: all schedules, all time steps and all derived rows of a
realisation stay in the same partition.

Provenance
----------
Every scenario carries its ``realisation_id``, ``schedule_id``,
``scenario_id``, the RNG ``seed`` that generated it, and the full set of
sampled inputs, so any row of the dataset can be regenerated exactly.

Synthetic data
--------------
The dataset is **synthetic**: it is produced by the validated simulator in
this repository, not measured in any field.  The sampled ranges express
*assumed prior uncertainty* for a generic deep saline aquifer.  Nothing here
is calibrated to observations and no field validation is claimed.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any

import numpy as np

from .config import Config
from .fluids import FluidProperties, RelPerm, RockProperties
from .grid import RadialGrid
from .impes import TwoPhaseModel, InjectionSchedule, SimulationFailure
from .outputs import extract_qois, time_series_frame
from .petrophysics import make_layered_rock
from .units import md_to_m2, cp_to_pas, YEAR, MPA


# --------------------------------------------------------------------------
def _mc(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """Plain Monte Carlo sample in [0,1)^d: independent uniform draws,
    step 2 of the Monte Carlo procedure of ``5-Uncertainty.pdf`` p.45
    ("generate a random number for each parameter").  An earlier version used
    Latin-hypercube sampling, which is not part of the supplied material."""
    return rng.random((n, d))


def _uni(u, lo, hi):
    return lo + u * (hi - lo)


def _logu(u, lo, hi):
    return float(np.exp(np.log(lo) + u * (np.log(hi) - np.log(lo))))


@dataclass
class Realisation:
    """One sampled reservoir description."""
    realisation_id: int
    seed: int
    k_median_mD: float
    V_DP: float
    phi_mean: float
    h_total_m: float
    r_e_m: float
    n_g: float
    n_a: float
    krg0: float
    S_ar: float
    mu_g_cP: float
    rho_g: float

    def to_dict(self) -> dict:
        return asdict(self)


def sample_realisations(cfg: Config) -> list[Realisation]:
    """Monte Carlo sample of the reservoir prior: permeability log-uniform
    (i.e. uniform in ln k), everything else uniform (``5-Uncertainty.pdf``
    p.42-45)."""
    sc = cfg.scenarios
    rng = np.random.default_rng(sc.seed)
    n = sc.n_realisations
    names = ["k_median_mD", "V_DP", "phi_mean", "h_total_m", "r_e_m",
             "n_g", "n_a", "krg0", "S_ar", "mu_g_cP", "rho_g"]
    u = _mc(n, len(names), rng)
    out = []
    for i in range(n):
        vals: dict[str, Any] = {}
        for j, nm in enumerate(names):
            lo, hi = getattr(sc, nm)
            vals[nm] = (_logu(u[i, j], lo, hi) if nm == "k_median_mD"
                        else float(_uni(u[i, j], lo, hi)))
        out.append(Realisation(realisation_id=i,
                               seed=int(sc.seed + 1000 * (i + 1)), **vals))
    return out


def reference_rate(cfg: Config, r: Realisation) -> float:
    """Injectivity- and storage-scaled reference CO2 mass rate [kg/s].

    Sampling absolute rates over a fixed interval is a poor design here: the
    same 60 kg/s that is unremarkable in a 500 mD, 80 m unit produces a
    physically meaningless >100 MPa buildup in a 20 mD, 20 m unit, so most of
    the sample lands in a regime no operator would ever propose and the
    surrogate spends its capacity there.

    Two independent mechanisms limit the rate, and the reference rate is the
    harmonic combination of the two (whichever binds, binds):

    **Flow resistance** -- steady radial Darcy, ``single_phase.steady_state_radial``::

        q_flow = rho_g * 2 pi k h dp_ref / (mu_a ln(r_e / r_w))

    The far field, which is most of ``ln(r_e/r_w)``, is still single-phase
    brine, so the BRINE viscosity sets this term; using ``mu_g`` here
    underestimates the resistance by ~10x.

    **Storage (compartment fill-up)** -- the closed-tank material balance of
    ``single_phase.tank_material_balance``::

        q_store = rho_g * dp_ref * c_t * V_p / t_inject

    In a sealed compartment this is usually the binding limit, and it is what
    makes the *shape* of a schedule matter: the buildup at time t carries a
    cumulative term (proportional to the mass already injected) plus an
    instantaneous term (proportional to the current rate), so the peak
    pressure of a plan depends on the order in which the mass is delivered,
    not only on the total.

    ``q_ref`` is a closed-form function of the reservoir description, so it is
    legitimately available to the surrogate as a feature -- it uses no
    simulator output.
    """
    g = cfg.grid
    k = md_to_m2(r.k_median_mD)
    dp_ref = cfg.schedule.dp_ref_MPa * MPA
    mu_ref = 6.0e-4                                   # brine viscosity [Pa.s]
    c_t = 4.5e-10 + 4.5e-10                           # c_rock + c_brine [1/Pa]
    q_flow = r.rho_g * (2.0 * np.pi * k * r.h_total_m * dp_ref
                        / (mu_ref * np.log(r.r_e_m / g.r_w_m)))
    V_p = np.pi * r.r_e_m ** 2 * r.h_total_m * r.phi_mean
    q_store = r.rho_g * dp_ref * c_t * V_p / (cfg.schedule.t_inject_years * YEAR)
    return float(1.0 / (1.0 / q_flow + 1.0 / q_store))


def sample_schedules(cfg: Config, realisation: Realisation,
                     n: int | None = None) -> list[dict]:
    """Sample piecewise-constant injection schedules for one realisation.

    **Orthogonal design.**  The obvious thing -- sample each period's rate
    independently -- is a poor experiment here: the overall *level* of a
    schedule and its *shape* then co-vary, the shape effect is not separately
    identifiable, and an L1-regularised surrogate simply zeroes every
    shape feature (measured: ``q_front_load``, ``q_cv`` and ``q_ramp`` all got
    coefficient 0 under the independent design, leaving the optimiser blind to
    exactly the decision it is supposed to make).

    So level and shape are sampled **separately**:

    * ``m``      log-uniform mean multiplier of ``q_ref``  -> sets the level
    * ``tilt``   uniform in [-1, 1] -> linear front/back loading
    * ``curve``  uniform in [-0.5, 0.5] -> quadratic bow

    The shape vector is normalised to mean 1, so it is exactly orthogonal to
    the level by construction, and ``q_front_load`` becomes a clean,
    independently varying regressor.
    """
    scfg = cfg.schedule
    n = n or cfg.scenarios.n_schedules_per_realisation
    rng = np.random.default_rng(realisation.seed + 7)
    u = _mc(n, 3, rng)
    t_inj = scfg.t_inject_years * YEAR
    t_tot = scfg.t_total_years * YEAR
    edges = np.concatenate([np.linspace(0.0, t_inj, scfg.n_periods + 1), [t_tot]])
    q_ref = reference_rate(cfg, realisation)
    out = []
    for s in range(n):
        m = _logu(u[s, 0], scfg.q_mult_min, scfg.q_mult_max)
        tilt = float(_uni(u[s, 1], -1.0, 1.0))
        curve = float(_uni(u[s, 2], -0.5, 0.5))
        shape = shape_vector(scfg.n_periods, tilt, curve)
        rates = np.clip(m * q_ref * shape, scfg.q_min_kg_s, scfg.q_max_kg_s)
        out.append({"schedule_id": s, "rates_kg_s": rates,
                    "t_edges_s": edges, "q_ref_kg_s": q_ref,
                    "level_mult": m, "tilt": tilt, "curve": curve})
    return out


def shape_vector(n_periods: int, tilt: float, curve: float) -> np.ndarray:
    """Normalised schedule shape with mean exactly 1.

    ``tilt > 0`` front-loads the plan, ``tilt < 0`` back-loads it; ``curve``
    adds a quadratic bow.  Clipped to stay positive, then renormalised.
    """
    x = np.linspace(-1.0, 1.0, n_periods)
    sh = 1.0 - tilt * x + curve * (x ** 2 - np.mean(x ** 2))
    sh = np.clip(sh, 0.05, None)
    return sh / sh.mean()


def make_schedule(cfg: Config, r: Realisation, rates_kg_s, schedule_id=-1) -> dict:
    """Wrap an explicit rate vector (used by the optimiser and the baseline)."""
    scfg = cfg.schedule
    edges = np.concatenate([np.linspace(0.0, scfg.t_inject_years * YEAR,
                                        scfg.n_periods + 1),
                            [scfg.t_total_years * YEAR]])
    return {"schedule_id": schedule_id,
            "rates_kg_s": np.asarray(rates_kg_s, float),
            "t_edges_s": edges, "q_ref_kg_s": reference_rate(cfg, r)}


def constant_schedule(cfg: Config, r: Realisation, total_mass_kg: float) -> dict:
    """Baseline: constant rate delivering ``total_mass_kg`` over the window."""
    q = total_mass_kg / (cfg.schedule.t_inject_years * YEAR)
    return make_schedule(cfg, r, np.full(cfg.schedule.n_periods, q),
                         schedule_id=-1)


# --------------------------------------------------------------------------
def fluid_models(r: Realisation):
    """Fluid and relative-permeability objects of one realisation -- shared
    by the simulator (:func:`build_model`) and the feature path
    (:func:`features.raw_inputs`) so both see the same ``krg0``, ``mu_g``..."""
    fl = FluidProperties(rho_g=r.rho_g, mu_g=cp_to_pas(r.mu_g_cP))
    rp = RelPerm(S_ar=r.S_ar, n_a=r.n_a, n_g=r.n_g, krg0=r.krg0)
    return fl, rp


def build_model(cfg: Config, r: Realisation):
    """Instantiate the simulator for one realisation."""
    g = cfg.grid
    rock = make_layered_rock(
        n_layers=g.n_layers, n_r=g.n_r, h_total=r.h_total_m,
        k_median=md_to_m2(r.k_median_mD), V_DP=r.V_DP, phi_mean=r.phi_mean,
        seed=r.seed)
    grids = [RadialGrid(n=g.n_r, r_w=g.r_w_m, r_e=r.r_e_m, h=rock.h[l],
                        r_near=g.r_near_m) for l in range(g.n_layers)]
    fl, rp = fluid_models(r)
    rk = RockProperties()
    s = cfg.solver
    model = TwoPhaseModel(
        grids, rock.k, rock.phi, fl, rp, rk, s.p_init_MPa * MPA,
        outer_bc=s.outer_bc, cfl=s.cfl, max_dS=s.max_dS,
        dt_init=s.dt_init_s, dt_max=s.dt_max_days * 86400.0,
        dt_min=s.dt_min_s, max_overshoot=s.max_overshoot,
        enforce_cfl=s.enforce_cfl)
    return model, grids, rock, fl, rp


def run_scenario(cfg: Config, r: Realisation, sched: dict,
                 *, want_series: bool = True) -> dict:
    """Run one scenario and return QoIs, rock summary and (optionally) the
    trajectory table.  Failures are returned as ``{'status': 'failed', ...}``
    rather than raised, so the caller can log them."""
    scenario_id = f"R{r.realisation_id:04d}_S{sched['schedule_id']:+03d}"
    try:
        model, grids, rock, fl, rp = build_model(cfg, r)
        # the sampled rates cover the injection window; the trailing period is
        # the post-injection shut-in (rate 0) up to t_total
        schedule = InjectionSchedule(sched["t_edges_s"],
                                     np.append(sched["rates_kg_s"], 0.0))
        report = np.linspace(0.0, schedule.t_end, cfg.schedule.n_report)
        res = model.run(schedule, report)
        q = extract_qois(res, grids, model,
                         p_limit=cfg.optim.p_limit_MPa * MPA)
        series = time_series_frame(res, grids, model) if want_series else None
    except SimulationFailure as exc:
        return {"status": "failed", "scenario_id": scenario_id,
                "realisation_id": r.realisation_id,
                "schedule_id": sched["schedule_id"],
                "error": str(exc), "error_type": "SimulationFailure"}
    except Exception as exc:                                   # pragma: no cover
        return {"status": "failed", "scenario_id": scenario_id,
                "realisation_id": r.realisation_id,
                "schedule_id": sched["schedule_id"],
                "error": repr(exc), "error_type": type(exc).__name__}

    from .features import raw_inputs
    r_hist = q.pop("_r_plume_history_m")
    row = {"status": "ok", "scenario_id": scenario_id,
           "realisation_id": r.realisation_id,
           "schedule_id": sched["schedule_id"]}
    # inputs through the SAME path the optimiser and interface use
    row.update(raw_inputs(cfg, r, sched["rates_kg_s"]))
    row.update(q)
    row["final_Sg_profile"] = _profile_string(res, grids, model)
    if series is not None:
        series.insert(0, "scenario_id", scenario_id)
        series.insert(1, "realisation_id", r.realisation_id)
        series.insert(2, "schedule_id", sched["schedule_id"])
    return {"status": "ok", "row": row, "series": series,
            "r_plume_history_m": r_hist}


def _profile_string(res, grids, model, n_bins: int = 24) -> str:
    """Final CO2 saturation, thickness-averaged over the layers and sampled at
    ``n_bins`` log-spaced radii between the well and 2 km (or ``r_e``),
    stored as a compact string.  Used only by the unsupervised studies of
    plume shape (PCA / clustering of profiles); never an ML input."""
    r_max = min(2000.0, grids[0].r_e)
    rr = np.geomspace(1.0, r_max, n_bins)
    h = np.array([g.h for g in grids])
    prof = sum(h[l] * np.interp(rr, grids[l].centres, res.Sg[-1, l])
               for l in range(len(grids))) / h.sum()
    return ";".join(f"{v:.5f}" for v in prof)
