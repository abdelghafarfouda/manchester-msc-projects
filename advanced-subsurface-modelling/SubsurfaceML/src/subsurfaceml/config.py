"""Configuration objects and YAML loading.

Every path is a :class:`pathlib.Path` and every path is configurable, so the
project runs unchanged on Windows, Linux or macOS.  Paths in the YAML files
may be absolute or relative to the **project root** (the directory containing
``config/``).
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml

from .units import md_to_m2, YEAR, MPA


def project_root() -> Path:
    """Directory containing ``config/``, ``src/`` and ``results/``."""
    return Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------
@dataclass
class Paths:
    root: Path = field(default_factory=project_root)
    results: Path = Path("results")
    figures: Path = Path("results/figures")
    data: Path = Path("results/data")
    models: Path = Path("results/models")
    metrics: Path = Path("results/metrics")
    reports: Path = Path("results/reports")
    #: Optional folder holding the supplied course material (``Models/`` and
    #: ``Data Science and machine learning/``), which is not part of the
    #: repository.  Set in the config or via ``SUBSURFACEML_COURSE_DIR``.  Only
    #: the supporting notebook s1 reads it, to check that the cited source
    #: files exist; nothing is copied from it.
    course_materials: Path | None = None

    def resolve(self) -> "Paths":
        r = Path(self.root)
        for f in ("results", "figures", "data", "models", "metrics", "reports"):
            p = Path(getattr(self, f))
            setattr(self, f, p if p.is_absolute() else r / p)
        if self.course_materials is not None:
            p = Path(self.course_materials)
            self.course_materials = p if p.is_absolute() else r / p
        return self

    def mkdirs(self) -> None:
        for f in ("results", "figures", "data", "models", "metrics", "reports"):
            Path(getattr(self, f)).mkdir(parents=True, exist_ok=True)


@dataclass
class GridConfig:
    n_r: int = 60
    n_layers: int = 4
    r_w_m: float = 0.15
    r_e_m: float = 5000.0
    r_near_m: float = 5.0
    h_total_m: float = 40.0


@dataclass
class SolverConfig:
    max_dS: float = 0.05
    cfl: float = 0.9
    dt_init_s: float = 3600.0
    dt_max_days: float = 180.0
    dt_min_s: float = 1.0
    max_overshoot: float = 1e-3
    enforce_cfl: bool = True
    outer_bc: str = "constant_pressure"
    p_init_MPa: float = 15.0
    validation_full: bool = False  # finest grid / well-block levels in V7


@dataclass
class ScheduleConfig:
    n_periods: int = 4
    t_inject_years: float = 5.0
    t_total_years: float = 15.0
    dp_ref_MPa: float = 4.0      # reference buildup used to size q_ref
    q_mult_min: float = 0.15     # log-uniform multiplier of q_ref
    q_mult_max: float = 2.0
    q_min_kg_s: float = 0.01     # numerical floor only (see CHANGELOG)
    q_max_kg_s: float = 250.0
    n_report: int = 41


@dataclass
class ScenarioConfig:
    """Prior ranges for the uncertain reservoir description.

    These are **assumed** ranges representing prior uncertainty for a generic
    deep saline aquifer; they are not derived from measurements of any real
    site.  See ``docs/ASSUMPTIONS.md``.
    """
    n_realisations: int = 120
    n_schedules_per_realisation: int = 3
    seed: int = 20260909

    k_median_mD: tuple = (20.0, 500.0)      # log-uniform
    V_DP: tuple = (0.05, 0.85)              # uniform
    phi_mean: tuple = (0.10, 0.28)          # uniform
    h_total_m: tuple = (20.0, 80.0)         # uniform
    r_e_m: tuple = (3000.0, 9000.0)         # uniform
    n_g: tuple = (1.5, 3.0)                 # Corey exponent, gas
    n_a: tuple = (2.0, 4.0)                 # Corey exponent, brine
    krg0: tuple = (0.20, 0.65)              # gas endpoint kr
    S_ar: tuple = (0.15, 0.35)              # irreducible brine
    mu_g_cP: tuple = (0.04, 0.08)           # CO2 viscosity
    rho_g: tuple = (600.0, 780.0)           # CO2 density kg/m3


@dataclass
class MLConfig:
    test_fraction: float = 0.25    # of realisations, untouched until the end
    calib_fraction: float = 0.20   # of the non-test realisations: error bands
    n_splits: int = 4              # GroupKFold folds inside training
    n_iter_search: int = 12        # RandomizedSearchCV budget per family
    random_state: int = 0
    targets: tuple = ("dp_bh_max_MPa", "r_plume_m95_m", "sweep_efficiency")
    families: tuple | None = None  # None = every course family in models.py
    band_lower_pct: float = 5.0    # empirical error band percentiles
    band_upper_pct: float = 95.0
    margin_warning_fraction: float = 0.8  # traffic-light "near limit" band
    run_studies: bool = True       # the ML-method studies of studies.py
    #: Revised design (2026-10), fixed by the development-set experiments of
    #: scripts/run_experiments.py before the independent test sets existed.
    feature_set: str = "all"       # baseline | rock | all  (features.py), pressure target
    feature_set_other: str = "all" # the same for the plume and sweep targets
    pressure_model: str = "hybrid" # hybrid (ROM x learned factor) | log
    interval_method: str = "reservoir_conformal"   # intervals.METHODS
    interval_alpha: float = 0.10   # nominal 90 % two-sided intervals


@dataclass
class OptimConfig:
    p_limit_MPa: float = 22.0        # ASSUMPTION: stated operating limit
    r_plume_limit_m: float = 2000.0  # ASSUMPTION: stated areal-review limit
    n_candidates: int = 4000
    n_shortlist: int = 8
    n_test_realisations: int = 6
    use_error_band: bool = True      # screen with the upper edge of the error band


@dataclass
class NumericsConfig:
    """Discretisation error of the dataset targets (numerics.py): a stratified
    sample of development cases re-simulated with refined grid, well block
    and time step."""
    run: bool = False
    cases_per_cell: int = 3        # per permeability tercile x intensity quartile
    finer_cases: int = 10          # subset also run at the 'finer' level


@dataclass
class EvaluationConfig:
    """Independent evaluation sets, generated only after every modelling
    choice has been fixed on the development reservoirs
    (``docs/EVALUATION_PROTOCOL.md``).  Their seeds differ from the
    development seed, and their realisation ids are offset so they can never
    be confused with development reservoirs."""
    final_test_seed: int = 20261004
    final_test_realisations: int = 100
    final_test_id_offset: int = 10000
    shift_seed: int = 20261005
    shift_realisations: int = 60
    shift_id_offset: int = 20000
    #: the one prior range changed for the distribution-shift set
    shift_k_median_mD: tuple = (10.0, 30.0)
    screening_reservoirs_final_test: int = 20
    screening_reservoirs_shift: int = 10
    screening_budget: int = 4      # simulator runs per reservoir and method
    screening_candidates: int = 4000


@dataclass
class Config:
    name: str = "demo"
    paths: Paths = field(default_factory=Paths)
    grid: GridConfig = field(default_factory=GridConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    scenarios: ScenarioConfig = field(default_factory=ScenarioConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    numerics: NumericsConfig = field(default_factory=NumericsConfig)
    n_jobs: int = -1

    # -- convenience -------------------------------------------------------
    @property
    def n_scenarios(self) -> int:
        return (self.scenarios.n_realisations
                * self.scenarios.n_schedules_per_realisation)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["paths"] = {k: (str(v) if v is not None else None)
                      for k, v in d["paths"].items()}
        return d

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, default=str))


_SECTIONS = {"grid": GridConfig, "solver": SolverConfig,
             "schedule": ScheduleConfig, "scenarios": ScenarioConfig,
             "ml": MLConfig, "optim": OptimConfig,
             "evaluation": EvaluationConfig, "numerics": NumericsConfig}


class ConfigError(ValueError):
    """Invalid configuration, with the offending key in the message."""


def validate_config(cfg: "Config") -> "Config":
    """Check ranges and consistency; raise :class:`ConfigError` early rather
    than failing half-way through a long run."""
    def need(cond, msg):
        if not cond:
            raise ConfigError(msg)
    g, s, sc, sch, ml, o = (cfg.grid, cfg.solver, cfg.scenarios, cfg.schedule,
                             cfg.ml, cfg.optim)
    need(g.n_r >= 5 and g.n_layers >= 1, "grid.n_r >= 5 and grid.n_layers >= 1")
    need(0 < g.r_w_m < g.r_near_m, "require 0 < grid.r_w_m < grid.r_near_m")
    need(s.outer_bc in ("closed", "constant_pressure"),
         "solver.outer_bc must be 'closed' or 'constant_pressure'")
    need(0 < s.cfl <= 1 and 0 < s.max_dS <= 1, "solver.cfl and solver.max_dS in (0, 1]")
    need(sch.t_inject_years > 0 and sch.t_total_years >= sch.t_inject_years,
         "schedule: 0 < t_inject_years <= t_total_years")
    need(0 < sch.q_mult_min < sch.q_mult_max, "schedule: 0 < q_mult_min < q_mult_max")
    need(sch.n_periods >= 1 and sch.n_report >= 3, "schedule.n_periods >= 1, n_report >= 3")
    for name in ("k_median_mD", "V_DP", "phi_mean", "h_total_m", "r_e_m", "n_g",
                 "n_a", "krg0", "S_ar", "mu_g_cP", "rho_g"):
        lo, hi = getattr(sc, name)
        need(lo <= hi, f"scenarios.{name}: lower bound {lo} > upper bound {hi}")
    need(sc.k_median_mD[0] > 0 and 0 <= sc.V_DP[0] and sc.V_DP[1] < 1,
         "scenarios: k_median_mD > 0 and 0 <= V_DP < 1")
    need(0 < sc.phi_mean[0] and sc.phi_mean[1] < 1, "scenarios.phi_mean in (0, 1)")
    need(sc.r_e_m[0] > 4 * g.r_near_m, "scenarios.r_e_m must exceed 4 x grid.r_near_m")
    need(sc.n_realisations >= 8, "scenarios.n_realisations >= 8 (grouped splits)")
    need(0 < ml.test_fraction < 0.5 and 0 < ml.calib_fraction < 0.5,
         "ml.test_fraction and ml.calib_fraction in (0, 0.5)")
    need(0 <= ml.band_lower_pct < ml.band_upper_pct <= 100, "ml band percentiles")
    need(o.p_limit_MPa > s.p_init_MPa, "optim.p_limit_MPa must exceed solver.p_init_MPa")
    need(ml.feature_set in ("baseline", "rock", "all")
         and ml.feature_set_other in ("baseline", "rock", "all"),
         "ml.feature_set / feature_set_other must be baseline, rock or all")
    need(ml.pressure_model in ("hybrid", "log"), "ml.pressure_model must be hybrid or log")
    need(ml.interval_method in ("empirical", "case_conformal", "reservoir_conformal",
                                "adaptive_conformal"), "ml.interval_method unknown")
    need(0 < ml.interval_alpha < 0.5, "ml.interval_alpha in (0, 0.5)")
    e = cfg.evaluation
    need(e.final_test_seed != sc.seed and e.shift_seed != sc.seed,
         "evaluation seeds must differ from the development seed")
    need(e.final_test_id_offset >= sc.n_realisations and e.shift_id_offset >= sc.n_realisations
         and e.final_test_id_offset != e.shift_id_offset,
         "evaluation realisation ids must not overlap the development ids")
    return cfg


def load_config(path: str | Path) -> Config:
    """Load and validate a YAML configuration file."""
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"configuration file not found: {path}")
    raw: dict[str, Any] = yaml.safe_load(path.read_text()) or {}
    unknown = set(raw) - set(_SECTIONS) - {"name", "paths", "n_jobs"}
    if unknown:
        raise ConfigError(f"unknown configuration section(s): {sorted(unknown)}")
    cfg = Config(name=raw.get("name", path.stem))

    p = raw.get("paths", {}) or {}
    paths = Paths(root=Path(p.get("root", project_root())))
    for k in ("results", "figures", "data", "models", "metrics", "reports"):
        if k in p:
            setattr(paths, k, Path(p[k]))
    cm = p.get("course_materials") or os.environ.get("SUBSURFACEML_COURSE_DIR")
    if cm:
        paths.course_materials = Path(os.path.expandvars(str(cm))).expanduser()
    cfg.paths = paths.resolve()

    for key, klass in _SECTIONS.items():
        if key in raw and raw[key]:
            cur = getattr(cfg, key)
            for k, v in raw[key].items():
                if not hasattr(cur, k):
                    raise ConfigError(f"unknown option {key}.{k}")
                setattr(cur, k, tuple(v) if isinstance(v, list) else v)
    if "n_jobs" in raw:
        cfg.n_jobs = int(raw["n_jobs"])
    return validate_config(cfg)


# --------------------------------------------------------------------------
def provenance() -> dict:
    """Environment fingerprint recorded with every generated dataset."""
    import numpy, scipy, sklearn, pandas
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "pandas": pandas.__version__,
        "scikit_learn": sklearn.__version__,
    }
    try:
        import xgboost
        info["xgboost"] = xgboost.__version__
    except Exception:
        info["xgboost"] = None
    try:
        info["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=project_root(),
            stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        info["git_commit"] = None
    return info
