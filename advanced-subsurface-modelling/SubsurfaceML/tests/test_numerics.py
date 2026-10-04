"""The dataset-case discretisation study: refinement levels and summaries."""
from __future__ import annotations

import pandas as pd
import pytest

from subsurfaceml.config import load_config, project_root
from subsurfaceml.numerics import MAX_STEPS, level_config, summarise


@pytest.fixture(scope="module")
def cfg():
    return load_config(project_root() / "config" / "study.yaml")


def test_levels_refine_only_what_they_name(cfg):
    p = level_config(cfg, "production")
    assert (p.grid.n_r, p.grid.r_near_m, p.solver.max_dS) == (cfg.grid.n_r, cfg.grid.r_near_m,
                                                              cfg.solver.max_dS)
    n = level_config(cfg, "n_r_x2")
    assert n.grid.n_r == 2 * cfg.grid.n_r and n.grid.r_near_m == cfg.grid.r_near_m
    w = level_config(cfg, "r_near_half")
    assert w.grid.r_near_m == cfg.grid.r_near_m / 2 and w.grid.n_r == cfg.grid.n_r
    t = level_config(cfg, "time_fine")
    assert t.solver.max_dS == cfg.solver.max_dS / 2 and t.solver.dt_max_days <= 30
    f = level_config(cfg, "fine")
    assert (f.grid.n_r, f.grid.r_near_m) == (2 * cfg.grid.n_r, cfg.grid.r_near_m / 2)
    assert cfg.grid.n_r == 60                       # the original is not modified


def test_refined_levels_get_a_larger_step_guard(cfg):
    assert level_config(cfg, "production").solver.max_steps == MAX_STEPS["production"]
    assert level_config(cfg, "fine").solver.max_steps > 10 * MAX_STEPS["production"]
    with pytest.raises(ValueError):
        level_config(cfg, "nonsense")


def test_summary_counts_label_flips_and_failures():
    rows = []
    for sid, prod, fine in (("a", 8.9, 9.1), ("b", 5.0, 5.01), ("c", 12.0, 11.9)):
        for lv, v in (("production", prod), ("fine", fine)):
            rows.append({"scenario_id": sid, "level": lv, "status": "ok",
                         "dp_bh_max_MPa": v, "r_plume_m95_m": 100.0, "sweep_efficiency": 0.01})
    rows.append({"scenario_id": "d", "level": "fine", "status": "failed"})
    rep = summarise(pd.DataFrame(rows), None, dp_limit_MPa=9.0)
    assert rep["n_failed_runs"] == 1
    assert rep["pressure_limit_label_flips"]["n_flipped"] == 1
    assert rep["pressure_limit_label_flips"]["flipped_cases"] == ["a"]
