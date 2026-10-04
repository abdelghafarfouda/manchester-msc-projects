"""Console entry point: ``subsurfaceml <command>``.

Commands
--------
``run``        the complete pipeline for a configuration
``validate``   the numerical verification suite only
``predict``    surrogate predictions for one reservoir + schedule (JSON input)
``simulate``   run the simulator for the same JSON input and compare
``dashboard``  print the Streamlit command for this configuration
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .config import ConfigError, load_config, project_root

EXAMPLE = project_root() / "examples" / "worked_example_input.json"


def _load_case(path):
    d = json.loads(Path(path).read_text())
    from .predict import realisation_from_dict
    return realisation_from_dict(d["reservoir"]), d["rates_kg_s"]


def main(argv=None) -> int:
    root = project_root()
    ap = argparse.ArgumentParser(prog="subsurfaceml", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    cfg_default = str(root / "config" / "demo.yaml")
    p = sub.add_parser("run", help="run the complete pipeline")
    p.add_argument("--config", default=cfg_default)
    for f in ("validation", "dataset", "numerics", "studies", "screening"):
        p.add_argument(f"--skip-{f}", action="store_true")
    p = sub.add_parser("validate", help="numerical verification only")
    p.add_argument("--config", default=cfg_default)
    for name in ("predict", "simulate"):
        p = sub.add_parser(name, help=f"{name} one case from a JSON file")
        p.add_argument("--config", default=cfg_default)
        p.add_argument("--input", default=str(EXAMPLE))
    p = sub.add_parser("dashboard", help="print the Streamlit command")
    p.add_argument("--config", default=cfg_default)
    a = ap.parse_args(argv)

    if a.cmd == "run":
        cmd = [sys.executable, str(root / "scripts" / "run_pipeline.py"), "--config", a.config]
        for f in ("validation", "dataset", "numerics", "studies", "screening"):
            if getattr(a, f"skip_{f}"):
                cmd.append(f"--skip-{f}")
        return subprocess.call(cmd)
    try:
        cfg = load_config(a.config)
    except ConfigError as exc:
        print(f"configuration error: {exc}")
        return 2
    if a.cmd == "validate":
        from .validation import run_all
        res = run_all(cfg, save=True)
        print(json.dumps(res["verdict"], indent=2))
        return 0 if res["verdict"]["ALL_PASS"] else 2
    if a.cmd == "predict":
        from .artifacts import ArtifactError
        from .predict import Predictor
        try:
            pr = Predictor(cfg)
        except ArtifactError as exc:
            print(f"model artifacts unavailable: {exc}")
            return 3
        r, rates = _load_case(a.input)
        out = pr.predict(r, [rates])
        res = {t: {"prediction": float(v["pred"][0]),
                   "interval_low": float(v["band_low"][0]),
                   "interval_high": float(v["band_high"][0]), "unit": v["unit"]}
               for t, v in out.items() if isinstance(v, dict) and "pred" in v}
        res["status"] = ("UNVERIFIED surrogate prediction - simulate before relying on it "
                         "(subsurfaceml simulate)")
        if "in_training_domain" in out:
            res["in_training_domain"] = bool(out["in_training_domain"][0])
            if not res["in_training_domain"]:
                res["domain_reasons"] = out["domain_reasons"][0]
        if "exceeds_pressure_limit" in out:
            e = out["exceeds_pressure_limit"]
            res["pressure_screen"] = {"flag_exceeds": bool(e["flag"][0]),
                                      "score": float(e["score"][0]),
                                      "threshold": float(e["threshold"]),
                                      "dp_limit_MPa": float(e["dp_limit_MPa"])}
        print(json.dumps(res, indent=2))
        return 0
    if a.cmd == "simulate":
        from .scenarios import make_schedule, run_scenario
        from .units import MPA
        r, rates = _load_case(a.input)
        o = run_scenario(cfg, r, make_schedule(cfg, r, rates), want_series=False)
        if o["status"] != "ok":
            print(json.dumps(o, indent=2, default=str))
            return 4
        w = o["row"]
        print(json.dumps({"dp_bh_max_MPa": w["dp_bh_max_Pa"] / MPA,
                          "r_plume_m95_m": w["r_plume_m95_m"],
                          "sweep_efficiency": w["sweep_efficiency"],
                          "mass_injected_Mt": w["mass_injected_kg"] / 1e9,
                          "mass_balance_error": w["mass_balance_error"],
                          "wall_time_s": w["wall_time_s"]}, indent=2))
        return 0
    if a.cmd == "dashboard":
        print(f'streamlit run "{root / "app" / "streamlit_app.py"}" -- --config "{a.config}"')
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
