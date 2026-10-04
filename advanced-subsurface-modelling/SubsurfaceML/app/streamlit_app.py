"""SubsurfaceML interface.

    streamlit run app/streamlit_app.py -- --config config/demo.yaml

The interface calls the same code as the pipeline: the simulator in
``impes.py``, the single feature path in ``features.py`` and the verified
model artifacts through ``predict.Predictor``.  It re-implements nothing.
If the model files are missing or incompatible it says so and shows the
command that rebuilds them.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from subsurfaceml.artifacts import ArtifactError
from subsurfaceml.config import load_config
from subsurfaceml.impes import InjectionSchedule
from subsurfaceml.predict import Predictor, realisation_from_dict
from subsurfaceml.scenarios import (build_model, make_schedule, reference_rate,
                                    run_scenario, sample_realisations)
from subsurfaceml.units import MPA, YEAR

st.set_page_config(page_title="SubsurfaceML", layout="wide")


def _cfg_path() -> str:
    argv = sys.argv[1:]
    return argv[argv.index("--config") + 1] if "--config" in argv else \
        str(ROOT / "config" / "demo.yaml")


@st.cache_resource(show_spinner=False)
def load(cfg_path: str):
    cfg = load_config(cfg_path)
    out = {"cfg": cfg, "predictor": None, "error": None}
    try:
        out["predictor"] = Predictor(cfg)
    except ArtifactError as exc:
        out["error"] = str(exc)
    s = Path(cfg.paths.metrics) / "summary.json"
    out["summary"] = json.loads(s.read_text()) if s.exists() else None
    r = Path(cfg.paths.reports) / "RESULTS.md"
    out["results_md"] = r.read_text() if r.exists() else None
    return out


S = load(_cfg_path())
cfg = S["cfg"]
st.title("SubsurfaceML - CO2 injection into a layered saline aquifer")
st.caption("Synthetic data from the in-repo IMPES simulator. No field data, no "
           "field validation. Pressure and plume limits are stated assumptions "
           f"(p_limit {cfg.optim.p_limit_MPa} MPa, plume limit "
           f"{cfg.optim.r_plume_limit_m} m).")
if S["error"]:
    st.error(S["error"])

tab = st.tabs(["Results", "Predict", "Simulate & compare", "Screen schedules"])

# ---------------------------------------------------------------- results
with tab[0]:
    if S["results_md"]:
        st.markdown(S["results_md"])
        figs = sorted(Path(cfg.paths.figures).glob("*.png"))
        pick = st.selectbox("figure", [f.name for f in figs]) if figs else None
        if pick:
            st.image(str(Path(cfg.paths.figures) / pick))
    else:
        st.info("No results yet. Run: python scripts/run_pipeline.py "
                f"--config config/{cfg.name}.yaml")


def reservoir_inputs(key):
    c1, c2, c3 = st.columns(3)
    d = {"k_median_mD": c1.number_input("median permeability [mD]", 1.0, 5000.0, 200.0, key=key + "k"),
         "V_DP": c1.slider("Dykstra-Parsons V_DP [-]", 0.0, 0.9, 0.5, key=key + "v"),
         "phi_mean": c1.slider("mean porosity [-]", 0.05, 0.35, 0.2, key=key + "p"),
         "h_total_m": c2.number_input("thickness [m]", 5.0, 200.0, 40.0, key=key + "h"),
         "r_e_m": c2.number_input("compartment radius [m]", 500.0, 10000.0, 2500.0, key=key + "r"),
         "n_g": c2.slider("CO2 Corey exponent", 1.0, 4.0, 2.0, key=key + "ng"),
         "n_a": c2.slider("brine Corey exponent", 1.0, 5.0, 3.0, key=key + "na"),
         "krg0": c3.slider("CO2 end-point kr", 0.1, 1.0, 0.4, key=key + "kr"),
         "S_ar": c3.slider("irreducible brine saturation", 0.05, 0.45, 0.25, key=key + "s"),
         "mu_g_cP": c3.number_input("CO2 viscosity [cP]", 0.02, 0.2, 0.06, key=key + "mu"),
         "rho_g": c3.number_input("CO2 density [kg/m3]", 400.0, 900.0, 700.0, key=key + "rho")}
    r = realisation_from_dict(d)
    q_ref = reference_rate(cfg, r)
    st.caption(f"reference rate q_ref = {q_ref:.1f} kg/s (closed form, no simulation)")
    cols = st.columns(cfg.schedule.n_periods)
    rates = np.array([c.number_input(f"q{i+1} [kg/s]", 0.0, 500.0, float(round(q_ref, 1)),
                                     key=key + f"q{i}") for i, c in enumerate(cols)])
    return r, rates


# ---------------------------------------------------------------- predict
with tab[1]:
    st.subheader("Surrogate prediction for a reservoir and schedule")
    r, rates = reservoir_inputs("p")
    if S["predictor"] is not None and st.button("Predict"):
        t0 = time.perf_counter()
        out = S["predictor"].predict(r, rates[None, :])
        dt = time.perf_counter() - t0
        rows = []
        for t in ("dp_bh_max_MPa", "r_plume_m95_m", "sweep_efficiency"):
            if t in out:
                o = out[t]
                rows.append({"quantity": t, "prediction": float(o["pred"][0]),
                             "90 % interval low": float(o["band_low"][0]),
                             "interval high": float(o["band_high"][0]), "unit": o["unit"]})
        st.table(pd.DataFrame(rows))
        if "in_training_domain" in out and not out["in_training_domain"][0]:
            st.warning("This reservoir is outside the training population "
                       f"({out['domain_reasons'][0]}); the prediction and its interval "
                       "have no support from the experiments - simulate instead.")
        if "exceeds_pressure_limit" in out:
            e = out["exceeds_pressure_limit"]
            st.write(f"Pressure-limit screen: **{'EXCEEDS' if e['flag'][0] else 'below'}** "
                     f"the {e['dp_limit_MPa']:.1f} MPa buildup limit "
                     f"(score {e['score'][0]:.3f}, threshold {e['threshold']:.3f}).")
        st.caption(f"{dt*1e3:.1f} ms including feature building and the analytical ROM. "
                   "These are UNVERIFIED surrogate predictions; intervals are calibrated on "
                   "held-out reservoirs (coverage measured in RESULTS.md) and say nothing "
                   "for reservoirs unlike the training population.")

# ---------------------------------------------------------------- simulate
with tab[2]:
    st.subheader("Run the simulator and compare with the surrogate")
    r, rates = reservoir_inputs("s")
    if st.button("Simulate (a few seconds)"):
        t0 = time.perf_counter()
        o = run_scenario(cfg, r, make_schedule(cfg, r, rates), want_series=True)
        dt = time.perf_counter() - t0
        if o["status"] != "ok":
            st.error(f"simulation failed: {o.get('error')}")
        else:
            w, ts = o["row"], o["series"]
            c1, c2 = st.columns(2)
            c1.metric("simulated dp_bh_max [MPa]", f"{w['dp_bh_max_Pa']/MPA:.2f}")
            c1.metric("simulated r_plume (95% mass) [m]", f"{w['r_plume_m95_m']:.0f}")
            c1.caption(f"{dt:.2f} s, {w['n_steps']} steps, mass balance error "
                       f"{w['mass_balance_error']:.1e}")
            if S["predictor"] is not None:
                p = S["predictor"].predict(r, rates[None, :])
                c2.metric("surrogate dp_bh_max [MPa]", f"{p['dp_bh_max_MPa']['pred'][0]:.2f}")
                c2.metric("surrogate r_plume [m]", f"{p['r_plume_m95_m']['pred'][0]:.0f}")
            fig, ax = plt.subplots(1, 2, figsize=(10, 3))
            ax[0].plot(ts.t_s / YEAR, ts.dp_bh_Pa / MPA)
            ax[0].set_xlabel("time [yr]"); ax[0].set_ylabel("BHP buildup [MPa]")
            ax[0].set_title("injection, then shut-in (diagnostic BHP)")
            ax[1].plot(ts.t_s / YEAR, ts.r_plume_m95_m)
            ax[1].set_xlabel("time [yr]"); ax[1].set_ylabel("plume radius, 95% mass [m]")
            st.pyplot(fig)

# ---------------------------------------------------------------- screening
with tab[3]:
    st.subheader("Verification-gated schedule screening on an independent test reservoir")
    st.caption("The surrogate proposes; the simulator decides. No schedule is "
               "recommended unless it has been simulated and met both stated limits "
               "(the 9 MPa build-up limit is a modelling assumption).")
    if S["predictor"] is None:
        st.info("Needs the trained models.")
    else:
        from subsurfaceml import screening as SC
        from subsurfaceml.features import features_for_schedules
        from subsurfaceml.final_eval import realisation_lookup
        from subsurfaceml.optimise import sample_candidates
        P = S["predictor"]
        look = realisation_lookup(cfg)
        ids = [k for k in sorted(look) if k >= cfg.evaluation.final_test_id_offset]
        rid = st.selectbox("independent reservoir (final test 10000+, shift 20000+)", ids)
        n = st.slider("candidates", 200, 5000, 1000, step=200)
        budget = st.slider("simulator budget", 1, 6, cfg.evaluation.screening_budget)
        if st.button("Screen, then verify with the simulator"):
            c, r = look[int(rid)]

            def predictor(R):
                X = features_for_schedules(c, r, R)
                _, p, hi = P.bands["dp_bh_max_MPa"].predict_interval(X)
                _, p2, hi2 = P.bands["r_plume_m95_m"].predict_interval(X)
                return {"dp": p, "dp_hi": hi, "r95": p2, "r95_hi": hi2}
            X0 = features_for_schedules(c, r, np.ones((1, c.schedule.n_periods)))
            in_dom = bool(P.domain.check(X0)["in_domain"].iloc[0]) if P.domain else True
            cands = sample_candidates(c, r, n, np.random.default_rng(int(rid)))
            rec = SC.recommend_surrogate(c, r, SC.make_simulator(c, r), budget, predictor,
                                         cands, in_domain=in_dom)
            st.write(f"**Status: {rec.status}**" + (f"  (flags: {', '.join(rec.flags)})"
                                                    if rec.flags else ""))
            if rec.recommended_rates_kg_s is not None:
                v = rec.verified
                st.success(f"Verified schedule {np.round(rec.recommended_rates_kg_s, 2)} kg/s: "
                           f"{v.mass_Mt:.3f} Mt, simulated build-up {v.dp_MPa:.2f} MPa, "
                           f"plume radius {v.r95_m:.0f} m.")
            else:
                st.error("No recommendation: " + (rec.notes or ""))
            st.table(pd.DataFrame([{"run": ch.label, "rates [kg/s]": np.round(ch.rates_kg_s, 2),
                                    "simulated build-up [MPa]": ch.dp_MPa,
                                    "simulated plume radius [m]": ch.r95_m,
                                    "mass [Mt]": ch.mass_Mt, "meets limits": ch.feasible}
                                   for ch in rec.checks]))
