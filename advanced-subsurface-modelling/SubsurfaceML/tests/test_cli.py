"""Command-line prediction: the documented example works, and malformed input
files are rejected with a one-line message and exit code 2 (no traceback,
no model loading)."""
from __future__ import annotations

import json

from subsurfaceml.cli import EXAMPLE, main
from subsurfaceml.config import project_root

DEMO = str(project_root() / "config" / "demo.yaml")


def test_documented_prediction_example_runs(capsys):
    assert main(["predict", "--config", DEMO, "--input", str(EXAMPLE)]) == 0
    out = json.loads(capsys.readouterr().out)
    p = out["dp_bh_max_MPa"]
    assert p["interval_low"] <= p["prediction"] <= p["interval_high"]
    assert out["status"].startswith("UNVERIFIED")


def test_malformed_inputs_give_a_clear_error(tmp_path, capsys):
    good = json.loads(EXAMPLE.read_text())
    bad_res = dict(good, reservoir=dict(good["reservoir"], k_median_mD=-5.0))
    cases = {"absent.json": None,
             "not_json.json": "{not json",
             "no_keys.json": json.dumps({"k_median_mD": 5.0}),
             "short_rates.json": json.dumps(dict(good, rates_kg_s=[1.0, 2.0])),
             "negative_rate.json": json.dumps(dict(good, rates_kg_s=[1.0, -2.0, 3.0, 4.0])),
             "bad_reservoir.json": json.dumps(bad_res)}
    for name, text in cases.items():
        f = tmp_path / name
        if text is not None:
            f.write_text(text)
        for cmd in ("predict", "simulate"):
            assert main([cmd, "--config", DEMO, "--input", str(f)]) == 2, (name, cmd)
            assert capsys.readouterr().out.startswith("input error:"), (name, cmd)
