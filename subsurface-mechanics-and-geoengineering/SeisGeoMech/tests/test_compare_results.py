"""scripts/compare_results.py must fail on every kind of difference it claims to catch."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("compare_results", ROOT / "scripts" / "compare_results.py")
cr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cr)


def _run(tmp_path, recorded, new, name="t.json"):
    a, b = tmp_path / "rec", tmp_path / "new"
    a.mkdir(exist_ok=True)
    b.mkdir(exist_ok=True)
    for folder, content in ((a, recorded), (b, new)):
        if content is not None:
            (folder / name).write_text(content if isinstance(content, str) else json.dumps(content))
    return cr.main([str(a), str(b)])


def test_identical_and_tolerated_differences_pass(tmp_path):
    assert _run(tmp_path, {"x": 1.0, "n": 3, "s": "a"}, {"x": 1.0 + 1e-12, "n": 3, "s": "a"}) == 0


def test_timestamp_is_skipped(tmp_path):
    assert _run(tmp_path, {"generated_utc": "a", "x": 1.0}, {"generated_utc": "b", "x": 1.0}) == 0


@pytest.mark.parametrize("new", [
    {"x": 1.0 + 1e-6, "n": 3, "s": "a"},          # outside tolerance
    {"x": 1.0, "n": 4, "s": "a"},                 # integer changed
    {"x": 1.0, "n": 3.0, "s": "a"},               # int became float
    {"x": 1.0, "n": 3, "s": "b"},                 # string changed
    {"x": 1.0, "n": 3, "s": "a", "extra": 1},     # extra key
    {"x": 1.0, "n": 3},                           # missing key
    {"x": float("inf"), "n": 3, "s": "a"},        # finite became infinite
    {"x": float("nan"), "n": 3, "s": "a"},        # finite became NaN
])
def test_json_differences_fail(tmp_path, new):
    assert _run(tmp_path, {"x": 1.0, "n": 3, "s": "a"}, new) == 1


def test_booleans_are_not_integers(tmp_path):
    assert _run(tmp_path, {"flag": True}, {"flag": 1}) == 1


def test_infinity_must_stay_the_same_infinity(tmp_path):
    assert _run(tmp_path, {"x": float("inf")}, {"x": float("inf")}) == 0
    assert _run(tmp_path, {"x": float("inf")}, {"x": 1e308}) == 1
    assert _run(tmp_path, {"x": float("inf")}, {"x": float("-inf")}) == 1


@pytest.mark.parametrize("new", [
    "a,b\n1,2.0\n3,4.5\n9,9.0\n",   # extra row
    "a,c\n1,2.0\n3,4.5\n",          # renamed column
    "a,b\n1,2.0\n3,4.6\n",          # value changed
    "a,b\n1,2.0\n3, 4.5\n",         # a cell that is no longer a clean number
])
def test_csv_differences_fail(tmp_path, new):
    assert _run(tmp_path, "a,b\n1,2.0\n3,4.5\n", new, name="t.csv") == 1


def test_missing_and_extra_files_fail(tmp_path):
    assert _run(tmp_path, {"x": 1.0}, None) == 1
    (tmp_path / "new" / "other.csv").write_text("a\n1\n")
    (tmp_path / "new" / "t.json").write_text(json.dumps({"x": 1.0}))
    assert cr.main([str(tmp_path / "rec"), str(tmp_path / "new")]) == 1


def test_expected_file_count_is_enforced(tmp_path):
    assert _run(tmp_path, {"x": 1.0}, {"x": 1.0}) == 0
    assert cr.main([str(tmp_path / "rec"), str(tmp_path / "new"), "--expect-files", "2"]) == 1
