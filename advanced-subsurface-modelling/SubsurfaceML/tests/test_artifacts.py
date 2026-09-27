"""Defect (e): model artifacts must be present, verified and compatible, and
their absence must produce an actionable message."""
from __future__ import annotations

import json

import pytest

from subsurfaceml.artifacts import ArtifactError, load_bundle, save_bundle
from subsurfaceml.config import load_config, project_root


@pytest.fixture()
def cfg(tmp_path):
    c = load_config(project_root() / "config" / "demo.yaml")
    c.paths.models = tmp_path / "models"
    c.paths.data = tmp_path / "data"
    return c


def test_roundtrip_and_manifest(cfg):
    man = save_bundle(cfg, {"surrogate_x.joblib": {"a": 1}})
    assert "sha256" in man["files"]["surrogate_x.joblib"]
    assert "rebuild_command" in man and man["environment"]["scikit_learn"]
    out = load_bundle(cfg)
    assert out["surrogate_x.joblib"] == {"a": 1}


def test_missing_models_give_a_rebuild_message(cfg):
    with pytest.raises(ArtifactError, match="run_pipeline.py"):
        load_bundle(cfg)


def test_modified_file_is_rejected(cfg):
    save_bundle(cfg, {"surrogate_x.joblib": {"a": 1}})
    p = cfg.paths.models / "surrogate_x.joblib"
    p.write_bytes(p.read_bytes() + b"x")
    with pytest.raises(ArtifactError, match="hash"):
        load_bundle(cfg)


def test_feature_list_and_version_mismatch_are_rejected(cfg):
    save_bundle(cfg, {"surrogate_x.joblib": 1})
    mp = cfg.paths.models / "manifest.json"
    man = json.loads(mp.read_text())
    man["features"] = ["something_else"]
    mp.write_text(json.dumps(man))
    with pytest.raises(ArtifactError, match="feature list"):
        load_bundle(cfg)
    save_bundle(cfg, {"surrogate_x.joblib": 1})
    man = json.loads(mp.read_text())
    man["environment"]["scikit_learn"] = "0.1.0"
    mp.write_text(json.dumps(man))
    with pytest.raises(ArtifactError, match="scikit-learn 0.1.0"):
        load_bundle(cfg)
