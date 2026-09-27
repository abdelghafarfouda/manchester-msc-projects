"""Saving and loading trained model artifacts with provenance.

``results/<config>/models/`` holds

* ``surrogate_<target>.joblib``     one fitted surrogate per target
* ``band_<target>.joblib``          its empirical error band
* ``pressure_classifier.joblib``    the binary pressure-limit screen
* ``manifest.json``                 SHA-256 of every file, the feature list,
                                    package versions, config name, seeds,
                                    dataset hash and the command that
                                    rebuilds them

:func:`load_bundle` refuses to return models whose files are missing, whose
hashes do not match the manifest, whose feature list differs from the
current :data:`features.FEATURES`, or that were saved with a different
scikit-learn major.minor version -- each with a message that says how to
rebuild (``python scripts/run_pipeline.py --config <cfg>``).
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import joblib

from .config import provenance
from .features import FEATURES


class ArtifactError(RuntimeError):
    """Missing, corrupted or incompatible model artifacts."""


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_bundle(cfg, objects: dict, *, dataset_file=None, extra=None) -> dict:
    """``objects``: ``{filename: python_object}``.  Writes them and the
    manifest; returns the manifest."""
    d = Path(cfg.paths.models)
    d.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, obj in objects.items():
        p = d / name
        joblib.dump(obj, p)
        files[name] = {"sha256": sha256(p), "bytes": p.stat().st_size}
    man = {"created": time.strftime("%Y-%m-%d %H:%M:%S"),
           "config_name": cfg.name,
           "features": list(FEATURES),
           "environment": provenance(),
           "seeds": {"scenario_master_seed": cfg.scenarios.seed,
                     "ml_random_state": cfg.ml.random_state},
           "rebuild_command": f"python scripts/run_pipeline.py --config config/{cfg.name}.yaml",
           "files": files}
    if dataset_file is not None and Path(dataset_file).exists():
        man["dataset"] = {"file": Path(dataset_file).name,
                          "sha256": sha256(dataset_file)}
    if extra:
        man.update(extra)
    (d / "manifest.json").write_text(json.dumps(man, indent=2, default=str))
    return man


def load_bundle(cfg, names=None, *, check_versions: bool = True) -> dict:
    """Load and verify artifacts.  Raises :class:`ArtifactError`."""
    d = Path(cfg.paths.models)
    rebuild = (f"Rebuild them with: python scripts/run_pipeline.py "
               f"--config config/{cfg.name}.yaml")
    mp = d / "manifest.json"
    if not mp.exists():
        raise ArtifactError(f"no model manifest at {mp}. {rebuild}")
    man = json.loads(mp.read_text())
    if man.get("features") != list(FEATURES):
        raise ArtifactError("the saved models were trained on a different "
                            f"feature list than the current code. {rebuild}")
    if check_versions:
        import sklearn
        saved = str(man.get("environment", {}).get("scikit_learn", ""))
        if saved.split(".")[:2] != sklearn.__version__.split(".")[:2]:
            raise ArtifactError(
                f"models were saved with scikit-learn {saved}, this environment "
                f"has {sklearn.__version__}; pickled estimators are not portable "
                f"across versions. Install the recorded version or {rebuild[0].lower()}{rebuild[1:]}")
    out = {"manifest": man}
    for name in (names or man["files"]):
        p = d / name
        if name not in man["files"]:
            raise ArtifactError(f"{name} is not listed in the manifest. {rebuild}")
        if not p.exists():
            raise ArtifactError(f"missing model file {p}. {rebuild}")
        if sha256(p) != man["files"][name]["sha256"]:
            raise ArtifactError(f"{p} does not match its recorded hash "
                                f"(modified or corrupted). {rebuild}")
        out[name] = joblib.load(p)
    return out
