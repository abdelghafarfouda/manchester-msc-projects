"""Heterogeneous permeability / porosity generation and heterogeneity metrics.

Implements the stratification and heterogeneity material of
``5-Uncertainty.pdf`` (pp. "Quantification of heterogeneity": variance,
coefficient of variation, Dykstra-Parsons) for a **layered, no-crossflow**
storage unit.

Dykstra-Parsons coefficient
---------------------------
For a log-normal permeability distribution the Dykstra-Parsons coefficient is

    V_DP = (k_50 - k_84.1) / k_50 = 1 - exp(-sigma_lnk)

so a target ``V_DP`` is reproduced exactly by setting
``sigma_lnk = -ln(1 - V_DP)``.  ``V_DP -> 0`` is homogeneous,
``V_DP -> 1`` is extremely stratified.

**Scope note.**  ``V_DP`` here is used as a *heterogeneity descriptor* of the
sampled layer permeabilities and as an ML feature.  It is **not** used as a
Dykstra-Parsons stratified-waterflood *performance correlation* (the classical
Johnson chart), which predicts vertical coverage for a linear waterflood with
mobility ratio M and would not apply to radial CO2 injection.  The full
stratified waterflood model is discussed as an extension in the report.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# --------------------------------------------------------------------------
# heterogeneity metrics
# --------------------------------------------------------------------------
def dykstra_parsons(k: np.ndarray) -> float:
    """Dykstra-Parsons coefficient of permeability variation.

    Estimated the log-normal way: fit ``sigma`` of ``ln k`` and return
    ``1 - exp(-sigma)``.  Equivalent to the (k50-k84.1)/k50 percentile
    definition when the sample is log-normal, but far more stable for the
    small layer counts used here.
    """
    k = np.asarray(k, float).ravel()
    k = k[k > 0]
    if k.size < 2:
        return 0.0
    sigma = float(np.std(np.log(k), ddof=1))
    return float(1.0 - np.exp(-sigma))


def dykstra_parsons_percentile(k: np.ndarray) -> float:
    """Percentile form ``(k50 - k84.1)/k50`` (needs a reasonably large sample)."""
    k = np.asarray(k, float).ravel()
    k = np.sort(k[k > 0])[::-1]
    if k.size < 3:
        return 0.0
    k50 = float(np.percentile(k, 50))
    k841 = float(np.percentile(k, 100 - 84.1))
    return float((k50 - k841) / k50) if k50 > 0 else 0.0


def coefficient_of_variation(x: np.ndarray) -> float:
    """Relative standard deviation (``5-Uncertainty.pdf``)."""
    x = np.asarray(x, float).ravel()
    m = float(np.mean(x))
    return float(np.std(x, ddof=1) / m) if m != 0 else 0.0


# --------------------------------------------------------------------------
# property generation
# --------------------------------------------------------------------------
@dataclass
class LayeredRock:
    """A no-crossflow layered storage unit.

    Attributes
    ----------
    k : (n_layers, n_r) array of absolute permeability [m^2]
    phi : (n_layers, n_r) array of porosity [-]
    h : (n_layers,) layer thicknesses [m]
    k_layer : (n_layers,) layer-average (thickness/geometric) permeability
    """

    k: np.ndarray
    phi: np.ndarray
    h: np.ndarray
    meta: dict = field(default_factory=dict)

    @property
    def n_layers(self) -> int:
        return self.k.shape[0]

    @property
    def n_r(self) -> int:
        return self.k.shape[1]

    @property
    def k_layer(self) -> np.ndarray:
        """Geometric-mean permeability of each layer (radial direction)."""
        return np.exp(np.mean(np.log(self.k), axis=1))

    @property
    def total_thickness(self) -> float:
        return float(np.sum(self.h))

    def summary(self) -> dict:
        kl = self.k_layer
        return {
            "n_layers": self.n_layers,
            "h_total_m": self.total_thickness,
            "k_geom_mean_m2": float(np.exp(np.mean(np.log(self.k)))),
            "k_arith_mean_m2": float(np.mean(self.k)),
            "k_harm_mean_m2": float(1.0 / np.mean(1.0 / self.k)),
            "k_min_layer_m2": float(np.min(kl)),
            "k_max_layer_m2": float(np.max(kl)),
            "V_DP_layers": dykstra_parsons(kl),
            "CV_k_layers": coefficient_of_variation(kl),
            "phi_mean": float(np.mean(self.phi)),
            "kh_total": float(np.sum(np.mean(self.k, axis=1) * self.h)),
        }


def make_layered_rock(n_layers: int, n_r: int, h_total: float,
                      k_median: float, V_DP: float,
                      phi_mean: float, *,
                      poro_perm_slope: float = 0.06,
                      seed: int | None = None) -> LayeredRock:
    """Generate a layered unit with log-normal layer permeabilities.

    Each layer is homogeneous in the radial direction; the heterogeneity is
    between layers (stratification), which is what the Dykstra-Parsons
    coefficient of ``5-Uncertainty.pdf`` p.23 measures.  Layer
    permeabilities are drawn from a log-normal distribution
    (``5-Uncertainty.pdf`` p.43) with ``sigma_lnk = -ln(1 - V_DP)``.

    Parameters
    ----------
    n_layers, n_r : grid size (layers x radial cells)
    h_total : total unit thickness [m], split equally between layers
    k_median : median (geometric-mean) permeability [m^2]
    V_DP : target Dykstra-Parsons coefficient of the layer permeabilities
    phi_mean : target mean porosity [-]
    poro_perm_slope : porosity tied to log-permeability,
        ``phi = phi_mean * (1 + slope * (ln k - mean ln k))`` -- a simple
        monotone porosity-permeability trend (the kind of relationship the
        ``poro-perm.csv`` examples of ``Lecture03/Lecture08.ipynb`` explore;
        that data file itself is not supplied).  0 gives constant porosity.
    seed : RNG seed (recorded in ``meta`` for provenance)

    An earlier version also generated a spatially correlated field *within*
    each layer (exponential covariance).  That technique is not in the
    supplied material and was removed to keep the model within scope.
    """
    rng = np.random.default_rng(seed)
    V_DP = float(np.clip(V_DP, 0.0, 0.95))
    sigma_layers = -np.log(1.0 - V_DP) if V_DP > 0 else 0.0
    ln_k_layer = np.log(k_median) + rng.normal(0.0, sigma_layers, size=n_layers)
    k = np.repeat(np.exp(ln_k_layer)[:, None], n_r, axis=1)

    lnk = np.log(k)
    phi = phi_mean * (1.0 + poro_perm_slope * (lnk - lnk.mean()))
    phi = np.clip(phi, 0.02, 0.40)

    h = np.full(n_layers, h_total / n_layers)
    meta = {
        "seed": seed,
        "k_median_m2": k_median,
        "V_DP_target": V_DP,
        "sigma_lnk_layers": sigma_layers,
        "phi_mean_target": phi_mean,
        "poro_perm_slope": poro_perm_slope,
        "generator": "make_layered_rock",
    }
    return LayeredRock(k=k, phi=phi, h=h, meta=meta)
