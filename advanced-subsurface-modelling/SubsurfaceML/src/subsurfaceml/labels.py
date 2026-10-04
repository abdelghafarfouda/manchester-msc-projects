"""Readable names and units for figures and tables.

Column names such as ``dp_bh_max_MPa`` are precise but unreadable on a
figure axis.  Every figure takes its labels from here, so one name is used
for one quantity throughout the project.
"""
from __future__ import annotations

LABELS = {
    # targets
    "dp_bh_max_MPa": "Peak bottom-hole pressure build-up [MPa]",
    "r_plume_m95_m": "Plume radius (95 % of CO$_2$ mass) [m]",
    "sweep_efficiency": "Swept pore-volume fraction [-]",
    "mass_retained_Mt": "CO$_2$ retained in the compartment [Mt]",
    # reservoir description
    "k_median_mD": "Median layer permeability (prior parameter) [mD]",
    "log10_k_mD": "log$_{10}$ median permeability [mD]",
    "V_DP": "Dykstra-Parsons coefficient (prior parameter) [-]",
    "V_DP_layers": "Dykstra-Parsons coefficient of the realised layers [-]",
    "phi_mean": "Mean porosity [-]",
    "h_total_m": "Total thickness [m]",
    "r_e_m": "Compartment radius [m]",
    "log10_k_arith_mD": "log$_{10}$ realised arithmetic-mean permeability [mD]",
    "log10_k_harm_mD": "log$_{10}$ realised harmonic-mean permeability [mD]",
    "log10_k_min_layer_mD": "log$_{10}$ permeability of the tightest layer [mD]",
    "log10_k_max_layer_mD": "log$_{10}$ permeability of the most permeable layer [mD]",
    "log10_kh_realised": "log$_{10}$ realised flow capacity kh [mD m]",
    "realised_vs_prior_k": "Realised / prior-expected mean permeability [-]",
    # schedule
    "log10_q_mult_mean": "log$_{10}$ mean rate / reference rate [-]",
    "q_mult_mean": "Mean rate / reference rate [-]",
    "q_front_load": "Share of the mass injected in the first half [-]",
    "r_fill_over_re": "Fill radius / compartment radius [-]",
    # ROM
    "log10_rom_dp_MPa": "log$_{10}$ analytical-ROM build-up [MPa]",
    "rom_dp_MPa": "Analytical-ROM build-up [MPa]",
    "rom_r_fill_max_m": "Largest layer fill radius from the ROM [m]",
    "rom_max_layer_share": "Largest layer share of injected CO$_2$ [-]",
}

SHORT = {
    "dp_bh_max_MPa": "Peak pressure build-up",
    "r_plume_m95_m": "Plume radius (95 % mass)",
    "sweep_efficiency": "Swept pore-volume fraction",
}

UNITS = {"dp_bh_max_MPa": "MPa", "r_plume_m95_m": "m", "sweep_efficiency": "-"}

MODEL_NAMES = {
    "mean": "mean value", "linear": "linear regression", "ridge": "ridge",
    "lasso": "lasso", "elasticnet": "elastic net", "knn": "k-nearest neighbours",
    "svr": "support-vector regression", "tree": "decision tree",
    "random_forest": "random forest", "gradient_boosting": "gradient boosting",
    "adaboost": "AdaBoost", "xgboost": "XGBoost", "rom": "analytical ROM",
}

VARIANT_NAMES = {
    "V0_published_features": "V0 published inputs",
    "V1_plus_realised_rock": "V1 + realised layers",
    "V2_plus_rom_feature": "V2 + ROM as input",
    "V3_hybrid_rom_residual": "V3 hybrid: ROM x learned factor",
    "V4_rom_only": "V4 analytical ROM alone",
    "V5_published_features_3x_search": "V5 published inputs, 3x search",
    "V2_plus_rock_and_rom_features": "V2 + realised layers + ROM inputs",
}


def label(name: str) -> str:
    """Readable label for a column name (falls back to the name itself)."""
    return LABELS.get(name, name.replace("_", " "))


def short(name: str) -> str:
    return SHORT.get(name, label(name))
