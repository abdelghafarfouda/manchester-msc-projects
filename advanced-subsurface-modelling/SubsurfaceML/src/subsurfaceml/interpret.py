"""Interpretation and unsupervised structure.

Nothing here is a causal claim.  Permutation importance, SHAP and the local
surrogate explanation measure how a *fitted model* uses its inputs on a
*particular data distribution*; the only causal statements in this project
come from the simulator's governing equations.

Course sources: permutation importance (``Lecture05``,
``E01_FeatureSelection``), SHAP (``Lecture08``, ``E02_SHAP``), LIME and
counterfactual / what-if analysis (``Lecture08``), PCA and kernel PCA
(``Lecture07``, ``E01_PCA``, ``PCA.ipynb``), k-means and DBSCAN with inertia,
silhouette, Davies-Bouldin and adjusted mutual information (``Lecture07``),
t-SNE and UMAP (``Lecture08``, ``E01_UMAP_tSNE``).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA, KernelPCA
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import (adjusted_mutual_info_score, davies_bouldin_score,
                             silhouette_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def permutation_importances(estimator, X, y, feature_names, *,
                            n_repeats: int = 20, random_state: int = 0,
                            scoring: str = "neg_root_mean_squared_error"
                            ) -> pd.DataFrame:
    """Permutation importance on **held-out** data (the test realisations)."""
    r = permutation_importance(estimator, X, y, n_repeats=n_repeats,
                               random_state=random_state, scoring=scoring,
                               n_jobs=-1)
    return (pd.DataFrame({"feature": feature_names,
                          "importance_mean": r.importances_mean,
                          "importance_std": r.importances_std})
            .sort_values("importance_mean", ascending=False)
            .reset_index(drop=True))


def _inner(estimator):
    return getattr(estimator, "regressor_", estimator)


def shap_summary(estimator, X, feature_names, *, max_samples: int = 300):
    """SHAP values of a tree-based pipeline on its preprocessed inputs
    (``E02_SHAP``: ``TreeExplainer`` on the final estimator).  Returns
    ``(values, X_used)`` or ``None`` for non-tree models."""
    try:
        import shap
    except Exception:                                        # pragma: no cover
        return None
    est = _inner(estimator)
    if not isinstance(est, Pipeline):
        return None
    model = est.named_steps.get("model")
    if model is None or not (hasattr(model, "estimators_")
                             or hasattr(model, "get_booster")
                             or hasattr(model, "tree_")):
        return None
    Xt = est[:-1].transform(X[:max_samples])
    try:
        vals = shap.TreeExplainer(model).shap_values(Xt, check_additivity=False)
    except Exception:                                        # pragma: no cover
        return None
    return np.asarray(vals), pd.DataFrame(Xt, columns=feature_names)


def tree_impurity_importance(estimator, feature_names):
    """Impurity-based importance of a fitted tree ensemble
    (``E03_TuningRandomforest`` "Feature importance"), or ``None``."""
    est = _inner(estimator)
    model = est.named_steps.get("model") if isinstance(est, Pipeline) else est
    if not hasattr(model, "feature_importances_"):
        return None
    return (pd.DataFrame({"feature": feature_names,
                          "importance": model.feature_importances_})
            .sort_values("importance", ascending=False).reset_index(drop=True))


def lime_explanation(predict, x0: pd.Series, X_ref: pd.DataFrame, *,
                     n_samples: int = 2000, kernel_width: float = 0.75,
                     random_state: int = 0) -> pd.DataFrame:
    """Local surrogate explanation in the spirit of LIME (``Lecture08``):
    perturb one case, weight the perturbations by their proximity to it, and
    fit a weighted linear model to the black-box predictions.  The slope of
    each standardised input is its *local* effect on the prediction.

    Implemented directly (the ``lime`` package could not be installed in the
    reference environment); the procedure is the one the lecture describes.
    """
    rng = np.random.default_rng(random_state)
    sd = X_ref.std().replace(0, 1.0).to_numpy()
    Z = rng.normal(size=(n_samples, len(x0)))
    Xp = pd.DataFrame(x0.to_numpy() + 0.5 * Z * sd, columns=X_ref.columns)
    f = np.asarray(predict(Xp), float)
    d = np.sqrt(np.sum((0.5 * Z) ** 2, axis=1) / len(x0))
    w = np.exp(-(d ** 2) / kernel_width ** 2)
    lin = Ridge(alpha=1e-3).fit(0.5 * Z, f, sample_weight=w)
    return (pd.DataFrame({"feature": X_ref.columns,
                          "local_effect_per_sd": lin.coef_})
            .assign(abs_effect=lambda t: t.local_effect_per_sd.abs())
            .sort_values("abs_effect", ascending=False).drop(columns="abs_effect")
            .reset_index(drop=True))


def what_if_rate_scaling(predict_fn, rates0: np.ndarray, limit: float,
                         factors=None) -> dict:
    """Counterfactual / what-if (``Lecture08``): scale the whole schedule
    and find the largest factor at which the predicted quantity stays below
    ``limit``.  ``predict_fn(rates_matrix) -> predictions``."""
    factors = np.linspace(0.2, 2.0, 91) if factors is None else np.asarray(factors)
    pred = np.asarray(predict_fn(np.outer(factors, rates0)), float)
    ok = pred <= limit
    f_max = float(factors[ok].max()) if ok.any() else None
    return {"factors": factors.tolist(), "predictions": pred.tolist(),
            "largest_factor_below_limit": f_max, "limit": float(limit)}


# --------------------------------------------------------------------------
def pca_regimes(X: np.ndarray, feature_names, n_components: int = 4) -> dict:
    """PCA of standardised inputs: explained variance, scree, loadings."""
    sc = StandardScaler().fit(X)
    Z = sc.transform(X)
    p = PCA(n_components=min(n_components, Z.shape[1])).fit(Z)
    load = pd.DataFrame(p.components_.T, index=feature_names,
                        columns=[f"PC{i+1}" for i in range(p.n_components_)])
    return {"pca": p, "scaler": sc, "scores": p.transform(Z),
            "explained_variance_ratio": p.explained_variance_ratio_.tolist(),
            "cumulative_variance": np.cumsum(p.explained_variance_ratio_).tolist(),
            "loadings": load,
            "top_loadings": {c: load[c].abs().sort_values(ascending=False)
                             .head(5).index.tolist() for c in load.columns}}


def cluster_regimes(scores: np.ndarray, k_range=(2, 3, 4, 5, 6),
                    random_state: int = 0) -> dict:
    """k-means with k chosen by silhouette; inertia (elbow) and
    Davies-Bouldin reported for every k (``Lecture07``)."""
    best, table = None, []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit(scores)
        s = float(silhouette_score(scores, km.labels_))
        table.append({"k": k, "silhouette": s, "inertia": float(km.inertia_),
                      "davies_bouldin": float(davies_bouldin_score(scores, km.labels_))})
        if best is None or s > best[1]:
            best = (km, s, k)
    return {"kmeans": best[0], "labels": best[0].labels_, "k": best[2],
            "silhouette": best[1], "table": pd.DataFrame(table)}


def describe_clusters(df: pd.DataFrame, labels, cols: list[str]) -> pd.DataFrame:
    d = df[cols].copy()
    d["cluster"] = labels
    out = d.groupby("cluster").median(numeric_only=True)
    out["n"] = d.groupby("cluster").size()
    return out.reset_index()


def parse_profiles(series: pd.Series) -> np.ndarray:
    """Final-saturation profiles stored as ';'-joined strings -> array."""
    return np.vstack([np.array(s.split(";"), float) for s in series])


def profile_structure(P: np.ndarray, labels_ref=None, random_state: int = 0) -> dict:
    """Unsupervised analysis of plume *shapes* (thickness-averaged final
    saturation versus log-radius): PCA (how many shape modes?), kernel PCA
    (does a non-linear projection separate them better?), k-means and DBSCAN
    (regimes and unusual shapes), and 2-D t-SNE / UMAP maps for inspection.
    ``labels_ref`` (e.g. the traffic-light class) is compared with the
    clusters by adjusted mutual information."""
    Z = StandardScaler().fit_transform(P)
    pca = PCA().fit(Z)
    cum = np.cumsum(pca.explained_variance_ratio_)
    k95 = int(np.searchsorted(cum, 0.95) + 1)
    S = pca.transform(Z)[:, :max(2, min(k95, 6))]
    # reconstruction error with k components (PCA as compression)
    rec = {}
    for k in (1, 2, 3, 5):
        p = PCA(n_components=k).fit(Z)
        rec[k] = float(np.mean((p.inverse_transform(p.transform(Z)) - Z) ** 2))
    kp = KernelPCA(n_components=2, kernel="rbf", gamma=1.0 / Z.shape[1],
                   random_state=random_state).fit_transform(Z)
    cl = cluster_regimes(S, random_state=random_state)
    km_k = KMeans(n_clusters=cl["k"], n_init=10, random_state=random_state)
    kp_lab = km_k.fit_predict(kp)
    # DBSCAN on the PCA scores: eps from the 90th percentile of 5-NN distance
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=5).fit(S)
    eps = float(np.percentile(nn.kneighbors(S)[0][:, -1], 90))
    db = DBSCAN(eps=eps, min_samples=5).fit(S)
    out = {"n_profiles": int(len(P)), "n_bins": int(P.shape[1]),
           "explained_variance_ratio": pca.explained_variance_ratio_[:8].tolist(),
           "n_components_95pct": k95,
           "reconstruction_mse_standardised": rec,
           "kmeans_on_pca": {"k": cl["k"], "silhouette": cl["silhouette"],
                             "table": cl["table"].to_dict("records")},
           "kmeans_on_kernel_pca_silhouette": float(silhouette_score(kp, kp_lab)),
           "dbscan": {"eps": eps, "min_samples": 5,
                      "n_clusters": int(len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)),
                      "n_noise": int(np.sum(db.labels_ == -1))},
           "_scores": S, "_kpca": kp, "_labels": cl["labels"],
           "_dbscan_labels": db.labels_}
    if labels_ref is not None:
        out["ami_kmeans_vs_reference"] = float(adjusted_mutual_info_score(labels_ref, cl["labels"]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from sklearn.manifold import TSNE
        out["_tsne"] = TSNE(n_components=2, perplexity=30, random_state=random_state,
                            init="pca").fit_transform(Z)
        try:
            import umap
            out["_umap"] = umap.UMAP(n_neighbors=15, min_dist=0.1,
                                     random_state=random_state).fit_transform(Z)
        except Exception as exc:                             # pragma: no cover
            out["umap_status"] = f"unavailable: {exc!r}"
    return out
