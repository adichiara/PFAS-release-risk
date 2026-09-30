"""Models and spatially grouped evaluation.

Evaluation holds out whole towns, so a block group is never scored by a model that
saw its neighbors. Every model is compared with two baselines (land area alone; area
plus population), because larger and busier places mechanically see more reported
releases. A permutation test gives the score expected from noise.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from .features import model_columns

AREA_COL = "land_km2"


def _scaled(estimator) -> object:
    # All features are non-negative counts, densities, distances or fractions.
    return make_pipeline(FunctionTransformer(np.log1p), StandardScaler(), estimator)


class ExposurePoisson(BaseEstimator, ClassifierMixin):
    """Poisson regression on releases per km² (land area as exposure).

    Fitting the rate y/area with sample weight = area is equivalent to a Poisson GLM with
    a log(area) offset, so area is accounted for rather than learned as risk.
    ``predict_proba`` returns P(at least one release) = 1 - exp(-expected count).
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def fit(self, X: pd.DataFrame, y):
        area = X[AREA_COL].to_numpy()
        self.model_ = _scaled(PoissonRegressor(alpha=self.alpha, max_iter=1000))
        self.model_.fit(X, np.asarray(y) / area, poissonregressor__sample_weight=area)
        self.classes_ = np.array([0, 1])
        return self

    def expected_count(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict(X) * X[AREA_COL].to_numpy()

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = 1 - np.exp(-self.expected_count(X))
        return np.column_stack([1 - p, p])


def model_specs(columns: list[str]) -> dict[str, tuple[list[str], object]]:
    logistic = LogisticRegression(C=0.1, class_weight="balanced", max_iter=5000)
    return {
        "baseline_area": ([AREA_COL], _scaled(logistic)),
        "baseline_area_population": ([AREA_COL, "pop_density", "housing_density"], _scaled(logistic)),
        "logistic": (columns, _scaled(logistic)),
        "poisson_rate": (columns, ExposurePoisson(alpha=1.0)),
        "gradient_boosting": (columns, HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=40,
            l2_regularization=1.0, class_weight="balanced", random_state=0)),
    }


def town_folds(towns: pd.Series, n_folds: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Randomly assign whole towns to folds (a shuffled GroupKFold)."""
    rng = np.random.default_rng(seed)
    unique = towns.unique()
    fold_of = dict(zip(unique, rng.permutation(len(unique)) % n_folds, strict=True))
    fold = towns.map(fold_of).to_numpy()
    return [(np.where(fold != k)[0], np.where(fold == k)[0]) for k in range(n_folds)]


def capture(scores: np.ndarray, releases: np.ndarray, budget: np.ndarray, frac: float) -> float:
    """Share of releases in the highest-scoring units whose cumulative ``budget`` <= frac."""
    order = np.argsort(-scores)
    spent = np.cumsum(budget[order]) / budget.sum()
    chosen = order[spent <= frac]
    return float(releases[chosen].sum() / releases.sum())


def score(y_count: np.ndarray, p: np.ndarray, area: np.ndarray) -> dict[str, float]:
    """Discrimination plus two targeting budgets (random targeting captures 10% under each).

    ``capture_top10pct_units``: releases in the 10% of block groups with the highest risk.
    ``capture_top10pct_area``: releases in the highest risk-per-km² block groups covering
    10% of land, i.e. what a field program with a fixed area budget would find.
    """
    y = (y_count > 0).astype(int)
    return {
        "roc_auc": roc_auc_score(y, p),
        "avg_precision": average_precision_score(y, p),
        "capture_top10pct_units": capture(p, y_count, np.ones_like(p), 0.10),
        "capture_top10pct_area": capture(p / area, y_count, area, 0.10),
    }


def out_of_fold(df: pd.DataFrame, cols: list[str], estimator, folds) -> np.ndarray:
    X, y = df[cols], df["releases"].to_numpy()
    target = y if isinstance(estimator, ExposurePoisson) else (y > 0).astype(int)
    p = np.zeros(len(df))
    for tr, te in folds:
        m = clone(estimator).fit(X.iloc[tr], target[tr])
        p[te] = m.predict_proba(X.iloc[te])[:, 1]
    return p


def evaluate(df: pd.DataFrame, repeats: int = 10, n_folds: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Repeated town-grouped CV. Returns (per-repeat metrics, mean out-of-fold scores)."""
    specs = model_specs(model_columns(df))
    y, area = df["releases"].to_numpy(), df[AREA_COL].to_numpy()
    rows, oof = [], {name: np.zeros(len(df)) for name in specs}
    for r in range(repeats):
        folds = town_folds(df["town"], n_folds, seed=r)
        for name, (cols, est) in specs.items():
            p = out_of_fold(df, cols, est, folds)
            oof[name] += p / repeats
            rows.append({"model": name, "repeat": r, **score(y, p, area)})
    return pd.DataFrame(rows), pd.DataFrame(oof, index=df.index)


def size_strata(df: pd.DataFrame, bins: int = 5) -> pd.Series:
    """Quintile of land area x quintile of population density (25 strata)."""
    a = pd.qcut(df[AREA_COL].rank(method="first"), bins, labels=False)
    p = pd.qcut(df["pop_density"].rank(method="first"), bins, labels=False)
    return a * bins + p


def permutation_null(df: pd.DataFrame, model: str, permutations: int = 20, n_folds: int = 5) -> pd.DataFrame:
    """Same CV with release counts shuffled among block groups of similar size and density.

    A plain shuffle is the wrong reference here: release reporting scales with land area and
    population, and the area-budget metric is itself sensitive to how releases are spread
    over block-group sizes. Shuffling within size x density strata keeps those relationships,
    so the null shows what a model reaches *without* information from the other features.
    """
    cols, est = model_specs(model_columns(df))[model]
    area = df[AREA_COL].to_numpy()
    strata = size_strata(df).to_numpy()
    rows = []
    for s in range(permutations):
        rng = np.random.default_rng(1000 + s)
        y = df["releases"].to_numpy().copy()
        for k in np.unique(strata):
            idx = np.where(strata == k)[0]
            y[idx] = rng.permutation(y[idx])
        shuffled = df.copy()
        shuffled["releases"] = y
        p = out_of_fold(shuffled, cols, est, town_folds(df["town"], n_folds, seed=s))
        rows.append({"model": model, "permutation": s, **score(shuffled["releases"].to_numpy(), p, area)})
    return pd.DataFrame(rows)


def fit_final(df: pd.DataFrame, model: str):
    cols, est = model_specs(model_columns(df))[model]
    y = df["releases"].to_numpy()
    target = y if isinstance(est, ExposurePoisson) else (y > 0).astype(int)
    return cols, clone(est).fit(df[cols], target)


def feature_group(column: str) -> str:
    """Collapse the count/near-count/distance trio of a source into one group name."""
    base = re.sub(r"_km$", "", re.sub(r"^(n2k_|n_|d_)", "", column))
    return {"landfill_frac": "landfill", "pop_density": "population", "housing_density": "population",
            "land_km2": "land_area"}.get(base, base)


def grouped_importance(df: pd.DataFrame, model: str, repeats: int = 3, n_folds: int = 5,
                       shuffles: int = 3) -> pd.DataFrame:
    """Drop in held-out ROC AUC when one feature group is shuffled.

    Uses the same town-grouped folds as evaluation: each fold's model scores its held-out towns
    with one group's columns shuffled together, and AUC is computed on the pooled out-of-fold
    predictions. Larger drops mean the model leans more on that group.
    """
    cols, est = model_specs(model_columns(df))[model]
    groups: dict[str, list[str]] = {}
    for c in cols:
        groups.setdefault(feature_group(c), []).append(c)
    X, y = df[cols], df["releases"].to_numpy()
    target = y if isinstance(est, ExposurePoisson) else (y > 0).astype(int)
    rows = []
    for r in range(repeats):
        rng = np.random.default_rng(2000 + r)
        base = np.zeros(len(df))
        shuffled = {(g, k): np.zeros(len(df)) for g in groups for k in range(shuffles)}
        for tr, te in town_folds(df["town"], n_folds, seed=r):
            m = clone(est).fit(X.iloc[tr], target[tr])
            held = X.iloc[te]
            base[te] = m.predict_proba(held)[:, 1]
            for g, gcols in groups.items():
                for k in range(shuffles):
                    perm = held.copy()
                    perm[gcols] = held[gcols].to_numpy()[rng.permutation(len(held))]
                    shuffled[(g, k)][te] = m.predict_proba(perm)[:, 1]
        auc = roc_auc_score(y > 0, base)
        for (g, k), p in shuffled.items():
            rows.append({"group": g, "repeat": r, "shuffle": k, "auc_drop": auc - roc_auc_score(y > 0, p)})
    out = pd.DataFrame(rows).groupby("group")["auc_drop"].agg(["mean", "std"])
    return out.sort_values("mean", ascending=False).reset_index()
