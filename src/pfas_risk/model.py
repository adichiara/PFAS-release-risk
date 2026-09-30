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
from sklearn.neighbors import BallTree
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from .features import model_columns

AREA_COL = "land_km2"
BASELINES = ("baseline_area", "baseline_area_population")


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


SMOOTH = "+smooth"
SMOOTH_RADIUS_M = 3_000


def neighbor_index(df: pd.DataFrame, radius_m: float = SMOOTH_RADIUS_M) -> list[np.ndarray]:
    """For each unit, the units whose interior points lie within ``radius_m`` (itself included)."""
    pts = df.geometry.representative_point()
    xy = np.column_stack([pts.x, pts.y])
    return list(BallTree(xy).query_radius(xy, r=radius_m))


def smooth(p: np.ndarray, neighbors: list[np.ndarray], rows: np.ndarray) -> np.ndarray:
    """Half the unit's own score plus half the mean score of its neighbors, for ``rows``."""
    return 0.5 * p[rows] + 0.5 * np.array([p[neighbors[i]].mean() for i in rows])


def split_model(name: str) -> tuple[str, bool]:
    """'gradient_boosting+smooth' -> ('gradient_boosting', True)."""
    return (name.removesuffix(SMOOTH), name.endswith(SMOOTH))


def _target(estimator, y: np.ndarray) -> np.ndarray:
    return y if isinstance(estimator, ExposurePoisson) else (y > 0).astype(int)


def out_of_fold(df: pd.DataFrame, cols: list[str], estimator, folds,
                neighbors: list[np.ndarray] | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    """Out-of-fold scores, and (with ``neighbors``) the same scores smoothed over nearby units.

    Smoothing happens inside each fold: held-out units are blended with that fold's own
    predictions for their neighbors, so no score depends on a model that saw the unit's label.
    """
    X, target = df[cols], _target(estimator, df["releases"].to_numpy())
    raw = np.zeros(len(df))
    smoothed = np.zeros(len(df)) if neighbors is not None else None
    for tr, te in folds:
        m = clone(estimator).fit(X.iloc[tr], target[tr])
        if neighbors is None:
            raw[te] = m.predict_proba(X.iloc[te])[:, 1]
            continue
        p_all = m.predict_proba(X)[:, 1]
        raw[te] = p_all[te]
        smoothed[te] = smooth(p_all, neighbors, te)
    return raw, smoothed


def evaluate(df: pd.DataFrame, repeats: int = 10, n_folds: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Repeated town-grouped CV. Returns (per-repeat metrics, mean out-of-fold scores).

    Each non-baseline model is also scored with neighbor smoothing, as '<model>+smooth'.
    """
    specs = model_specs(model_columns(df))
    neighbors = neighbor_index(df)
    y, area = df["releases"].to_numpy(), df[AREA_COL].to_numpy()
    rows, oof = [], {}
    for r in range(repeats):
        folds = town_folds(df["town"], n_folds, seed=r)
        for name, (cols, est) in specs.items():
            raw, smoothed = out_of_fold(df, cols, est, folds, None if name in BASELINES else neighbors)
            for label, p in [(name, raw), (name + SMOOTH, smoothed)]:
                if p is None:
                    continue
                oof[label] = oof.get(label, 0) + p / repeats
                rows.append({"model": label, "repeat": r, **score(y, p, area)})
    return pd.DataFrame(rows), pd.DataFrame(oof, index=df.index)


def forward_test(df: pd.DataFrame, y_before: np.ndarray, new_after: np.ndarray, models: list[str]) -> dict:
    """Train on releases reported before a cutoff; score units whose first release came after.

    ``y_before`` is each unit's release count before the cutoff; ``new_after`` flags units with
    no earlier release that have one after it. Only units without an earlier release are scored.
    """
    specs = model_specs(model_columns(df))
    neighbors = neighbor_index(df)
    area = df[AREA_COL].to_numpy()
    rows = np.where(y_before == 0)[0]
    out = {}
    for name in models:
        base, smoothed = split_model(name)
        cols, est = specs[base]
        m = clone(est).fit(df[cols], _target(est, y_before))
        p_all = m.predict_proba(df[cols])[:, 1]
        p = smooth(p_all, neighbors, rows) if smoothed else p_all[rows]
        out[name] = {k: round(float(v), 4) for k, v in score(new_after[rows].astype(int), p, area[rows]).items()}
    return out


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
    base, smoothed = split_model(model)
    cols, est = model_specs(model_columns(df))[base]
    neighbors = neighbor_index(df) if smoothed else None
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
        raw, sm = out_of_fold(shuffled, cols, est, town_folds(df["town"], n_folds, seed=s), neighbors)
        p = sm if smoothed else raw
        rows.append({"model": model, "permutation": s, **score(shuffled["releases"].to_numpy(), p, area)})
    return pd.DataFrame(rows)


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
    cols, est = model_specs(model_columns(df))[split_model(model)[0]]
    groups: dict[str, list[str]] = {}
    for c in cols:
        groups.setdefault(feature_group(c), []).append(c)
    X, y = df[cols], df["releases"].to_numpy()
    target = _target(est, y)
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
