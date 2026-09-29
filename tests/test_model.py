import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point, box

from pfas_risk.features import point_measures
from pfas_risk.model import ExposurePoisson, capture, evaluate, town_folds


def test_town_folds_never_split_a_town():
    towns = pd.Series(np.repeat([f"t{i}" for i in range(23)], 7))
    folds = town_folds(towns, n_folds=5, seed=3)
    covered = np.concatenate([te for _, te in folds])
    assert sorted(covered) == list(range(len(towns)))
    for tr, te in folds:
        assert not set(towns.iloc[tr]) & set(towns.iloc[te])


def test_capture_respects_budget():
    scores = np.array([0.9, 0.8, 0.1, 0.05])
    releases = np.array([1, 0, 3, 0])
    assert capture(scores, releases, np.ones(4), 0.25) == 0.25
    # Area budget: the first unit alone is 60% of the area, so nothing fits in 50%.
    assert capture(scores, releases, np.array([6.0, 1, 1, 2]), 0.5) == 0.0


def test_exposure_poisson_accounts_for_area():
    # Same rate per km² everywhere: expected counts should scale with area.
    rng = np.random.default_rng(0)
    area = rng.uniform(0.5, 20, 400)
    X = pd.DataFrame({"land_km2": area, "noise": rng.uniform(0, 1, 400)})
    y = rng.poisson(0.2 * area)
    m = ExposurePoisson(alpha=1e-6).fit(X, y)
    mu = m.expected_count(X)
    assert np.corrcoef(mu, area)[0, 1] > 0.99
    assert mu.sum() == pytest.approx(y.sum(), rel=0.05)


def test_point_measures_counts_and_distance():
    bg = gpd.GeoDataFrame({"geoid": ["a", "b"]},
                          geometry=[box(0, 0, 1000, 1000), box(5000, 0, 6000, 1000)], crs="EPSG:26986")
    bg = bg.set_index("geoid")
    pts = gpd.GeoDataFrame(geometry=[Point(500, 500), Point(2500, 500)], crs="EPSG:26986")
    m = point_measures(bg, pts, "x")
    assert m.loc["a", "n_x"] == 1 and m.loc["b", "n_x"] == 0
    assert m.loc["a", "n2k_x"] == 2 and m.loc["b", "n2k_x"] == 0   # 2500 is 1.5 km from a, 2.5 km from b
    assert m.loc["b", "d_x_km"] == pytest.approx(3.0, abs=0.01)


def test_evaluate_runs_on_synthetic_data():
    rng = np.random.default_rng(1)
    n = 600
    df = pd.DataFrame({
        "land_km2": rng.uniform(0.2, 10, n),
        "pop_density": rng.uniform(10, 5000, n),
        "housing_density": rng.uniform(5, 2000, n),
        "n_fire_station": rng.poisson(0.3, n),
        "town": rng.choice([f"t{i}" for i in range(40)], n),
    })
    df["releases"] = rng.poisson(0.02 * df["land_km2"] * (1 + 3 * df["n_fire_station"]))
    metrics, oof = evaluate(df, repeats=2)
    assert set(metrics["model"]) == {"baseline_area", "baseline_area_population", "logistic",
                                     "poisson_rate", "gradient_boosting"}
    assert oof.shape == (n, 5) and oof.notna().all().all()
