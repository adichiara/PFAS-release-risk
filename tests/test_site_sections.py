import numpy as np
import pandas as pd

from pfas_risk.model import feature_group, grouped_importance
from pfas_risk.site_sections import capture_curve, group_label


def test_capture_curve_orders_by_risk_per_area():
    scores = np.array([0.9, 0.1, 0.5])
    area = np.array([1.0, 1.0, 2.0])        # risk per km²: 0.9, 0.1, 0.25
    releases = np.array([2, 0, 2])
    grid = np.array([0.0, 0.25, 0.75, 1.0])
    out = capture_curve(scores, releases, area, grid)
    assert out[0] == 0 and out[-1] == 1
    assert out[1] == 0.5                      # first 25% of land = unit 0 = 2 of 4 releases
    assert out[2] == 1.0                      # next 50% = unit 2 = the other 2


def test_feature_groups_and_labels():
    assert feature_group("n2k_ind_petroleum") == feature_group("d_ind_petroleum_km") == "ind_petroleum"
    assert feature_group("pop_density") == feature_group("housing_density") == "population"
    assert group_label("ind_paper_printing") == "Industry: paper and printing"


def test_grouped_importance_finds_the_informative_group():
    rng = np.random.default_rng(0)
    n = 800
    df = pd.DataFrame({
        "land_km2": rng.uniform(0.5, 5, n), "pop_density": rng.uniform(10, 5000, n),
        "housing_density": rng.uniform(5, 2000, n), "n_signal": rng.poisson(0.5, n),
        "n_noise": rng.poisson(0.5, n), "town": rng.choice([f"t{i}" for i in range(40)], n),
    })
    df["releases"] = rng.poisson(0.03 + 0.5 * df["n_signal"])
    imp = grouped_importance(df, "logistic", repeats=1, shuffles=2).set_index("group")["mean"]
    assert imp.idxmax() == "signal"
    assert imp["signal"] > imp["noise"]
