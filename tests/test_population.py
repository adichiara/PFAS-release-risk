import numpy as np
import pandas as pd

from pfas_risk.population import cell_tiers, summarize


def test_cell_tiers_cover_the_right_share_of_land():
    idx = [f"hex1_{i:05d}" for i in range(100)]
    df = pd.DataFrame({"land_km2": np.ones(100)}, index=idx)
    t = cell_tiers(df, pd.Series(np.linspace(1, 0.01, 100), index=idx))
    assert t["top10"].sum() == 10 and t["top25"].sum() == 25
    assert t.loc["hex1_00000", "top10"] == 1 and t.loc["hex1_00050", "top25"] == 0


def test_summarize_separates_density_from_group_membership():
    # Two density bands; high-risk cells are all in the dense band. Group A lives only in the
    # dense band, so its raw share is high but it is exactly what density predicts (ratio 1).
    n = 20
    b = pd.DataFrame({
        "POP20": np.full(n, 100.0),
        "ALAND20": np.r_[np.full(10, 1e5), np.full(10, 1e7)],        # dense, then sparse
        "top10": np.r_[np.full(5, 1.0), np.zeros(15)],
        "top25": np.r_[np.full(10, 1.0), np.zeros(10)],
        "pct": np.linspace(99, 1, n),
        "EJ": ["Yes"] * 10 + [None] * 10,
        "EJ_CRIT_DE": ["Minority"] * 10 + [None] * 10,
        "public_water": [True] * 15 + [False] * 5,
    }, index=[f"25{i:013d}" for i in range(n)])
    rows = {r["group"]: r for r in summarize(b)}
    assert rows["All residents"]["top10"] == 0.25
    assert rows["In EJ block groups"]["top10"] == 0.5
    assert rows["In EJ block groups"]["top10_vs_density"] == 1.0
    assert rows["Outside community water service (private wells)"]["top10"] == 0.0
