import numpy as np
import pandas as pd
from scipy.special import expit

from pfas_risk import private_wells


def _blocks(rng, towns=30, per=20):
    town = np.repeat([f"T{i}" for i in range(towns)], per)
    base = np.repeat(rng.normal(-1.0, 0.8, towns), per) + rng.normal(0, 0.3, towns * per)
    return pd.DataFrame({"TOWN": town, "POP20": rng.integers(10, 200, towns * per).astype(float),
                         "p_over20": expit(base)}), base


def test_calibration_recovers_a_known_offset(monkeypatch):
    rng = np.random.default_rng(0)
    blocks, base = _blocks(rng)
    # Private wells are truly 1.2 log-odds below the model; simulate each town's test results.
    true_town = private_wells.town_means(pd.Series(expit(base - 1.2)), blocks["POP20"], blocks["TOWN"])
    n = np.full(len(true_town), 400)
    prog = pd.DataFrame({"n": n, "k": rng.binomial(n, true_town.to_numpy()),
                         "random_invites": 0, "targeted_invites": 1}, index=true_town.index)
    monkeypatch.setattr(private_wells, "program_results", lambda: prog)
    p, stats = private_wells.calibrate(blocks)
    assert abs(stats["offset"] + 1.2) < 0.15
    assert stats["loto_log_loss"]["calibrated_model"] < stats["loto_log_loss"]["uncalibrated_model"]
    # Sampled towns are pulled toward their own results.
    got = private_wells.town_means(p, blocks["POP20"], blocks["TOWN"])
    observed = prog["k"] / prog["n"]
    assert np.corrcoef(got[observed.index], observed)[0, 1] > 0.9


def test_prior_weight_is_large_when_towns_match_the_model():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.02, 0.2, 200)
    n = np.full(200, 50)
    exact = private_wells.fit_prior_weight(p, rng.binomial(n, p), n)
    noisy = private_wells.fit_prior_weight(p, rng.binomial(n, rng.permutation(p)), n)
    assert exact > noisy
