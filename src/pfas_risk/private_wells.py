"""Private-well estimates checked and calibrated against MassDEP's private-well sampling program.

From 2020 to 2022 MassDEP tested 1,649 private wells in 83 towns where most residents use
private wells, and published, per town, how many wells were sampled and how many had PFAS6
at or above 20 ng/L (MassDEP's ArcGIS layer behind its Private Wells PFAS Sampling Program
dashboard). Invitations were mostly targeted near suspected sources, so the program's rates
likely run higher than those of private wells in general.

The groundwater model (``groundwater.py``) learns from public wells, which turn out to be
more contaminated than private wells: it ranks towns well against the program's results but
predicts about 2.5 times their rate. So:

1. **Calibrate:** shift the model's log-odds by one offset, fit by maximum likelihood to the
   towns' counts (a slope as well did not help under leave-one-town-out validation).
2. **Blend:** in sampled towns, combine the calibrated estimate with the town's own results,
   ``(k + m * p) / (n + m)``, with the prior weight ``m`` (in wells) fit by beta-binomial
   maximum likelihood; each block's odds are scaled so its town matches that blend.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import betaln, expit, logit
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from .sources import download


def program_results() -> pd.DataFrame:
    """Towns sampled in MassDEP's program: wells sampled (n) and at or above 20 ng/L (k)."""
    t = pd.read_csv(download("massdep_private_wells"))
    t = t[(t["TOWN_SAMPLED"] == 1) & (t["wells_sampl"] > 0)]
    return pd.DataFrame({"n": t["wells_sampl"].to_numpy(), "k": t["Wells_over_20"].fillna(0).to_numpy(),
                         "random_invites": t["RAND_INVI"].fillna(0).to_numpy(),
                         "targeted_invites": t["TARG_INV"].fillna(0).to_numpy()},
                        index=t["TOWN"].str.upper().to_numpy())


def _logit(p: pd.Series | np.ndarray) -> np.ndarray:
    return logit(np.clip(np.asarray(p, dtype=float), 1e-4, 1 - 1e-4))


def town_means(p: pd.Series, pop: pd.Series, town: pd.Series) -> pd.Series:
    """Population-weighted mean of block probabilities per town."""
    return (p * pop).groupby(town).sum() / pop.groupby(town).sum()


def _binomial_nll(p: np.ndarray, k: np.ndarray, n: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(k * np.log(p) + (n - k) * np.log(1 - p)).sum())


def fit_offset(lg: np.ndarray, pop: np.ndarray, town: np.ndarray, prog: pd.DataFrame) -> float:
    """Log-odds shift that best matches the program's town counts."""
    towns = prog.index.to_numpy()

    def nll(b: float) -> float:
        tm = town_means(pd.Series(expit(lg + b)), pd.Series(pop), pd.Series(town))
        return _binomial_nll(tm.reindex(towns).to_numpy(), prog["k"].to_numpy(), prog["n"].to_numpy())

    return float(minimize_scalar(nll, bounds=(-5, 5), method="bounded").x)


def fit_prior_weight(model_p: np.ndarray, k: np.ndarray, n: np.ndarray) -> float:
    """Beta-binomial prior strength (in wells) around the model's town rates."""
    def nll(log_m: float) -> float:
        m = np.exp(log_m)
        a, b = model_p * m, (1 - model_p) * m
        return float(-(betaln(k + a, n - k + b) - betaln(a, b)).sum())

    return float(np.exp(minimize_scalar(nll, bounds=(-2, 8), method="bounded").x))


def calibrate(private: pd.DataFrame) -> tuple[pd.Series, dict]:
    """Calibrated, blended P(PFAS6 >= 20 ng/L) per private-well block, and validation stats.

    ``private`` needs POP20, TOWN and p_over20 (the public-well model's probability).
    """
    prog = program_results()
    town = private["TOWN"].str.upper()
    sampled = prog.index.intersection(town.unique())
    prog = prog.loc[sampled]
    lg, pop = _logit(private["p_over20"]), private["POP20"].to_numpy()
    k, n = prog["k"].to_numpy(), prog["n"].to_numpy()

    raw_town = town_means(private["p_over20"], private["POP20"], town).reindex(sampled).to_numpy()
    offset = fit_offset(lg, pop, town.to_numpy(), prog)

    # Leave-one-town-out: refit the offset without each town, then predict it.
    loto = []
    for name in sampled:
        b = fit_offset(lg, pop, town.to_numpy(), prog.drop(index=name))
        tm = town_means(pd.Series(expit(lg + b), index=private.index), private["POP20"], town)
        loto.append(tm[name])
    loto = np.array(loto)
    flat = k.sum() / n.sum()

    p_cal = pd.Series(expit(lg + offset), index=private.index)
    cal_town = town_means(p_cal, private["POP20"], town)
    m = fit_prior_weight(cal_town.reindex(sampled).to_numpy(), k, n)
    blended = pd.Series((k + m * cal_town.reindex(sampled).to_numpy()) / (n + m), index=sampled)
    # Scale each sampled town's block odds so the town's mean matches the blend.
    adjusted = p_cal.copy()
    for name in sampled:
        rows = town == name
        target = blended[name]

        def gap(s: float, rows=rows, target=target) -> float:
            p = expit(_logit(p_cal[rows]) + s)
            return (np.average(p, weights=private.loc[rows, "POP20"]) - target) ** 2

        shift = minimize_scalar(gap, bounds=(-6, 6), method="bounded").x
        adjusted[rows] = expit(_logit(p_cal[rows]) + shift)

    stats = {
        "towns": len(sampled), "wells": int(n.sum()), "wells_over20": int(k.sum()),
        "observed_share": round(float(flat), 3),
        "targeted_share_of_invitations": round(float(prog["targeted_invites"].sum()
                                                    / (prog["targeted_invites"] + prog["random_invites"]).sum()), 2),
        "model_share_before_calibration": round(float(np.average(raw_town, weights=n)), 3),
        "spearman_towns": round(float(spearmanr(raw_town, k / n).statistic), 2),
        # Defined only when some tested towns had no well at 20 ng/L or more.
        "auc_towns_any_over20": (round(float(roc_auc_score(k > 0, raw_town)), 2)
                                 if 0 < (k > 0).sum() < len(k) else None),
        "loto_log_loss": {"calibrated_model": round(_binomial_nll(loto, k, n), 1),
                          "uncalibrated_model": round(_binomial_nll(raw_town, k, n), 1),
                          "flat_rate": round(_binomial_nll(np.full(len(k), flat), k, n), 1)},
        "offset": round(offset, 3), "prior_weight_wells": round(m, 1),
        # Per tested town: wells sampled (n) and at 20+ (k); the model's town rate before calibration,
        # the calibrated rate predicted without that town (out of sample), and the final blend.
        "town_table": [
            {"town": str(name).title(), "n": int(nn), "k": int(kk), "model": round(float(r), 4),
             "loto": round(float(lo), 4), "blended": round(float(blended[name]), 4)}
            for name, nn, kk, r, lo in zip(sampled, n, k, raw_town, loto, strict=True)],
    }
    return adjusted, stats
