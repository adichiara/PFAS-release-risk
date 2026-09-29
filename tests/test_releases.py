import pandas as pd
import pytest

from pfas_risk.releases import _standardize, is_pfas, pfas_rtns


@pytest.mark.parametrize("name", [
    "PFAS6 (SUMMATION)", "PERFLUROHEXANSULFONIC ACID PFHXS", "TOTAL REGULATED PFAS COMPOUNDS (PPT)",
    "PER- AND POLYFLUOROALKYL SUBSTANCES (PFAS)", "PERFLUOROOCTANESULFONIC ACID", "pfoa", "HFPO-DA (GenX)",
])
def test_pfas_names_match(name):
    assert is_pfas(name)


@pytest.mark.parametrize("name", ["BENZENE", "#2 FUEL OIL", "TETRACHLOROETHYLENE", "1,4-DIOXANE", None])
def test_other_chemicals_do_not_match(name):
    assert not is_pfas(name)


def test_standardize_maps_portal_style_columns():
    df = pd.DataFrame(columns=["Release Tracking Number", "Release Town", "Release Address", "Chemical Name"])
    assert set(_standardize(df).columns) == {"rtn", "town", "address", "chemical"}


def test_pfas_rtns_merges_database_and_seed():
    release = pd.DataFrame({
        "RTN": ["1-1", "1-1", "2-2", "3-3", "4-4"],        # 1-1 repeated verbatim, as in RELEASE.DBF
        "TOWN": ["A", "A", "B", "C", "D"], "ADDRESS": ["1 X St"] * 5, "SITE_NAME": ["s"] * 5,
        "OFC_NOTIF": ["02/01/2021"] * 5, "CURRENT_ST": ["TIERI", "TIERI", "RTNCLOSED", "DPS", "RAO"],
        "PRIM_ID": [None, None, "1-1", None, None],       # 2-2 was closed into 1-1
    })
    chemical = pd.DataFrame({
        "RTN": ["1-1", "1-1", "2-2", "3-3"],
        "CHEMICAL": ["PFOS", "PFAS6", "PFOA", "BENZENE"],
    })
    seed = pd.DataFrame({"rtn": ["1-1", "4-4"], "town": ["A", "D"], "address": ["1 X St", "2 Y St"],
                         "site_name": ["s", "t"], "notification_date": ["2020-12-09", "2020-12-09"],
                         "chemical": ["PFOS", "PFAS"]})
    out = pfas_rtns(release, chemical, seed).set_index("rtn")
    assert list(out.index) == ["1-1", "4-4"]              # 2-2 linked away, 3-3 not PFAS
    assert out.loc["1-1", "chemical"] == "PFAS6; PFOS"
    assert out.loc["1-1", "source"] == "MassDEP database"
    assert out.loc["4-4", "source"] == "2021 list only"
    assert out["notification_date"].notna().all()         # both date formats parse


def test_standardize_reports_missing_columns():
    with pytest.raises(ValueError, match="chemical"):
        _standardize(pd.DataFrame(columns=["RTN", "Town", "Address"]))
