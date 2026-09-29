import pandas as pd
import pytest

from pfas_risk.releases import _standardize, is_pfas


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


def test_standardize_reports_missing_columns():
    with pytest.raises(ValueError, match="chemical"):
        _standardize(pd.DataFrame(columns=["RTN", "Town", "Address"]))
