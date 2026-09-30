import pandas as pd

from pfas_risk.industry import sector_facilities


def test_sector_facilities_match_any_naics_prefix():
    fac = pd.DataFrame({
        "REGISTRY_ID": ["a", "b", "c"], "PRIMARY_NAME": ["Mill", "Plater", "Bakery"],
        "LATITUDE83": [42.3, 42.4, 42.5], "LONGITUDE83": [-71.1, -71.2, -71.3],
    })
    naics = pd.DataFrame({
        "REGISTRY_ID": ["a", "b", "b", "c"],
        "NAICS_CODE": ["313210", "423510", "332813", "311811"],   # b is a plater by a secondary code
    })
    sectors = {"textiles": {"naics": ["313", "314"]}, "metal_finishing": {"naics": ["3328"]}}
    out = sector_facilities(fac, naics, sectors)
    assert set(out) == {"ind_textiles", "ind_metal_finishing"}
    assert out["ind_textiles"]["REGISTRY_ID"].tolist() == ["a"]
    assert out["ind_metal_finishing"]["REGISTRY_ID"].tolist() == ["b"]
    assert out["ind_textiles"].crs == "EPSG:26986"
