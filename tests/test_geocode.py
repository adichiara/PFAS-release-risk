import pandas as pd

from pfas_risk.geocode import geocode, normalize_street, split_address

INDEX = pd.DataFrame({
    "num": pd.array([10, 20, 30, 5, pd.NA], dtype="Int64"),
    "street": ["MAIN ST", "MAIN ST", "MAIN ST", "N PLEASANT ST", "RTE 6"],
    "town": ["HUDSON", "HUDSON", "HUDSON", "NATICK", "TRURO"],
    "community": ["HUDSON", "HUDSON", "HUDSON", "SOUTH NATICK", "NORTH TRURO"],
    "x": [1.0, 2.0, 3.0, 50.0, 70.0],
    "y": [10.0, 20.0, 30.0, 500.0, 700.0],
})


def test_normalize_street_abbreviates_suffixes_and_directions():
    assert normalize_street("North Pleasant Street") == "N PLEASANT ST"
    assert normalize_street("Route 6") == "RTE 6"
    assert normalize_street("Iyannough Rd.") == "IYANNOUGH RD"


def test_split_address_handles_ranges_and_missing_numbers():
    assert split_address("121-125 Liberty Street") == (121, "LIBERTY ST")
    assert split_address("12A Main St") == (12, "MAIN ST")
    assert split_address("Water Street") == (None, "WATER ST")


def test_match_levels():
    df = pd.DataFrame({
        "address": ["20 Main Street", "24 Main St", "Main Street", "5 North Pleasant St", "Route 6", "1 Nowhere Rd"],
        "town": ["Hudson", "HUDSON", "Hudson", "South Natick", "Truro", "Hudson"],
    })
    out = geocode(df, index=INDEX)
    assert out["geocode_precision"].tolist() == ["address", "street", "street", "address", "street", "unmatched"]
    assert (out.loc[0, "x"], out.loc[0, "y"]) == (2.0, 20.0)
    assert out.loc[1, "x"] == 2.0          # nearest house number on the street (20)
    assert out.loc[2, "x"] == 2.0          # no number: street median
    assert out.loc[3, "x"] == 50.0         # town field held the village name
    assert pd.isna(out.loc[5, "x"])
