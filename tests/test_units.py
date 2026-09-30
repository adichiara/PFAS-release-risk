import numpy as np
import pytest
import shapely

from pfas_risk.features import unit_key
from pfas_risk.units import hexagons


def test_hexagons_have_the_requested_area_and_tile_without_gaps():
    cells = hexagons((0, 0, 20_000, 20_000), cell_km2=4.0)
    assert np.allclose(cells.area, 4e6, rtol=1e-9)
    inner = shapely.box(2_000, 2_000, 18_000, 18_000)
    covered = shapely.union_all(cells.geometry.values)
    assert inner.difference(covered).area < 1.0                      # no gaps (m², float slivers only)
    overlap = cells.area.sum() - covered.area
    assert overlap == pytest.approx(0, abs=1.0)                      # no overlaps (m²)


def test_unit_keys():
    assert unit_key("bg") == "bg"
    assert unit_key("hex", 4.0) == "hex4"
    assert unit_key("hex", 0.5) == "hex0.5"
    with pytest.raises(ValueError):
        unit_key("tract")
