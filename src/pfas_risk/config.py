"""Project paths and constants."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "config"
DATA_DIR = Path(os.environ.get("PFAS_RISK_DATA", ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
SEED_DIR = DATA_DIR / "seed"
INTERIM_DIR = DATA_DIR / "interim"
OUTPUT_DIR = Path(os.environ.get("PFAS_RISK_OUTPUT", ROOT / "outputs"))

# Massachusetts State Plane (meters). Every MassGIS layer ships in it, and all
# distances and areas are computed in it.
CRS = "EPSG:26986"
WGS84 = "EPSG:4326"

SEED_RELEASES = SEED_DIR / "massdep_pfas_releases_2021-11-07.csv"


@lru_cache
def catalog() -> dict:
    with open(CONFIG_DIR / "sources.yaml") as f:
        return yaml.safe_load(f)


def source(name: str) -> dict:
    try:
        return catalog()["sources"][name]
    except KeyError as e:
        raise KeyError(f"Unknown source '{name}' (see config/sources.yaml)") from e
