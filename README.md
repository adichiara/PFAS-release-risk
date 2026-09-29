# PFAS release risk (Massachusetts)

Ranks Massachusetts 2020 census block groups by their relative risk of a reported PFAS
release, using only publicly available data. It is a rebuild of the release risk model from
the 2021 WPI Data Science / MassDEP graduate capstone
([GQP-TeamMassDEP/Mass_PFAS-Analysis](https://github.com/GQP-TeamMassDEP/Mass_PFAS-Analysis)),
fixing what limited that version:

| 2021 model | This rebuild |
|---|---|
| Aggregated to 64 groups built from tract-number prefixes (`GEOID[:7]`), which are not towns or Census places; group size alone predicted the label (AUC 0.76) | Predicts on 5,109 block groups; land area is modeled as exposure or compared against explicitly |
| Random train/test split | Whole towns held out in cross-validation |
| Compared against chance | Compared against size-only baselines and a size- and density-matched permutation null |
| Industry features from a non-public business list | Public sources only (see `config/sources.yaml`) |
| Addresses geocoded with a web service (one landed ~1,900 km away) | Offline matching against MassGIS address points, with a precision flag per site |

## Current status

**Preliminary.** The release list (the thing we predict) still comes from the 52 PFAS
releases (RTNs) compiled in November 2021 (`data/seed/`). MassDEP's release
database is public but has to be exported by hand (see below). Many more PFAS releases
have been reported since 2021, and more releases matter more than any modeling change.

Latest evaluation: [`outputs/evaluation.md`](outputs/evaluation.md). Interactive map:
`outputs/risk_map.html`.

In short, land area alone explains most of what the data shows, because larger block groups
contain more reported releases. The best model (gradient boosting) finds about 26% of
releases in the highest-risk 10% of land, against 18% for area alone. Against a permutation
null that keeps size and density effects the result is borderline (empirical p ≈ 0.05–0.10
with 20 permutations). Treat the map as a starting point, not a finding.

## Run it

```bash
pip install -e ".[dev]"
pfas-risk download        # public MassGIS layers (~1 GB, mostly address points and roads)
pfas-risk run             # geocode releases, build features, cross-validate, write outputs/
pytest
```

Steps can also be run one at a time: `pfas-risk features`, `pfas-risk evaluate`, `pfas-risk map`.
Add `-v` for progress logging. A full run takes about 15 minutes.

### Updating the release list

1. Search MassDEP's [Waste Site & Reportable Release database](https://eeaonline.eea.state.ma.us/portal#!/search/wastesite)
   and export the results, including address, town and chemical fields.
2. Save the export as `data/raw/massdep_releases.csv`. Column names are matched flexibly
   (`src/pfas_risk/releases.py`), and non-PFAS chemicals are filtered out by pattern.
3. Rerun `pfas-risk run`, then check `geocode_precision` for unmatched addresses. Place
   them by hand in `data/seed/geocode_overrides.csv` (x/y in EPSG:26986, with a note).

## Method

**Unit.** 2020 census block groups (MassGIS), in Massachusetts State Plane meters.

**Response.** Number of PFAS release sites (RTNs) located in each block group. Addresses are
matched to MassGIS address points: `address` (exact), `street` (nearest house number on the
street), `override` (placed by hand) or `unmatched` (excluded).

**Features.** For each point source (fire stations, MassDEP major facilities, hazardous-waste
large-quantity generators, air-permitted facilities, underground storage tanks): the count
inside the block group, the count within 2 km, and the distance to the nearest one. Also:
distance to the nearest landfill and the landfill share of area, the share of area over high-
or medium-yield aquifers, major-road density, population and housing density, land area and
water share.

**Models.** Two baselines (land area; area + population and housing density), L2 logistic
regression, a Poisson rate model with land area as exposure, and gradient boosting.

**Evaluation.** Five-fold cross-validation that holds out whole towns, repeated 10 times with
different town-to-fold assignments. Metrics:

- ROC AUC and average precision
- share of releases in the top 10% of block groups
- share of releases in the highest risk-per-km² block groups covering 10% of land

The model is selected on the last metric. Its significance comes from 20 permutations that
shuffle release labels only among block groups in the same area x population-density
quintile, so the null keeps the size effects and tests whether anything else adds signal.

**Map.** The map shows out-of-fold scores, so each block group is colored by a model trained
without its town and known release sites are not simply echoed back. Colors show the
percentile of risk per km².

## Data

Every source, its publisher, URL and status is in [`config/sources.yaml`](config/sources.yaml).
Downloads are recorded with a sha256 in `data/raw/manifest.json`.

Planned additions (public, identified, not yet wired in): EPA ECHO/FRS facilities by
PFAS-related NAICS codes (the public replacement for the 2021 industry list), airports,
military installations (DoD MIRTA), 2016 land cover, and EPA UCMR 5 results for context.

## Layout

```
config/sources.yaml       data catalog
data/seed/                2021 release list, hand-placed geocodes
src/pfas_risk/
  sources.py              download + read public layers
  geocode.py              offline address matching
  releases.py             load and filter PFAS releases
  features.py             block-group feature table
  model.py                models, town-grouped CV, permutation null
  mapping.py              interactive Leaflet map
  cli.py                  `pfas-risk` command
tests/
outputs/                  evaluation.md / .json, risk_map.html
```
