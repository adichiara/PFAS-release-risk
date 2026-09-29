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

**Preliminary.** The release list (`data/releases/massdep_pfas_releases.csv`) has 214 PFAS
releases (RTNs): 194 from MassDEP's May 2026 bulk download of all waste site notifications
and chemicals, plus 20 sites from the 2021 project list that the current database no
longer tags with a PFAS chemical. 194 of the 214 are located (see Method). About 22% of RTNs
notified since 2019 have no chemical recorded at all, so some PFAS releases are
necessarily missing from the labels.

**Site:** https://adichiara.github.io/PFAS-release-risk/ (results, method and data sources,
with a link to the interactive map). Latest evaluation:
[`outputs/evaluation.md`](outputs/evaluation.md).

In short, land area alone explains most of what the data shows, because larger block groups
contain more reported releases. The best model (gradient boosting) finds about 26% of
releases in the highest-risk 10% of land, against 18% for area alone. Against a permutation
null that keeps size and density effects the result is borderline (empirical p ≈ 0.05–0.10
with 20 permutations). Treat the map as a starting point, not a finding.

## Run it

```bash
pip install -e ".[dev]"
pfas-risk download        # public MassGIS layers (~1 GB, mostly address points and roads)
pfas-risk run             # geocode releases, build features, cross-validate, write outputs/ and docs/
pytest
```

Steps can also be run one at a time: `pfas-risk features`, `pfas-risk evaluate`, `pfas-risk map`,
`pfas-risk site`.
Add `-v` for progress logging. A full run takes about 15 minutes.

### Updating the release list

1. Download **Downloadable Data: Waste Site Cleanup Notifications & Status** from
   [Downloadable Contaminated Site Lists](https://www.mass.gov/info-details/downloadable-contaminated-site-lists)
   in a browser (mass.gov refuses scripted downloads) and save it as
   `data/raw/massdep_release_data.zip`.
2. Run `pfas-risk releases`. It keeps RTNs whose chemical list matches PFAS names (any
   spelling), adds 2021-list sites no longer tagged, drops RTNs MassDEP closed into another
   RTN on the list, and looks up MassDEP's published site coordinates (reusing ones already
   fetched). The result is `data/releases/massdep_pfas_releases.csv`; commit it.
3. Run `pfas-risk run`, then check the `geocode_precision` log line. Unplaced sites can be
   placed by hand in `data/seed/geocode_overrides.csv` (x/y in EPSG:26986, with a note).

## Method

**Unit.** 2020 census block groups (MassGIS), in Massachusetts State Plane meters.

**Response.** Number of PFAS release sites (RTNs) located in each block group. Sites are
placed with MassDEP's published coordinates (`massdep`) unless those fall more than 1 km
outside the site's stated town. Otherwise the address is matched to MassGIS address points:
`address` (exact), `street` (nearest house number on the street, after dropping qualifiers
like "Near" or "Off"), `override` (placed by hand) or `unmatched` (excluded; mostly RTNs
listed as "MULTIPLE LOCATIONS").

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
data/releases/            PFAS release list built from the MassDEP download
data/seed/                2021 release list, hand-placed geocodes
src/pfas_risk/
  sources.py              download + read public layers
  geocode.py              offline address matching
  releases.py             load and filter PFAS releases
  features.py             block-group feature table
  model.py                models, town-grouped CV, permutation null
  mapping.py              interactive Leaflet map
  site.py                 GitHub Pages landing page
  cli.py                  `pfas-risk` command
tests/
outputs/                  evaluation.md / .json (other outputs are regenerated, not committed)
docs/                     published site: index.html, map.html
```

## Publishing

`pfas-risk site` (also run by `pfas-risk run`) writes the site to `docs/`. The
`Deploy site to GitHub Pages` workflow publishes `docs/` whenever it changes on `main` (Pages
source: GitHub Actions); it can also be run by hand from the Actions tab. Commit `docs/` after
each run to update the site.
