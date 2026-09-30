# PFAS release risk (Massachusetts)

Ranks Massachusetts by the relative risk of a reported PFAS release, on a grid of equal-area
1 km² hexagons (with 2 km² and 4 km² hexagons and 2020 census block groups as comparisons),
using only publicly available data. It is a rebuild of the release risk model from
the 2021 WPI Data Science / MassDEP graduate capstone
([GQP-TeamMassDEP/Mass_PFAS-Analysis](https://github.com/GQP-TeamMassDEP/Mass_PFAS-Analysis)),
fixing what limited that version:

| 2021 model | This rebuild |
|---|---|
| Aggregated to 64 groups built from tract-number prefixes (`GEOID[:7]`), which are not towns or Census places; group size alone predicted the label (AUC 0.76) | Predicts on 21,774 equal-area 1 km² hexagons, where size no longer predicts the label (AUC 0.50), and on larger cells and 5,109 block groups for comparison |
| Random train/test split | Whole towns held out in cross-validation |
| Compared against chance | Compared against size and density baselines, a size- and density-matched permutation null, and a forward-in-time test on releases reported later |
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
with links to the interactive maps). Latest evaluations:
[`outputs/hex1/evaluation.md`](outputs/hex1/evaluation.md) (primary),
[`outputs/hex2/`](outputs/hex2/evaluation.md), [`outputs/hex4/`](outputs/hex4/evaluation.md)
and [`outputs/bg/`](outputs/bg/evaluation.md).

On **1 km² hexagons** land area alone predicts nothing (ROC AUC 0.50), so the features have to
carry the signal. The selected model (logistic regression) finds about 53% of releases in the
top-risk 10% of land, against 12% for population and housing density (ROC AUC 0.84 vs 0.66).
It beat all 20 permutations of a null that keeps size and density effects, on every metric
(p < 0.05, the smallest p 20 permutations can show).

**Forward in time:** trained only on the 104 releases reported before 2023, it put 44% of the
77 cells with a first release reported since then in its top-risk 10% of land, against 10% for
density (ROC AUC 0.79 vs 0.65).

**Cell size.** Smaller cells rank better with the same features:

| Unit | Top-risk 10% of land: CV (baseline) | Forward test (baseline) | ROC AUC, CV |
|---|---|---|---|
| 1 km² hexagons | 53% (12%) | 44% (10%) | 0.84 |
| 2 km² hexagons | 48% (11%) | 34% (9%) | 0.82 |
| 4 km² hexagons | 43% (10%) | 37% (10%) | 0.78 |
| Block groups | 36% (13%, land area) | 34% (15%) | 0.86 (land area alone 0.83) |

The forward test scores only 61–77 cells, so its differences between sizes are noisy.

**What drives it.** Drinking-water PFAS results and MassDEP major facilities rank first, then
electronics, chemical/plastics and petroleum facilities, hazardous-waste generators and
landfills. The drinking-water features need care: 34 of the 194 releases (18%) sit within
250 m of a supply well that had PFAS6 ≥ 20 ng/L before 2023, mostly releases found through
drinking-water testing and filed at the well, so part of their cross-validation gain is
recognizing those. For releases reported since 2023 the overlap is smaller (8 of 96). Without
the drinking-water features, 1 km² cells still reach 41% in cross-validation and 43% in the
forward test, so the finer grid's gain doesn't depend on them.

Scores are relative risk of a *reported* release, which also reflects where investigations
happen.

## Run it

```bash
pip install -e ".[dev]"
pfas-risk download        # public MassGIS layers (~1 GB, mostly address points and roads)
pfas-risk run             # geocode releases, build features, cross-validate, write outputs/ and docs/
pytest
```

Steps can also be run one at a time: `pfas-risk features`, `pfas-risk evaluate`, `pfas-risk map`,
`pfas-risk site`. The unit is 1 km² hexagons by default; put `--unit bg` before the command for
block groups, or `--cell-km2 4` for another cell size. Results go to `outputs/<unit>/` (for
example `outputs/hex1/`, `outputs/bg/`); `pfas-risk site` leads with `hex1` and compares every
other evaluated unit below it.
Add `-v` for progress logging. A full run of one unit takes 10–45 minutes (the permutation
null is most of it).

### Updating the release list

This is automated: the **Refresh data and model** workflow runs every Monday (or on demand
from the Actions tab, with an option to force a rebuild). It downloads MassDEP's data and
stops if the file is unchanged. Otherwise it rebuilds the release list, reruns the model,
commits `data/releases/`, `outputs/` and `docs/` to `main`, and redeploys the site.

To do the same by hand:

1. `pfas-risk fetch-releases` downloads MassDEP's **Waste Site Cleanup Notifications &
   Status** zip (from [Downloadable Contaminated Site Lists](https://www.mass.gov/info-details/downloadable-contaminated-site-lists))
   to `data/raw/massdep_release_data.zip` and reports whether it changed. mass.gov refuses
   plain scripted requests, so this falls back to a headless browser: install it with
   `pip install -e ".[refresh]" && python -m playwright install chromium` (or point
   `PFAS_RISK_CHROMIUM` at an existing Chromium). Saving the file from a browser also works.
2. `pfas-risk releases` keeps RTNs whose chemical list matches PFAS names (any spelling),
   adds 2021-list sites no longer tagged, drops RTNs MassDEP closed into another RTN on the
   list, and looks up MassDEP's published coordinates for RTNs new to the list. It writes
   `data/releases/massdep_pfas_releases.csv` and `massdep_source.json` (the download's
   sha256 and date); commit both.
3. `pfas-risk run`, then check the `geocode_precision` log line. Unplaced sites can be
   placed by hand in `data/seed/geocode_overrides.csv` (x/y in EPSG:26986, with a note).

## Method

**Units.** Evaluated separately:

- *Equal-area hexagons* (1 km² is the default and the main map; 2 km² and 4 km² are compared;
  `--cell-km2` to change), clipped
  to the state. Every full cell has the same exposure, so the size effect drops out.
  Population, housing, land and water area come from 2020 census blocks weighted by the
  share of each block inside a cell, and each cell is assigned the town holding most of its
  land for town-grouped validation.
- *2020 census block groups* (MassGIS, `--unit bg`). They range from under 0.1 km² to over
  200 km², and land area alone explains much of which ones contain a reported release.

All work is in Massachusetts State Plane meters.

**Response.** Number of PFAS release sites (RTNs) located in each cell. Sites are
placed with MassDEP's published coordinates (`massdep`) unless those fall more than 1 km
outside the site's stated town. Otherwise the address is matched to MassGIS address points:
`address` (exact), `street` (nearest house number on the street, after dropping qualifiers
like "Near" or "Off"), `override` (placed by hand) or `unmatched` (excluded; mostly RTNs
listed as "MULTIPLE LOCATIONS").

**Features.** For each point source, the count inside the cell, the count within 2 km,
and the distance to the nearest one. Point sources are fire stations, MassDEP major
facilities, hazardous-waste large-quantity generators, air-permitted facilities, underground
storage tanks, and facilities in eight PFAS-related industry sectors from EPA's Facility
Registry Service: textiles/leather, paper/printing, chemicals/plastics, metal finishing,
electronics, petroleum, aviation/military and waste/wastewater. Sectors are defined by NAICS
prefix in [`config/pfas_sectors.yaml`](config/pfas_sectors.yaml), with the reason for each. And
public water supply sources (wells and intakes) with PFAS6 results: all tested sources,
sources with PFAS6 detected (≥ 2 ng/L) and sources at or above the 20 ng/L state standard,
using only samples collected before the forward-test cutoff. Also:
distance to the nearest landfill and the landfill share of area, the share of area over high-
or medium-yield aquifers, major-road density, population and housing density, land area and
water share.

**Models.** Two baselines (land area; area + population and housing density), L2 logistic
regression, a Poisson rate model with land area as exposure, and gradient boosting. Each model
also has a smoothed variant that averages a cell's score with the mean score of cells within
3 km (half and half). Smoothing happens inside each fold: the fold's model scores every cell,
then the held-out cells are smoothed, so no score depends on a model that saw that cell.

**Evaluation.** Five-fold cross-validation that holds out whole towns, repeated 10 times with
different town-to-fold assignments. Metrics:

- ROC AUC and average precision
- share of releases in the top 10% of cells
- share of releases in the highest risk-per-km² cells covering 10% of land

The model is selected on the last metric. Its significance comes from 20 permutations that
shuffle release labels only among cells in the same area x population-density
quintile, so the null keeps the size effects and tests whether anything else adds signal.

The **forward test** trains on releases notified before 2023-01-01 (`--cutoff`) and scores
only cells with no earlier release, asking whether it ranks the cells that get their first
release later ahead of the others.

**Map.** The map shows out-of-fold scores, so each cell is colored by a model trained
without its town and known release sites are not simply echoed back. Colors show the
percentile of risk per km². Hexagons clipped at the coast or state line count as a full cell
there, so dividing by a small land area doesn't push them to the top.

## Data

Every source, its publisher, URL and status is in [`config/sources.yaml`](config/sources.yaml).
Downloads are recorded with a sha256 in `data/raw/manifest.json`.

Drinking-water results come from MassDEP's data on the EEA Data Portal API, which returns at most
100 records per query and ignores paging, so `pfas-risk download` queries each water system
(raw and finished water separately; about 20 minutes). Systems with more results are
represented by the 100 the API returns. Results are tied to a source by sampling-point code,
or to all of the system's sources when sampled at a plant or entry point.

**Tested and left out** (`status: evaluated` in the catalog, with the measured change): FAA
public-use airports and military airfields, DoD military installations (MIRTA), EPA Toxics
Release Inventory reporters (no Massachusetts facility has reported a PFAS chemical) and
municipal sewer service areas. None improved the hexagon model; airports and military sites
are already represented by the aviation/military NAICS sector. No public layer exists for fire
training academies or biosolids land application.

Planned additions (public, identified, not yet wired in): 2016 land cover and EPA UCMR 5
results.

## Layout

```
config/sources.yaml       data catalog
data/releases/            PFAS release list built from the MassDEP download
data/seed/                2021 release list, hand-placed geocodes
src/pfas_risk/
  sources.py              download + read public layers
  geocode.py              offline address matching
  releases.py             load and filter PFAS releases
  units.py                block groups and the equal-area hexagon grid
  features.py             feature table for a unit
  pfas_sources.py         drinking-water, airport, military, TRI and sewer layers
  api_sources.py          fetchers for sources served by web APIs
  model.py                models, town-grouped CV, permutation null
  mapping.py              interactive Leaflet map
  site.py                 GitHub Pages landing page
  cli.py                  `pfas-risk` command
tests/
outputs/<unit>/           evaluation.md / .json per unit (other outputs are regenerated, not committed)
docs/                     published site: index.html, map.html (hexagons), map_bg.html, data/
```

## Publishing

`pfas-risk site` (also run by `pfas-risk run`) writes the site to `docs/`. The
`Deploy site to GitHub Pages` workflow publishes `docs/` whenever it changes on `main` (Pages
source: GitHub Actions); it can also be run by hand from the Actions tab. The weekly refresh
workflow commits `docs/` and starts this deploy itself. The **CI** workflow runs lint and
tests on every pull request.
