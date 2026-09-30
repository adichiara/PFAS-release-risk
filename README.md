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

In short, with 194 located releases in 140 block groups the model now clearly adds
information beyond size. Gradient boosting finds about 32% of releases in the highest-risk
10% of land, against 13% for land area alone and 16% for area plus density (ROC AUC 0.84
vs 0.83). It beat all 20 permutations of a null that keeps size and density effects, on
every metric (p < 0.05, the smallest p 20 permutations can show). Scores are still relative
risk of a *reported* release, which also reflects where investigations happen.

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

**Unit.** 2020 census block groups (MassGIS), in Massachusetts State Plane meters.

**Response.** Number of PFAS release sites (RTNs) located in each block group. Sites are
placed with MassDEP's published coordinates (`massdep`) unless those fall more than 1 km
outside the site's stated town. Otherwise the address is matched to MassGIS address points:
`address` (exact), `street` (nearest house number on the street, after dropping qualifiers
like "Near" or "Off"), `override` (placed by hand) or `unmatched` (excluded; mostly RTNs
listed as "MULTIPLE LOCATIONS").

**Features.** For each point source, the count inside the block group, the count within 2 km,
and the distance to the nearest one. Point sources are fire stations, MassDEP major
facilities, hazardous-waste large-quantity generators, air-permitted facilities, underground
storage tanks, and facilities in eight PFAS-related industry sectors from EPA's Facility
Registry Service: textiles/leather, paper/printing, chemicals/plastics, metal finishing,
electronics, petroleum, aviation/military and waste/wastewater. Sectors are defined by NAICS
prefix in [`config/pfas_sectors.yaml`](config/pfas_sectors.yaml), with the reason for each. Also:
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

Planned additions (public, identified, not yet wired in): 2016 land cover and EPA UCMR 5
results for context. Airports and military sites are covered through their NAICS codes in the
EPA facility data; FAA airport and DoD MIRTA layers could sharpen those.

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
source: GitHub Actions); it can also be run by hand from the Actions tab. The weekly refresh
workflow commits `docs/` and starts this deploy itself. The **CI** workflow runs lint and
tests on every pull request.
