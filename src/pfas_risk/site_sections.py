"""Charts, tables and downloads for the landing page.

Charts are inline SVG built here, styled by the page's CSS tokens (so they follow light and
dark mode) and carry native hover titles; the site needs no JavaScript libraries.
"""

from __future__ import annotations

import html
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from .config import CRS
from .industry import sector_definitions
from .model import risk_density

GROUP_LABELS = {
    "land_area": "Land area",
    "population": "Population and housing density",
    "water_frac": "Water share of area",
    "fire_station": "Fire stations",
    "bwp_major": "MassDEP major facilities",
    "haz_waste_lqg": "Hazardous-waste generators",
    "air_permit": "Air-permitted facilities",
    "ust": "Underground storage tanks",
    "landfill": "Landfills",
    "aquifer_frac": "High/medium-yield aquifers",
    "major_road_km_per_km2": "Major roads",
    "drinking_water": "Drinking-water PFAS (pre-2023)",
}
SECTOR_LABELS = {
    "textiles_leather": "Textiles and leather",
    "paper_printing": "Paper and printing",
    "chemicals_plastics": "Chemicals and plastics",
    "metal_finishing": "Metal finishing",
    "electronics": "Electronics",
    "petroleum": "Petroleum",
    "aviation_military": "Aviation and military",
    "waste_wastewater": "Waste and wastewater",
}


def group_label(group: str) -> str:
    if group.startswith("ind_"):
        return "Industry: " + SECTOR_LABELS.get(group[4:], group[4:]).lower()
    return GROUP_LABELS.get(group, group.replace("_", " "))


def _pct(v: float) -> str:
    return f"{v * 100:.0f}%"


# ---- Capture curve -----------------------------------------------------------

def capture_curve(scores: np.ndarray, releases: np.ndarray, area: np.ndarray,
                  grid: np.ndarray) -> np.ndarray:
    """Share of releases found when targeting the highest risk-per-km² land first."""
    order = np.argsort(-(scores / area))
    land = np.cumsum(area[order]) / area.sum()
    found = np.cumsum(releases[order]) / releases.sum()
    return np.interp(grid, np.concatenate([[0], land]), np.concatenate([[0], found]))


def capture_chart(df: pd.DataFrame, oof: pd.DataFrame, model: str, base_col: str = "baseline_area",
                  base_label: str = "Land area alone") -> str:
    grid = np.linspace(0, 1, 101)
    area, y = df["land_km2"].to_numpy(), df["releases"].to_numpy()
    curves = {
        "model": capture_curve(oof[model].to_numpy(), y, area, grid),
        "base": capture_curve(oof[base_col].to_numpy(), y, area, grid),
    }
    W, H, L, R, T, B = 640, 360, 52, 120, 16, 44
    sx = lambda v: L + v * (W - L - R)  # noqa: E731
    sy = lambda v: T + (1 - v) * (H - T - B)  # noqa: E731

    def path(vals):
        return "M" + " L".join(f"{sx(x):.1f},{sy(v):.1f}" for x, v in zip(grid, vals, strict=True))

    ticks = [0, 0.25, 0.5, 0.75, 1]
    grid_lines = "".join(
        f'<line class="grid" x1="{sx(0)}" x2="{sx(1)}" y1="{sy(t):.1f}" y2="{sy(t):.1f}"/>'
        f'<text class="tick" x="{L - 8}" y="{sy(t) + 4:.1f}" text-anchor="end">{_pct(t)}</text>'
        f'<text class="tick" x="{sx(t):.1f}" y="{H - B + 18}" text-anchor="middle">{_pct(t)}</text>'
        for t in ticks)
    guide = (f'<line class="guide" x1="{sx(0.1):.1f}" x2="{sx(0.1):.1f}" y1="{sy(0)}" y2="{sy(1)}"/>'
             f'<text class="tick" x="{sx(0.1) + 4:.1f}" y="{sy(1) + 12}">10% of land</text>')
    hover = []
    for x in [0.05, 0.1, 0.2, 0.3, 0.5, 0.75]:
        i = round(x * 100)
        m, b = curves["model"][i], curves["base"][i]
        tip = f"Top-risk {_pct(x)} of land: model finds {_pct(m)} of releases, {base_label.lower()} {_pct(b)}"
        for v, cls in [(m, "dot model"), (b, "dot base")]:
            hover.append(f'<circle class="{cls}" cx="{sx(x):.1f}" cy="{sy(v):.1f}" r="4">'
                         f'<title>{tip}</title></circle>')
    # Direct labels in open space: above-left of the model curve, below-right of the baseline.
    m20, b35 = curves["model"][20], curves["base"][35]
    labels = (f'<text class="label" x="{sx(0.2) - 8:.1f}" y="{sy(m20) - 10:.1f}" text-anchor="end">Model</text>'
              f'<text class="label" x="{sx(0.35) + 10:.1f}" y="{sy(b35) + 16:.1f}">{base_label}</text>'
              f'<text class="label muted" x="{sx(0.8):.1f}" y="{sy(0.8) + 18:.1f}">Random</text>')
    return (f'<svg class="chart" viewBox="0 0 {W} {H}" role="img" aria-label="Capture curve: share of '
            f'releases found versus share of land targeted, model versus {base_label.lower()}">'
            f'{grid_lines}{guide}'
            f'<line class="random" x1="{sx(0)}" y1="{sy(0)}" x2="{sx(1)}" y2="{sy(1)}"/>'
            f'<path class="line base" d="{path(curves["base"])}"/>'
            f'<path class="line model" d="{path(curves["model"])}"/>'
            f'{"".join(hover)}{labels}'
            f'<text class="axis" x="{sx(0.5):.1f}" y="{H - 6}" text-anchor="middle">'
            f'Share of land targeted, highest risk per km² first</text>'
            f'<text class="axis" transform="translate(14 {sy(0.5):.1f}) rotate(-90)" text-anchor="middle">'
            f'Share of releases found</text></svg>')


# ---- Importance ---------------------------------------------------------------

def importance_chart(importance: list[dict], top: int = 12) -> str:
    rows = sorted(importance, key=lambda r: -r["mean"])[:top]
    W, row_h, L, R, T = 640, 26, 250, 70, 8
    H = T + row_h * len(rows) + 30
    vmax = max(max(r["mean"] + (r["std"] or 0) for r in rows), 1e-3)
    sx = lambda v: L + max(v, 0) / vmax * (W - L - R)  # noqa: E731
    parts = []
    for i, r in enumerate(rows):
        y = T + i * row_h
        label = html.escape(group_label(r["group"]))
        tip = f"{label}: AUC drops by {r['mean']:.3f} (sd {r['std']:.3f}) when shuffled"
        parts.append(
            f'<text class="tick" x="{L - 8}" y="{y + 16}" text-anchor="end">{label}</text>'
            f'<rect class="bar" x="{L}" y="{y + 5}" width="{sx(r["mean"]) - L:.1f}" height="14" rx="3">'
            f'<title>{tip}</title></rect>'
            f'<text class="value" x="{sx(r["mean"]) + 6:.1f}" y="{y + 16}">{r["mean"]:.3f}</text>')
    axis_y = T + row_h * len(rows) + 22
    return (f'<svg class="chart" viewBox="0 0 {W} {H}" role="img" aria-label="Drop in ROC AUC when each '
            f'feature group is shuffled">{"".join(parts)}'
            f'<text class="axis" x="{L}" y="{axis_y}">Drop in held-out ROC AUC when the group is shuffled</text>'
            f'</svg>')


# ---- Releases by year -----------------------------------------------------------

def releases_by_year_chart(releases: pd.DataFrame) -> str:
    years = pd.to_datetime(releases["notification_date"], errors="coerce").dt.year.dropna().astype(int)
    first = 2016
    counts = years.clip(lower=first - 1).value_counts().sort_index()
    counts = counts.reindex(range(first - 1, years.max() + 1), fill_value=0)
    W, H, L, R, T, B = 640, 240, 36, 12, 18, 40
    n = len(counts)
    step = (W - L - R) / n
    vmax = max(counts.max(), 1)
    parts = []
    for i, (yr, c) in enumerate(counts.items()):
        x = L + i * step + step * 0.18
        w = step * 0.64
        h = c / vmax * (H - T - B)
        label = f"before {first}" if yr == first - 1 else str(yr)
        parts.append(
            f'<rect class="bar" x="{x:.1f}" y="{H - B - h:.1f}" width="{w:.1f}" height="{max(h, 0.5):.1f}" '
            f'rx="3"><title>{label}: {c} PFAS release{"s" if c != 1 else ""} reported</title></rect>'
            f'<text class="value" x="{x + w / 2:.1f}" y="{H - B - h - 5:.1f}" text-anchor="middle">{c}</text>'
            f'<text class="tick" x="{x + w / 2:.1f}" y="{H - B + 16}" text-anchor="middle">'
            f'{"<" + str(first) if yr == first - 1 else yr}</text>')
    note = (f'<text class="axis" x="{W - R}" y="{H - 6}" text-anchor="end">'
            f'{years.max()} covers part of the year</text>')
    return (f'<svg class="chart" viewBox="0 0 {W} {H}" role="img" aria-label="PFAS releases reported to '
            f'MassDEP by year of notification">{"".join(parts)}{note}</svg>')


# ---- Watch list -------------------------------------------------------------------

def watchlist(df: gpd.GeoDataFrame, oof: pd.DataFrame, model: str, releases: gpd.GeoDataFrame,
              n: int = 20) -> pd.DataFrame:
    """Highest risk-per-km² units with no reported release, with nearby context."""
    d = df.copy()
    d["density"] = risk_density(oof[model], d)
    d["pct"] = d["density"].rank(pct=True) * 100
    top = d[d["releases"] == 0].sort_values("density", ascending=False).head(n)
    rep = gpd.GeoDataFrame(geometry=top.representative_point(), crs=CRS)
    located = releases[releases["x"].notna()][["geometry"]]
    near = gpd.sjoin_nearest(rep, located, distance_col="dist")["dist"]
    near = near[~near.index.duplicated()]
    sectors = []
    for _, row in top.iterrows():
        counts = {s: int(row[f"n2k_ind_{s}"]) for s in sector_definitions() if f"n2k_ind_{s}" in row}
        total = sum(counts.values())
        lead = sorted(((c, s) for s, c in counts.items() if c > 0), reverse=True)[:3]
        detail = ", ".join(f"{SECTOR_LABELS[s].lower()} {c}" for c, s in lead)
        sectors.append(f"{total} ({detail})" if total else "0")
    return pd.DataFrame({
        "geoid": top.index, "town": top["town"].values, "risk_pct": top["pct"].round(1).values,
        "land_km2": top["land_km2"].round(2).values, "population": top["POP20"].astype(int).values,
        "nearest_release_km": (near.reindex(top.index) / 1000).round(1).values,
        "industries_within_2km": sectors,
    })


def watchlist_rows(w: pd.DataFrame) -> str:
    return "\n".join(
        f'<tr><td>{html.escape(str(r.town))}</td><td>{r.geoid}</td><td>{r.risk_pct:.1f}</td>'
        f'<td>{r.land_km2:.2f}</td><td>{r.population:,}</td><td>{r.nearest_release_km:.1f}</td>'
        f'<td class="wrap">{html.escape(r.industries_within_2km)}</td></tr>'
        for r in w.itertuples())


# ---- Downloads ------------------------------------------------------------------------

def write_downloads(site_dir: Path, key: str, plural: str, df: pd.DataFrame, oof: pd.DataFrame, model: str,
                    releases: pd.DataFrame) -> list[tuple[str, str]]:
    out = site_dir / "data"
    out.mkdir(parents=True, exist_ok=True)
    scores = pd.DataFrame({
        "geoid": df.index, "town": df["town"].values, "land_km2": df["land_km2"].round(4).values,
        "population": df["POP20"].values, "reported_pfas_releases": df["releases"].values,
        "risk_score": oof[model].round(5).values,
        "risk_per_km2_percentile": risk_density(oof[model], df).rank(pct=True).mul(100).round(2).values,
    })
    scores.rename(columns={"geoid": "unit_id"}).to_csv(out / f"{key}_risk.csv", index=False)
    cols = [c for c in ["rtn", "town", "address", "site_name", "notification_date", "chemical", "status",
                        "source", "geocode_precision", "x", "y"] if c in releases]
    releases[cols].to_csv(out / "pfas_releases.csv", index=False)
    return [(f"data/{key}_risk.csv", f"Risk scores for {len(scores):,} {plural}"),
            ("data/pfas_releases.csv", f"PFAS release list with locations ({len(releases)} rows)")]


# ---- Other spatial units ----------------------------------------------------------------

def unit_comparison_rows(units: list[tuple[str, dict]]) -> str:
    """One row per unit: its best model against its two size baselines."""
    rows = []
    for label, r in units:
        best = r["mean"][r["best_model"]]
        area, dens = r["mean"]["baseline_area"], r["mean"]["baseline_area_population"]
        p = r["null"]["capture_top10pct_area"]["p_value"]
        rows.append(
            f"<tr><td>{html.escape(label)}</td><td>{r['units']:,}</td><td>{r['release_units']}</td>"
            f"<td>{best['roc_auc']:.3f} / {dens['roc_auc']:.3f}</td>"
            f"<td>{best['avg_precision']:.3f} / {dens['avg_precision']:.3f}</td>"
            f"<td>{_pct(best['capture_top10pct_area'])} / {_pct(area['capture_top10pct_area'])} / "
            f"{_pct(dens['capture_top10pct_area'])}</td><td>{p:.2f}</td></tr>")
    return "\n".join(rows)
