"""Front page of the site: PFAS in drinking water (docs/index.html).

Reads the outputs of `pfas-risk exposure` (outputs/water/) and copies the drinking-water map and
downloads next to the page. The reported-release analysis moves to docs/releases.html.
"""

from __future__ import annotations

import html
import json
import re
import shutil
from pathlib import Path

import pandas as pd

from .config import OUTPUT_DIR
from .drinking_water import BANDS

TEMPLATE = Path(__file__).with_name("water_site_template.html")
STYLE_SOURCE = Path(__file__).with_name("site_template.html")
WATER_DIR = OUTPUT_DIR / "water"
BAND_LABELS = [label for _, _, label in BANDS]


def _style() -> str:
    """The release page's stylesheet, so both pages share one design."""
    return re.search(r"<style>.*?</style>", STYLE_SOURCE.read_text(), re.S).group(0)


def _people(n: float) -> str:
    if n >= 1e6:
        return f"{n / 1e6:.1f} million"
    return f"{round(n, -3):,.0f}" if n >= 10_000 else f"{n:,.0f}"


def _kpis(h: dict) -> str:
    now = h["public_by_current_band"].get(BAND_LABELS[3], 0)
    peak = h["public_by_peak_band"].get(BAND_LABELS[3], 0)
    detected = h["public_water"] - h["public_by_current_band"].get(BAND_LABELS[0], 0)
    tiles = [
        (_people(now), "on public water now at or above the 20 ng/L state standard"),
        (_people(peak), "on systems that reached it in their worst year"),
        (_people(detected), "with PFAS6 detected in their tap water now"),
        (_people(h["private_expected_over20"]),
         f"estimated among {_people(h['private_wells'])} private-well residents (rough)"),
    ]
    return "".join(f'<div class="kpi"><div class="v">{v}</div><div class="l">{html.escape(t)}</div></div>'
                   for v, t in tiles)


def stack_chart(h: dict) -> tuple[str, str]:
    """100% horizontal bars of public-water residents by PFAS6 band: now and highest year."""
    W, L, R, bar, gap, top = 720, 110, 16, 34, 18, 10
    rows = [("Now", h["public_by_current_band"]), ("Highest year", h["public_by_peak_band"])]
    parts = []
    for i, (label, counts) in enumerate(rows):
        y = top + i * (bar + gap)
        total = sum(counts.get(b, 0) for b in BAND_LABELS)
        x = L
        parts.append(f'<text class="tick" x="{L - 10}" y="{y + bar / 2 + 4}" text-anchor="end">{label}</text>')
        for k, b in enumerate(BAND_LABELS):
            n = counts.get(b, 0)
            w = (W - L - R) * n / total
            if w <= 0:
                continue
            tip = f"{b}: {n:,.0f} people ({n / total:.0%})"
            parts.append(f'<rect class="seg-{k}" x="{x:.1f}" y="{y}" width="{w:.1f}" height="{bar}">'
                         f'<title>{html.escape(tip)}</title></rect>')
            if w > 44:
                parts.append(f'<text class="value" x="{x + w / 2:.1f}" y="{y + bar / 2 + 4}" '
                             f'text-anchor="middle" style="fill:{"#0b0b0b" if k < 2 else "#ffffff"}">{n / total:.0%}</text>')
            x += w
    H = top + len(rows) * (bar + gap)
    svg = (f'<svg class="chart stack" viewBox="0 0 {W} {H}" role="img" aria-label="Public-water residents '
           f'by PFAS6 level in their tap water, now and in each system\'s highest year">{"".join(parts)}</svg>')
    names = ["Not detected", "2 to <10 ng/L", "10 to <20 ng/L", "20+ ng/L (state standard)"]
    legend = "".join(f'<span><i class="sw seg-{k}"></i>{html.escape(b)}</span>' for k, b in enumerate(names))
    return svg, legend


def _num(v) -> str:
    return "" if pd.isna(v) else ("ND" if v < 2 else f"{v:.1f}")


def _system_rows(systems: pd.DataFrame, n: int = 25) -> str:
    top = systems[systems["pfas6_current_ng_l"] >= 10].sort_values("population_served", ascending=False).head(n)
    return "\n".join(
        f'<tr><td class="wrap">{html.escape(r.system)}'
        f'{" <span class=note>(buys its water)</span>" if r.results_from == "purchased" else ""}</td>'
        f"<td>{r.population_served:,.0f}</td><td>{_num(r.pfas6_current_ng_l)}</td>"
        f"<td>{_num(r.pfas6_peak_ng_l)}</td><td>{_num(r.ucmr5_pfas6_ng_l)}</td></tr>"
        for r in top.itertuples())


def _gw_rows(v: dict) -> str:
    labels = {"logistic": "Model", "baseline_population": "Development alone (population within 1 km)"}
    rows = [f'<tr><td class="wrap">{labels[m]}</td><td>{v["over20"][m]["roc_auc"]:.2f}</td>'
            f'<td>{v["detect"][m]["roc_auc"]:.2f}</td></tr>' for m in labels]
    rows.append(f'<tr><td class="wrap">Share of wells in the category</td><td>{v["over20"]["prevalence"]:.0%}</td>'
                f'<td>{v["detect"]["prevalence"]:.0%}</td></tr>')
    return "\n".join(rows)


def _pct(v: float) -> str:
    return "" if pd.isna(v) else (f"{v:.1%}" if v < 0.1 else f"{v:.0%}")


def _group_rows(groups: list[dict]) -> str:
    rows = []
    for g in groups:
        cls = ' class="selected"' if g["group"] == "All residents" else ""
        rows.append(
            f'<tr{cls}><td class="wrap">{html.escape(g["group"])}</td><td>{g["residents"]:,}</td>'
            f'<td>{_pct(g["current_over20"])}</td><td>{_pct(g["current_10_20"])}</td>'
            f'<td>{_pct(g["current_detected"])}</td><td>{_pct(g["peak_over20"])}</td>'
            f'<td>{g["private_wells"]:,}</td><td>{_pct(g["private_expected_over20"])}</td></tr>')
    return "\n".join(rows)


def build_water_page(site_dir: Path, repo_url: str, source_rows: str) -> Path | None:
    report_path = WATER_DIR / "exposure.json"
    if not report_path.exists():
        return None
    r = json.loads(report_path.read_text())
    h = r["headline"]
    systems = pd.read_csv(WATER_DIR / "water_systems_pfas6.csv", dtype={"pws_id": str})
    data = site_dir / "data"
    data.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(WATER_DIR / "water_map.html", site_dir / "water_map.html")
    downloads = [("block_group_drinking_water.csv",
                  "Drinking water by 2020 block group: public-water PFAS6 now and at peak, private-well share and estimate, EJ fields"),
                 ("water_systems_pfas6.csv", "PFAS6 levels for every community water system, with the UCMR 5 cross-check")]
    for name, _ in downloads:
        shutil.copyfile(WATER_DIR / name, data / name)
    chart, legend = stack_chart(h)
    u = r["ucmr5_check"]
    bands = r["private_bands"]
    page = (TEMPLATE.read_text()
            .replace("__STYLE__", _style())
            .replace("__REPO__", repo_url)
            .replace("__KPIS__", _kpis(h))
            .replace("__STACK_CHART__", chart)
            .replace("__STACK_LEGEND__", legend)
            .replace("__UCMR5_NOTE__", f"Cross-check: for the {u['systems']} systems also tested under EPA's UCMR 5 "
                     f"(2023 to 2025), the state and federal PFAS6 levels correlate at {u['correlation']:.2f}. UCMR 5 "
                     "reports each compound only above 3 to 4 ng/L, so its sums run lower.")
            .replace("__SYSTEM_ROWS__", _system_rows(systems))
            .replace("__PRIVATE_POP__", _people(h["private_wells"]))
            .replace("__N_WELLS__", f"{r['groundwater_validation']['wells']:,}")
            .replace("__GW_ROWS__", _gw_rows(r["groundwater_validation"]))
            .replace("__PRIVATE_ROWS__", "\n".join(f"<tr><td>{html.escape(k)}</td><td>{v:,}</td></tr>"
                                                   for k, v in reversed(list(bands.items()))))
            .replace("__GROUP_ROWS__", _group_rows(r["groups"]))
            .replace("__SOURCE_ROWS__", source_rows)
            .replace("__DOWNLOADS__", "".join(f'<li><a href="data/{f}">{html.escape(t)}</a></li>' for f, t in downloads))
            .replace("__RUN_DATE__", r["run_date"]))
    (site_dir / "index.html").write_text(page)
    return site_dir / "index.html"
