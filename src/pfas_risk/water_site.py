"""Drinking-water pages of the site.

- docs/index.html: the private-well study: data, groundwater model, check against MassDEP's
  private-well testing, calibration and results (study_template.html).
- docs/analysis.html: measured PFAS6 in public water (water_site_template.html).

Both read the outputs of `pfas-risk exposure` (outputs/water/); the map and downloads are
copied next to them. The reported-release analysis is docs/releases.html.
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
STUDY_TEMPLATE = Path(__file__).with_name("study_template.html")
NAV_ITEMS = [("index.html", "Private-well study"), ("water_map.html", "Map"), ("analysis.html", "Public water"),
             ("releases.html", "Reported releases")]


def nav(current: str, repo_url: str) -> str:
    """Site navigation shared by every page; ``current`` is this page's file name."""
    links = []
    for href, label in NAV_ITEMS:
        mark = ' aria-current="page"' if href == current else ""
        links.append(f'<a href="{href}"{mark}>{label}</a>')
    return f'<nav class="site" aria-label="Site">{"".join(links)}<a href="{repo_url}">Code and data</a></nav>'


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


def _kpis(h: dict, mwra: dict) -> str:
    now = h["public_by_current_band"].get(BAND_LABELS[3], 0)
    peak = h["public_by_peak_band"].get(BAND_LABELS[3], 0)
    detected = h["public_water"] - h["public_by_current_band"].get(BAND_LABELS[0], 0)
    tiles = [
        (_people(now), "on public water now at or above the 20 ng/L state standard"),
        (_people(peak), "on systems that reached it in their worst year"),
        (_people(detected), f"of {_people(h['public_water'])} with PFAS6 detected in their tap water now"),
        (_people(mwra["population"]), f"in {mwra['systems']} systems on MWRA water, PFAS6 not detected"),
    ]
    return _tiles(tiles)


def _tiles(tiles: list[tuple[str, str]]) -> str:
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


def _pct(v: float) -> str:
    return "" if pd.isna(v) else (f"{v:.1%}" if v < 0.1 else f"{v:.0%}")


def _group_rows(groups: list[dict]) -> str:
    """Public-water columns by EJ group."""
    rows = []
    for g in groups:
        cls = ' class="selected"' if g["group"] == "All residents" else ""
        rows.append(
            f'<tr{cls}><td class="wrap">{html.escape(g["group"])}</td><td>{g["public_water"]:,}</td>'
            f'<td>{_pct(g["current_over20"])}</td><td>{_pct(g["current_10_20"])}</td>'
            f'<td>{_pct(g["current_detected"])}</td><td>{_pct(g["peak_over20"])}</td></tr>')
    return "\n".join(rows)


def _private_group_rows(groups: list[dict]) -> str:
    rows = []
    for g in groups:
        cls = ' class="selected"' if g["group"] == "All residents" else ""
        rows.append(f'<tr{cls}><td class="wrap">{html.escape(g["group"])}</td><td>{g["private_wells"]:,}</td>'
                    f'<td>{_pct(g["private_wells"] / g["residents"])}</td>'
                    f'<td>{_pct(g["private_expected_over20"])}</td></tr>')
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
    page = (TEMPLATE.read_text()
            .replace("__STYLE__", _style())
            .replace("__REPO__", repo_url)
            .replace("__KPIS__", _kpis(h, r["mwra"]))
            .replace("__STACK_CHART__", chart)
            .replace("__STACK_LEGEND__", legend)
            .replace("__UCMR5_NOTE__", f"Cross-check: for the {u['systems']} systems also tested under EPA's UCMR 5 "
                     f"(2023 to 2025), the state and federal PFAS6 levels correlate at {u['correlation']:.2f}. UCMR 5 "
                     "reports each compound only above 3 to 4 ng/L, so its sums run lower.")
            .replace("__SYSTEM_ROWS__", _system_rows(systems))
            .replace("__GROUP_ROWS__", _group_rows(r["groups"]))
            .replace("__SOURCE_ROWS__", source_rows)
            .replace("__MWRA_POP__", _people(r["mwra"]["population"]))
            .replace("__MWRA_SYSTEMS__", str(r["mwra"]["systems"]))
            .replace("__DOWNLOADS__", _downloads(downloads))
            .replace("__RUN_DATE__", r["run_date"])
            .replace("__NAV__", nav("analysis.html", repo_url)))
    (site_dir / "analysis.html").write_text(page)
    return build_study_page(site_dir, repo_url, r, source_rows, downloads[:1])


def _downloads(items: list[tuple[str, str]]) -> str:
    return "".join(f'<li><a href="data/{f}">{html.escape(t)}</a></li>' for f, t in items)


def scatter_chart(towns: list[dict]) -> str:
    """Tested towns: calibrated rate predicted without the town (x) against its measured rate (y)."""
    W, H, L, B, T, R = 720, 400, 52, 44, 12, 16
    top = max(0.3, max(max(t["loto"], t["k"] / t["n"]) for t in towns) * 1.05)
    ticks = [x / 10 for x in range(0, int(top * 10) + 1)]
    sx = lambda v: L + (W - L - R) * v / top  # noqa: E731
    sy = lambda v: H - B - (H - B - T) * v / top  # noqa: E731
    parts = []
    for v in ticks:
        parts.append(f'<line class="grid" x1="{L}" x2="{W - R}" y1="{sy(v):.1f}" y2="{sy(v):.1f}"/>'
                     f'<text class="tick" x="{L - 8}" y="{sy(v) + 4:.1f}" text-anchor="end">{v:.0%}</text>'
                     f'<text class="tick" x="{sx(v):.1f}" y="{H - B + 18}" text-anchor="middle">{v:.0%}</text>')
    parts.append(f'<line class="ref" x1="{sx(0):.1f}" y1="{sy(0):.1f}" x2="{sx(top):.1f}" y2="{sy(top):.1f}"/>'
                 f'<text class="tick" x="{sx(top * 0.9) - 6:.1f}" y="{sy(top * 0.9) - 8:.1f}" text-anchor="end">'
                 "predicted = measured</text>")
    for t in sorted(towns, key=lambda t: -t["n"]):
        r = 3 + 1.6 * t["n"] ** 0.5
        tip = (f'{t["town"]}: {t["k"]} of {t["n"]} tested wells at 20+ ({t["k"] / t["n"]:.0%}); '
               f'predicted {t["loto"]:.1%} out of sample, {t["model"]:.1%} before calibration')
        parts.append(f'<circle class="pt" cx="{sx(t["loto"]):.1f}" cy="{sy(t["k"] / t["n"]):.1f}" r="{r:.1f}">'
                     f'<title>{html.escape(tip)}</title></circle>')
    parts.append(f'<text class="tick" x="{(L + W - R) / 2}" y="{H - 6}" text-anchor="middle">'
                 "Predicted share at 20 ng/L or more, out of sample</text>"
                 f'<text class="tick" transform="translate(14 {(H - B + T) / 2}) rotate(-90)" text-anchor="middle">'
                 "Measured share</text>")
    return (f'<svg class="chart scatter" viewBox="0 0 {W} {H}" role="img" aria-label="Predicted against measured '
            f'share of private wells at or above 20 ng/L for {len(towns)} tested towns">{"".join(parts)}</svg>')


def band_chart(bands: dict) -> str:
    """Horizontal bars: private-well residents by estimated probability of PFAS6 at 20 ng/L or more."""
    W, L, R, bar, gap, top = 720, 120, 110, 30, 12, 6
    peak = max(bands.values())
    parts = []
    for i, (label, n) in enumerate(bands.items()):
        y = top + i * (bar + gap)
        w = max((W - L - R) * n / peak, 2)
        parts.append(f'<text class="tick" x="{L - 10}" y="{y + bar / 2 + 5}" text-anchor="end">{html.escape(label)}</text>'
                     f'<rect class="b{i}" x="{L}" y="{y}" width="{w:.1f}" height="{bar}" rx="3">'
                     f'<title>{html.escape(label)}: {n:,} residents</title></rect>'
                     f'<text class="value" x="{L + w + 8:.1f}" y="{y + bar / 2 + 5}">{n:,}</text>')
    H = top + len(bands) * (bar + gap)
    return (f'<svg class="chart bars" viewBox="0 0 {W} {H}" role="img" aria-label="Private-well residents by '
            f'estimated probability of PFAS6 at or above 20 ng/L">{"".join(parts)}</svg>')


def _town_rows(towns: list[dict], tested: dict, n: int = 20) -> str:
    rows = []
    for t in towns[:n]:
        p = tested.get(t["town"])
        prog = f'{p["k"]} / {p["n"]}' if p else '<span class="note">not tested</span>'
        rows.append(f'<tr><td>{html.escape(t["town"])}</td><td>{t["residents"]:,}</td><td>{_pct(t["share"])}</td>'
                    f'<td>{t["expected"]:,}</td><td>{_pct(t["model_share"])}</td><td>{prog}</td></tr>')
    return "\n".join(rows)


def build_study_page(site_dir: Path, repo_url: str, r: dict, source_rows: str,
                     extra_downloads: list[tuple[str, str]]) -> Path:
    h, pv, gw = r["headline"], r["private_well_validation"], r["groundwater_validation"]
    loss = pv["loto_log_loss"]
    tested = {t["town"]: t for t in pv["town_table"]}
    towns = r["private_towns"]
    data = site_dir / "data"
    pd.DataFrame(towns).rename(columns={"share": "est_share_over20", "model_share": "share_before_calibration",
                                        "expected": "expected_residents_over20"}).to_csv(
        data / "private_well_towns.csv", index=False)
    pd.DataFrame(pv["town_table"]).rename(columns={
        "n": "wells_tested", "k": "wells_over20", "model": "model_rate_uncalibrated",
        "loto": "calibrated_rate_out_of_sample", "blended": "blended_rate"}).to_csv(
        data / "private_well_program_towns.csv", index=False)
    downloads = [("private_well_towns.csv", "Private-well residents and estimates by town"),
                 ("private_well_program_towns.csv",
                  "MassDEP program results by tested town, with the model's uncalibrated, out-of-sample and blended rates"),
                 *extra_downloads]
    share = h["private_expected_over20"] / h["private_wells"]
    pw_rows = "\n".join(f'<tr><td class="wrap">{html.escape(a)}</td><td class="wrap">{b}</td></tr>' for a, b in [
        ("Rank agreement between the model's town rates and measured town rates (Spearman)",
         f"{pv['spearman_towns']:.2f}"),
        *([("Separating towns with any well at 20 ng/L or more from those with none (ROC AUC)",
            f"{pv['auc_towns_any_over20']:.2f}")] if pv.get("auc_towns_any_over20") is not None else []),
        ("Binomial log loss of town counts, each town predicted by a calibration fit on the others (lower is better)",
         (f"{loss['calibrated_model']:.0f} (uncalibrated {loss['uncalibrated_model']:.0f}; "
          f"one statewide rate {loss['flat_rate']:.0f})")),
    ])
    gw_labels = {"logistic": "Logistic model (all features)", "gradient_boosting": "Gradient boosting (same features)",
                 "baseline_population": "Population within 1 km alone"}
    gw_rows = "\n".join(
        [f'<tr><td class="wrap">{label}</td><td>{gw["over20"][m]["roc_auc"]:.2f}</td>'
         f'<td>{gw["detect"][m]["roc_auc"]:.2f}</td></tr>' for m, label in gw_labels.items() if m in gw["over20"]]
        + [(f'<tr><td class="wrap">Share of wells in the class</td><td>{gw["over20"]["prevalence"]:.0%}</td>'
            f'<td>{gw["detect"]["prevalence"]:.0%}</td></tr>')])
    kpis = _tiles([
        (_people(h["private_wells"]), "residents outside community water service, on private wells"),
        (_people(h["private_expected_over20"]), f"expected at 20 ng/L or more ({share:.0%}), calibrated"),
        (f"{pv['wells']:,}", f"private wells tested by MassDEP in {pv['towns']} towns, used to check and calibrate"),
        (f"{pv['spearman_towns']:.2f}", "rank agreement with measured town rates, before calibration"),
    ])
    m = pv["prior_weight_wells"]
    page = (STUDY_TEMPLATE.read_text()
            .replace("__STYLE__", _style())
            .replace("__NAV__", nav("index.html", repo_url))
            .replace("__REPO__", repo_url)
            .replace("__KPIS__", kpis)
            .replace("__PRIVATE_POP__", _people(h["private_wells"]))
            .replace("__N_WELLS__", f"{gw['wells']:,}")
            .replace("__GW_PREV20__", f"{gw['over20']['prevalence']:.0%}")
            .replace("__GW_PREVDET__", f"{gw['detect']['prevalence']:.0%}")
            .replace("__GW_ROWS__", gw_rows)
            .replace("__PW_WELLS__", f"{pv['wells']:,}")
            .replace("__PW_TOWNS__", str(pv["towns"]))
            .replace("__PW_OVER__", str(pv["wells_over20"]))
            .replace("__PW_SHARE__", f"{pv['observed_share']:.0%}")
            .replace("__PW_MODEL_SHARE__", f"{pv['model_share_before_calibration']:.0%}")
            .replace("__PW_ROWS__", pw_rows)
            .replace("__SCATTER__", scatter_chart(pv["town_table"]))
            .replace("__PW_TARGETED__", f"{pv['targeted_share_of_invitations']:.0%}")
            .replace("__OFFSET__", f"{pv['offset']:.2f}")
            .replace("__PRIOR__", f"{m:.0f}")
            .replace("__BLEND_WEIGHT__", f"{40 / (40 + m):.0%}")
            .replace("__PRIVATE_20__", _people(h["private_expected_over20"]))
            .replace("__PRIVATE_SHARE__", f"{share:.0%}")
            .replace("__PRIVATE_20_MODEL__", _people(h["private_expected_over20_uncalibrated"]))
            .replace("__BAND_CHART__", band_chart(r["private_bands"]))
            .replace("__TOWN_ROWS__", _town_rows(towns, tested))
            .replace("__GROUP_ROWS__", _private_group_rows(r["groups"]))
            .replace("__DATA_DATE__", r["latest_result"])
            .replace("__SOURCE_ROWS__", source_rows)
            .replace("__DOWNLOADS__", _downloads(downloads))
            .replace("__RUN_DATE__", r["run_date"]))
    (site_dir / "index.html").write_text(page)
    return site_dir / "index.html"
