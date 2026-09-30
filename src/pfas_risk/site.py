"""Static GitHub Pages site: a landing page plus interactive maps, written to docs/.

The page leads with the primary unit (equal-area hexagons, where land area no longer drives
the result) and compares the other evaluated units below it.
.github/workflows/pages.yml publishes docs/ whenever it changes on main.
"""

from __future__ import annotations

import html
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import pandas as pd

from .config import ROOT, catalog
from .fetch import SOURCE_RECORD
from .site_sections import (
    capture_chart,
    importance_chart,
    releases_by_year_chart,
    unit_comparison_rows,
    watchlist,
    watchlist_rows,
    write_downloads,
)

SITE_DIR = ROOT / "docs"
TEMPLATE = Path(__file__).with_name("site_template.html")
REPO_URL = "https://github.com/adichiara/PFAS-release-risk"
PRIMARY_UNIT = "hex4"

MODEL_LABELS = {
    "baseline_area": "Baseline: land area",
    "baseline_area_population": "Baseline: area + density",
    "logistic": "Logistic regression",
    "logistic+smooth": "Logistic regression, smoothed",
    "poisson_rate": "Poisson rate (area as exposure)",
    "poisson_rate+smooth": "Poisson rate, smoothed",
    "gradient_boosting": "Gradient boosting",
    "gradient_boosting+smooth": "Gradient boosting, smoothed",
}


@dataclass
class UnitResult:
    key: str                      # 'hex4', 'bg', ...
    labels: tuple[str, str]       # ('4 km² hexagon', '4 km² hexagons')
    report: dict
    df: gpd.GeoDataFrame
    oof: pd.DataFrame
    map_path: Path

    @property
    def is_hex(self) -> bool:
        return self.key.startswith("hex")

    @property
    def baseline(self) -> tuple[str, str]:
        """The size baseline to beat: area + density for equal cells, land area otherwise."""
        return (("baseline_area_population", "Area + density") if self.is_hex
                else ("baseline_area", "Land area alone"))

    @property
    def map_file(self) -> str:
        return "map.html" if self.key == PRIMARY_UNIT else f"map_{self.key}.html"


def _metric_labels(plural: str) -> dict[str, str]:
    return {
        "roc_auc": "ROC AUC",
        "avg_precision": "Average precision",
        "capture_top10pct_units": f"Releases in top 10% of {plural}",
        "capture_top10pct_area": "Releases in top-risk 10% of land",
    }


def _fmt(metric: str, v: float) -> str:
    return f"{v * 100:.0f}%" if metric.startswith("capture") else f"{v:.3f}"


def _model_rows(report: dict, plural: str) -> str:
    rows = []
    for model, label in MODEL_LABELS.items():
        m = report["mean"].get(model)
        if m is None:
            continue
        cls = ' class="selected"' if model == report["best_model"] else ""
        cells = "".join(f"<td>{_fmt(k, m[k])}</td>" for k in _metric_labels(plural))
        rows.append(f"<tr{cls}><td>{html.escape(label)}</td>{cells}</tr>")
    return "\n".join(rows)


def _null_rows(report: dict, plural: str) -> str:
    labels = _metric_labels(plural)
    return "\n".join(
        f"<tr><td>{labels[k]}</td><td>{_fmt(k, v['observed'])}</td><td>{_fmt(k, v['null_mean'])}</td>"
        f"<td>{_fmt(k, v['null_p95'])}</td><td>{v['p_value']:.2f}</td></tr>"
        for k, v in report["null"].items())


def _forward_rows(report: dict, plural: str) -> str:
    fwd = report.get("forward")
    if not fwd:
        return ""
    labels = _metric_labels(plural)
    rows = []
    for model, m in fwd["results"].items():
        cls = ' class="selected"' if model == report["best_model"] else ""
        cells = "".join(f"<td>{_fmt(k, m[k])}</td>" for k in labels)
        rows.append(f"<tr{cls}><td>{html.escape(MODEL_LABELS.get(model, model))}</td>{cells}</tr>")
    return "\n".join(rows)


def _kpis(u: UnitResult) -> str:
    r = u.report
    best = r["mean"][r["best_model"]]
    base_col, base_label = u.baseline
    tiles = [
        (f"{r['located_releases']}", "located PFAS releases"),
        (_fmt("capture_top10pct_area", best["capture_top10pct_area"]),
         "of releases in the model's top-risk 10% of land"),
        (_fmt("capture_top10pct_area", r["mean"][base_col]["capture_top10pct_area"]),
         f"for {base_label.lower()}"),
    ]
    fwd = r.get("forward")
    if fwd:
        tiles.append((_fmt("capture_top10pct_area", fwd["results"][r["best_model"]]["capture_top10pct_area"]),
                      (f"of new {fwd['cutoff'][:4]}+ release cells found in the top-risk 10% of land "
                       f"by a model trained only on earlier reports")))
    return "".join(f'<div class="kpi"><div class="v">{v}</div><div class="l">{html.escape(label)}</div></div>'
                   for v, label in tiles)


STATUS_LABELS = {"available": "used", "evaluated": "tested, not used"}


def _source_rows() -> str:
    rows = []
    for meta in catalog()["sources"].values():
        title = html.escape(meta["title"])
        url = meta.get("url") or (f"{catalog()['massgis_base']}/{meta['path']}" if "path" in meta else None)
        name = f'<a href="{html.escape(url)}">{title}</a>' if url else title
        status = STATUS_LABELS.get(meta["status"], meta["status"])
        result = f'<br><span class="note">{html.escape(meta["result"])}</span>' if "result" in meta else ""
        rows.append(f'<tr><td class="wrap">{name}{result}</td><td class="wrap">{html.escape(meta["publisher"])}</td>'
                    f'<td><span class="status">{html.escape(status)}</span></td></tr>')
    return "\n".join(rows)


def _data_date() -> str:
    if SOURCE_RECORD.exists():
        return f" on MassDEP release data dated {json.loads(SOURCE_RECORD.read_text())['tables_dated']}"
    return ""


def _legend(base_label: str) -> str:
    return (f'<div class="legend"><span><i></i>Model</span><span><i class="base"></i>{base_label}</span>'
            f'<span><i class="random"></i>Random targeting</span></div>')


def _other_units_section(primary: UnitResult, others: list[UnitResult]) -> str:
    if not others:
        return ""
    table = unit_comparison_rows([(u.labels[1].capitalize(), u.report) for u in [primary, *others]])
    parts = [f"""
  <h2>Compared with other units</h2>
  <p>The same features, validation and permutation test on other spatial units. Census block groups range
     from under 0.1 km² to over 200 km², so land area alone explains much of which ones contain a reported
     release; equal-area hexagons remove that effect, which is why they lead this page.</p>
  <div class="scroll"><table>
    <thead><tr><th>Unit</th><th>Cells</th><th>With a release</th>
      <th>ROC AUC<br><span class="note">model / area + density</span></th>
      <th>Avg precision<br><span class="note">model / area + density</span></th>
      <th>Releases in top-risk 10% of land<br><span class="note">model / land area / area + density</span></th>
      <th><i>p</i> vs size-matched null</th></tr></thead>
    <tbody>{table}</tbody>
  </table></div>"""]
    for u in others:
        base_col, base_label = u.baseline
        parts.append(f"""
  <h3>{html.escape(u.labels[1].capitalize())}</h3>
  <p>Capture curve against {html.escape(base_label.lower())}. <a href="{u.map_file}">Open the
     {html.escape(u.labels[1])} map</a>.</p>
  <div class="figure">{capture_chart(u.df, u.oof, u.report["best_model"], base_col, base_label)}
    {_legend(base_label)}
  </div>""")
    return "".join(parts)


def build_site(primary: UnitResult, others: list[UnitResult], releases: gpd.GeoDataFrame) -> Path:
    r = primary.report
    model = r["best_model"]
    singular, plural = primary.labels
    base_col, base_label = primary.baseline
    SITE_DIR.mkdir(exist_ok=True)
    downloads = write_downloads(SITE_DIR, primary.key, plural, primary.df, primary.oof, model, releases)
    source = ("MassDEP's release database plus the 2021 project list"
              if r["release_source"] == "massdep_pfas_releases.csv"
              else "the 2021 list of MassDEP PFAS release sites")
    fwd = r.get("forward") or {}
    importance = r.get("importance") or []
    page = (TEMPLATE.read_text()
            .replace("__N_RELEASES__", str(r["located_releases"]))
            .replace("__SOURCE__", source)
            .replace("__REPO__", REPO_URL)
            .replace("__KPIS__", _kpis(primary))
            .replace("__UNIT_COUNT__", f"{r['units']:,}")
            .replace("__UNITS__", plural)
            .replace("__UNIT__", singular)
            .replace("__UNIT_TITLE__", "Cell" if primary.is_hex else singular.capitalize())
            .replace("__BASE_LABEL__", base_label)
            .replace("__BASE_LABEL_LC__", base_label.lower())
            .replace("__REPEATS__", str(r["repeats"]))
            .replace("__MODEL_ROWS__", _model_rows(r, plural))
            .replace("__N_NULL__", str(r.get("null_permutations", 20)))
            .replace("__NULL_ROWS__", _null_rows(r, plural))
            .replace("__FORWARD_ROWS__", _forward_rows(r, plural))
            .replace("__FWD_CUTOFF_YEAR__", str(fwd.get("cutoff", "2023"))[:4])
            .replace("__FWD_TRAIN__", str(fwd.get("train_releases", "")))
            .replace("__FWD_NEW__", str(fwd.get("new_units", "")))
            .replace("__SOURCE_ROWS__", _source_rows())
            .replace("__RUN_DATE__", r["run_date"])
            .replace("__DATA_DATE__", _data_date())
            .replace("__CAPTURE_CHART__", capture_chart(primary.df, primary.oof, model, base_col, base_label))
            .replace("__IMPORTANCE_CHART__", importance_chart(importance) if importance else
                     "<p class='note'>Run <code>pfas-risk evaluate</code> to compute importance.</p>")
            .replace("__WATCH_ROWS__", watchlist_rows(watchlist(primary.df, primary.oof, model, releases)))
            .replace("__YEAR_CHART__", releases_by_year_chart(releases))
            .replace("__N_ALL_RELEASES__", str(len(releases)))
            .replace("__OTHER_UNITS__", _other_units_section(primary, others))
            .replace("__DOWNLOADS__", "".join(f'<li><a href="{h}">{html.escape(t)}</a></li>'
                                              for h, t in downloads)))
    (SITE_DIR / "index.html").write_text(page)
    for u in [primary, *others]:
        if u.map_path.exists():
            shutil.copyfile(u.map_path, SITE_DIR / u.map_file)
    return SITE_DIR / "index.html"
