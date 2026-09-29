"""Static GitHub Pages site: a landing page plus the interactive map, written to docs/.

.github/workflows/pages.yml publishes docs/ whenever it changes on main.
"""

from __future__ import annotations

import html
import json
import shutil
from pathlib import Path

from .config import OUTPUT_DIR, ROOT, catalog
from .fetch import SOURCE_RECORD

SITE_DIR = ROOT / "docs"
TEMPLATE = Path(__file__).with_name("site_template.html")
REPO_URL = "https://github.com/adichiara/PFAS-release-risk"

MODEL_LABELS = {
    "baseline_area": "Baseline: land area",
    "baseline_area_population": "Baseline: area + density",
    "logistic": "Logistic regression",
    "poisson_rate": "Poisson rate (area as exposure)",
    "gradient_boosting": "Gradient boosting",
}
METRIC_LABELS = {
    "roc_auc": "ROC AUC",
    "avg_precision": "Average precision",
    "capture_top10pct_units": "Releases in top 10% of block groups",
    "capture_top10pct_area": "Releases in top-risk 10% of land",
}


def _fmt(metric: str, v: float) -> str:
    return f"{v * 100:.0f}%" if metric.startswith("capture") else f"{v:.3f}"


def _model_rows(report: dict) -> str:
    rows = []
    for model, label in MODEL_LABELS.items():
        m = report["mean"].get(model)
        if m is None:
            continue
        cls = ' class="selected"' if model == report["best_model"] else ""
        cells = "".join(f"<td>{_fmt(k, m[k])}</td>" for k in METRIC_LABELS)
        rows.append(f"<tr{cls}><td>{html.escape(label)}</td>{cells}</tr>")
    return "\n".join(rows)


def _null_rows(report: dict) -> str:
    return "\n".join(
        f"<tr><td>{METRIC_LABELS[k]}</td><td>{_fmt(k, v['observed'])}</td><td>{_fmt(k, v['null_mean'])}</td>"
        f"<td>{_fmt(k, v['null_p95'])}</td><td>{v['p_value']:.2f}</td></tr>"
        for k, v in report["null"].items())


def _kpis(report: dict) -> str:
    best, base = report["mean"][report["best_model"]], report["mean"]["baseline_area"]
    tiles = [
        (f"{report['located_releases']}", "located PFAS releases"),
        (_fmt("capture_top10pct_area", best["capture_top10pct_area"]),
         "of releases in the model's top-risk 10% of land"),
        (_fmt("capture_top10pct_area", base["capture_top10pct_area"]), "for land area alone"),
    ]
    return "".join(f'<div class="kpi"><div class="v">{v}</div><div class="l">{html.escape(label)}</div></div>'
                   for v, label in tiles)


def _source_rows() -> str:
    rows = []
    for meta in catalog()["sources"].values():
        title = html.escape(meta["title"])
        url = meta.get("url") or (f"{catalog()['massgis_base']}/{meta['path']}" if "path" in meta else None)
        name = f'<a href="{html.escape(url)}">{title}</a>' if url else title
        rows.append(f'<tr><td class="wrap">{name}</td><td class="wrap">{html.escape(meta["publisher"])}</td>'
                    f'<td><span class="status">{html.escape(meta["status"])}</span></td></tr>')
    return "\n".join(rows)


def _data_date() -> str:
    if SOURCE_RECORD.exists():
        return f" on MassDEP release data dated {json.loads(SOURCE_RECORD.read_text())['tables_dated']}"
    return ""


def build_site() -> Path:
    report = json.loads((OUTPUT_DIR / "evaluation.json").read_text())
    source = ("MassDEP's release database plus the 2021 project list"
              if report["release_source"] == "massdep_pfas_releases.csv"
              else "the 2021 list of MassDEP PFAS release sites")
    page = (TEMPLATE.read_text()
            .replace("__N_RELEASES__", str(report["located_releases"]))
            .replace("__SOURCE__", source)
            .replace("__REPO__", REPO_URL)
            .replace("__KPIS__", _kpis(report))
            .replace("__REPEATS__", str(report["repeats"]))
            .replace("__MODEL_ROWS__", _model_rows(report))
            .replace("__N_NULL__", str(report.get("null_permutations", 20)))
            .replace("__NULL_ROWS__", _null_rows(report))
            .replace("__SOURCE_ROWS__", _source_rows())
            .replace("__RUN_DATE__", report["run_date"])
            .replace("__DATA_DATE__", _data_date()))
    SITE_DIR.mkdir(exist_ok=True)
    (SITE_DIR / "index.html").write_text(page)
    shutil.copyfile(OUTPUT_DIR / "risk_map.html", SITE_DIR / "map.html")
    return SITE_DIR / "index.html"
