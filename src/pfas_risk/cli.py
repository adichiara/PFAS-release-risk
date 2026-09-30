"""Command line: ``pfas-risk {download,features,evaluate,map,run}``."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone

import pandas as pd

from .config import OUTPUT_DIR
from .features import attach_releases, build_features, unit_key
from .fetch import fetch_release_zip, recorded_sha256, sha256
from .mapping import risk_map
from .model import evaluate, grouped_importance, permutation_null
from .releases import build_release_list, locate_releases
from .site import build_site
from .sources import download_all

log = logging.getLogger("pfas_risk")

BASELINES = ("baseline_area", "baseline_area_population")
SELECTION_METRIC = "capture_top10pct_area"


UNIT_LABELS = {"bg": ("block group", "block groups")}


def unit_labels(key: str) -> tuple[str, str]:
    """(singular, plural) for 'bg' or e.g. 'hex4'."""
    if key in UNIT_LABELS:
        return UNIT_LABELS[key]
    size = key.removeprefix("hex")
    return f"{size} km² hexagon", f"{size} km² hexagons"


def _key(args) -> str:
    return unit_key(args.unit, args.cell_km2)


def _out_dir(args):
    """outputs/ for block groups (the published default), outputs/<key>/ for other units."""
    key = _key(args)
    return OUTPUT_DIR if key == "bg" else OUTPUT_DIR / key


def _dataset(args=None, force: bool = False, unit: str = "bg", cell_km2: float = 4.0):
    if args is not None:
        unit, cell_km2 = args.unit, args.cell_km2
    releases = locate_releases()
    return attach_releases(build_features(force=force, unit=unit, cell_km2=cell_km2), releases), releases


def cmd_download(args) -> None:
    for p in download_all(force=args.force):
        print(p)


def cmd_fetch_releases(args) -> None:
    """Download the MassDEP zip; print `changed=true|false` (for GitHub Actions outputs)."""
    path = fetch_release_zip()
    changed = sha256(path) != recorded_sha256()
    print(f"downloaded {path} ({path.stat().st_size:,} bytes)")
    print(f"changed={'true' if changed else 'false'}")


def cmd_releases(args) -> None:
    print(build_release_list(args.zip, coordinates=not args.no_coordinates,
                             refresh_coordinates=args.refresh_coordinates))


def cmd_features(args) -> None:
    df, _ = _dataset(args, force=args.force)
    _, plural = unit_labels(_key(args))
    print(f"{len(df)} {plural}, {int(df['releases'].sum())} located releases in "
          f"{int((df['releases'] > 0).sum())} {plural}; "
          f"{df.attrs['unlocated_releases']} releases could not be located "
          f"({df.attrs['releases_outside_units']} fell outside the units)")


def cmd_evaluate(args) -> None:
    df, releases = _dataset(args)
    out_dir = _out_dir(args)
    metrics, oof = evaluate(df, repeats=args.repeats)
    summary = metrics.drop(columns="repeat").groupby("model").agg(["mean", "std"])
    candidates = summary.drop(index=list(BASELINES))
    best = candidates[(SELECTION_METRIC, "mean")].idxmax()
    null = permutation_null(df, best, permutations=args.null)
    importance = grouped_importance(df, best)

    out_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out_dir / "cv_metrics.csv", index=False)
    null.to_csv(out_dir / "null_metrics.csv", index=False)
    oof.to_parquet(out_dir / "oof_scores.parquet")
    importance.to_csv(out_dir / "importance.csv", index=False)
    report = {
        "run_date": datetime.now(timezone.utc).date().isoformat(),
        "unit": _key(args),
        "release_source": releases.attrs.get("source"),
        "units": len(df),
        "located_releases": int(df["releases"].sum()),
        "release_units": int((df["releases"] > 0).sum()),
        "unlocated_releases": df.attrs["unlocated_releases"],
        "repeats": args.repeats,
        "null_permutations": args.null,
        "selection_metric": SELECTION_METRIC,
        "best_model": best,
        "mean": metrics.drop(columns="repeat").groupby("model").mean().round(4).to_dict("index"),
        "null": _null_summary(metrics[metrics["model"] == best], null),
        "importance": importance.round(4).to_dict("records"),
    }
    (out_dir / "evaluation.json").write_text(json.dumps(report, indent=2))
    (out_dir / "evaluation.md").write_text(_markdown(report, summary, null))
    print((out_dir / "evaluation.md").read_text())


METRIC_COLS = ["roc_auc", "avg_precision", "capture_top10pct_units", "capture_top10pct_area"]


def _null_summary(observed: pd.DataFrame, null: pd.DataFrame) -> dict:
    """Null mean, 95th percentile and one-sided empirical p-value per metric."""
    out = {}
    for c in METRIC_COLS:
        obs = observed[c].mean()
        out[c] = {"observed": round(obs, 4), "null_mean": round(null[c].mean(), 4),
                  "null_p95": round(null[c].quantile(0.95), 4),
                  "p_value": round((1 + (null[c] >= obs).sum()) / (1 + len(null)), 4)}
    return out


def _markdown(report: dict, summary: pd.DataFrame, null: pd.DataFrame) -> str:
    cols = METRIC_COLS
    _, plural = unit_labels(report["unit"])
    lines = [
        f"# Evaluation: {plural}",
        "",
        (f"Run {report['run_date']} on `{report['release_source']}`: {report['located_releases']} located "
         f"releases in {report['release_units']} of {report['units']} {plural} "
         f"({report['unlocated_releases']} could not be located)."),
        "",
        (f"Town-grouped 5-fold cross-validation, {report['repeats']} repeats (mean ± sd). "
         "Compare models with the two baselines, not with 10%: releases are not spread evenly "
         "over land or over units, so chance capture depends on the budget."),
        "",
        (f"| model | ROC AUC | avg precision | releases in top 10% of {plural} "
         "| releases in top-risk 10% of land |"),
        "|---|---|---|---|---|",
    ]
    for model, row in summary.iterrows():
        cells = [f"{row[(c, 'mean')]:.3f} ± {row[(c, 'std')]:.3f}" for c in cols]
        lines.append(f"| {model} | " + " | ".join(cells) + " |")
    lines += [
        "",
        f"Selected model (best non-baseline on `{report['selection_metric']}`): **{report['best_model']}**.",
        "",
        "## Does it beat size and density alone?",
        "",
        (f"Permutation null for {report['best_model']}: release labels shuffled {len(null)} times among "
         f"{plural} in the same land-area x population-density quintile, then the same "
         "cross-validation. This keeps the size and density effects and removes everything else."),
        "",
        "| metric | observed | null mean | null 95th pct | p |",
        "|---|---|---|---|---|",
    ]
    for c, v in report["null"].items():
        lines.append(f"| {c} | {v['observed']:.3f} | {v['null_mean']:.3f} | {v['null_p95']:.3f} | {v['p_value']:.3f} |")
    lines.append("")
    return "\n".join(lines)


def cmd_map(args) -> None:
    out_dir = _out_dir(args)
    report = json.loads((out_dir / "evaluation.json").read_text())
    model = args.model or report["best_model"]
    df, releases = _dataset(args)
    oof = pd.read_parquet(out_dir / "oof_scores.parquet")
    note = ("MassDEP PFAS release list" if report["release_source"] == "massdep_pfas_releases.csv"
            else "2021 seed list of MassDEP PFAS RTNs")
    path = risk_map(df, oof[model], releases, model, report["mean"][model], report["mean"]["baseline_area"],
                    out_dir / "risk_map.html", note, unit_labels(_key(args)))
    print(path)


def cmd_site(args) -> None:
    """The site always leads with block groups; other units' results are added when present."""
    df, releases = _dataset()
    oof = pd.read_parquet(OUTPUT_DIR / "oof_scores.parquet")
    others = []
    for report_path in sorted(OUTPUT_DIR.glob("*/evaluation.json")):
        report = json.loads(report_path.read_text())
        key = report.get("unit", report_path.parent.name)
        unit, size = ("hex", float(key.removeprefix("hex"))) if key.startswith("hex") else ("bg", 4.0)
        udf, _ = _dataset(unit=unit, cell_km2=size)
        others.append((key, unit_labels(key), report, udf, pd.read_parquet(report_path.parent / "oof_scores.parquet"),
                       report_path.parent / "risk_map.html"))
    print(build_site(df, oof, releases, others))


def cmd_run(args) -> None:
    cmd_evaluate(args)
    args.model = None
    cmd_map(args)
    cmd_site(args)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="pfas-risk", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--unit", choices=["bg", "hex"], default="bg",
                        help="spatial unit: census block groups (default) or equal-area hexagons")
    parser.add_argument("--cell-km2", type=float, default=4.0, help="hexagon area in km² (default 4)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("download", help="fetch all automatically available public sources")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_download)

    p = sub.add_parser("fetch-releases", help="download the MassDEP bulk data and report whether it changed")
    p.set_defaults(func=cmd_fetch_releases)

    p = sub.add_parser("releases", help="build the PFAS release list from the MassDEP bulk download")
    p.add_argument("--zip", help="path to the MassDEP download (default: data/raw/massdep_release_data.zip)")
    p.add_argument("--no-coordinates", action="store_true", help="skip MassDEP site coordinate lookups")
    p.add_argument("--refresh-coordinates", action="store_true",
                   help="look up coordinates for every RTN, not just ones new to the list")
    p.set_defaults(func=cmd_releases)

    p = sub.add_parser("features", help="build the feature table for the chosen unit")
    p.add_argument("--force", action="store_true", help="rebuild even if cached")
    p.set_defaults(func=cmd_features)

    for name, func, help_ in [("evaluate", cmd_evaluate, "cross-validate all models"),
                              ("run", cmd_run, "evaluate, then build the map and site")]:
        p = sub.add_parser(name, help=help_)
        p.add_argument("--repeats", type=int, default=10)
        p.add_argument("--null", type=int, default=20, help="label permutations for the null test")
        p.set_defaults(func=func)

    p = sub.add_parser("map", help="write outputs/risk_map.html")
    p.add_argument("--model", help="model to map (default: the selected model)")
    p.set_defaults(func=cmd_map)

    p = sub.add_parser("site", help="write the GitHub Pages site to docs/")
    p.set_defaults(func=cmd_site)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
