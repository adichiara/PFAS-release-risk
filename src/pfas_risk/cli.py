"""Command line: ``pfas-risk {download,features,evaluate,map,run}``."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone

import pandas as pd

from .config import FORWARD_CUTOFF, OUTPUT_DIR
from .exposure import block_exposure, exposure_report, system_table
from .exposure import block_group_table as exposure_block_groups
from .features import attach_releases, build_features, unit_key
from .fetch import fetch_release_zip, recorded_sha256, sha256
from .mapping import BASELINE_TEXT, risk_map
from .model import BASELINES, evaluate, forward_test, grouped_importance, permutation_null
from .population import population_report
from .releases import build_release_list, locate_releases
from .site import PRIMARY_UNIT, UnitResult, build_site
from .sources import download_all
from .water_map import water_map

log = logging.getLogger("pfas_risk")

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
    return OUTPUT_DIR / _key(args)


def _dataset(args=None, force: bool = False, unit: str = "hex", cell_km2: float = 1.0):
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
    forward = _forward(df, releases, best, args.cutoff)

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
        "forward": forward,
    }
    (out_dir / "evaluation.json").write_text(json.dumps(report, indent=2))
    (out_dir / "evaluation.md").write_text(_markdown(report, summary, null))
    print((out_dir / "evaluation.md").read_text())


def _forward(df, releases, best: str, cutoff: str) -> dict:
    """Train on releases reported before ``cutoff``; score units whose first release came after."""
    dates = pd.to_datetime(releases["notification_date"], errors="coerce")
    features = df.drop(columns="releases")
    before = attach_releases(features, releases[dates < cutoff])["releases"].to_numpy()
    after = attach_releases(features, releases[dates >= cutoff])["releases"].to_numpy()
    new_after = (after > 0) & (before == 0)
    return {
        "cutoff": cutoff,
        "train_releases": int((dates < cutoff).sum()),
        "test_releases": int((dates >= cutoff).sum()),
        "new_units": int(new_after.sum()),
        "results": forward_test(df, before, new_after, [best, *BASELINES]),
    }


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
    fwd = report.get("forward")
    if fwd:
        lines += [
            "",
            "## Does it predict new reports?",
            "",
            (f"Trained on the {fwd['train_releases']} releases reported before {fwd['cutoff']}; scored on the "
             f"{fwd['new_units']} {plural} whose first release was reported on or after it (only {plural} "
             "without an earlier release are scored)."),
            "",
            "| model | " + " | ".join(cols) + " |",
            "|---|" + "---|" * len(cols),
        ]
        for model, m in fwd["results"].items():
            lines.append(f"| {model} | " + " | ".join(f"{m[c]:.3f}" for c in cols) + " |")
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
    kind = "hex" if args.unit == "hex" else "bg"
    base = report["mean"]["baseline_area_population" if kind == "hex" else "baseline_area"]
    path = risk_map(df, oof[model], releases, model, report["mean"][model], base,
                    out_dir / "risk_map.html", note, unit_labels(_key(args)), BASELINE_TEXT[kind],
                    hex_km2=args.cell_km2 if kind == "hex" else None)
    print(path)


def cmd_site(args) -> None:
    """Lead with the primary unit (1 km² hexagons); add every other evaluated unit for comparison."""
    releases = locate_releases()
    results = []
    for report_path in sorted(OUTPUT_DIR.glob("*/evaluation.json")):
        report = json.loads(report_path.read_text())
        key = report["unit"]
        unit, size = ("hex", float(key.removeprefix("hex"))) if key.startswith("hex") else ("bg", 4.0)
        df = attach_releases(build_features(unit=unit, cell_km2=size), releases)
        results.append(UnitResult(key, unit_labels(key), report, df,
                                  pd.read_parquet(report_path.parent / "oof_scores.parquet"),
                                  report_path.parent / "risk_map.html"))
    if not results:
        raise SystemExit("no evaluated units in outputs/; run `pfas-risk --unit hex evaluate` first")
    primary = next((u for u in results if u.key == PRIMARY_UNIT), results[0])
    # Other hexagon sizes, smallest first, then block groups.
    others = sorted((u for u in results if u is not primary),
                    key=lambda u: (not u.is_hex, float(u.key[3:]) if u.is_hex else 0.0))
    print(build_site(primary, others, releases))


def cmd_population(args) -> None:
    """Carry the unit's out-of-fold scores to census blocks; summarize who lives in higher-risk areas."""
    if args.unit != "hex":
        raise SystemExit("population summaries need an equal-area unit: use --unit hex")
    out_dir = _out_dir(args)
    report = json.loads((out_dir / "evaluation.json").read_text())
    oof = pd.read_parquet(out_dir / "oof_scores.parquet")
    df = build_features(unit="hex", cell_km2=args.cell_km2)
    rows, block_groups = population_report(df, oof[report["best_model"]])
    (out_dir / "population.json").write_text(json.dumps(
        {"unit": _key(args), "model": report["best_model"], "run_date": report["run_date"], "groups": rows},
        indent=2))
    block_groups.to_csv(out_dir / "block_group_risk.csv")
    for r in rows:
        print(f"{r['group']:50s} {r['residents']:>9,}  top 10% of land: {r['top10']:.1%}"
              f"  (x{r['top10_vs_density']} vs density alone)")


def cmd_exposure(args) -> None:
    """Drinking-water exposure: public systems (measured) and private wells (groundwater model)."""
    out_dir = OUTPUT_DIR / "water"
    out_dir.mkdir(parents=True, exist_ok=True)
    blocks = block_exposure()
    systems = system_table()
    report = exposure_report(blocks, systems)
    report["run_date"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    (out_dir / "exposure.json").write_text(json.dumps(report, indent=2))
    exposure_block_groups(blocks).to_csv(out_dir / "block_group_drinking_water.csv")
    systems.to_csv(out_dir / "water_systems_pfas6.csv", index=False)
    print(water_map(blocks, locate_releases(), out_dir / "water_map.html"))
    h = report["headline"]
    print(f"public water {h['public_water']:,}: now >= 20 ng/L "
          f"{h['public_by_current_band'].get('20 or more (state standard)', 0):,}; "
          f"private wells {h['private_wells']:,}: expected >= 20 ng/L {h['private_expected_over20']:,}")


def cmd_run(args) -> None:
    cmd_evaluate(args)
    args.model = None
    cmd_map(args)
    if _key(args) == PRIMARY_UNIT:
        cmd_population(args)
    cmd_site(args)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="pfas-risk", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--unit", choices=["bg", "hex"], default="hex",
                        help="spatial unit: equal-area hexagons (default) or census block groups")
    parser.add_argument("--cell-km2", type=float, default=1.0, help="hexagon area in km² (default 1)")
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
        p.add_argument("--cutoff", default=FORWARD_CUTOFF,
                       help="forward test: train on releases before this date (default %(default)s)")
        p.set_defaults(func=func)

    p = sub.add_parser("map", help="write outputs/<unit>/risk_map.html")
    p.add_argument("--model", help="model to map (default: the selected model)")
    p.set_defaults(func=cmd_map)

    p = sub.add_parser("exposure", help="drinking-water PFAS exposure: public systems and private wells")
    p.set_defaults(func=cmd_exposure)

    p = sub.add_parser("population", help="who lives in higher-risk areas (census blocks, EJ, private wells)")
    p.set_defaults(func=cmd_population)

    p = sub.add_parser("site", help="write the GitHub Pages site to docs/")
    p.set_defaults(func=cmd_site)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
