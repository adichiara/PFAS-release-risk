"""Command line: ``pfas-risk {download,features,evaluate,map,run}``."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date

import pandas as pd

from .config import OUTPUT_DIR
from .features import attach_releases, build_features
from .mapping import risk_map
from .model import evaluate, permutation_null
from .releases import locate_releases
from .site import build_site
from .sources import download_all

log = logging.getLogger("pfas_risk")

BASELINES = ("baseline_area", "baseline_area_population")
SELECTION_METRIC = "capture_top10pct_area"


def _dataset(force: bool = False):
    releases = locate_releases()
    return attach_releases(build_features(force=force), releases), releases


def cmd_download(args) -> None:
    for p in download_all(force=args.force):
        print(p)


def cmd_features(args) -> None:
    df, releases = _dataset(force=args.force)
    print(f"{len(df)} block groups, {int(df['releases'].sum())} located releases in "
          f"{int((df['releases'] > 0).sum())} block groups; "
          f"{df.attrs['unlocated_releases']} releases could not be located")


def cmd_evaluate(args) -> None:
    df, releases = _dataset()
    metrics, oof = evaluate(df, repeats=args.repeats)
    summary = metrics.drop(columns="repeat").groupby("model").agg(["mean", "std"])
    candidates = summary.drop(index=list(BASELINES))
    best = candidates[(SELECTION_METRIC, "mean")].idxmax()
    null = permutation_null(df, best, permutations=args.null)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(OUTPUT_DIR / "cv_metrics.csv", index=False)
    null.to_csv(OUTPUT_DIR / "null_metrics.csv", index=False)
    oof.to_parquet(OUTPUT_DIR / "oof_scores.parquet")
    report = {
        "run_date": date.today().isoformat(),
        "release_source": releases.attrs.get("source"),
        "block_groups": len(df),
        "located_releases": int(df["releases"].sum()),
        "release_block_groups": int((df["releases"] > 0).sum()),
        "unlocated_releases": df.attrs["unlocated_releases"],
        "repeats": args.repeats,
        "null_permutations": args.null,
        "selection_metric": SELECTION_METRIC,
        "best_model": best,
        "mean": metrics.drop(columns="repeat").groupby("model").mean().round(4).to_dict("index"),
        "null": _null_summary(metrics[metrics["model"] == best], null),
    }
    (OUTPUT_DIR / "evaluation.json").write_text(json.dumps(report, indent=2))
    (OUTPUT_DIR / "evaluation.md").write_text(_markdown(report, summary, null))
    print((OUTPUT_DIR / "evaluation.md").read_text())


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
    lines = [
        "# Evaluation",
        "",
        f"Run {report['run_date']} on `{report['release_source']}`: {report['located_releases']} located "
        f"releases in {report['release_block_groups']} of {report['block_groups']} block groups "
        f"({report['unlocated_releases']} could not be located).",
        "",
        f"Town-grouped 5-fold cross-validation, {report['repeats']} repeats (mean ± sd). "
        "Compare models with the two baselines, not with 10%: releases are not spread evenly "
        "over land or over block groups, so chance capture depends on the budget.",
        "",
        "| model | ROC AUC | avg precision | releases in top 10% of block groups "
        "| releases in top-risk 10% of land |",
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
        f"Permutation null for {report['best_model']}: release labels shuffled {len(null)} times among "
        "block groups in the same land-area x population-density quintile, then the same "
        "cross-validation. This keeps the size and density effects and removes everything else.",
        "",
        "| metric | observed | null mean | null 95th pct | p |",
        "|---|---|---|---|---|",
    ]
    for c, v in report["null"].items():
        lines.append(f"| {c} | {v['observed']:.3f} | {v['null_mean']:.3f} | {v['null_p95']:.3f} | {v['p_value']:.3f} |")
    lines.append("")
    return "\n".join(lines)


def cmd_map(args) -> None:
    report = json.loads((OUTPUT_DIR / "evaluation.json").read_text())
    model = args.model or report["best_model"]
    df, releases = _dataset()
    oof = pd.read_parquet(OUTPUT_DIR / "oof_scores.parquet")
    note = ("MassDEP export" if report["release_source"] == "massdep_releases.csv"
            else "2021 seed list of MassDEP PFAS RTNs")
    path = risk_map(df, oof[model], releases, model, report["mean"][model], report["mean"]["baseline_area"],
                    OUTPUT_DIR / "risk_map.html", note)
    print(path)


def cmd_site(args) -> None:
    print(build_site())


def cmd_run(args) -> None:
    cmd_evaluate(args)
    args.model = None
    cmd_map(args)
    cmd_site(args)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="pfas-risk", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("download", help="fetch all automatically available public sources")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_download)

    p = sub.add_parser("features", help="build the block-group feature table")
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
