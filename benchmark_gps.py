"""Compare predeclared GPS candidates on held-out industry doses, without outcomes."""

import argparse
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.stats import norm

from labor_did import Config
from labor_did.data import load_data, prepare_panel, prepare_year
from labor_did.io import write_json
from labor_did.models import fit_gps
from labor_did.pipeline import ROOT


def candidates(base):
    yield (
        "legacy_spline5_reference",
        replace(base, gps_model="spline", gps_n_knots=5, gps_fit_scope="legacy"),
    )
    yield "nested_linear", replace(base, gps_model="linear", gps_fit_scope="nested")
    for knots in [3, 5]:
        yield (
            f"nested_spline{knots}",
            replace(
                base, gps_model="spline", gps_n_knots=knots, gps_fit_scope="nested"
            ),
        )
    for depth, leaf in [(2, 10), (3, 8), (3, 15), (4, 10)]:
        yield (
            f"rf_depth{depth}_leaf{leaf}",
            replace(
                base,
                gps_model="random_forest",
                gps_fit_scope="nested",
                gps_params={
                    "n_estimators": 200,
                    "max_depth": depth,
                    "min_samples_leaf": leaf,
                    "max_features": 0.75,
                },
            ),
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", nargs="+", type=int, default=[2023, 2025])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", type=Path, default=ROOT / "results/gps_benchmark")
    args = parser.parse_args()
    config = Config.load(ROOT / "settings.json")
    config = replace(
        config,
        n_repeats=args.repeats,
        cache_dir=str(ROOT / ".cache"),
        data_path=str(ROOT / config.data_path),
    )
    unknown = set(args.years) - set(config.years)
    if unknown:
        parser.error(f"Years absent from analysis settings: {sorted(unknown)}")
    raw, _ = load_data(config.data_path, config.cache_dir)
    data, sample = prepare_panel(raw, config)
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "sample.json", sample)
    rows = []
    for year in args.years:
        frame = prepare_year(data, year, config)
        first = frame.groupby("ind_code", sort=True).head(1).index
        for name, c in candidates(config):
            print(f"{year}: {name}", flush=True)
            started = perf_counter()
            mean, sigma, audit = fit_gps(frame, c)
            y = frame.loc[first, c.dose].to_numpy(float)
            m, s = mean[first], sigma[first]
            z = (y - m) / s
            rows.append(
                {
                    "year": year,
                    "candidate": name,
                    "n_industries": len(y),
                    "features": len(audit["features"]),
                    "repeats": c.n_repeats,
                    "nested": name != "legacy_spline5_reference",
                    "mse": float(np.mean((y - m) ** 2)),
                    "gaussian_nll": float(-np.mean(norm.logpdf(y, m, s))),
                    "coverage_68": float(np.mean(np.abs(z) <= 1)),
                    "coverage_95": float(np.mean(np.abs(z) <= 1.96)),
                    "median_sigma": float(np.median(s)),
                    "observed_density_floor_fraction": float(
                        np.mean(norm.pdf(y, m, s) < c.density_floor)
                    ),
                    "seconds": perf_counter() - started,
                }
            )
            pd.DataFrame(rows).to_csv(args.out / "gps_benchmark.csv", index=False)
            write_json(
                args.out / f"{year}_{name}.json",
                {"config": c.to_dict(), "audit": audit},
            )
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
