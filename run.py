"""One command for a single specification or an explicit Cartesian grid."""

import argparse
import itertools
import json
from pathlib import Path

import pandas as pd

from labor_did import Config, run
from labor_did.io import write_json
from labor_did.pipeline import PRIMARY, ROOT, holm


def assignment(value):
    try:
        key, raw = value.split("=", 1)
        return key, json.loads(raw)
    except (ValueError, json.JSONDecodeError) as exc:
        raise argparse.ArgumentTypeError(
            "Use KEY=JSON, for example bandwidth_scale=0.4"
        ) from exc


def main(argv=None):
    parser = argparse.ArgumentParser(description="Continuous DiD — 1772 baseline")
    parser.add_argument("--config", type=Path, default=ROOT / "settings.json")
    parser.add_argument(
        "--set", action="append", type=assignment, default=[], metavar="KEY=JSON"
    )
    parser.add_argument(
        "--grid", action="append", type=assignment, default=[], metavar="KEY=[VALUES]"
    )
    parser.add_argument("--drop-gps", nargs="+", default=[], metavar="VARIABLE")
    parser.add_argument("--drop-mu", nargs="+", default=[], metavar="VARIABLE")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Preview with smaller fitting budgets; changes the configuration",
    )
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    args = parser.parse_args(argv)
    values = Config.load(args.config).to_dict()
    if args.quick:
        values.update(
            label="preview",
            n_splits=2,
            inner_splits=2,
            n_repeats=1,
            mu_search_n_iter=0,
            mu_params={"n_estimators": 40, "max_depth": 2, "learning_rate": 0.1},
            n_bootstrap=399,
            diff_step=0.005,
        )
    values.update(dict(args.set))
    for field, removed in [
        ("gps_features", args.drop_gps),
        ("mu_features", args.drop_mu),
    ]:
        unknown = set(removed) - set(values[field])
        if unknown:
            parser.error(f"Variables absent from {field}: {sorted(unknown)}")
        values[field] = [name for name in values[field] if name not in removed]
    keys = [key for key, _ in args.grid]
    if len(keys) != len(set(keys)):
        parser.error("Each grid key must occur once")
    for key, choices in args.grid:
        if not isinstance(choices, list) or not choices:
            parser.error(f"Grid {key} must be a nonempty JSON list")
    combinations = itertools.product(*(choices for _, choices in args.grid))
    # Validate the complete plan before any expensive fit or output mutation.
    try:
        configs = [
            Config.from_dict({**values, **dict(zip(keys, combo))})
            for combo in combinations
        ]
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    if len({c.fingerprint for c in configs}) != len(configs):
        parser.error("The grid contains duplicate configurations")
    args.out.mkdir(parents=True, exist_ok=True)
    results, attempts = [], []
    for i, config in enumerate(configs, 1):
        folder = args.out / f"{config.label}_{config.fingerprint}"
        print(f"\n[{i}/{len(configs)}] {config.label} → {folder}", flush=True)
        attempt = {
            "config_id": config.fingerprint,
            "output": str(folder.resolve()),
            "configuration": config.to_dict(),
        }
        try:
            table = run(config, folder)
            attempt["status"] = "completed"
            results.append(
                table.loc[table.effect.isin(PRIMARY)].assign(
                    config_id=config.fingerprint, label=config.label
                )
            )
        except Exception as exc:
            attempt.update(status="failed", error=str(exc))
            print(f"Failed: {exc}", flush=True)
        attempts.append(attempt)
        # Save every attempted setting and all completed results after each case.
        write_json(
            args.out / "attempts.json", {"planned": len(configs), "attempts": attempts}
        )
        if results:
            combined = pd.concat(results, ignore_index=True)
            family_size = sum(4 * len(c.years) for c in configs)
            combined["p_holm_batch"] = holm(combined.p_value, family_size)
            combined.to_csv(args.out / "comparison.csv", index=False)
    return 1 if any(a["status"] == "failed" for a in attempts) else 0


if __name__ == "__main__":
    raise SystemExit(main())
