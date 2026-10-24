"""Run one paper-aligned specification or an explicit, fully recorded grid."""

import argparse
import itertools
import json
from pathlib import Path

import pandas as pd

from labor_did.io import write_json
from paper_did.config import PaperConfig
from paper_did.pipeline import ROOT, run
from run import assignment


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / "paper_settings.json")
    p.add_argument(
        "--set", action="append", type=assignment, default=[], metavar="KEY=JSON"
    )
    p.add_argument(
        "--grid", action="append", type=assignment, default=[], metavar="KEY=[VALUES]"
    )
    p.add_argument("--out", type=Path)
    args = p.parse_args(argv)
    try:
        values = json.loads(args.config.read_text())
        values.update(dict(args.set))
        keys = [k for k, _ in args.grid]
        if len(keys) != len(set(keys)):
            raise ValueError("Each grid key must occur once")
        if any(not isinstance(v, list) or not v for _, v in args.grid):
            raise ValueError("Each grid needs a nonempty JSON list")
        configs = [
            PaperConfig(**{**values, **dict(zip(keys, combo))})
            for combo in itertools.product(*(v for _, v in args.grid))
        ]
        if len({c.fingerprint for c in configs}) != len(configs):
            raise ValueError("Duplicate grid configurations")
    except (ValueError, TypeError) as e:
        p.error(str(e))
    base = args.out or ROOT / "results/paper_comparison"
    attempts = []
    tables = []
    for c in configs:
        out = base / f"{c.label}_{c.fingerprint}" if args.grid else args.out
        attempt = dict(configuration=c.to_dict(), output=str(out) if out else None)
        try:
            result = run(c, out)
            attempt["status"] = (
                "completed"
                if result.status.eq("completed").all()
                else "contains_failed_estimates"
            )
            tables.append(result.assign(config_id=c.fingerprint))
            print(
                result.loc[
                    result.effect.isin(["total", "sme", "large", "moderation"]),
                    [
                        "year",
                        "estimand",
                        "effect",
                        "estimate",
                        "se",
                        "p_value",
                        "status",
                    ],
                ].to_string(index=False)
            )
        except Exception as exc:
            attempt.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            print(attempt["error"], flush=True)
        attempts.append(attempt)
        if args.grid:
            base.mkdir(parents=True, exist_ok=True)
            write_json(
                base / "attempts.json", dict(planned=len(configs), attempts=attempts)
            )
            if tables:
                pd.concat(tables, ignore_index=True).to_csv(
                    base / "comparison.csv", index=False
                )
    return int(any(a["status"] != "completed" for a in attempts))


if __name__ == "__main__":
    raise SystemExit(main())
