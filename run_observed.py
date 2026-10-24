"""Run psmethod=1 ACRT directly at observed doses; default configuration is 1538."""

import argparse
import json
from pathlib import Path

from paper_did.config import PaperConfig
from paper_did.pipeline import ROOT
from ps1_observed import run
from run import assignment


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / "observed_settings.json")
    p.add_argument("--set", action="append", type=assignment, default=[], metavar="KEY=JSON")
    p.add_argument("--support", choices=["supported", "strict"], default="supported")
    p.add_argument("--out", type=Path, default=ROOT / "results/ps1_observed")
    args = p.parse_args(argv)
    try:
        values = json.loads(args.config.read_text(encoding="utf-8"))
        values.update(dict(args.set))
        c = PaperConfig(**values)
        table = run(c, args.out, args.support)
    except (OSError, ValueError, TypeError) as e:
        p.error(str(e))
    print(table[["year", "effect", "estimate", "se", "p_value", "coverage", "status"]].to_string(index=False))
    print(f"Results: {args.out.resolve()}")
    return 0 if table.status.eq("completed").all() else 2


if __name__ == "__main__":
    raise SystemExit(main())
