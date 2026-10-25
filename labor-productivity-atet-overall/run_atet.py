"""Run overall ATET against fixed reference doses from any working directory."""
import argparse
import json
from pathlib import Path

from atet_did.config import ATETConfig
from atet_did.pipeline import ROOT, run


def assignment(value):
    try:
        key, raw = value.split("=", 1)
        return key, json.loads(raw)
    except (ValueError, json.JSONDecodeError) as exc:
        raise argparse.ArgumentTypeError("Use KEY=JSON") from exc


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "atet_settings.json")
    parser.add_argument("--set", action="append", type=assignment, default=[], metavar="KEY=JSON")
    parser.add_argument("--support", choices=["strict", "supported"], default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        values = json.loads(args.config.read_text(encoding="utf-8-sig"))
        values.update(dict(args.set))
        if args.support is not None:
            values["support_policy"] = args.support
        table, out = run(ATETConfig(**values), args.out)
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    print(table[["reference_percent", "year", "effect", "estimate", "se", "p_value", "coverage", "status"]].to_string(index=False))
    print(f"Results: {out}")
    return 0 if table.status.eq("completed").all() else 2


if __name__ == "__main__":
    raise SystemExit(main())
