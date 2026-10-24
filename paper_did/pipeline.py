"""Reproducible one-specification run with annual checkpoints and complete outputs."""

import hashlib
import html
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd

from labor_did import Config as LegacyConfig
from labor_did.data import load_data, prepare_panel, prepare_year
from labor_did.io import versions, write_json
from labor_did.pipeline import holm

from .config import PaperConfig
from .estimation import PRIMARY, compute_year
from .models import source_hash
from .size import classify_size

ROOT = Path(__file__).resolve().parents[1]


def prepare(raw, c):
    # Reuse only the established balanced sample/size classification/industry encoding.
    # No legacy GPS, ESS scan, bandwidth or density-floor calculation enters estimation.
    legacy = LegacyConfig(
        data_path=c.data_path,
        base_year=c.base_year,
        years=c.years,
        dose=c.dose,
        outcome=c.outcome,
        dose_transform=c.dose_transform,
        covariate_time=c.covariate_time,
        pre_sign=c.pre_sign,
        gps_features=c.gps_features,
        mu_features=c.mu_features,
        include_industry_dummies=c.include_industry_dummies,
        industry_scheme=c.industry_scheme,
        industry_min_groups=c.industry_min_groups,
        industry_min_obs=c.industry_min_obs,
        min_ess_prime=0,
        mu_model="xgboost",
    )
    panel, sample = prepare_panel(raw, legacy)
    panel, size_audit = classify_size(panel, c)
    sample["size_classification"] = size_audit
    if c.size_missing_policy == "unclassified":
        sample["size_missing_handling"] = "Missing size observations remain in total and are unclassified for size contrasts"
    sample["covariate_time"] = c.covariate_time
    sample["mu_includes_firm_controls"] = c.include_firm_controls
    sample["missing_policy"] = c.missing_policy
    return panel, sample, legacy


def plots(table, out, c):
    from labor_did.plot import plt

    for estimand in table.estimand.unique():
        part = table[table.estimand.eq(estimand) & table.effect.isin(PRIMARY)]
        for pair_id, sub in part.groupby("pair_id"):
            fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
            for ax, effects in zip(axes, [["total"], ["sme", "large"], ["moderation"]]):
                for effect in effects:
                    g = sub[sub.effect.eq(effect)].sort_values("year")
                    g = pd.concat(
                        [
                            g,
                            pd.DataFrame(
                                [
                                    dict(
                                        year=c.base_year,
                                        estimate=0.0,
                                        ci_lower=0.0,
                                        ci_upper=0.0,
                                    )
                                ]
                            ),
                        ]
                    ).sort_values("year")
                    ax.errorbar(
                        g.year.to_numpy(),
                        g.estimate.to_numpy(),
                        yerr=[
                            (g.estimate - g.ci_lower).to_numpy(),
                            (g.ci_upper - g.estimate).to_numpy(),
                        ],
                        fmt="o-",
                        capsize=3,
                        label=effect,
                    )
                ax.axhline(0, color="gray", ls="--", lw=0.8)
                ax.axvline(c.base_year, color="gray", ls=":", lw=0.8)
                ax.set(xlabel="Year", ylabel=estimand)
                ax.legend()
                ax.grid(alpha=0.2)
            fig.suptitle(f"{c.label} | {estimand} | baseline {c.base_year}")
            fig.savefig(out / f"{estimand}_{pair_id}.png", dpi=160)
            plt.close(fig)


def run(config=None, output_dir=None):
    c = config if isinstance(config, PaperConfig) else PaperConfig(**(config or {}))
    c = replace(
        c,
        data_path=str((ROOT / c.data_path).resolve()),
        cache_dir=str((ROOT / c.cache_dir).resolve()),
    )
    out = Path(output_dir or ROOT / "results" / f"{c.label}_{c.fingerprint}")
    out.mkdir(parents=True, exist_ok=True)
    raw, digest = load_data(c.data_path, c.cache_dir)
    code = source_hash()
    run_key = hashlib.sha256(
        json.dumps([c.to_dict(), digest, code, versions()], sort_keys=True).encode()
    ).hexdigest()
    status_file = out / "status.json"
    if status_file.exists():
        status = json.loads(status_file.read_text())
        if status.get("run_key") != run_key:
            raise FileExistsError(
                "Output belongs to different code/settings; choose another --out"
            )
        if status.get("status") == "completed":
            return pd.read_csv(out / "estimates.csv")
    write_json(out / "config.json", c.to_dict())
    status = dict(
        status="running",
        run_key=run_key,
        source_sha256=code,
        data_sha256=digest,
        versions=versions(),
    )
    write_json(status_file, status)
    tables = []
    audits = []
    try:
        panel, sample, legacy = prepare(raw, c)
        del raw
        write_json(out / "sample.json", sample)
        print(
            f"{c.label}: {sample['analysis_firms']} firms / {sample['analysis_industries']} industries",
            flush=True,
        )
        for year in c.years:
            folder = out / "years" / str(year)
            marker = folder / "completed.json"
            if (
                marker.exists()
                and json.loads(marker.read_text()).get("run_key") == run_key
            ):
                t = pd.read_csv(folder / "estimates.csv")
                a = json.loads((folder / "audit.json").read_text())
                print(f"  {year}: checkpoint", flush=True)
            else:
                frame = prepare_year(panel, year, legacy)
                t, _, a = compute_year(frame, year, c, folder)
                write_json(marker, dict(run_key=run_key))
            tables.append(t)
            audits.append(a)
            pd.concat(tables, ignore_index=True).to_csv(
                out / "estimates.partial.csv", index=False
            )
        table = pd.concat(tables, ignore_index=True)
        primary = table.effect.isin(PRIMARY)
        table.loc[primary, "p_holm_within_run"] = holm(
            table.loc[primary, "p_value"], int(primary.sum())
        )
        table.to_csv(out / "estimates.csv", index=False)
        table[primary].to_csv(out / "summary.csv", index=False)
        write_json(out / "audit.json", dict(sample=sample, years=audits))
        plots(table, out, c)
        image_names = [p.name for p in out.glob("*.png")]
        intro = (
            "ATET follows the author score; finite_difference_ACRT is an additional numerical derivative and firm-weighted averaging extension. ESS is diagnostic only. Baseline controls: "
            + str(c.covariate_time)
        )
        report = (
            '<!doctype html><html lang="en"><meta charset="utf-8"><title>Paper-aligned DiD</title><style>body{font-family:sans-serif;margin:32px}table{border-collapse:collapse;font-size:12px}th,td{padding:6px;border:1px solid #ddd}img{max-width:100%}</style><h1>'
            + html.escape(c.label)
            + "</h1><p>"
            + html.escape(intro)
            + "</p>"
        )
        report += "".join(
            '<img src="' + html.escape(name) + '">' for name in image_names
        ) + table[primary].to_html(index=False)
        report += (
            "<h2>Settings</h2><pre>"
            + html.escape(json.dumps(c.to_dict(), indent=2, ensure_ascii=False))
            + "</pre></html>"
        )
        (out / "report.html").write_text(report)
        status.update(
            status="completed", failed_results=int(table.status.eq("failed").sum())
        )
        write_json(status_file, status)
        print(f"Completed: {out}", flush=True)
        return table
    except Exception as exc:
        status.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write_json(status_file, status)
        raise
