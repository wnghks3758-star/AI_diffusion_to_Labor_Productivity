"""Frozen 200-specification exploration with early gates and selective retention.

Rejected estimates, logs and fitted models are temporary and deleted. Only aggregate
rejection counts survive; the pre-generated configuration plan remains reproducible.
"""

import argparse
import contextlib
import json
import os
import shutil
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from labor_did.config import INDUSTRY_CONTROLS
from labor_did.data import load_data, prepare_year
from labor_did.io import file_hash, versions, write_json
from paper_did.config import PaperConfig
from paper_did.estimation import PRIMARY, compute_year
from paper_did.models import source_hash
from paper_did.pipeline import ROOT, plots, prepare

SEED = 2026102303
DEST = ROOT / "audit/results/paper_search_200"
ESTIMAND = "finite_difference_ACRT"
SCREEN_ORDER = (2023, 2020, 2021, 2024, 2025)
CORE = [
    "pre_all_nonsignificant",
    "initial_significant_decline",
    "recovery",
    "sme_crosses_positive_earlier",
    "moderation_negative_significant",
]


def primary(table):
    return table[table.estimand.eq(ESTIMAND) & table.effect.isin(PRIMARY)].copy()


def gate(table, year, alpha=0.05):
    """Require all four valid pre-period tests, then negative/significant 2023 total."""
    part = primary(table)
    if year in (2020, 2021):
        if len(part) != 4 or set(part.effect) != set(PRIMARY):
            return "numerical_failure"
        if not np.isfinite(part[["estimate", "p_value"]].to_numpy()).all():
            return "numerical_failure"
        return None if part.p_value.ge(alpha).all() else f"pre_{year}"
    row = part[part.effect.eq("total")]
    if len(row) != 1 or not np.isfinite(row[["estimate", "p_value"]].to_numpy()).all():
        return "numerical_failure"
    r = row.iloc[0]
    return None if r.estimate < 0 and r.p_value < alpha else "initial_2023"


def evaluate(table, p_column="p_value", alpha=0.05):
    t = primary(table).set_index(["effect", "year"])
    years = [2020, 2021, 2023, 2024, 2025]

    def v(effect, year, column):
        try:
            return float(t.loc[(effect, year), column])
        except (KeyError, TypeError, ValueError):
            return np.nan

    complete = len(t) == 20 and all(
        np.isfinite(v(e, y, k))
        for e in PRIMARY
        for y in years
        for k in ("estimate", p_column)
    )

    def sig(e, y, sign):
        return sign * v(e, y, "estimate") > 0 and v(e, y, p_column) < alpha

    post = [2023, 2024, 2025]
    result = dict(complete=complete, inference_column=p_column)
    result["pre_all_nonsignificant"] = complete and all(
        v(e, y, p_column) >= alpha for e in PRIMARY for y in [2020, 2021]
    )
    result["initial_significant_decline"] = complete and sig("total", 2023, -1)
    result["recovery"] = complete and all(
        all(np.diff([v(e, y, "estimate") for y in post]) >= 0)
        and v(e, 2025, "estimate") > v(e, 2023, "estimate")
        for e in ["total", "sme", "large"]
    )

    def crossing(e):
        return next((y for y in post if v(e, y, "estimate") > 0), None)

    sme, large = crossing("sme"), crossing("large")
    result.update(sme_first_positive_year=sme, large_first_positive_year=large)
    result["sme_crosses_positive_earlier"] = (
        complete and sme is not None and (large is None or sme < large)
    )
    result["sme_2025_significantly_positive"] = complete and sig("sme", 2025, 1)
    persistent = all(sig("moderation", y, -1) for y in post)
    strengthening = sig("moderation", 2025, -1) and (
        v("moderation", 2025, p_column) <= v("moderation", 2023, p_column)
    )
    result["moderation_negative_significant"] = (
        complete
        and all(v("moderation", y, "estimate") < 0 for y in post)
        and (persistent or strengthening)
    )
    result["pattern_match"] = all(result[k] for k in CORE)
    result["preferred_match"] = (
        result["pattern_match"] and result["sme_2025_significantly_positive"]
    )
    result["criteria_passed"] = sum(
        bool(result[k]) for k in CORE + ["sme_2025_significantly_positive"]
    )
    result["sme_2025_estimate"] = v("sme", 2025, "estimate")
    result["sme_2025_p"] = v("sme", 2025, p_column)
    return result


def make_plan(dest=DEST):
    dest = Path(dest)
    path = dest / "plan.json"
    if path.exists():
        raise FileExistsError("Frozen plan already exists")
    base = PaperConfig().to_dict()
    families = [("baseline", base)]
    for field in ["gps_features", "mu_industry_features", "mu_firm_features"]:
        for removed in base[field]:
            families.append(
                (
                    f"remove:{field}:{removed}",
                    {**base, field: [x for x in base[field] if x != removed]},
                )
            )
    assert len(families) == 31
    rng = np.random.default_rng(SEED)
    models = [
        ("random_forest", "xgboost"),
        ("lasso", "xgboost"),
        ("ridge", "xgboost"),
        ("linear", "xgboost"),
        ("xgboost", "xgboost"),
        ("spline", "random_forest"),
        ("spline", "lasso"),
        ("spline", "ridge"),
        ("spline", "spline"),
    ]
    for i, (gps, mu) in enumerate(models):
        selected = {}
        for field, pool in [
            ("gps_features", INDUSTRY_CONTROLS),
            ("mu_industry_features", INDUSTRY_CONTROLS),
            ("mu_firm_features", base["mu_firm_features"]),
        ]:
            selected[field] = [x for x in pool if rng.random() < 0.7]
            if not selected[field]:
                selected[field] = [pool[0]]
        params = lambda model: (
            {"n_estimators": 150, "max_depth": 3, "min_samples_leaf": 8}
            if model == "random_forest"
            else {"alpha": 0.01}
            if model == "lasso"
            else {"alpha": 1.0}
            if model == "ridge"
            else {}
        )
        families.append(
            (
                f"individual_subset_{i + 1}",
                {
                    **base,
                    **selected,
                    "gps_model": gps,
                    "mu_model": mu,
                    "gps_params": params(gps),
                    "mu_params": params(mu),
                    "n_splits": [3, 4, 5][i % 3],
                    "seed": [1, 42, 2026][i % 3],
                    "spline_n_knots": [3, 4, 5][i % 3],
                    "spline_degree": 2 if i % 2 else 3,
                    "industry_min_obs": [2, 4, 8][i % 3],
                    "industry_min_groups": [2, 4, 6][i % 3],
                    "include_industry_dummies": i != 3,
                    "industry_scheme": "major" if i == 4 else "refined",
                    "include_firm_controls": i != 7,
                },
            )
        )
    variants = [(0.7, 0.1), (0.45, 0.1), (0.55, 0.05), (0.85, 0.1), (1.0, 0.15)]
    entries = []
    for f, (name, values) in enumerate(families):
        for j, (bandwidth, trim) in enumerate(variants):
            n = len(entries) + 1
            c = PaperConfig(
                **{
                    **values,
                    "bandwidth_factor": bandwidth,
                    "trim": trim,
                    "label": f"paper_search_{n:03d}",
                }
            )
            entries.append(
                dict(
                    number=n,
                    family=f,
                    family_name=name,
                    variant=j,
                    config=c.to_dict(),
                    fingerprint=c.fingerprint,
                )
            )
    assert len(entries) == 200 and len({e["fingerprint"] for e in entries}) == 200
    plan = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        seed=SEED,
        count=200,
        source_sha256=source_hash(),
        runner_sha256=file_hash(__file__),
        data_sha256=file_hash(ROOT / base["data_path"]),
        versions=versions(),
        hypothesis_estimand=ESTIMAND,
        alpha=0.05,
        correction_family_size=4000,
        stop_rule="Each specification stops at first failed gate; screen 2023 total decline, then pre-2020/pre-2021; all 200 are attempted.",
        screening_order=list(SCREEN_ORDER),
        retention="Only gate-passed results retained. Rejected estimates/logs/models deleted; aggregate counts and predetermined plan retained.",
        criteria=dict(
            pre="All 4 effects in 2020 and 2021 p>=.05",
            initial="2023 total estimate<0 and p<.05",
            recovery="2023<=2024<=2025 and 2025>2023 for total/SME/Large",
            crossing="First positive SME year precedes Large (Large may never cross)",
            moderation="Large-SME negative in all post years; significant all years OR significant in 2025 with p2025<=p2023",
            preferred="SME2025 estimate>0 and p<.05 in addition to full pattern",
        ),
        entries=entries,
    )
    write_json(path, plan)
    return plan


def run_family(entries, dest):
    """Temporary annual results and shared nuisance cache disappear on rejection."""
    os.chdir(ROOT)
    dest = Path(dest)
    scratch = dest / ".scratch"
    scratch.mkdir(exist_ok=True)
    counts = Counter()
    retained = []
    with tempfile.TemporaryDirectory(prefix="family_", dir=scratch) as tmp:
        folder = Path(tmp)
        with (
            (folder / "worker.log").open("w") as log,
            contextlib.redirect_stdout(log),
            contextlib.redirect_stderr(log),
        ):
            first = entries[0]["config"]
            raw, _ = load_data(str(ROOT / first["data_path"]), str(folder / "cache"))
            for entry in entries:
                c = PaperConfig(**entry["config"])
                c = replace(
                    c,
                    data_path=str(ROOT / c.data_path),
                    cache_dir=str(folder / "cache"),
                )
                work = folder / c.label
                tables, audits = [], []
                try:
                    panel, sample, legacy = prepare(raw, c)
                    reject = None
                    for year in [y for y in SCREEN_ORDER if y in c.years]:
                        table, _, audit = compute_year(
                            prepare_year(panel, year, legacy),
                            year,
                            c,
                            work / "years" / str(year),
                        )
                        tables.append(table)
                        audits.append(audit)
                        if year in [2020, 2021, 2023]:
                            reject = gate(table, year, c.alpha)
                            if reject:
                                break
                    if reject:
                        counts[reject] += 1
                    else:
                        combined = pd.concat(tables, ignore_index=True)
                        combined["p_bonferroni_search"] = np.where(
                            combined.estimand.eq(ESTIMAND)
                            & combined.effect.isin(PRIMARY),
                            np.minimum(1.0, combined.p_value * 4000),
                            np.nan,
                        )
                        combined.to_csv(work / "estimates.csv", index=False)
                        primary(combined).to_csv(work / "summary.csv", index=False)
                        write_json(work / "config.json", entry["config"])
                        write_json(work / "sample.json", sample)
                        write_json(
                            work / "audit.json", dict(sample=sample, years=audits)
                        )
                        write_json(work / "pattern_flags.json", evaluate(combined))
                        write_json(
                            work / "adjusted_flags.json",
                            evaluate(combined, "p_bonferroni_search"),
                        )
                        plots(combined, work, c)
                        destination = dest / "retained" / c.label
                        destination.parent.mkdir(exist_ok=True)
                        if destination.exists():
                            shutil.rmtree(destination)
                        shutil.move(str(work), destination)
                        counts["retained"] += 1
                        retained.append(c.label)
                except Exception:
                    counts["execution_error"] += 1
                finally:
                    if work.exists():
                        shutil.rmtree(work)
    return dict(family=entries[0]["family"], counts=dict(counts), retained=retained)


def search(dest=DEST, workers=3):
    dest = Path(dest)
    plan = json.loads((dest / "plan.json").read_text())
    assert plan["source_sha256"] == source_hash(), "Numerical code changed after plan"
    assert plan["runner_sha256"] == file_hash(__file__), "Runner changed after plan"
    assert plan["data_sha256"] == file_hash(
        ROOT / plan["entries"][0]["config"]["data_path"]
    )
    progress_path = dest / "progress.json"
    progress = (
        json.loads(progress_path.read_text())
        if progress_path.exists()
        else dict(completed_families=[], counts={}, retained=[])
    )
    families = {}
    for entry in plan["entries"]:
        if entry["family"] not in progress["completed_families"]:
            families.setdefault(entry["family"], []).append(entry)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = [
            pool.submit(run_family, entries, str(dest)) for entries in families.values()
        ]
        for future in as_completed(pending):
            result = future.result()
            progress["completed_families"].append(result["family"])
            counts = Counter(progress["counts"])
            counts.update(result["counts"])
            progress["counts"] = dict(counts)
            progress["retained"] = sorted(
                set(progress["retained"] + result["retained"])
            )
            progress["attempted"] = sum(counts.values())
            progress["updated_utc"] = datetime.now(timezone.utc).isoformat()
            progress["status"] = (
                "completed" if progress["attempted"] == 200 else "running"
            )
            write_json(progress_path, progress)
            print(
                json.dumps(
                    {k: v for k, v in progress.items() if k != "completed_families"}
                ),
                flush=True,
            )
    shutil.rmtree(dest / ".scratch", ignore_errors=True)
    return progress


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["plan", "run"])
    parser.add_argument("--out", type=Path, default=DEST)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if args.command == "plan":
        p = make_plan(args.out)
        print(f"Frozen {p['count']} configurations: {args.out}")
    else:
        search(args.out, args.workers)
