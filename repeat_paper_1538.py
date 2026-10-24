"""Predeclared three-case sensitivity comparison with 50 repeated industry splits."""

import argparse
import contextlib
import json
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from labor_did.data import load_data, prepare_year
from labor_did.io import file_hash, write_json
from paper_did.config import PaperConfig
from paper_did.estimation import compute_year
from paper_did.models import source_hash
from paper_did.pipeline import ROOT, prepare
from search_paper import SCREEN_ORDER, evaluate, gate, primary

PARENT = ROOT / "audit/results/paper_search_1501_1700/retained/paper_search_1538"
DEST = ROOT / "results/paper_1538_sensitivity"
KEYS = ["year", "estimand", "pair_id", "effect"]
PANEL = LEGACY = BASE = WORK = None


def median_repeats(tables, alpha=.05):
    """Median point; sqrt(median(SE_r^2 + (theta_r - median(theta))^2))."""
    if not tables:
        raise ValueError("At least one repetition is required")
    first = tables[0].reset_index(drop=True)
    keys = first[KEYS]
    if keys.duplicated().any() or any(not t.reset_index(drop=True)[KEYS].equals(keys) for t in tables):
        raise ValueError("Repetition rows must align exactly; missing/duplicate effects cannot be dropped")
    out = first.copy()
    points = np.stack([t.estimate.to_numpy() for t in tables])
    errors = np.stack([t.se.to_numpy() for t in tables])
    valid = np.isfinite(points).all(axis=0) & np.isfinite(errors).all(axis=0)
    center = np.median(points, axis=0)
    variance = np.median(errors**2 + (points - center)**2, axis=0)
    center[~valid] = np.nan
    se = np.sqrt(variance)
    se[~valid] = np.nan
    z = norm.ppf(1 - alpha / 2)
    out["estimate"], out["se"] = center, se
    out["p_value"] = 2 * norm.sf(np.divide(np.abs(center), se, out=np.full_like(center, np.nan), where=se > 0))
    out["ci_lower"], out["ci_upper"] = center - z * se, center + z * se
    out["status"] = np.where(valid, "completed", "failed")
    out["error"] = np.where(valid, "", "At least one repetition failed; no repetition was discarded")
    for col in ["bootstrap_se", "percentile_lower", "percentile_upper"]:
        if col in out and len(tables) > 1:
            out[col] = np.nan
    for col in ["coverage", "n_used_firms", "n_in_range_firms"]:
        if col in out:
            out[col] = np.min(np.stack([t[col].to_numpy() for t in tables]), axis=0)
    out["repetitions"] = len(tables)
    out["aggregation"] = "median_with_split_variance"
    out["split_sd"] = np.std(points, axis=0, ddof=1) if len(tables) > 1 else 0.
    out["split_min"], out["split_max"] = np.min(points, axis=0), np.max(points, axis=0)
    return out


def make_plan(dest=DEST, parent=PARENT):
    dest, parent = Path(dest), Path(parent)
    if (dest / "plan.json").exists():
        raise FileExistsError("Frozen sensitivity plan already exists")
    original = json.loads((parent / "config.json").read_text())
    c = PaperConfig(**original)
    seeds = {}
    for role in ["gps", "mu"]:
        base = c.fold_settings(role)[1]
        rng = np.random.default_rng(np.random.SeedSequence([20261024, 1538, base]))
        seeds[role] = [base] + rng.choice(2**31 - 1, size=49, replace=False).astype(int).tolist()
        assert len(set(seeds[role])) == 50
    entries = []
    for label, step, repeats in [("diff_1e5", 1e-5, 1), ("repeat_50", c.diff_step, 50), ("both", 1e-5, 50)]:
        cfg = replace(c, diff_step=step, label=f"paper_1538_{label}").to_dict()
        entries.append(dict(label=label, config=cfg, repetitions=repeats))
    plan = dict(created_utc=datetime.now(timezone.utc).isoformat(), parent_path=str(parent.relative_to(ROOT)),
                parent_config_sha256=file_hash(parent / "config.json"), parent_estimates_sha256=file_hash(parent / "estimates.csv"),
                source_sha256=source_hash(), runner_sha256=file_hash(__file__), data_sha256=file_hash(ROOT / c.data_path),
                entries=entries, seeds=seeds, screening_order=list(SCREEN_ORDER),
                aggregation="median_with_split_variance", standard_error="sqrt(median(se_r**2 + (theta_r - median(theta))**2))",
                repeat_design="Same original 4/3 industry fold counts; role-specific outer split seeds vary; model random states fixed; repetition 0 is original 1538",
                correction_family_size=60, cumulative_correction_family_size=34060,
                retention="Apply gates to the combined estimator, never select individual repetitions. Delete rejected case estimates and stop later years. Retain only status counts and frozen settings for rejected cases.")
    write_json(dest / "plan.json", plan)
    return plan


def init_worker(panel, legacy, base, work):
    global PANEL, LEGACY, BASE, WORK
    PANEL, LEGACY, BASE, WORK = panel, legacy, base, Path(work)


def repeat_year(task):
    repeat, year, steps, gps_seed, mu_seed = task
    root = WORK / f"rep_{repeat:02d}"
    cache = root / "cache"
    frame = prepare_year(PANEL, year, LEGACY)
    result = {}
    root.mkdir(parents=True, exist_ok=True)
    with (root / "worker.log").open("a") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        for step in steps:
            c = replace(BASE, diff_step=step, gps_fold_seed=gps_seed, mu_fold_seed=mu_seed, cache_dir=str(cache))
            folder = root / f"step_{step:g}" / "years" / str(year)
            marker = folder / "completed.json"
            if marker.exists():
                table = pd.read_csv(folder / "estimates.csv")
            else:
                table, _, _ = compute_year(frame, year, c, folder)
                # The dense support grid is reconstructed from settings; compact interval audit is retained.
                (folder / "computability.csv").unlink(missing_ok=True)
                write_json(marker, dict(repetition=repeat, year=year, diff_step=step, gps_fold_seed=gps_seed, mu_fold_seed=mu_seed))
            result[step] = table
    shutil.rmtree(cache / "nuisance", ignore_errors=True)
    return repeat, result


def run(dest=DEST, workers=3):
    dest = Path(dest)
    plan = json.loads((dest / "plan.json").read_text())
    assert plan["source_sha256"] == source_hash() and plan["runner_sha256"] == file_hash(__file__)
    parent = ROOT / plan["parent_path"]
    assert file_hash(parent / "config.json") == plan["parent_config_sha256"]
    assert file_hash(parent / "estimates.csv") == plan["parent_estimates_sha256"]
    base = PaperConfig(**json.loads((parent / "config.json").read_text()))
    assert file_hash(ROOT / base.data_path) == plan["data_sha256"]
    work = dest / ".scratch"
    raw, _ = load_data(str(ROOT / base.data_path), str(work / "input"))
    panel, sample, legacy = prepare(raw, base)
    del raw
    progress_file = dest / "progress.json"
    progress = json.loads(progress_file.read_text()) if progress_file.exists() else dict(cases={}, completed_years=[], completed_repetitions={})
    if progress.get("status") == "completed":
        return progress
    retained = dest / "retained"
    retained.mkdir(exist_ok=True)
    active = {e["label"]: e for e in plan["entries"] if progress["cases"].get(e["label"], {}).get("status") not in ["rejected", "completed"]}
    with ProcessPoolExecutor(max_workers=workers, initializer=init_worker, initargs=(panel, legacy, base, work)) as pool:
        for year in plan["screening_order"]:
            if year in progress["completed_years"] or not active:
                continue
            results = {label: {} for label in active}
            tasks = []
            for rep in range(max(e["repetitions"] for e in active.values())):
                steps = sorted({e["config"]["diff_step"] for e in active.values() if rep < e["repetitions"]})
                tasks.append((rep, year, steps, plan["seeds"]["gps"][rep], plan["seeds"]["mu"][rep]))
            done = 0
            for future in as_completed([pool.submit(repeat_year, task) for task in tasks]):
                rep, data = future.result()
                done += 1
                for label, entry in active.items():
                    if rep < entry["repetitions"]:
                        results[label][rep] = data[entry["config"]["diff_step"]]
                progress["completed_repetitions"][str(year)] = done
                progress["status"] = "running"
                write_json(progress_file, progress)
                print(json.dumps(dict(year=year, completed=done, planned=len(tasks))), flush=True)
            rejected = []
            for label, entry in active.items():
                n = entry["repetitions"]
                assert set(results[label]) == set(range(n))
                aggregate = median_repeats([results[label][rep] for rep in range(n)], base.alpha)
                reject = gate(aggregate, year) if year in [2023, 2020, 2021] else None
                folder = retained / label
                if reject:
                    shutil.rmtree(folder, ignore_errors=True)
                    progress["cases"][label] = dict(status="rejected", reason=reject, stopped_year=year, repetitions=n)
                    rejected.append(label)
                else:
                    folder.mkdir(exist_ok=True)
                    write_json(folder / "config.json", entry["config"])
                    write_json(folder / "sample.json", sample)
                    write_json(folder / "repeat_design.json", dict(repetitions=n, seeds={r: plan["seeds"][r][:n] for r in ["gps", "mu"]}, aggregation=plan["aggregation"]))
                    target = folder / "years" / str(year)
                    target.mkdir(parents=True, exist_ok=True)
                    aggregate.to_csv(target / "estimates.csv", index=False)
                    for rep in range(n):
                        src = work / f"rep_{rep:02d}" / f"step_{entry['config']['diff_step']:g}" / "years" / str(year)
                        dst = folder / "repetitions" / f"rep_{rep:02d}" / "years" / str(year)
                        shutil.copytree(src, dst, dirs_exist_ok=True)
                    progress["cases"][label] = dict(status="running", repetitions=n, last_year=year)
                print(json.dumps(dict(case=label, year=year, status=progress["cases"][label])), flush=True)
            for label in rejected:
                del active[label]
            progress["completed_years"].append(year)
            write_json(progress_file, progress)
            # Surviving cases have their compact diagnostic copies; rejected estimates disappear here.
            for rep in range(50):
                root = work / f"rep_{rep:02d}"
                if root.exists():
                    for path in root.glob("step_*"):
                        shutil.rmtree(path)
                    (root / "worker.log").unlink(missing_ok=True)
    for label, entry in active.items():
        folder = retained / label
        table = pd.concat([pd.read_csv(folder / "years" / str(y) / "estimates.csv") for y in base.years], ignore_index=True)
        mask = table.estimand.eq("finite_difference_ACRT") & table.effect.isin(["total", "sme", "large", "moderation"])
        table["p_bonferroni_sensitivity"] = np.where(mask, np.minimum(1., table.p_value * 60), np.nan)
        table["p_bonferroni_cumulative"] = np.where(mask, np.minimum(1., table.p_value * 34060), np.nan)
        table.to_csv(folder / "estimates.csv", index=False)
        primary(table).to_csv(folder / "summary.csv", index=False)
        write_json(folder / "pattern_flags.json", evaluate(table))
        progress["cases"][label] = dict(status="completed", repetitions=entry["repetitions"])
    progress["status"] = "completed"
    write_json(progress_file, progress)
    shutil.rmtree(work, ignore_errors=True)
    return progress


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["plan", "run"])
    parser.add_argument("--out", type=Path, default=DEST)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    print(json.dumps(make_plan(args.out) if args.command == "plan" else run(args.out, args.workers), ensure_ascii=False))
