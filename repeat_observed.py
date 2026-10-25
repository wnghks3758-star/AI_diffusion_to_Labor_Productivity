"""Fifty repeated PS1 observed-dose estimates with fixed common averaging targets."""

import argparse
import contextlib
import hashlib
import json
import platform
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from labor_did.data import load_data, prepare_year
from labor_did.io import file_hash, versions, write_json
from labor_did.support import scopes
from observed_acrt import aggregate_results, evaluate_pairs
from paper_did import models
from paper_did.config import PaperConfig
from paper_did.models import columns_for, fit_gps_oof, fit_nuisance
from paper_did.numerics import paper_bandwidth
from paper_did.pipeline import ROOT, prepare
from paper_did.support import computable_grid
from ps1_observed import implementation_hash
from repeat_paper_1538 import median_repeats
from search_paper import evaluate

DEST = ROOT / "results/ps1_observed_repeat50"
FRAMES = BASE = DESTINATION = None


def make_plan(dest=DEST, config_path=None, count=50):
    dest = Path(dest)
    if (dest / "plan.json").exists():
        raise FileExistsError("A frozen repeat plan already exists")
    if count < 2:
        raise ValueError("At least two repetitions are required")
    path = Path(config_path or ROOT / "observed_settings.json")
    c = PaperConfig(**json.loads(path.read_text(encoding="utf-8")))
    if c.covariate_time != "baseline" or c.gps_model != "xgboost" or c.mu_model != "xgboost":
        raise ValueError("This stability experiment requires baseline controls and two XGBoost learners")
    if c.gps_features != c.mu_industry_features:
        raise ValueError("GPS and Mu industry features must match")
    channels = {}
    original = dict(gps_fold=c.fold_settings("gps")[1], mu_fold=c.fold_settings("mu")[1],
                    gps_model=c.gps_params.get("random_state", c.seed), mu_model=c.mu_params.get("random_state", c.seed))
    for i, (role, first) in enumerate(original.items()):
        rng = np.random.default_rng(np.random.SeedSequence([20261024, 1538, 50, i]))
        values = [int(first)]
        while len(values) < count:
            candidate = int(rng.integers(0, 2**31-1))
            if candidate not in values:
                values.append(candidate)
        channels[role] = values
    entries = [dict(repetition=r, seeds={k: v[r] for k, v in channels.items()}) for r in range(count)]
    plan = dict(created_utc=datetime.now(timezone.utc).isoformat(), count=count, config=c.to_dict(), entries=entries,
                data_sha256=file_hash(ROOT / c.data_path), base_source_sha256=implementation_hash(),
                runner_sha256=file_hash(__file__), aggregation_sha256=file_hash(Path(__file__).with_name("repeat_paper_1538.py")),
                aggregation="median_with_split_and_learner_variation", se_formula="sqrt(median(se_r**2 + (theta_r - median(theta))**2))",
                repeat_design="Repetition 0 is original 1538; vary GPS/Mu outer-fold seeds and both learner random states; other settings fixed.",
                common_target="Intersection of computationally supported observed-dose targets across all repetitions, by scope; selected using GPS/kernel/trim only.",
                retention="All repetitions and all five years, including nonsignificant results. No gate screening, interpolation, density floor or ESS cutoff.",
                os=platform.system(), versions=versions())
    write_json(dest / "plan.json", plan)
    pd.DataFrame([dict(repetition=e["repetition"], **e["seeds"]) for e in entries]).to_csv(dest / "seeds.csv", index=False)
    return plan


def init_worker(frames, base, dest):
    global FRAMES, BASE, DESTINATION
    FRAMES, BASE, DESTINATION = frames, base, Path(dest)


def repeat_config(entry):
    seeds = entry["seeds"]
    return replace(BASE, gps_fold_seed=seeds["gps_fold"], mu_fold_seed=seeds["mu_fold"],
                   gps_params={**BASE.gps_params, "random_state": seeds["gps_model"]},
                   mu_params={**BASE.mu_params, "random_state": seeds["mu_model"]},
                   cache_dir=str(DESTINATION / ".scratch" / f"cache_{entry['repetition']:02d}"),
                   label=f"ps1_observed_repeat_{entry['repetition']:02d}")


def support_preflight(entry):
    c = repeat_config(entry)
    frame = FRAMES[c.years[0]]
    ind = frame.groupby("ind_code", as_index=False).first()
    columns = columns_for(frame, c.gps_features, c)
    source = hashlib.sha256(Path(models.__file__).read_bytes() + Path(models.__file__).with_name("numerics.py").read_bytes()).hexdigest()
    gps = fit_gps_oof(ind, columns, c, source)
    mapping = frame.ind_code.map(pd.Series(np.arange(len(ind)), index=ind.ind_code)).to_numpy(int)
    stub = dict(mean=gps["mean"][mapping], variance=gps["variance"][mapping])
    doses = np.sort(frame[c.dose].unique())
    h = paper_bandwidth(ind[c.dose], c.bandwidth_factor)
    flags, _ = computable_grid(frame, doses, h, stub, c)
    available = {scope: sorted(set(sub[c.dose].unique()) & set(doses[flags[scope]])) for scope, sub in scopes(frame).items()}
    result = dict(repetition=entry["repetition"], seeds=entry["seeds"], targets=available, gps_cache_key=gps["cache_key"])
    write_json(DESTINATION / "preflight" / f"rep_{entry['repetition']:02d}.json", result)
    return result


def run_repetition(entry):
    rep = entry["repetition"]
    c = repeat_config(entry)
    out = DESTINATION / "repetitions" / f"rep_{rep:02d}"
    common = json.loads((DESTINATION / "common_targets.json").read_text())["targets"]
    preflight = json.loads((DESTINATION / "preflight" / f"rep_{rep:02d}.json").read_text())
    all_tables = []
    out.mkdir(parents=True, exist_ok=True)
    with (out / "worker.log").open("a") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        write_json(out / "config.json", c.to_dict())
        for year, frame in FRAMES.items():
            folder = out / "years" / str(year)
            marker = folder / "completed.json"
            if marker.exists():
                all_tables.append(pd.read_csv(folder / "estimates.csv"))
                continue
            folder.mkdir(parents=True, exist_ok=True)
            h = paper_bandwidth(frame.groupby("ind_code")[c.dose].first(), c.bandwidth_factor)
            fitted = fit_nuisance(frame, c)
            if fitted["audit"]["gps_cache_key"] != preflight["gps_cache_key"]:
                raise ValueError("GPS inputs changed across years; common baseline support is no longer applicable")
            doses = np.sort(frame[c.dose].unique())
            pairs = [(float(d), float(d-c.diff_step)) for d in doses]
            results = evaluate_pairs(frame, pairs, fitted, c, h)
            native = {}
            diagnostics = []
            for scope, sub in scopes(frame).items():
                native[scope] = []
                for d in sorted(sub[c.dose].unique()):
                    r = results[(float(d), float(d-c.diff_step))][0][scope]
                    okay = r["status"] == "completed" and np.isfinite([r["estimate"], r["se"]]).all()
                    if okay:
                        native[scope].append(d)
                    diagnostics.append(dict(dose=d, scope=scope, target_firms=int(sub.loc[sub[c.dose].eq(d), "firm_code"].nunique()),
                                            in_common=d in common[scope], **{k: r.get(k, np.nan) for k in
                                            ["estimate", "se", "p_value", "status", "error", "ess_treat", "ess_control", "max_industry_weight", "n_trimmed"]}))
                if not set(common[scope]).issubset(native[scope]):
                    raise ValueError("A predeclared common target failed; no repetition or target may be silently dropped")
            tables, scores = [], {}
            for target, selected in [("common", common), ("native", native)]:
                t, v = aggregate_results(frame, results, c, year, h, selected)
                if not np.isfinite(t[["estimate", "se", "p_value"]]).all().all():
                    raise ValueError("A repeat estimate or its inference failed; all failures must be resolved or reported")
                tables.append(t.assign(target=target, repetition=rep))
                scores[target] = v
            table = pd.concat(tables, ignore_index=True)
            table.to_csv(folder / "estimates.csv", index=False)
            pd.DataFrame(diagnostics).to_csv(folder / "endpoints.csv", index=False)
            np.savez_compressed(folder / "cluster_scores.npz", industries=np.sort(frame.ind_code.unique()), **scores)
            write_json(folder / "audit.json", dict(year=year, psmethod=1, evaluation="observed_doses", diff_step=c.diff_step,
                       bandwidth=h, kernel_halfwidth=np.sqrt(5)*h, nuisance=fitted["audit"], seeds=entry["seeds"],
                       density_floor=None, ess_filter=None, interpolation=False))
            write_json(marker, dict(completed=True))
            all_tables.append(table)
            shutil.rmtree(Path(c.cache_dir) / "nuisance", ignore_errors=True)
    combined = pd.concat(all_tables, ignore_index=True)
    combined.to_csv(out / "estimates.csv", index=False)
    (out / "worker.log").unlink(missing_ok=True)
    return rep


def search(dest=DEST, workers=3):
    dest = Path(dest)
    plan = json.loads((dest / "plan.json").read_text())
    assert file_hash(__file__) == plan["runner_sha256"] and implementation_hash() == plan["base_source_sha256"]
    assert file_hash(Path(__file__).with_name("repeat_paper_1538.py")) == plan["aggregation_sha256"]
    c = PaperConfig(**plan["config"])
    assert file_hash(ROOT / c.data_path) == plan["data_sha256"]
    progress_file = dest / "progress.json"
    progress = json.loads(progress_file.read_text()) if progress_file.exists() else dict(status="preflight", completed=[], preflight_completed=[])
    plan_sha256 = file_hash(dest / "plan.json")
    if progress_file.exists() and progress.get("plan_sha256") != plan_sha256:
        raise ValueError("The frozen repeat plan changed; use a new output folder")
    progress["plan_sha256"] = plan_sha256
    if progress["status"] == "completed":
        return progress
    c = replace(c, data_path=str(ROOT / c.data_path))
    raw, _ = load_data(c.data_path, str(dest / ".scratch/input"))
    panel, sample, legacy = prepare(raw, c)
    del raw
    frames = {y: prepare_year(panel, y, legacy).reset_index(drop=True) for y in c.years}
    write_json(dest / "sample.json", sample)
    write_json(progress_file, progress)
    with ProcessPoolExecutor(max_workers=workers, initializer=init_worker, initargs=(frames, c, dest)) as pool:
        if not (dest / "common_targets.json").exists():
            jobs = [pool.submit(support_preflight, e) for e in plan["entries"] if e["repetition"] not in progress["preflight_completed"]]
            for future in as_completed(jobs):
                result = future.result()
                progress["preflight_completed"].append(result["repetition"])
                write_json(progress_file, progress)
                print(f"Support preflight {len(progress['preflight_completed'])}/{plan['count']}", flush=True)
            sets = [json.loads((dest / "preflight" / f"rep_{e['repetition']:02d}.json").read_text())["targets"] for e in plan["entries"]]
            common = {s: sorted(set.intersection(*(set(t[s]) for t in sets))) for s in scopes(frames[c.years[0]])}
            if any(not d for d in common.values()):
                raise ValueError("No shared supported averaging target; cannot compare a common estimand")
            rows = []
            for scope, sub in scopes(frames[c.years[0]]).items():
                used = sub.loc[sub[c.dose].isin(common[scope]), "firm_code"].nunique()
                rows.append(dict(scope=scope, n_target_firms=sub.firm_code.nunique(), n_used_firms=used,
                                 coverage=used/sub.firm_code.nunique(), dose_points=len(common[scope])))
            write_json(dest / "common_targets.json", dict(targets=common, coverage=rows))
            pd.DataFrame(rows).to_csv(dest / "common_coverage.csv", index=False)
        progress["status"] = "running"
        write_json(progress_file, progress)
        jobs = [pool.submit(run_repetition, e) for e in plan["entries"] if e["repetition"] not in progress["completed"]]
        for future in as_completed(jobs):
            rep = future.result()
            progress["completed"].append(rep)
            write_json(progress_file, progress)
            print(f"Completed repeat {len(progress['completed'])}/{plan['count']} (rep {rep}; all {len(c.years)} years)", flush=True)
    tables = [pd.read_csv(dest / "repetitions" / f"rep_{r:02d}" / "estimates.csv") for r in range(plan["count"])]
    all_results = pd.concat(tables, ignore_index=True)
    all_results.to_csv(dest / "repeat_estimates.csv", index=False)
    combined = []
    for target in ["common", "native"]:
        t = median_repeats([t[t.target.eq(target)].reset_index(drop=True) for t in tables], c.alpha)
        t = t.drop(columns="repetition", errors="ignore")
        combined.append(t.assign(target=target, aggregation=plan["aggregation"]))
    pd.concat(combined, ignore_index=True).to_csv(dest / "combined_estimates.csv", index=False)
    flags = [dict(repetition=r, **evaluate(t[t.target.eq("common")])) for r, t in enumerate(tables)]
    pd.DataFrame(flags).to_csv(dest / "repeat_flags.csv", index=False)
    progress["status"] = "completed"
    write_json(progress_file, progress)
    shutil.rmtree(dest / ".scratch", ignore_errors=True)
    return progress


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["plan", "run"])
    p.add_argument("--out", type=Path, default=DEST)
    p.add_argument("--config", type=Path, default=ROOT / "observed_settings.json")
    p.add_argument("--count", type=int, default=50)
    p.add_argument("--workers", type=int, default=3)
    args = p.parse_args()
    result = make_plan(args.out, args.config, args.count) if args.command == "plan" else search(args.out, args.workers)
    print(json.dumps(result, ensure_ascii=False))
